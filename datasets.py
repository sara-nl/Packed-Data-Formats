# Standard libraries
from pathlib import Path
import glob
import csv
import os
import json
import io
import pickle
import PIL.Image
import zipfile
import tarfile

# For opening packed data formats
import h5py
import lmdb

# Libraries for fast image reading [OPTIONAL]
try:
    import pyspng
except ImportError:
    pyspng = None

try:
    import turbojpeg
except ImportError:
    turbojpeg = None

if turbojpeg:
    try:
        turbojpeg_path = "/opt/TurboVNC/java/libturbojpeg.so"
        # Test
        turbojpeg_decoder = turbojpeg.TurboJPEG(turbojpeg_path)
    except OSError:
        print("turboJPEG was installed but not found. Continuing without")

# Deep Learning
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for PNG/JPEG to pass to a PyTorch dataloader

    Args:
        path (str): location where the images are stored on disk
        transform (obj): torchvision.transforms object or None
        prefix (str): only for ImageNet we need custom prefix ILSVRC2012_val_
        offset_index (int): only for ImageNet we need offset (000000001.jpeg => offset_index is 8)

    Returns:
        torch Dataset: to pass to a dataloader
    """

    def __init__(self, path, cache=False, transform=None, prefix="", offset_index=0):
        print("Initializing Image dataset with path: ", path)
        self.img_dir = Path(path)
        self.labels = []
        self.cached_images = []
        self.cache = cache
        self.transform = transform

        # For ImageNet we need a custom prefix (ILSVRC2012_val_)
        self.prefix = prefix

        # For ImageNet: we have format of 00000001.jpeg (fill to 8)
        self.zfill_len = 0 if prefix == "" else 8
        self.offset_index = offset_index

        # Find image extension
        index = offset_index
        img_path_example = os.path.join(
            self.img_dir, self.prefix + str(index).zfill(self.zfill_len)
        )
        img_path = list(glob.glob(img_path_example + "*"))
        assert len(img_path) == 1
        self.file_ext = os.path.splitext(img_path[0])[1]

        # Read all labels and store them in the class instance
        self.read_label_file()
        if cache:
            self.cache_images()

    def read_label_file(self):
        """
        Read label function
        Assumes a single label file containing the ground-truth labels for all samples in txt or csv
        """
        label_file = list(self.img_dir.glob("*.csv")) + list(self.img_dir.glob("*.txt"))
        assert len(label_file) == 1
        label_file = label_file[0]

        with open(label_file, "r") as csvfile:
            reader = csv.reader(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            for row in reader:
                self.labels.append(int(row[0]))

    def read_image(self, img_path):
        """
        Read image and return image
        img_path: path where image is located on disk
        """
        fname = open(img_path, "rb")
        jpgs = (".jpg", ".jpeg")
        if self.file_ext == ".png" and pyspng and not self.encoder_info:
            image = pyspng.load(fname.read())
        elif self.file_ext.lower() in jpgs and turbojpeg:
            try:
                image = turbojpeg_decoder.decode(fname.read(), pixel_format=0)
            except IOError:  # Catch jpgs which are actually encoded as PNG
                image = PIL.Image.open(fname).convert("RGB")
        else:
            image = PIL.Image.open(img_path).convert("RGB")
        return image

    def cache_images(self):
        for index in range(len(self.labels)):
            index += self.offset_index
            img_path = os.path.join(
                self.img_dir,
                self.prefix + str(index).zfill(self.zfill_len) + self.file_ext,
            )
            image = self.read_image(img_path)
            self.cached_images.append(image)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.cache:
            image = self.cached_images[index]
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)

            return image, label

        index += self.offset_index
        img_path = os.path.join(
            self.img_dir, self.prefix + str(index).zfill(self.zfill_len) + self.file_ext
        )

        image = self.read_image(img_path)

        if self.transform:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        return image, label

    def __len__(self):
        return len(self.labels)


# TODO:
# - Multiple .h5 files of data
# - (Partially) cached data
class H5Dataset(torch.utils.data.Dataset):
    """Dataset for packed HDF5/H5 files to pass to a PyTorch dataloader

    Args:
        path (str): location where the images are stored on disk
        transform (obj): torchvision.transforms object or None
        load_encoded (bool): whether the images within the .h5 file are encoded or saved as bytes directly
    Returns:
        torch Dataset: to pass to a dataloader
    """

    def __init__(self, path, cache=False, transform=None, load_encoded=False):
        print("Initializing HDF5 dataset with path: ", path)
        self.file_path = path
        self.cache = cache
        self.transform = transform
        self.load_encoded = load_encoded

        # Hardcoded key, value pair within the .h5 files
        self.h5_key_samples = "images"
        self.h5_key_labels = "labels"

        self.labels = []
        self.dataset = None

        # Initial check (or for caching)
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file["labels"])
            if cache:
                self.cached_images = list(file["images"])
                self.cached_labels = list(file["labels"])

    def __getitem__(self, index):
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
        # Why fill dataset at getitem rather than init?
        # Creates dataset first time getitem is called

        if self.cache:
            image = self.cached_images[index]
            label = self.cached_labels[index]
            if self.load_encoded:
                image = PIL.Image.open(io.BytesIO(image))
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)
            return image, label

        # Each worker (which are forked after the init) need to have their own file handle
        if self.dataset is None:
            data = h5py.File(self.file_path, "r")
            self.dataset = data.get(self.h5_key_samples)
            self.labels = data.get(self.h5_key_labels)

        image = self.dataset[index]
        if self.load_encoded:
            image = PIL.Image.open(io.BytesIO(image))

        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        return image, label

    def __len__(self):
        return self.dataset_len


class LMDBDataset(torch.utils.data.Dataset):
    """Dataset for packed LMDB files to pass to a PyTorch dataloader

    Args:
        path (str): location where the images are stored on disk
        transform (obj): torchvision.transforms object or None
        load_encoded (bool): whether the images within the .h5 file are encoded or saved as bytes directly
    Returns:
        torch Dataset: to pass to a dataloader
    """

    def __init__(self, path, cache=False, transform=None, load_encoded=False):
        print("Initializing LMDB dataset with path: ", path)
        self.path = path
        self.cache = cache
        self.transform = transform
        self.load_encoded = load_encoded

        self.cached_images = []
        self.cached_labels = []
        self.env = None
        self.txn = None

        # Is this necessary?
        self.env = lmdb.open(
            self.path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Extract all keys
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            if cache:
                for key in txn.cursor().iternext(keys=True, values=False):
                    image, label = pickle.loads(txn.get(key))
                    self.cached_images.append(image)
                    self.cached_labels.append(label)

            self.keys = [
                k for k in txn.cursor().iternext(keys=True, values=False)
            ]  # https://github.com/jnwatson/py-lmdb/issues/195

    def __getitem__(self, index):
        if self.cache:
            image = self.cached_images[index]
            label = self.cached_labels[index]

            if self.load_encoded:
                image = PIL.Image.open(io.BytesIO(image))
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)
            return image, label

        # Each worker (which are forked after the init) need to have their own file handle
        if self.txn is None:
            self.txn = self.env.begin()

        # Load from LMDB
        image, label = pickle.loads(self.txn.get(self.keys[index]))

        if self.load_encoded:
            image = PIL.Image.open(io.BytesIO(image))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image)

        return image, label

    def __len__(self):
        return self.length


class ZIPDataset(torch.utils.data.Dataset):
    """Dataset for packed ZIP to pass to a PyTorch dataloader

    Args:
        path (str): location where the images are stored on disk
        transform (obj): torchvision.transforms object or None
        load_encoded (bool): whether the images within the .zip file are encoded or saved as bytes directly
    Returns:
        torch Dataset: to pass to a dataloader
    """

    def __init__(self, path, cache=False, transform=None, load_encoded=False):
        print(f"Initializing ZIP dataset for: {path}")
        self.path = path
        self.transform = transform
        self.load_encoded = load_encoded

        # Each worker needs to get the keys of the ZIP file
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        self.zip_handle = {worker: zipfile.ZipFile(path)}
        self.members = sorted(self.zip_handle[worker].namelist())

        # Retrieve samples in list and label files in list
        self._get_all_samples()

        label_fname = ""  # Or None. If None then no label file is there
        if len(self.label_fname) >= 1 and label_fname is not None:
            self._parse_label_file(worker)
        else:
            self._get_filler_labels()

    def _get_all_samples(self, label_exts=(".json")):
        """Sort labels and images from the names
        Args:
            label_exts: the file extensions which will be categorized as labels
        Returns:
            None: filled lists of self.samples and self.label_fname
        """
        self.samples = []
        self.label_fname = []
        PIL.Image.init()

        for m in self.members:
            if m.lower().endswith(tuple(PIL.Image.EXTENSION.keys())):
                self.samples.append(m)
            elif m.lower().endswith(label_exts):
                self.label_fname.append(m)
        self.file_ext = os.path.splitext(self.samples[0])[1]

        self.length = len(self.samples)

    def _parse_label_file(self, worker, label_fname=""):
        """Sort labels and images from the names
        Args:
            worker (int): worker id
            label_fname (str): hardcoded label file name
        Returns:
            None: parsed labels
        """
        # If no hardcoded label filename is given, use the first one from the list by _get_all_samples()
        if label_fname == "":
            if len(self.label_fname) != 1:
                print(
                    f"WARNING: found {len(self.label_fname)} label files - using only the first"
                )

            label_fname = self.label_fname[0]

        # Parse labels from json file
        label_file = self.zip_handle[worker].open(label_fname, "r")
        labels = json.load(label_file)["labels"]
        labels = dict(sorted(labels, key=lambda item: item[1]))
        labels = [labels[fname.replace("\\", "/")] for fname in self.samples]
        self.labels = np.array(labels, dtype=np.uint8)

    def _get_filler_labels(self):
        self.labels = [0] * self.length

    def _get_file(self, fname):
        """Retrieve file handle for a given file name within the ZIP file"""
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.zip_handle:
            self.zip_handle[worker] = zipfile.ZipFile(self.path)

        return self.zip_handle[worker].open(fname, "r")

    def _get_image(self, fname):
        fname = self._get_file(fname)
        jpgs = (".jpg", ".jpeg")
        if not self.load_encoded:
            # In case of having the image saved as bytes:
            image = np.frombuffer(fname.read(), dtype=np.uint8).reshape(256, 256, 3)
        else:
            if self.file_ext.lower() == ".png" and pyspng:
                image = pyspng.load(fname.read())
            elif self.file_ext.lower() in jpgs and turbojpeg:
                try:
                    image = turbojpeg_decoder.decode(fname.read(), pixel_format=0)
                except IOError:  # Catch jpgs which are actually encoded as PNG
                    image = PIL.Image.open(fname).convert("RGB")
            else:
                image = PIL.Image.open(fname.read()).convert("RGB")

        return image

    def _get_label(self, index):
        return self.labels[index]

    def __getitem__(self, index):
        fname_image = self.samples[index]
        image = self._get_image(fname_image)
        label = self._get_label(index)

        if self.transform:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        return image, label

    def __len__(self):
        return self.length

    def __del__(self):
        """Clean all file handles of the workers on exit"""
        for o in self.zip_handle.values():
            o.close()

    def __getstate__(self):
        """Serialize without the ZipFile references, for multiprocessing compatibility"""
        state = dict(self.__dict__)
        state["zip_handle"] = {}
        return state


class TARDataset(torch.utils.data.Dataset):
    """Dataset for packed ZIP to pass to a PyTorch dataloader

    Args:
        path (str): location where the images are stored on disk
        transform (obj): torchvision.transforms object or None
        load_encoded (bool): whether the images within the .tar file are encoded or saved as bytes directly
        label_file (str): path to save the cached getmembers() output as this may take a while for larger dataset
    Returns:
        torch Dataset: to pass to a dataloader
    """

    def __init__(self, path, transform=None, load_encoded=False, label_file=None):
        print(f"Initializing TAR dataset for: {path}")
        self.path = path
        self.transform = transform
        self.load_encoded = load_encoded

        # First uncompress because .gz cannot be read in parallel
        if self.path.endswith(".tar.gz"):
            print(
                "WARNING: tar file is compressed which drastically impacts performance -> gunzip -k <file>"
            )

        # TAR is not thread-safe so give file handle to each worker
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        self.tar_handle = {worker: tarfile.open(path)}

        # Store headers of all files and folders by name
        # self.members = sorted(self.tar_handle[worker].getmembers(), key=lambda m: m.name)

        # store headers of all files and folders by name
        if label_file:
            with open(label_file, "rb") as fp:
                self.members_by_name = pickle.load(fp)

        else:
            # get.members() takes very long for larger TAR archives so cache the members in a byte file
            self.members_by_name = {
                m.name: m
                for m in sorted(
                    self.tar_handle[worker].getmembers(), key=lambda m: m.name
                )
            }
            with open(os.path.join(Path(path).parent, "members"), "wb") as fp:
                pickle.dump(self.members_by_name, fp)
            print(
                f"Finished create a members file. Please add the following path to the init as label_file next time: ",
                Path(path).parent + "members",
            )

        self._get_all_samples()
        label_fname = None  # or "String"

        if len(self.label_fname) >= 1 and label_fname is not None:
            self._parse_label_file(worker, label_fname)
        else:
            self._get_filler_labels()

    def _get_all_samples(self, label_exts=(".json")):
        """Sort labels and images from the names
        Args:
            label_exts: the file extensions which will be categorized as labels
        Returns:
            None: filled lists of self.samples and self.label_fname
        """
        self.members_by_name = {m.name: m for m in self.members}
        self.samples = []
        self.label_fname = []
        PIL.Image.init()

        for m in self.members_by_name.values():
            m_name = m.name
            if m_name.lower().endswith(tuple(PIL.Image.EXTENSION.keys())):
                self.samples.append(m_name)
            elif m_name.lower().endswith(label_exts):
                self.label_fname.append(m_name)
        self.file_ext = os.path.splitext(self.samples[0])[1]

        self.length = len(self.samples)

    def _parse_label_file(self, worker, label_fname=""):
        """Sort labels and images from the names
        Args:
            worker (int): worker id
            label_fname (str): hardcoded label file name
        Returns:
            None: parsed labels
        """
        # If no hardcoded label filename is given, use the first one from the list by _get_all_samples()
        if label_fname == "":
            if len(self.label_fname) != 1:
                print(
                    f"WARNING: found {len(self.label_fname)} label files - using only the first"
                )

            label_fname = self.label_fname[0]

        # Parse labels from json file
        label_file = self.tar_handle[worker].extractfile(label_fname)
        labels = json.load(label_file)["labels"]
        labels = dict(sorted(labels, key=lambda item: item[1]))

        labels = [labels[fname.replace("\\", "/")] for fname in self.samples]
        self.labels = np.array(labels, dtype=np.uint8)

    def _get_filler_labels(self):
        # Placeholder for datasets with no labels like FFHQ
        self.labels = [0] * self.length

    def _get_file(self, fname):
        """Retrieve file handle for a given file name within the ZIP file"""
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_handle:
            self.tar_handle[worker] = tarfile.open(self.path)

        return self.tar_handle[worker].extractfile(self.members_by_name[fname])

    def _get_image(self, fname):
        fname = self._get_file(fname)
        jpgs = (".jpg", ".jpeg")
        if not self.load_encoded:
            # In case of having the image saved as bytes:
            image = np.frombuffer(fname.read(), dtype=np.uint8).reshape(256, 256, 3)
        else:
            if self.file_ext.lower() == ".png" and pyspng:
                image = pyspng.load(fname.read())
            elif self.file_ext.lower() in jpgs and turbojpeg:
                try:
                    image = turbojpeg_decoder.decode(fname.read(), pixel_format=0)
                except IOError:  # Catch jpgs which are actually encoded as PNG
                    image = PIL.Image.open(fname).convert("RGB")
            else:
                image = PIL.Image.open(fname.read()).convert("RGB")

        return image

    def _get_label(self, index):
        return self.labels[index]

    def __getitem__(self, index):
        fname_image = self.samples[index]
        image = self._get_image(fname_image)
        label = self._get_label(index)

        if self.transform:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        return image, label

    def __len__(self):
        return self.length

    def __del__(self):
        """Clean all file handles of the workers on exit"""
        if hasattr(self, "tar_handle"):
            for o in self.tar_handle.values():
                o.close()

    def __getstate__(self):
        """Serialize without the ZipFile references, for multiprocessing compatibility"""
        state = dict(self.__dict__)
        state["tar_handle"] = {}
        return state
