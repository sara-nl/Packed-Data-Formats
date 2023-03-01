import os
import glob
import pickle
from pathlib import Path
import csv
import tarfile
import json
import io

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

import numpy as np
import PIL.Image
import torch
from torchvision import transforms
from torch.utils.data import get_worker_info


def transform(new_size, to_tensor=False):
    transform_list = []
    if to_tensor:
        transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Resize((new_size, new_size)))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transform_list)
    return transform


def collate_fn(batch):
    samples = np.asarray([item[0] for item in batch])
    labels = np.asarray([item[1] for item in batch])
    return samples, labels


def collate_fn_encoder_info(batch):
    samples = np.asarray([item[0] for item in batch])
    labels = np.asarray([item[1] for item in batch])
    encoder_info = np.asarray([item[2] for item in batch])
    return samples, labels, encoder_info


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        cache=False,
        transform=None,
        prefix="",
        offset_index=0,
        encoder_info=False,
    ):
        self.img_dir = Path(path)
        self.labels = []
        self.cached_images = []
        self.cache = cache
        self.transform = transform
        self.encoder_info = encoder_info
        self.info = None

        self.prefix = prefix
        self.offset_index = offset_index

        self.zfill_len = 0 if prefix == "" else 8
        index = offset_index
        img_path_example = os.path.join(
            self.img_dir, self.prefix + str(index).zfill(self.zfill_len)
        )
        img_path = list(glob.glob(img_path_example + "*"))
        assert len(img_path) == 1
        self.file_ext = os.path.splitext(img_path[0])[1]

        self.read_label_file()
        if cache:
            self.cache_images()

    def read_label_file(self):
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
            image = PIL.Image.open(
                fname
            )  # .convert("RGB") <- loses all image information like dpi, format etc.
            if self.encoder_info:
                self.info = self.fill_encoder_info(image)
            image = image.convert("RGB")

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

    @staticmethod
    def fill_encoder_info(image):
        if hasattr(image, "encoderinfo"):  # never happens? so always empty info
            info = image.encoderinfo
        else:
            info = {}
        encoder_info = {
            "mode": "RGB",
            "format": image.format,
            "progressive": info.get("progressive", False)
            or info.get("progression", False),  # progressive
            "smooth": info.get("smooth", 0),  # smooth
            "optimize": info.get("optimize", False),  # optimize
            "streamtype": info.get("streamtype", 0),  # streamtype
            "dpi": [round(x) for x in info.get("dpi", (0, 0))],  # dpi
            "layer": getattr(image, "layer", None),  # layer
            "layers": getattr(image, "layers", None),  # layers
            "quantization": getattr(image, "quantization", None),  # quantization
        }
        return encoder_info

    def __getitem__(self, index):
        label = self.labels[index]
        if self.cache:
            image = self.cached_images[index]
            if self.transform:
                image = self.transform(image)
            return image, label

        index += self.offset_index

        img_path = os.path.join(
            self.img_dir, self.prefix + str(index).zfill(self.zfill_len) + self.file_ext
        )
        image = self.read_image(img_path)
        if self.transform:
            image = self.transform(image)
        # else:
        #    image = transforms.functional.to_tensor(image)

        # else:
        #    image = image

        image = np.asarray(image)
        if self.info:
            self.info["size"] = (image.shape[1], image.shape[0])
            return image, label, self.info

        return image, label

    def __len__(self):
        return len(self.labels)


class TARDataset(torch.utils.data.Dataset):
    def __init__(
        self, path, transform=None, file_ext=None, encoder_info=False, label_file=None
    ):
        self.path = path
        self.file_ext = file_ext
        self.encoder_info = encoder_info
        self.info = None
        print(f"Initializing TAR dataset for: {self.path}")

        # assert(self.path.endswith(".tar"))

        # Tar Dataset not thread-safe so we give handle to multiple workers in the init when the workers are forked
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_handle = {worker: tarfile.open(path)}

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

        label_fname = None  # or "string"

        if len(self.label_fname) >= 1 and label_fname is not None:
            self._parse_label_file(worker, label_fname)
        else:
            self._get_all_labels()

        self.transform = transform

    def _get_all_samples(self):
        self.samples = []
        self.label_fname = []
        PIL.Image.init()
        label_exts = (".json", ".txt", ".csv")
        for m in self.members_by_name.values():
            m_name = m.name
            if m_name.lower().endswith(tuple(PIL.Image.EXTENSION.keys())):
                self.samples.append(m_name)
            elif m_name.lower().endswith(label_exts):
                self.label_fname.append(m_name)
        self.file_ext = os.path.splitext(self.samples[0])[1]

        self.length = len(self.samples)

    def _parse_label_file(self, worker, label_fname=""):
        if label_fname == "":
            if len(self.label_fname) != 1:
                print(
                    f"WARNING: found {len(self.label_fname)} label files - using only the first"
                )

            label_fname = self.label_fname[0]
        label_file = self.tar_handle[worker].extractfile(label_fname)
        labels = json.load(label_file)["labels"]
        labels = dict(sorted(labels, key=lambda item: item[1]))

        labels = [labels[fname.replace("\\", "/")] for fname in self.samples]
        self.labels = np.array(labels, dtype=np.uint8)

    def _get_all_labels(self):
        # PLACEHOLDER
        self.labels = [0] * self.length

    def _get_file(self, name):
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_handle:
            self.tar_handle[worker] = tarfile.open(self.path)

        return self.tar_handle[worker].extractfile(self.members_by_name[name])

    def _get_image(self, fname):
        f_handle = self._get_file(fname)
        jpgs = (".jpg", ".jpeg")
        if self.file_ext.lower() == ".png" and pyspng and not self.encoder_info:
            image = pyspng.load(f_handle.read())
        elif self.file_ext.lower() in jpgs and turbojpeg:
            try:
                image = turbojpeg_decoder.decode(fname.read(), pixel_format=0)
            except IOError:  # Catch jpgs which are actually encoded as PNG
                image = PIL.Image.open(fname).convert("RGB")
        else:
            image = PIL.Image.open(
                f_handle
            )  # .convert("RGB") <- loses all original image information like dpi, format etc.
            if self.encoder_info:
                self.info = self.fill_encoder_info(image)
            image = image.convert("RGB")

        return image

    def _get_label(self, index):
        return self.labels[index]  # Placeholder

    @staticmethod
    def fill_encoder_info(image):
        if hasattr(image, "encoderinfo"):
            info = image.encoderinfo
        else:
            info = {}
        encoder_info = {
            "mode": image.mode,
            "format": image.format,
            "progressive": info.get("progressive", False)
            or info.get("progression", False),  # progressive
            "smooth": info.get("smooth", 0),  # smooth
            "optimize": info.get("optimize", False),  # optimize
            "streamtype": info.get("streamtype", 0),  # streamtype
            "dpi": [round(x) for x in info.get("dpi", (0, 0))],  # dpi
            "layer": getattr(image, "layer", None),  # layer
            "layers": getattr(image, "layers", None),  # layers
            "quantization": getattr(image, "quantization", None),  # quantization
        }
        return encoder_info

    def __getitem__(self, index):
        fname_image = self.samples[index]
        image = self._get_image(fname_image)
        label = self._get_label(index)

        if self.transform:
            image = self.transform(image)
        else:
            image = np.asarray(image)

        if self.info:
            self.info["size"] = (image.shape[1], image.shape[0])
            return image, label, self.info
        return image, label

    def __len__(self):
        return self.length

    def __del__(self):
        """Clean all file handles of the workers on exit"""
        if hasattr(self, "tar_handle"):
            for o in self.tar_handle.values():
                o.close()

    def __getstate__(self):
        """Serialize without the TarFile references, for multiprocessing compatibility."""
        state = dict(self.__dict__)
        state["tar_handle"] = {}
        return state
