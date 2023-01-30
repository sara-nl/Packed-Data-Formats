import os
import time
import shutil
import distutils.dir_util

import numpy as np
import torch
import matplotlib.pyplot as plt

#from torchtext.data.functional import to_map_style_dataset

from tfrecord.torch.dataset import TFRecordDataset

from petastorm.pytorch import DataLoader as psDataLoader
from petastorm import make_reader, TransformSpec


from args import parse_args

from datasets import *
from util import TransformCV2, transform as transform_fn



def benchmark_petastorm(path, epochs, batch_size, num_workers, format, len_dataset, resize_dim=256, pin_memory=False, cache=False, warm_start=False, persistent_workers=False, shuffle=True, device="cuda"):
    """ Perform a benchmark on petastorm/parquet files by running a dataloader through a number of epochs
        Args:
            epochs (int): number of epochs (number of times of looping through the dataset)
            batch_size (int): batch_size of dataloader
            num_workers (list): number of workers (processes) for dataloader
            dataset_name (str): name of dataset
            resize_dim (int): resizing for transform function
            pin_memory (bool): pin_memory of dataloader
            cache (bool): flag for enabling caching of data (Petastorm setting)
            warm_start (bool): flag for running an epoch without timing this
            shuffle (bool): shuffle of data before dataloader
        Returns:
            dict: timings per number of worker for the epochs
        """
    print("Start experiment (GPU) on {} with: {} epochs and batch size {}".format(format, epochs, batch_size))

    results = {"format": format, "epochs": epochs, "num_workers": num_workers, "batch_size": batch_size, "time": []}

    len_dataset = np.ceil(len_dataset / batch_size)

    cache_type = "local-disk" if cache else "null"
    cache_row_size_estimate = 256 if cache else None
    cache_size_limit = 3e+10 if cache else None

    transform_fn_ = transform_fn(resize_dim, to_tensor=True)

    def _transform_row(row):
        result_row = {
            "image": transform_fn_(row["image"]),
            "label": row["label"]
        }
        return result_row

    ps_transform = TransformSpec(_transform_row)


    for num_worker in num_workers:
        if num_worker == 0:
            pool_type = "dummy"
        else:
            pool_type = "thread" #thread
            
        with psDataLoader(make_reader(path, 
                                      reader_pool_type = pool_type, 
                                      num_epochs = epochs,
                                      shuffle_rows = shuffle,
                                      shuffle_row_groups = shuffle, 
                                      transform_spec = ps_transform,
                                      workers_count = num_worker, 
                                      cache_type = cache_type,
                                      cache_size_limit = cache_size_limit,
                                      cache_row_size_estimate = cache_row_size_estimate),
                                      batch_size=batch_size) as train_loader:
            if not warm_start:
                start = time.time()
                print("Finished warm start")
            # Only calls this once but does shuffle and continues len(dataset) * num_epochs
            timer_per_epoch = time.time()
            for i, batch in enumerate(train_loader):
                images, labels = map(lambda tensor: tensor.to(device, non_blocking=pin_memory), (batch["image"], batch["label"]))
                #if (i % len_dataset) == 0 and warm_start:
                if i == len_dataset:
                    start = time.time()
                    print(f"Epoch {0} finished in {time.time() - timer_per_epoch}")

            end = time.time()
            results["time"].append(end - start)
            results["throughput"] = len_dataset * epochs / np.array(results["time"])
            print("Finish with: {} second, num_workers={}".format(end - start, num_worker))


    return results


def benchmark_gpu(dataset, epochs, batch_size, num_workers, format, persistent_workers=False, pin_memory=True, warm_start=False, shuffle=True, device="cuda"):
    """ Perform a benchmark on GPU files by running a dataloader through a number of epochs
        Args:
            epochs (int): number of epochs (number of times of looping through the dataset)
            batch_size (int): batch_size of dataloader
            num_workers (list): number of workers (processes) for dataloader
            dataset_name (str): name of dataset
            persistent_workers (bool): flag for reforking the workers on each batch call(?)
            pin_memory (bool): pin_memory of dataloader
            warm_start (bool): flag for running an epoch without timing this
            shuffle (bool): shuffle of data before dataloader
            device (str): tensors to be pushed to GPU ("cpu" or "cuda")
        Returns:
            dict: timings per number of worker for the epochs
        """
    print("Start experiment (GPU) on {} with: {} epochs and batch size {}".format(format, epochs, batch_size))

    results = {"format": format, 
               "epochs": epochs, 
               "num_workers": num_workers, 
               "batch_size": batch_size, 
               "time": [], 
               "throughput": []}

    for num_worker in num_workers:
        dataset.zipfile = None
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            num_workers=num_worker, 
            batch_size=batch_size,
            shuffle=shuffle,
            persistent_workers=num_worker > 0 and persistent_workers,
            pin_memory=pin_memory,
        )

        if not warm_start:
            start = time.time()

        for epoch in range(epochs):
            if epoch == 1 and warm_start:
                start = time.time()
            timer_per_epoch = time.time()
            for i, (images, labels) in enumerate(dataloader):
                images, labels = map(lambda tensor: tensor.to(device, non_blocking=pin_memory), (images, labels))

            print(f"Epoch {epoch} finished in {time.time() - timer_per_epoch}")
        end = time.time()
        results["time"].append(end - start)
        results["throughput"] = len(dataset) * epochs / np.array(results["time"])
        print("Finish with: {} second, num_workers={}".format(end - start, num_worker))
    return results


def benchmark_gpu_tfrecords(dataset, epochs, batch_size, num_workers, format, len_dataset, persistent_workers=False, shuffle=False, pin_memory=True, warm_start=False, device="cuda"):
    print("Start experiment {} with: {} epochs and batch size {}".format(format, epochs, batch_size))

    results = {"format": format, 
               "epochs": epochs, 
               "num_workers": num_workers, 
               "batch_size": batch_size, 
               "time": [], 
               "throughput": []}

    for num_worker in num_workers:
        #shuffle=True not possible due to iterabledataset unless converted to map style dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            num_workers=num_worker, 
            batch_size=batch_size,
            shuffle=False,
            persistent_workers=num_worker > 0 and persistent_workers,
            pin_memory=pin_memory
        )

        if not warm_start:
            start = time.time()

        for epoch in range(epochs):
            if epoch == 1 and warm_start:
                start = time.time()
            timer_per_epoch = time.time()
            for i, batch in enumerate(dataloader):
                images, labels = map(lambda tensor: tensor.to(device, non_blocking=pin_memory), (batch["image"], batch["label"]))

            print(f"Epoch {epoch} finished in {time.time() - timer_per_epoch}")
        end = time.time()
        results["time"].append(end - start)
        results["throughput"] = len_dataset * epochs / np.array(results["time"])
        print("Finish with: {} second, num_workers={}".format(end - start, num_worker))
    return results


def plot_benchmarks(results, x_label, title=None, log=False, savename=None):
    ''' Plot the result with x axis being the number of workers and y-axis the throughput '''
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    all_plots = []
    epochs = results[0]["epochs"]
    for result in results:
        temp, = plt.plot(result["num_workers"], result["throughput"], "x", label=result["format"])
        all_plots.append(temp)

    if title is None:
        title = "Benchmark for {} epochs with batch size {}".format(results[0]["epochs"], results[0]["batch_size"])
    y_label = "Throughput (images/s)"

    if log:
        plt.yscale("log")
        title = "Log " + title

    if savename is None:
        savename = "figures/{}.png".format(title)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.legend(handles=all_plots)
 
    plt.savefig(savename)
    plt.show()

def save_results_to_file(results, filename):
    ''' Save dictionary of results to a given filename '''
    str_write = ""
    with open(filename, "w") as f:
        for result in results:
            str_write += result["format"] + "\n"
            for w, t in zip(result["num_workers"], result["time"]):
                str_write += "Finished in: {} seconds with num workers: {}\n".format(t, w)
        f.write(str_write) 
    print("Results saved in {}".format(filename))


def prepare_data(data_paths, formats, location="home"):
    new_data_paths = {}
    for format in formats:

        # Check location
        format_path = data_paths[format]
        if location == "/scratch-local/":
            user = os.getenv("USER")
            scratch_id = os.getenv("SLURM_JOB_ID")
            if scratch_id:
                new_path = "/scratch-local/{}.{}/".format(user, scratch_id)
            else:
                new_path = "/scratch-local/{}/".format(user)

            print("Copying {} to {}..".format(format_path, new_path))
            format_path = copy_data_to_folder(format_path, new_path=new_path)

        elif location == "/scratch-nvme/1/":
            user = os.getenv("USER")
            new_path = "/scratch-nvme/1/" + user
            print("Copying {} to {}..".format(format_path, new_path))
            format_path = copy_data_to_folder(format_path, new_path=new_path)

        

        if format == "Petastorm":
            if location == "home":
                format_path = "file://" + os.getcwd() + "/" + format_path
            else:
                format_path = "file:///" + format_path
        
        new_data_paths[format] = format_path

    return new_data_paths

def copy_data_to_folder(old_path, new_path):
    if isinstance(old_path, list):
        new_path_out = []
        for p in old_path:
            new_path_out.append(copy_data_to_folder(p, new_path))
        return new_path_out

    parent_dir = os.path.join(new_path, os.path.dirname(old_path))
    new_path_out = os.path.join(new_path, old_path)
    if not os.path.exists(new_path_out):
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.isdir(old_path):
            distutils.dir_util.copy_tree(old_path, new_path_out)
        else:
            shutil.copy2(old_path, new_path_out)
    return new_path_out


def run_benchmarks(dataset_name, data_paths, epochs, batch_size, num_workers, cache, load_encoded, transform, **dataloader_kwargs):
    results = []
    if dataset_name == "CIFAR10":
        orig_dim = 32
        resize_dim = 32 # Keep original resolution
    elif dataset_name == "ImageNet10k":
        orig_dim = 256
        resize_dim = 256
    elif dataset_name == "FFHQ":
        orig_dim = 1024
        resize_dim = 256 

    # Add your dataset length
    lengths = {"CIFAR10": 50000,
               "ImageNet10k": 10000,
               "FFHQ": 70000}


    # TODO: maybe nicer to put the to_tensor() in the else here...
    for format, path in data_paths.items():
        transform_fn_ = transform_fn(resize_dim, to_tensor=True) if transform else None
        len_dataset = lengths[dataset_name]

        if format == "Image":
            if dataset_name == "ImageNet10k":
                dataset = ImageDataset(path, transform=transform_fn_, cache=cache, prefix="ILSVRC2012_val_", offset_index=1)
            else:
                dataset = ImageDataset(path, transform=transform_fn_, cache=cache)
        elif format == "ZIP":
            dataset = ZIPDataset(path, cache=cache, transform=transform_fn_, load_encoded=load_encoded)
        elif format == "TAR":
            dataset = TARDataset(path, cache=cache, transform=transform_fn_, load_encoded=load_encoded)
        elif format == "HDF5":
            dataset = H5Dataset(path, cache=cache, transform=transform_fn_, load_encoded=load_encoded)
        elif format == "LMDB":
            dataset = LMDBDataset(path, cache=cache, transform=transform_fn_, load_encoded=load_encoded)
        elif format == "TFRecords":
            description = {"image": "byte", "label": "int"}
            if transform:
                resize_dim = resize_dim
            else:
                resize_dim = None
            transform_fn_ = TransformCV2(batch_size, orig_dim=orig_dim, resize_dim=resize_dim)
            if dataloader_kwargs["shuffle"] == True:
                dataset = TFRecordDataset(path[0], path[1], description, transform=transform_fn_, shuffle_queue_size=16)
            else:
                dataset = TFRecordDataset(path[0], path[1], description, transform=transform_fn_)
            dataloader_kwargs["shuffle"] = False
        
        if format == "Petastorm":
            results_dataset = benchmark_petastorm(path, epochs, batch_size, num_workers, format, len_dataset, resize_dim=resize_dim, **dataloader_kwargs)
        elif format == "TFRecords":
            results_dataset = benchmark_gpu_tfrecords(dataset, epochs, batch_size, num_workers, format, len_dataset, **dataloader_kwargs)
        else:
            results_dataset = benchmark_gpu(dataset, epochs, batch_size, num_workers, format, **dataloader_kwargs)

        results.append(results_dataset)


    return results



def main():
    args = parse_args()
    for arg in vars(args):
        print("-- {} : {}".format(arg, getattr(args, arg)))

    dataset = args.dataset
    device = args.device
    cache = bool(args.cache)
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    persistent_workers = args.persistent_workers
    transform = args.transform
    load_encoded = args.load_encoded
    
    # Hardcoded data prefix location where data is stored
    prefix = "data/"
    #prefix = "/scratch-shared/{}/".format(os.getenv("USER"))
    dataset_path = dataset.lower()

    data_path_image = f"{prefix}/{dataset_path}/disk/"
    data_path_h5 = f"{prefix}/{dataset_path}/hdf5/part0.h5"
    data_path_lmdb = f"{prefix}/{dataset_path}/lmdb/part0.lmdb"
    data_path_zip = f"{prefix}/{dataset_path}/zip/part0.zip"
    data_path_tar = f"{prefix}/{dataset_path}/tar/part0.tar"
    data_path_petastorm = f"{prefix}/{dataset_path}/parquet/"
    data_path_tfrecords = f"{prefix}/{dataset_path}/tfrecords/part0.tfrecords"
    index_path = f"{prefix}/{dataset_path}/tfrecords/data.index"


    data_paths = {"Image": data_path_image, 
                  "ZIP": data_path_zip, 
                  "TAR": data_path_tar,
                  "HDF5": data_path_h5, 
                  "LMDB": data_path_lmdb, 
                  "Petastorm": data_path_petastorm, 
                  "TFRecords": [data_path_tfrecords, index_path]
                  }


    dataloader_kwargs = {"device": device,
                         "persistent_workers": persistent_workers,
                         "warm_start": bool(args.warm_start),
                         "pin_memory": bool(args.pin_memory),
                         "shuffle": bool(args.shuffle),
                        }


    # Copy files to system monitor if necessary and prepare data paths
    data_paths = prepare_data(data_paths, args.format, args.location)
    results = run_benchmarks(dataset, data_paths, epochs, batch_size, num_workers, cache, load_encoded, transform, **dataloader_kwargs)


    x_label = "Number of processes"

    title = "results/{}_e{}_bs{}_c{}_ws{}_pm{}_pw{}_t{}".format(dataset, epochs, batch_size, int(cache), args.warm_start, args.pin_memory, persistent_workers, args.transform)
    log_file = title + ".txt"
    save_results_to_file(results, log_file)


    plot_benchmarks(results, x_label, log=False)


if __name__ == "__main__":
    main()


    
