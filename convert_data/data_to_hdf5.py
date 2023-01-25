import os
from pathlib import Path
import io
import os

import numpy as np
import h5py
import torch
import PIL.Image

from utils_convert import ImageDataset, TARDataset, transform, collate_fn, collate_fn_encoder_info


def generate_hdf5_data_bytes(dataset, path, encoder_info=True, num_files=4):
    num_images = len(dataset)
    num_files = np.linspace(0, num_images, num_files+1, dtype=int)[1:]
    num_file = 0

    if encoder_info:
        collate_fn_ = collate_fn_encoder_info
    else:
        collate_fn_ = collate_fn

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=1, collate_fn=collate_fn_)
    for i, batch in enumerate(dataloader):
        if i == 0:
            file = h5py.File(path + f"/part{num_file}.h5", "w")
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            # Create a dataset in the file
            div = num_files[0]
            im_dataset = file.create_dataset("images",  (div, ),  dtype=dt)
            label_dataset = file.create_dataset("labels", (div, ), dtype=np.uint8)

        image = batch[0][0]
        
        image_bits = io.BytesIO()
        if encoder_info:
            info = batch[2][0]
            image_pil = PIL.Image.new("RGB", info["size"]) # force mode to RGB
            image_pil.putdata(PIL.Image.fromarray(image).getdata())
            image_pil.format = info["format"]
            image_pil.progressive = info["progressive"]
            image_pil.progression = info["progressive"]
            image_pil.smooth = info["smooth"]
            image_pil.optimize = info["optimize"]
            image_pil.streamtype = info["streamtype"] 
            image_pil.dpi = info["dpi"]
            image_pil.layer = info["layer"]
            image_pil.layers = info["layers"]
            image_pil.quantization = info["quantization"]

            


            format = image_pil.format.lower()
            if format == "png":
                quality = "png"
            elif format == "jpeg":
                quality = "keep"

            image_pil.save(image_bits, format=image_pil.format.lower(), quality=quality) # slight discrepancy because compression algorithm might not be same?

        else:
            image_pil = PIL.Image.fromarray(image)
            image_pil.save(image_bits, format="png")

        image_byte = image_bits.getvalue()
        image_pil.close()

        index = i % div 


        im_dataset[index] = np.frombuffer(image_byte, dtype="uint8")
        label = batch[1][0].astype(np.uint8)
        label_dataset[index] = label

        if i+1 == num_files[num_file]:
            file.close()

            if i+1 == num_images:
                break
                
            num_file+=1

            # Create a new HDF5 file
            file = h5py.File(path + f"/part{num_file}.h5", "w")

            # Create a dataset in the file
            im_dataset = file.create_dataset("images",  (div, ),  dtype=dt)
            label_dataset = file.create_dataset("labels", (div, ), dtype=np.uint8)
    return

def generate_hdf5_data(dataset, path, num_files=4):

    num_images = len(dataset)
    num_files = np.linspace(0, num_images, num_files+1, dtype=int)[1:]

    num_file = 0

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=1, collate_fn=collate_fn)

    for i, (image, label) in enumerate(dataloader):
        image = image[0]
        if i == 0:
            file = h5py.File(path + f"/part{num_file}.h5", "w")
            # Create a dataset in the file
            div = num_files[0]
            im_dataset = file.create_dataset("images",  (div, *image.shape),  dtype=np.uint8)
            label_dataset = file.create_dataset("labels", (div, ), dtype=np.uint8)



        index = i % div 


        im_dataset[index] = image
        label = label[0].astype(np.uint8)
        label_dataset[index] = label

        if i+1 == num_files[num_file]:
            file.close()

            if i+1 == num_images:
                break
                
            num_file+=1

            # Create a new HDF5 file
            file = h5py.File(path + f"/part{num_file}.h5", "w")

            # Create a dataset in the file
            im_dataset = file.create_dataset("images",  np.shape(image),  dtype=np.uint8)
            label_dataset = file.create_dataset("labels", (div, ), dtype=np.uint8)
    return

def cifar10_to_hdf5(num_files=1, save_encoded=False, encoder_info=False):
    output_path = "data/cifar10/hdf5/"
    Path(output_path).mkdir(parents=True, exist_ok=True)


    data_path = "data/cifar10/disk/"
    dataset = ImageDataset(data_path, encoder_info=encoder_info)

    if save_encoded:
        generate_hdf5_data_bytes(dataset, output_path, num_files=num_files, encoder_info=encoder_info)

    else:
        generate_hdf5_data(dataset, output_path, num_files=num_files)

def imagenet10k_to_hdf5(num_files=1, save_encoded=False, resize=True, encoder_info=False):
    output_path = "..data/imagenet10k/hdf5/"
    Path(output_path).mkdir(parents=True, exist_ok=True)


    if resize:
        resize_dim = 256
        transform_ = transform(resize_dim)
    else:
        transform_ = None

    data_path = "..data/imagenet10k/disk/"
    dataset = ImageDataset(data_path, prefix="ILSVRC2012_val_", transform=transform_, offset_index=1, encoder_info=encoder_info)

    if save_encoded:
        generate_hdf5_data_bytes(dataset, output_path, num_files=num_files, encoder_info=encoder_info)
    else:
        generate_hdf5_data(dataset, output_path, num_files=num_files)


def ffhq_to_hdf5(num_files=1, save_encoded=False, encoder_info=False):
    output_path = "/scratch-shared/{}/ffhq/hdf5".format(os.getenv("USER"))
    Path(output_path).mkdir(parents=True, exist_ok=True)

    data_path = "/scratch-shared/{}/ffhq/tar/ffhq_images.tar".format(os.getenv("USER"))
    dataset = TARDataset(data_path, encoder_info=encoder_info, label_file="/scratch-shared/{}/ffhq/tar/members".format(os.getenv("USER")))

    if save_encoded:
        generate_hdf5_data_bytes(dataset, output_path, num_files=num_files, encoder_info=encoder_info)
    else:
        generate_hdf5_data(dataset, output_path, num_files=num_files)

if __name__ == "__main__":
    '''
    Creates .h5 file(s) from a given dataset.

    1. Provide the output path to the hdf5 file and the path to the input files
    2. Choose number of files to split the data into to
    3. Create a torch dataset instance to iterate through
    4. Choose by saving the images in bytes or numpy arrays
       Converting and saving the bytes is 8 times slower but the files are 2 times smaller for images of 256x256x3
       The byte version serializes the image with lossless PNG or the original JPEG compression
    '''

    # Number of partitions/shard/files to subdivide the dataset into
    num_files = 1
    # Flag to save as bytes or H5 arrays
    save_encoded = False
    resize = False # resize must be true for HDF5 for ImageNet10k
    encoder_info = False
    cifar10_to_hdf5(num_files, save_encoded, encoder_info=encoder_info)
    #imagenet10k_to_hdf5(num_files, save_encoded, encoder_info=encoder_info)
    #ffhq_to_hdf5(num_files, save_encoded, encoder_info)