from pathlib import Path
import io
import os

import numpy as np 

import pickle # Python 3 automatically uses pickle C accelerator

import lmdb
import torch
import PIL.Image

from utils_convert import ImageDataset, TARDataset, transform, collate_fn, collate_fn_encoder_info


def generate_lmdb_data(dataset, path, num_files=4, save_encoded=False, encoder_info=False):
    num_images = len(dataset)

    num_files = np.linspace(0, num_images, num_files+1, dtype=int)[1:]
    num_file = 0

    num_images_per_part = num_files[0]
    if encoder_info:
        collate_fn_ = collate_fn_encoder_info
    else:
        collate_fn_ = collate_fn
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=1, collate_fn=collate_fn_)

    for i, batch in enumerate(dataloader):
        
        image, label = batch[0][0], batch[1][0]
        if i == 0:
            # We need to set the map size prior
            # Create a new LMDB DB for all the images, assumes all the same resolution or you have to find the largest file     
            map_size = num_images_per_part * image.nbytes * 10

            env = lmdb.open(path + f"/part{num_file}.lmdb", map_size=map_size, subdir=False)

        with env.begin(write=True) as txn: #TODO: this takes much time?
            key = f"{i:08}"
            if save_encoded:
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
                txn.put(key.encode("ascii"), pickle.dumps((image_byte, label)))
                
            else:
                #memfile = io.BytesIO()
                #np.save(memfile, image)
                #image_bytes = bytearray(memfile.getvalue())
                #image_bytes = image.tobytes()
                txn.put(key.encode("ascii"), pickle.dumps((image, label)))

        if i+1 == num_files[num_file]:
            env.close()

            if i+1 == num_images:
                break   

            num_file+=1
            env = lmdb.open(path + f"/part{num_file}.lmdb", map_size=map_size, subdir=False)
    env.close()
    return

def cifar10_to_lmdb(num_files, save_encoded, encoder_info=False):
    output_path = "../data/cifar10/lmdb"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    data_path = "../data/cifar10/disk/"
    dataset = ImageDataset(data_path, encoder_info=encoder_info)
    generate_lmdb_data(dataset, output_path, num_files=num_files, save_encoded=save_encoded, encoder_info=encoder_info)


def imagenet10k_to_lmdb(num_files, save_encoded=False, resize=True, encoder_info=False):
    output_path = "../data/imagenet10k/lmdb"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # ImageNet has various resolutions so resize to a fixed size
    if resize:
        resize_dim = 256
        transform_ = transform(resize_dim)
    else:
        transform_ = None

    data_path = "../data/imagenet10k/disk/"
    dataset = ImageDataset(data_path, prefix="ILSVRC2012_val_", transform=transform_, offset_index=1, encoder_info=encoder_info)
    generate_lmdb_data(dataset, output_path, num_files=num_files, save_encoded=save_encoded, encoder_info=encoder_info)


def ffhq_to_lmdb(num_files, save_encoded=False, encoder_info=False):
    output_path = "/scratch-shared/{}/ffhq/lmdb".format(os.getenv("USER"))
    Path(output_path).mkdir(parents=True, exist_ok=True)

    data_path = "/scratch-shared/{}/ffhq/tar/ffhq_images.tar".format(os.getenv("USER"))
    dataset = TARDataset(data_path, encoder_info=encoder_info, label_file="/scratch-shared/{}/ffhq/tar/members".format(os.getenv("USER")))

    generate_lmdb_data(dataset, output_path, num_files=num_files, save_encoded=save_encoded, encoder_info=encoder_info)


if __name__ == "__main__":  
    '''
    Creates a SINGLE lmdb folder with .mdb files from a given dataset.

    1. Provide the output path to the hdf5 file and the path to the input files
    2. Choose number of files to split the data into to
    3. Create a torch dataset instance to iterate through
    4. Choose by saving the images in bytes or numpy arrays
       Converting and saving the bytes is 8 times slower but the files are 2 times smaller for images of 256x256x3
       The byte version serializes the image with lossless PNG compression
    '''
    # Number of partitions/shard/files to subdivide the dataset into
    num_files = 1
    # Flag to save as bytes or H5 arrays
    save_encoded = False
    # Flag to use the original encoding
    encoder_info = False
    # Flag to resize the samples to a common resolution
    resize = False
    #ifar10_to_lmdb(num_files, save_encoded, encoder_info=encoder_info)
    #imagenet10k_to_lmdb(num_files, save_encoded, encoder_info=encoder_info)
    #ffhq_to_lmdb(num_files, save_encoded, encoder_info)

        