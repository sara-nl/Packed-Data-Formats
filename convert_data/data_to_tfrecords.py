import os
from pathlib import Path
import struct
import io

import numpy as np
import tensorflow as tf
import torch
import PIL.Image

from utils_convert import ImageDataset, TARDataset, transform, collate_fn, collate_fn_encoder_info

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords_data(dataset, path, num_files=4, png_encoded=True, encoder_info=False):
    num_images = len(dataset)
    num_files = np.linspace(0, num_images, num_files+1, dtype=int)[1:]

    num_file = 0

    if encoder_info:
        collate_fn_ = collate_fn_encoder_info
    else:
        collate_fn_ = collate_fn
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=1, collate_fn=collate_fn_)

    for i,  batch in enumerate(dataloader):
        image = batch[0][0]
        label = batch[1][0]
       
        if i == 0:
            fname = path + f"/part{num_file}.tfrecords"
            tf_file = tf.io.TFRecordWriter(fname)


        if png_encoded:

            if encoder_info:
                info = batch[2][0]
                image_pil = PIL.Image.new("RGB", info["size"]) # force grayscale to RGB
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

                image_bits = io.BytesIO()

                format = image_pil.format.lower()
                if format == "png":
                    quality = "png"
                elif format == "jpeg":
                    quality = "keep"

                image_pil.save(image_bits, format=image_pil.format.lower(), quality=quality) # slight discrepancy because compression algorithm might not be same?
                image_bytes = image_bits.getvalue()
            else:
                image_pil = PIL.Image.fromarray(image)
                image_pil.save(image_bits, format="png")
            image_pil.close()

        else:
            #memfile = io.BytesIO()
            #np.save(memfile, image)
            #image_bytes = bytearray(memfile.getvalue())
            image_bytes = image.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
                "image": _bytes_feature(image_bytes),
                "label": _int64_feature(label),
            }))
        tf_file.write(example.SerializeToString())


        if i+1 == num_files[num_file]:
            tf_file.close()

            if i+1 == num_images:
                break
                
            num_file+=1
            
            fname = path + f"/part{num_file}.tfrecords"
            tf_file = tf.io.TFRecordWriter(fname)
    return


def create_index_file(tfrecord_dir, index_file):
    tf_files = list(Path(tfrecord_dir).glob("*.tfrecords"))
    outfile = open(index_file, "w")

    for tfrecord_file in tf_files:
        infile = open(tfrecord_file, "rb")


        while True:
            current = infile.tell()
            try:
                byte_len = infile.read(8)
                if len(byte_len) == 0:
                    break
                infile.read(4)
                proto_len = struct.unpack("q", byte_len)[0] # q = non-zero integer
                infile.read(proto_len)
                infile.read(4)
                outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
            except:
                print("Failed to parse TFRecord.")
                break
        infile.close()
    outfile.close()



def cifar10_to_tfrecords(num_files, png_encoded=False, encoder_info=False):
    output_path = "data/cifar10/tfrecords/"
    index_file = "data/cifar10/tfrecords/data.index"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    data_path = "data/cifar10/disk/"
    dataset = ImageDataset(data_path, encoder_info=encoder_info)
    generate_tfrecords_data(dataset, output_path, num_files=num_files, png_encoded=png_encoded, encoder_info=encoder_info)
    create_index_file(output_path, index_file)


def imagenet10k_to_tfrecords(num_files, png_encoded=False, resize=False, encoder_info=False):
    output_path = "data/imagenet10k/tfrecords/"
    index_file = "data/imagenet10k/tfrecords/data.index"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # ImageNet has various resolutions so resize to a fixed size
    if resize:
        resize_dim = 256
        transform_ = transform(resize_dim)
    else:
        transform_ = None

    data_path = "data/imagenet10k/disk/"
    dataset = ImageDataset(data_path, prefix="ILSVRC2012_val_", transform=transform_, offset_index=1, encoder_info=encoder_info)
    generate_tfrecords_data(dataset, output_path, num_files=num_files, png_encoded=png_encoded, encoder_info=encoder_info)
    create_index_file(output_path, index_file)


def ffhq_to_tfrecords(num_files, png_encoded=False, encoder_info=False):
    #output_path = "data/ffhq/hdf5/"
    output_path = "/scratch-shared/thomaso/ffhq/tfrecords/"
    index_file = "/scratch-shared/thomaso/ffhq/tfrecords/data.index"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    data_path = "/scratch-shared/thomaso/ffhq/tar/ffhq_images.tar"
    dataset = TARDataset(data_path, encoder_info=encoder_info)

    generate_tfrecords_data(dataset, output_path, num_files=num_files, png_encoded=png_encoded, encoder_info=encoder_info)
    create_index_file(output_path, index_file)

if __name__ == "__main__":
    '''
    Creates .tfrecords file(s) from a given dataset.

    1. Provide the output path to the hdf5 file and the path to the input files
    2. Choose number of files to split the data into to
    3. Create a torch dataset instance to iterate through
    4. Choose by saving the images in bytes or numpy arrays
       Converting and saving the bytes is 8 times slower but the files are 2 times smaller for images of 256x256x3
       The byte version serializes the image with lossless PNG or the original JPEG compression
    '''
    num_files = 1
    png_encoded = False
    resize = True
    encoder_info = False
    cifar10_to_tfrecords(num_files, png_encoded)
    #imagenet10k_to_tfrecords(num_files, png_encoded, resize, encoder_info)
    #ffhq_to_tfrecords(num_files, png_encoded)
