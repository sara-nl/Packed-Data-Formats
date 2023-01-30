import os
import numpy as np

from petastorm.codecs import CompressedImageCodec, \
        NdarrayCodec, ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema,\
        UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType


from utils_convert import ImageDataset, TARDataset, transform



def generate_parquet_data(dataset, output_url, rowgroup_size_mb=128, num_procs=32, num_partitions=4, save_encoded=False):
    if save_encoded:
        codec = CompressedImageCodec("png")
    else:
        codec = NdarrayCodec() # equivalent to  memfile = BytesIO(), np.save(memfile, value), bytearray(memfile.getvalue())

    image_shape = np.array(dataset[0][0]).shape


    # Create schema to store the data structurally
    MySchema = Unischema("MySchema", [
        UnischemaField("image", np.uint8,
                       image_shape, codec, False),
        UnischemaField("label", np.uint8,
                       (), ScalarCodec(IntegerType()), False),
    ])

    # Create Spark session (underlying Java)
    # .master sets the local machine (instead of spark cluster) with #num_procs being number of executors
    # .spark.driver.memory sets maximum gigabyte memory to temporarily cache files
    spark = SparkSession \
        .builder \
        .appName("Dataset creation") \
        .master(f"local[{num_procs}]") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    sc = spark.sparkContext    

    num_samples = len(dataset)


    def row_generator(i, dataset):
        sample = dataset[i]
        return {
            MySchema.image.name: sample[0],
            MySchema.label.name: sample[1],
            } 

    # Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    with materialize_dataset(spark, output_url,
                             MySchema, rowgroup_size_mb):
                             

        rows_rdd = sc.parallelize(range(num_samples)) \
            .map(lambda x: row_generator(x, dataset)) \
            .map(lambda x: dict_to_spark_row(MySchema, x))

        spark.createDataFrame(rows_rdd, 
                              MySchema.as_spark_schema()) \
            .coalesce(num_partitions) \
            .write \
            .mode("overwrite") \
            .parquet(output_url)



def generate_cifar10_parquet(num_files, save_encoded):
    rowgroup_size_mb = 128
    output_url = "file:///home/{}/CompressionProject/data/cifar10/parquet_encoded/".format(os.getenv("USER"))

    data_path = "../data/cifar10/disk/"
    dataset = ImageDataset(data_path)
    generate_parquet_data(dataset, output_url, rowgroup_size_mb, num_partitions=num_files, save_encoded=save_encoded)

def generate_imagenet_parquet(num_files, save_encoded=False, resize=True):
    rowgroup_size_mb = 128
    output_url = "file:///home/{}/CompressionProject/data/imagenet10k/parquet/".format(os.getenv("USER"))

    # Unischema requires same dimension so must be resized!
    if resize:
        resize_dim = 256
        transform_ = transform(resize_dim)
    else:
        transform_ = None

    data_path = "../data/imagenet10k/disk/"
    dataset = ImageDataset(data_path, prefix="ILSVRC2012_val_", transform=transform_, offset_index=1)
    generate_parquet_data(dataset, output_url, rowgroup_size_mb, num_partitions=num_files, save_encoded=save_encoded)

def generate_ffhq_parquet(num_files, save_encoded=False):
    rowgroup_size_mb = 256
    output_url = "file:///scratch-shared/{}/ffhq/parquet/".format(os.getenv("USER"))

    data_path = "/scratch-shared/{}/ffhq/tar/ffhq_images.tar".format(os.getenv("USER"))
    dataset = TARDataset(data_path, label_file="/scratch-shared/{}/ffhq/tar/members".format(os.getenv("USER")))
    generate_parquet_data(dataset, output_url, rowgroup_size_mb, num_partitions=num_files, save_encoded=save_encoded)

if __name__ == "__main__":
    '''
    Creates .parquet file(s) from a given dataset.

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
    # Flag to resize the samples to a common resolution
    resize = False
    #generate_cifar10_parquet(num_files, save_encoded)
    #generate_imagenet_parquet(num_files, save_encoded, resize=resize)
    generate_ffhq_parquet(num_files, save_encoded)
    