# Packed Data Formats for Pytorch
For more information please refer to: https://servicedesk.surf.nl/wiki/display/WIKI/Best+Practice+for+Data+Formats+in+Deep+Learning

A variety of packed data formats are tested on efficiency in terms of disk space, write speed and most importantly read speed in an image dataloader setting.

The packed data formats are:
- Individual png and jpeg files (baseline)
- HDF5
- LMDB
- Petastorm/Parquet
- TAR
- TFRecords
- ZIP

The datasets used for benchmarks are:
- CIFAR10
- ImageNet (10K random samples)
- FFHQ

These datasets can be found on the managed datasets on the system.

## Installation

Please find the Python packages in `requirements.txt`. Keep in mind, you might not need all of them if you plan to only use a single packed data format.
For TFRecords, a custom implementation to integrate TFRecords in PyTorch is used. Please refer to https://github.com/vahidk/tfrecord.

#### Clone repository to your own machine or download from browser
```
git clone git@github.com:sara-nl/Packed-Data-Formats.git
```

#### Change into repository directory on your machine
```
cd Packed-Data-Formats
```

#### Create virtual environment (named *venv* here)
```
python -m virtualenv venv
```

#### Activate virtual environment
```
source venv/bin/activate
```

#### Install packages from the repository's requirements
```
pip install -r requirements.txt
```

## Usage
To benchmark different datasets with a variety of settings and different locations use:
```python
python benchmark.py -l <path/to/save/data> -f <data format> -d <dataset>
```

To convert your existing data to a specific packed data format, refer to `convert_data/`. Example:
```
cd convert_data/
python data_to_hdf5.py
```



## TODO
- Benchmark on groups/buckets of files
- Support (partial) caching of the data
- Extend to tabular data
- Include memmap 
- Improve codebase
