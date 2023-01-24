# Packed Data Formats for Pytorch

A variety of packed data formats are tested on efficiency in terms of disk space, write speed and most importantly read speed in the dataloader. 

The packed data formats are:
- Individual png and jpeg files (baseline)
- TAR
- ZIP
- LMDB
- HDF5
- Petastorm/Parquet
- TFRecords

## Installation

Please find the Python packages in `requirements.txt`. Keep in mind, you might not need all of them if you plan to only use a single packed data format.

#### Clone repository to your own machine or download from browser
```
git clone <repo>
```

#### Change into repository directory on your machine
```
cd <repo>
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

```python
python benchmark.py -l <path/to/save/data> -f <data format> -d <dataset>
```

## Links
TODO: link to knowledgebase