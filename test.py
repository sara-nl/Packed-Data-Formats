import zipfile 
import PIL.Image
import pyspng


img_path = "data/cifar10/disk/0/png"
zip_path = "data/cifar10/zip_encoded/part0.zip"

zip_file = zipfile.ZipFile(zip_path)
zip_names = zip_file.namelist()
zip_ex = zip_names[0]

with zip_file.open(zip_ex, "r") as f:
    img_py = pyspng.load(f.read())
    print(img_py.shape)

zip_ex_f = zip_file.open(zip_ex)
img_py = pyspng.load(zip_ex_f.read())
print(img_py.shape)