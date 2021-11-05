# MIT License
# Copyright (c) 2021 
# Earthquake Engineering and Structural Dynamics (EESD), EPFL 

import urllib
import zipfile
import os

# create a folder for our data
up_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(up_dir, 'dataset')
os.mkdir(data_dir)

zip_path=os.path.join(data_dir, 'DIC_crack_dataset.zip')
data = urllib.request.urlretrieve('https://zenodo.org/record/4307686/files/DIC_crack_dataset.zip?download=1', zip_path)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

os.remove(zip_path)