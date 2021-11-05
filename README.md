# Deep Crack Segmentation

This repo contains the source codes for segmenting cracks on DIC images presented in [Rezaie et al.](https://doi.org/10.1016/j.conbuildmat.2020.120474)

Link to the publication: [click here](https://doi.org/10.1016/j.conbuildmat.2020.120474)

Link to the crack dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4307686.svg)](https://doi.org/10.5281/zenodo.4307686)


# How to use it?

## 1. Clone repository

All necessary data and codes are inside the ``src`` directory. 

## 2. Install Conda or Miniconda

Link to install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html

Link to install miniconda: https://docs.conda.io/en/latest/miniconda.html

## 3. Create a conda environment 


Run the following commands in your terminal to install and activate the environment.

``bash
conda env create -f environment.yml
``

``bash
conda activate deepcrack
``

## 4. Download the crack dataset

To download and create the dataset directory containing train/validation/test images run the following python script:

```bash
    python download_dataset.py
```

This will create a new directory called ```dataset```. 


## 5. Train a model

To train a deep model you can run the following command:
```bash
    python run.py --model_type=TernausNet16 --lr=2e-4 --weight_decay=0 --num_epochs=100 --pretrained=1  --batch_size=1
```


# Citation

If you find this implementation useful, please cite us as:

```
@article{REZAIE2020120474,
title = {Comparison of crack segmentation using digital image correlation measurements and deep learning},
journal = {Construction and Building Materials},
volume = {261},
pages = {120474},
year = {2020},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2020.120474},
url = {https://www.sciencedirect.com/science/article/pii/S095006182032479X},
author = {Amir Rezaie and Radhakrishna Achanta and Michele Godio and Katrin Beyer},
keywords = {Crack segmentation, Digital image correlation, Deep learning, Threshold method, Masonry},
}
```
