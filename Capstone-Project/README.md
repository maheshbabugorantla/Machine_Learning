# [**Kaggle Pet Adoption Prediction**](https://www.kaggle.com/c/petfinder-adoption-prediction/data)

## Description

In this competition you will predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder. Sometimes a profile represents a group of pets. In this case, the speed of adoption is determined by the speed at which all of the pets are adopted. The data included consists of text, tabular, and image data.

This is a Kernels-only competition. At the end of the competition, test data will be replaced in their entirety with new data of approximately the same size, and your kernels will be rerun on the new data.

## How to use

1. [Download](https://www.kaggle.com/c/10686/download-all) the dataset first, unzip and store it in a folder called `data`

This is how the folder hierarchy should look like

> boldened words are folders

- **Capstone Project**
  - **data**
    - breed_labels.csv
    - color_labels.csv
    - state_labels.csv
    - **test**
    - **test_images**
    - **test_metadata**
    - **test_sentiment**
    - **train**
    - **train_images**
    - **train_metadata**
    - **train_sentiment**

2. Open the Jupyter Notebook `Kaggle-Notebook.ipynb` and run all the cells to the replicate the same results as reported in the [`Capstone-Project-Report.md`](Capstone-Project-Report.md)

## Tools and Libraries used

All the development code is written in python and organized into a working solution using Jupyter Notebook

### Python Libraries

1. numpy
2. pandas
3. scikit-learn
4. xgboost-gpu
