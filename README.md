# Project overview

The project is right now initialized. Due to legal reasons the data-files will not be published via this repository. 
What you need in order to run the project properly is a ```./data/PSP_Jan_Feb_2019.xlsx``` file.

This repository was built using ```cona```. To make the code fully reproducible you have to install the specified conda environment from the file ```environment.yml``` first. 
This is done using:

```console
    conda env create -f environment.yml
    conda activate kfex
```

Most of the plots in the data understanding phase are created using ```plotly```. Those plots do not get displayed correctly in GitHub, because only static content is supported. However, if you wish to have an interactive view of the provided notebooks you can do that via [nbviewer.org](https://nbviewer.org). For example if you want a dynamic view of the file ```02_Data_Understanding.ipynb``` you can achieve it via the following [URL](https://nbviewer.org/github/TNMayer/IU_DLMDSME01/blob/main/02_Data_Understanding.ipynb)