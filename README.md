# Mergers and Acquisitions Forecasting via Projection of Interlocking Director Network

## Introduction
This repository is the official implementation of the paper "Mergers and Acquisitions Forecasting via Projection of Interlocking Director Network".
In this repo we provide the code to reproduce the experiments of the paper, with all its variants.

## Installation
You can use the package manager ```pip``` to install the required packages.
<br>
```pip install -r requirements.txt```

## Experiments
To reproduce the experiments of the paper, you can run the following command:
<br>
```python 7_ratio_search.py``` with ratio=10.
<br>
<br>
The run assumes you already have all preprocessed data, supplied in each folder. 
Note that the folder themselves have the preprocessing code we used for our row data (even if the row data is not always supplied).
<br>
<br>
The ratio variable stands for the ratio between 
the negative and positive samples in the training set.
After experimenting with different ratios, 
we found that the best ratio is 10. You are more than welcome to experiment with different ratios yourselves and even update us if you get better results.

## Results
For each ratio we ran, we saved the results in a costume 
results folder (for example: for ratio 10 the results 
will be saved at folder results_10).
<br>
Each results folder includes the following files:
* A trained model for each year the model was examined on. 
The format is: "model_for_YYYY.pkl".
* A dictionary of the roc details for each year the model was examined on.
The format is: "roc_details_YYYY.pkl".
* A dictionary of the scores the model achieved, by variant methods, for each year the model was examined on.
The format is: "scores_YYYY.pkl".
<br><br>
Overall, the ```results``` folder summarizes each (results folder) results in a plot or csv file. 

<br>
<br>