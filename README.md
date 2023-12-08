# CSCI 467 Project

## Setup
Run the following commands to create a new conda environment for the final project

```
conda create --name cs467 python=3.11
conda activate cs467
pip install -r requirements.txt 
```


## Datasets

### Import dataset
- create empty directory with name **data**
- [dataset1](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data), once downloaded, add **True.csv** and **Fake.csv** to the data folder
- [dataset2](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/code), once downloaded, add **WELFake_Dataset.csv** to the data folder

There are two dataset for this project, we can specify the project we work on.

I provide three options:

1. **dataset=1** means we're using a old dataset where news are from 2015 - 2018
2. **dataset=2** means we're using a relatively newer dataset from WELFake
3. **dataset=3** means we're using combination of previous two datasets

We can specify the dataset we want to train/dev/test on by setting:
```
python file.py ...other parameters... --dataset 3
```

By default, we use **dataset=3**


## Baseline
There are three baselines that you can run:

#### MostCommonWords
```
python baseline.py --method MostCommonWords
```
(long waiting time...)

#### NumPunctuation
```
python baseline.py --method NumPunctuation
```

#### NumCaps
```
python baseline.py --method NumCaps
```


## Naive Bayes
Run naive bayes method with default list laplace smoothing hyperparameters:

alphas = [1000,100,10,1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

```
python naive_bayes.py
```

or, you can specify the alphas you want to test

```
python naive_bayes.py --alphas alpha1 alpha2 ...
```


## Clean up
Feel free to remove this conda environment after running all the code

```
conda deactivate cs467
conda remove --name cs467 --all
```