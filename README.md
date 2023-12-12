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
create an empty directory with name **data**
- [dataset1](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data), once downloaded, add **True.csv** and **Fake.csv** to the data folder
- [dataset2](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/code), once downloaded, add **WELFake_Dataset.csv** to the data folder

There are two datasets for this project, we can specify the project we work on.

I provide three options:

1. **dataset=1** means we're using the old dataset where news is from 2015 - 2018
2. **dataset=2** means we're using a relatively newer dataset from WELFake
3. **dataset=3** means we're using the combination of the previous two datasets

We can specify the dataset we want to train/dev/test on by setting:
```
python file.py ...other parameters... --dataset 2
```

By default, we use **dataset=2**


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
Run the Naive Bayes method with default list Laplace smoothing hyperparameters:

alphas = [1000,100,10,1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

```
python naive_bayes.py
```

or, you can specify the alphas you want to test

```
python naive_bayes.py --alphas alpha1 alpha2 ...
```

## LSTM

### Train mode

The LSTM method contains a bunch of arguments to specify certain hyperparameters during training (Check lstm.py parser for more detail on running the code)

- learning rate
- number of episodes
- batch size
- hidden layer dimension for LSTM
- dropout probability
- whether or not to use the pre-trained word vector
```
python lstm.py train --arg [val]
```

We could also save the model that you trained by setting this argument
```
python lstm.py train --save
```

### Run mode
If we want to test existing models, run this
```
python lstm.py run --model [name of model]
```

If we don't want to run the test (which takes time) and only want to get the splot
```
python lstm.py run --model [name of model] --notest --plot
```

I have trained multiple models (including one used in the paper), you can find them in the model's subdirectory. The **models_info.json** file includes all the names, hyperparameters, and train/dev/test performance metrics.

### Tune mode
If we want to plot different models' performance for different hyperparameters, we can use the tune mode to plot the figure
```
python lstm.py tune --hyperparameter [name of hyperparameter]
```

## Clean up
Feel free to remove this conda environment after running all the code

```
conda deactivate cs467
conda remove --name cs467 --all
```