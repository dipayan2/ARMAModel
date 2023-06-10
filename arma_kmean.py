import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
#from sklearn.model_selection import train_test_split
import random

################################
## data
##
utilData = pd.read_csv('./Util_Run1.csv')
#print(utilData.shape)
print(utilData.head())
utilData['memLoad'] = utilData['memUse']/utilData['memTot']
utilData['cpuScore'] = utilData['cpuload']*utilData['cpufreq']/1400000
utilData['cpuF'] = utilData['cpufreq']/1000000



################################
## model
##
nr_sample = 2000
train = utilData.iloc[0:nr_sample]
test  = utilData.iloc[nr_sample:]

nr_chunk_sample = 10
nr_chunk = int(nr_sample / nr_chunk_sample)
chunk_model = []

nr_model = 7
models = []

nr_kmean_epoch = 60
pr_kmean_update = 0.1

#model_order = (2, 0, [2])
model_order = (3, 0, 0)
endog_key = 'cpuload'
exog_key = 'cpuF'

def get_sample0(m):
    # sample indexes whose chunk_model is not m
    index = [i for i in range(nr_sample) if chunk_model[int(i / nr_chunk_sample)] != m]
    y = train[endog_key].copy()
    x = train[exog_key].copy()
    y.loc[index] = 0
    x.loc[index] = 0
    return y, x

def get_sample1(m):
    # sample indexes whose chunk_model is not m
    y, x = [], []
    for i in range(nr_chunk):
        if chunk_model[i] == m:
            for j in range(i*nr_chunk_sample, (i+1)*nr_chunk_sample):
                y.append(train.loc[j, endog_key]) 
                x.append(train.loc[j, exog_key]) 
            for j in range(nr_chunk_sample):
                y.append(0) 
                x.append(0) 
    return y, x

def get_sample2(m):
    # sample indexes whose chunk_model is not m
    index = [i for i in range(1, nr_sample) if chunk_model[int(i / nr_chunk_sample)] != m]
    y = train[endog_key].copy()
    x = train[exog_key].copy()
    y.loc[index] = np.nan
    x.loc[index] = np.nan
    return y.interpolate(), x.interpolate()
    # return y.interpolate('spline', order=2), x.interpolate('spline', order=2)


def fit_models():
    get_sample = get_sample1 # find a way to use unrelated chunks of samples
    for m in range(nr_model):
        y, x = get_sample(m)
        model = ARIMA(y, order=model_order).fit()
        models.append(model)
        print('mse', model.mse)

def init_chunks():
    random.seed(1)
    for m in range(nr_chunk):
        chunk_model.append(random.randrange(nr_model))

def update_chunks():
    for c in range(nr_chunk):
        if random.random() > pr_kmean_update: #update slowly
            continue

        mses = []
        for m in range(nr_model):
            start, end = c*nr_chunk_sample, (c+1)*nr_chunk_sample
            res = models[m].apply(train[endog_key][start:end], refit=False)
            #res = models[m].apply(train[endog_key][start:end], exog=train[exog_key][start:end], refit=False)
            mses.append(res.mse)
        best = np.argmin(mses)
        chunk_model[c] = best


def estimate_models():
    # k-mean
    init_chunks()
    for i in range(nr_kmean_epoch):
        print('[', i, '/', nr_kmean_epoch, ']--------------')
        fit_models()
        update_chunks()


estimate_models()
for i in range(nr_model):
	print("Model Num: ", i)
	print(models[i].summary())
