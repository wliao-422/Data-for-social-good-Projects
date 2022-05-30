# Machine Learning
# Regression Analyses
# Author: WL
# Fragile Families: what factors  are more important/predictive in governing the trajectory of children’s development in fragile families?

import os
os.getcwd()
import pandas as pd
import numpy as np
def fillMissing(inputcsv, outputcsv):
    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    
    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1
    # write filled outputcsv
    df.to_csv(outputcsv, index=False)
# Usage:
# fillMissing('background.csv', 'output.csv')

# here, load in the two data sets.
bg = pd.read_csv('output.csv', low_memory=False)
tr = pd.read_csv('train.csv', low_memory=False)

# merge the two data frames to create a training data set with predictors and outcomes.
tr['train'] = 1
trbg = tr.merge(bg,'outer',left_on='challengeID',right_on='challengeID')

# convert categorical variable to numeric
def getdum(data):
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    cat_colnames = list(set(cols)-set(num_cols))
    data_with_dummies = pd.get_dummies(data,columns=cat_colnames)
    return data_with_dummies


# get the columns with categorical data
# get_dummies
trbg_with_dummies = getdum(trbg.ix[:,8:12952])
trbg_with_dummies.to_csv("trbg_with_dummies.csv", index =False)

# now, two regularized regression methods - Ridge regression and Lasso regression.
# (1) ridge regression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

def ridge_reg_pred(datax, datay, yvar,alpha):
    ridgereg = Ridge(alpha=1)
    
    print(datax.shape)
    print(datay.shape)
    
    datax1 = datax[datay['train']==1]
    datay1 = datay[datay['train']==1]
    data1_noNA = datax1.loc[datay1[yvar]>0,:]
    
    print(data1_noNA.shape)
    
    X = data1_noNA
    Xval_validation = X.values[-100:,:]
    y = datay.loc[datay[yvar]>0,yvar]
    yval_validation = y.values[-100:]
    Xval_training = X.values[:-100,:]
    yval_training = y.values[:-100]
    ridgereg.fit(Xval_training,yval_training)
    y_pred_validation = ridgereg.predict(Xval_validation)
    y_pred_training = ridgereg.predict(Xval_training)
    
    
    rsq_validation = r2_score(yval_validation, y_pred_validation)
    rsq_training = r2_score(yval_training, y_pred_training)
    
    datax2 = datax[datay['train']!=1]
    Xval_pred = datax2.values

    y_pred = ridgereg.predict(Xval_pred)
        
    return y_pred,rsq_validation,rsq_training

### Model fitting

gpa_ridge_pred,gpa_ridge_rsq_validation,gpa_ridge_rsq_training = ridge_reg_pred(trbg_with_dummies, trbg, 'gpa', 1)
grit_ridge_pred,grit_ridge_rsq_validation,grit_ridge_rsq_training = ridge_reg_pred(trbg_with_dummies, trbg, 'grit', 1)
mH_ridge_pred,mH_ridge_rsq_validation,mH_ridge_rsq_training = ridge_reg_pred(trbg_with_dummies, trbg, 'materialHardship', 1)

gpa_ridge_rsq_training,grit_ridge_rsq_training,mH_ridge_rsq_training

# (2) Lasso regression
from sklearn.linear_model import Lasso
def lasso_reg_pred(datax, datay, yvar,alpha):
    lassoreg = Lasso(alpha=alpha)
    
    print(datax.shape)
    print(datay.shape)
    
    datax1 = datax[datay['train']==1]
    datay1 = datay[datay['train']==1]
    data1_noNA = datax1.loc[datay1[yvar]>0,:]
    
    print(data1_noNA.shape)
    
    X = data1_noNA
    Xval_validation = X.values[-100:,:]
    y = datay.loc[datay[yvar]>0,yvar]
    yval_validation = y.values[-100:]
    Xval_training = X.values[:-100,:]
    yval_training = y.values[:-100]
    lassoreg.fit(Xval_training,yval_training)
    y_pred_validation = lassoreg.predict(Xval_validation)
    y_pred_training = lassoreg.predict(Xval_training)
    
    
    rsq_validation = r2_score(yval_validation, y_pred_validation)
    rsq_training = r2_score(yval_training, y_pred_training)
    
    datax2 = datax[datay['train']!=1]
    Xval_pred = datax2.values

    y_pred = lassoreg.predict(Xval_pred)
        
    return y_pred,rsq_validation,rsq_training

gpa_lasso_pred,gpa_lasso_rsq_validation,gpa_lasso_rsq_training = lasso_reg_pred(trbg_with_dummies, trbg, 'gpa', 1)
grit_lasso_pred,grit_lasso_rsq_validation,grit_lasso_rsq_training = lasso_reg_pred(trbg_with_dummies, trbg, 'grit', 1)
mH_lasso_pred,mH_lasso_rsq_validation,mH_lasso_rsq_training = lasso_reg_pred(trbg_with_dummies, trbg, 'materialHardship', 1)

gpa_lasso_rsq_training,grit_lasso_rsq_training, mH_lasso_rsq_training

# Generate prediction dataset for submission.
sel = trbg[trbg['train']!=1]
challengeIDpred = sel['challengeID']

prediction_ridge = pd.DataFrame({"challengeID":challengeIDpred,
                           "gpa":gpa_ridge_pred, 
                           "grit":grit_ridge_pred, 
                           "materialHardship":mH_ridge_pred,
                           "eviction":np.nan,
                            "layoff":np.nan,
                            "jobTraining":np.nan},
                         columns=['challengeID','gpa','grit','materialHardship','eviction','layoff','jobTraining'])
prediction_sub_ridge = prediction_ridge.append(tr.ix[:,:-1])

prediction_lasso = pd.DataFrame({"challengeID":challengeIDpred,
                           "gpa":gpa_lasso_pred, 
                           "grit":grit_lasso_pred, 
                           "materialHardship":mH_lasso_pred,
                           "eviction":np.nan,
                            "layoff":np.nan,
                            "jobTraining":np.nan},
                         columns=['challengeID','gpa','grit','materialHardship','eviction','layoff','jobTraining'])
prediction_sub_lasso = prediction_lasso.append(tr.ix[:,:-1])

prediction_sub = prediction_sub_ridge

prediction_sub_sort = prediction_sub.sort_values(['challengeID'])
prediction_sub_sort
prediction_sub_sort.to_csv('FFC_Ridge_Dummies_Liao_April2017.csv', index=False)

prediction_sub = prediction_sub_lasso

prediction_sub_sort = prediction_sub.sort_values(['challengeID'])
prediction_sub_sort
prediction_sub_sort.to_csv('FFC_Lasso_Dummies_Liao_April2017.csv', index=False)
