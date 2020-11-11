"""@package docstring
This module contains functions to align different streams of Data. 

L_*** functions are Low level functions that work with pandas dataframe containing only the data and time as index
H_*** functions are High level functions that accept as argument pandas dataframe with standard "app" structure (with columns 'subject_id' and 'condition') and will call L_*** functions
HH_*** functions are even Higher level functions that accept a list of pandas dataframe with standard "app" structure for each data stream and will call H_*** functions
"""

import numpy as np
import pandas as pd

import warnings

def L_align(X1, X2):
    """
    Low Level Align, align two pandas timeseries (time as index)

    :param X1: dataframe
    :param X2: dataframe
    :return: aligned dataframe based on indices 
    """

    data = X1.join(X2, how = 'outer', sort = True)
    #Interpolate missing values (will interpolate labels but also X1)
    data = data.interpolate('index', limit_area ='inside')

    #Get only the data interpolated (remove NaN)
    data = data.loc[data.iloc[:,0].dropna().index.intersection(data.iloc[:,-1].dropna().index)]

    #Remove original labels index (X1 was interpolated at these rows, we don't want it)
    data = data[~data.index.isin(X2.index.intersection(data.index))]
    return data

def H_align(X,labels):
    '''
    High Level Align
    Align labels and X and interpolate missing values of labels for each recording independently

    :param X: dataframe with standard "app" structure
    :param labels: dataframe with standard "app" structure
    :return: X and labels aligned
    '''
    Label_is = []
    X_is = []

    #Raw data
    if 'subject_id' in X.columns and 'condition' in X.columns:
        Xsubj = set(X['subject_id'])
        Xcdt = set(X['condition'])
        for subj in Xsubj:
            for cdt_nb in Xcdt:
                #Get data with correct cdt and subj
                X_i = X[X['condition'] == cdt_nb]
                X_i = X_i[X_i['subject_id'] == subj] 

                #same with labels
                Label_i = labels[labels['condition'] == cdt_nb]
                Label_i = Label_i[Label_i['subject_id'] == subj] 
                del Label_i['subject_id']
                del Label_i['condition']
                
                
                #align X and labels for thisspecific cdt and subj
                data = L_align(X_i,Label_i)
                
                #X back but cut with labels
                X_i_cut = data[X_i.columns]
                
                #Labels and columns of subject and condition
                labels_i_aligned = data[list(Label_i.columns.values)  + ['subject_id','condition']]

                Label_is.append(labels_i_aligned)
                X_is.append(X_i_cut)

    #Features
    if 'Recording' in X.columns:
        warnings.warn('The use of recording is deprecated')
        Xrecs = set(X['Recording'])
        for rec in Xrecs:
            subj,cdt,nb = rec.split('_')
            X_i = X[X['Recording'] == rec]

            Label_i = labels[labels['condition'] == cdt+'_'+nb]
            Label_i = Label_i[Label_i['subject_id'] == subj] 
            del Label_i['subject_id']
            del Label_i['condition']

            data = L_align(X_i,Label_i)
            
            X_i_cut = data[X_i.columns]
            labels_i_aligned = data[list(Label_i.columns.values)  + ['Recording']]

            X_i_cut = X_i_cut[~X_i_cut.index.duplicated(keep='first')]
            labels_i_aligned = labels_i_aligned[~labels_i_aligned.index.duplicated(keep='first')]

            Label_is.append(labels_i_aligned)
            X_is.append(X_i_cut)
    else:
        pass

    X = pd.concat(X_is, axis = 0)
    labels = pd.concat(Label_is, axis = 0)

    return X, labels

def HH_Align(Tables,Labels):
    '''
    High-High Level Align
    Align labels and X and interpolate missing values of labels for each recording independently for several streams of data

    :param X: dataframe with standard "app" structure
    :param labels: dataframe with standard "app" structure
    :return: X and labels aligned
    '''
    for stream_idx in range(len(Tables)):
        Tables[stream_idx],labels_out = H_align(Tables[stream_idx],Labels)
        
    return Tables, labels_out

def H_align_with_windows(Windowed_Tables,Labels):
    """
    This function will align targets with windows of the first stream, the last time value of labels of each window is extracted. The number of labels should be same as the number of windows of each stream.
    It should be called for each set (train,val,test) and will remove 'subject_id', 'condition' and 'set'
    Warning: works for only one stream of data

    :param Windowed_Tables: list of pandas dataframe (windows of data)
    :param Labels: dataframe with standard "app" structure
    :return: Labels for each window (without 'subject_id' and 'condition' and 'set')
    """
    indices = []
    for i in range(len(Windowed_Tables[0])):
        indices.append(Windowed_Tables[0][i].index.values[-1])
    
    return Labels.loc[indices].iloc[:,:-3]
