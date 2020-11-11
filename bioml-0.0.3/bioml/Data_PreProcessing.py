"""@package docstring
This module contains functions to perform different steps of preprocessing

L_*** functions are Low level functions that work with pandas dataframe containing only the data and time as index
H_*** functions are High level functions that create pandas dataframe with standard "app" structure (with columns 'subject_id' and 'condition') and will call L_*** functions
HH_*** functions are even Higher level functions that accept a list of pandas dataframe with standard "app" structure and will call H_*** functions
"""

import numpy as np
import pandas as pd
import scipy.signal as sgn

import sys
import warnings
from sklearn import preprocessing

def H_filter_data(Tables, FSs, recs):
    """
    Apply function 'L_Filter_EMG' to filter EMG data 
    
    :param Tables:  List of dataframes containing data for each stream in standard "app" structure 
    :param FSs: List of sampling frequencies for each stream
    :param recs: List of selected recordings
    :return: List of dataframes containing data for each stream in standard "app" structure filtered
    """
    
    for stream_idx in range(len(Tables)):  
        for rec in recs: 
            #Window by window to mimic pseudo real time (Do it at receive time ?)
            # SEE THIS FOR RT : https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
            
            subj, cdt, nb = rec.split('_')
            cdt_nb = cdt + '_' + nb

            warnings.warn('The filtering will not work in RT')
            Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb),Tables[stream_idx].columns[:-2]] = L_Filter_EMG(Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb),Tables[stream_idx].columns[:-2]],
                                                                                                                                            float(FSs[stream_idx])
                                                                                                                                            )
                                                                                                
    return Tables

def H_standardize_training_data(Tables,recs):
    """
    Standardize training data (x - mean)/std for each channel independently

    :param Tables:  List of dataframes containing data for each stream in standard "app" structure 
    :param recs: List of selected recordings
    :return: List of dataframes containing data for each stream in standard "app" structure standardized and sklearn StandardScaler objects for later use (one per stream and recording)
    """

    #One mean,std per stream and per recording
    standard_scalers = [[[] for _ in range(len(recs))] for _ in range(len(Tables))]

    for stream_idx in range(len(Tables)):  
        for rec_idx, rec in enumerate(recs): 

            subj, cdt, nb = rec.split('_')
            cdt_nb = cdt + '_' + nb
            
            #Create and apply standard scaler
            Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb)  & (Tables[stream_idx]['set'] == 'train'),Tables[stream_idx].columns[:-3]], standardscaler = L_standardize(Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb) & (Tables[stream_idx]['set'] == 'train'),Tables[stream_idx].columns[:-3]])

            #Save for use on other sets (Val and test)
            standard_scalers[stream_idx][rec_idx] = standardscaler

    return Tables, standard_scalers

#should just be standardize based on saved scaler and be called several times as needed
def H_standardize_val_test_data(Tables, recs, standard_scalers):
    """
    Standardize validation and test data based on StandardScaler obtained on training data

    :param Tables:  List of dataframes containing data for each stream in standard "app" structure 
    :param recs: List of selected recordings
    :param standard_scalers: sklearn StandardScaler object
    :return: List of dataframes containing data for each stream in standard "app" structure standardized with standard_scalers
    """
    warnings.warn('Change this function to work after data cut')
    for stream_idx in range(len(Tables)):  
        for rec_idx, rec in enumerate(recs): 

            subj, cdt, nb = rec.split('_')
            cdt_nb = cdt + '_' + nb

            #apply standard scaler on validation set
            if len(Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb)  & (Tables[stream_idx]['set'] == 'validation')]) != 0:
                Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb)  & (Tables[stream_idx]['set'] == 'validation'),Tables[stream_idx].columns[:-3]] = standard_scalers[stream_idx][rec_idx].transform(Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb)  & (Tables[stream_idx]['set'] == 'validation'),Tables[stream_idx].columns[:-3]])   
            #Apply standard scaler on test set
            Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb)  & (Tables[stream_idx]['set'] == 'test'),Tables[stream_idx].columns[:-3]] = standard_scalers[stream_idx][rec_idx].transform(Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb)  & (Tables[stream_idx]['set'] == 'test'),Tables[stream_idx].columns[:-3]])   
    return Tables

def H_standardize_val_test_data_online(online_data, subj, cdt_nb, standard_scalers):
    """
    Standardize validation and test data based on StandardScaler obtained on training data ONLINE ONLY

    :param online_data:  List of dataframes containing data for each stream obtained in real time (no subj_id and condition) 
    :param subj: value of subject in column 'subject_id' to retrieve (not used)
    :param cdt_nb: value of cdt_nb in column 'cdt_nb' to retrieve (not used)
    :param standard_scalers: sklearn StandardScaler object
    :return: List of dataframes containing data for each stream in standard "app" structure standardized with standard_scalers
    """
    for stream_idx in range(len(online_data)):  

        #Apply standard scaler on online_data
        online_data[stream_idx].iloc[:,:] = standard_scalers[stream_idx][0].transform(online_data[stream_idx].iloc[:,:])   
    return online_data

def H_NormRef_Data(Tables, channels, references):
    """
    Normalize channels based on ref electrodes 

    :param Tables:  List of dataframes containing data for each stream in standard "app" structure 
    :param channels: List, channels selected
    :param references: List, references electrodes if any
    :return: List of dataframes containing data for each stream in standard "app" structure standardized with reference electrodes
    """
    for stream_idx in range(len(Tables)):  
        if references[stream_idx] != []:
            Tables[stream_idx] = L_NormRef_Data(Tables[stream_idx],
                                                    channels[stream_idx], 
                                                    references[stream_idx])
    return Tables

def H_cut_time_windows(Tables,FSs,win_len,step):
    """
    This function will create overlaping time windows from the Tables data. It will delete columns 'subj_id, 'condition' and 'set' from the data (NO TRACE OF ORIGINAL RECORDING THE WINDOW CAME FROM)
    
    :param Tables:  List of dataframes containing data for each stream in standard "app" structure 
    :param FSs: List of sampling frequencies for each stream
    :param winlen: float, length of the window (in ms)
    :param step: float, length of the the step between windows (in ms) if step >= winlen it means that windows are not overlapped
    :return: 3 lists of dataframes (one for each set)
    """

    datasets = [[],[],[]]
    
    for i,dataset in enumerate(['train','validation','test']):
        datasets[i] = [[] * len(Tables)]
        for stream_idx in range(len(Tables)):
            for subj in set(Tables[stream_idx]['subject_id']):
                for cdt_nb in set(Tables[stream_idx]['condition']):
                    datasets[i][stream_idx].extend(
                        L_cut_time_windows(
                            Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb) & (Tables[stream_idx]['set'] == dataset),Tables[stream_idx].columns[:-3]],
                                                          FSs[stream_idx],
                                                          win_len,
                                                          step))

    Train = datasets[0]
    Val = datasets[1]
    Test = datasets[2]

    return Train,Val,Test


def H_shift_labels(Labels,nb_win,future):
    """
    Shifts labels in the future or past (should be used after window creation of nb_win is the number of samples)

    :param Labels: dataframe with standard "app" structure
    :param nb_win: float, number of windows to shift the data
    :param future: bool, if true it will shift labels to predict the future, else it will predict the past
    :return: dataframe with standard "app" structure shifted
    """
    Labels.iloc[:,:] = L_shift_labels(Labels.values,nb_win,future)
    return Labels



def L_NormRef_Data(data, Selected_Channels, Ref_Electrodes):
    """ Normalize data with reference electrodes for one cluster
    
    :param data: dataframe
    :param Selected_Channels: List of selected channels
    :param Ref_Electrodes: List of channels corresponding to reference electrodes if any
    :return: data normalized by the corresponding channels
    """

    print(Selected_Channels, Ref_Electrodes)

    #Normalize each selected channels by selected reference electrodes  
    if Ref_Electrodes != [] and Ref_Electrodes != None and Ref_Electrodes != ['']: 
        references = [data[ref_chan].copy() for ref_chan in Ref_Electrodes]

        for channel in Selected_Channels:
            print(channel)
            #Normalize channels with reference electrodes when there is one
            data[channel] -= np.mean(references ,axis = 0)
            # Mean of several channels into one channel (for EEG processing)
            #data[channel] = data[[ref_chan for ref_chan in Ref_Electrodes]].mean(axis = 1) 
            # Create dipole for EEG processing
            #data[channel] = data[[Ref_Electrodes[0],Ref_Electrodes[1],Ref_Electrodes[2],Ref_Electrodes[3]]].mean(axis = 1) - data[[Ref_Electrodes[4],Ref_Electrodes[5],Ref_Electrodes[6],Ref_Electrodes[7]]].mean(axis = 1)
        return data
 
###############################################################################################################
def L_Filter_EMG(data, fs): 
    """ Filter EMG data with notch filter at 50Hz for electromagnetic noise, Band-pass filter at between 15 and 500Hz to remove moving artefact,

    :param data: dataFrame
    :param fs: float, Sampling frequency of data
    :return: data filtered
    """
    #50Hz notch filter
    b_notch, a_notch = sgn.iirnotch(w0 = 50, Q = 10, fs=fs)
    #Apply 
    data = sgn.lfilter(b_notch,a_notch, data, axis = 0)

    #bandpass filter 15-500 Hz
    sos_bandpass = sgn.iirfilter(N = 17, Wn = [15,500], rs=60, btype='band',
                        analog=False, ftype='cheby2', fs=fs, output='sos')
    #Apply
    data = sgn.sosfilt(sos_bandpass, data, axis = 0)
  
    return data

def L_standardize(data):
    """ standardize data: (x-mean)/std

    :param data: dataFrame
    :return: data standardized
    """
    standard_scaler = preprocessing.StandardScaler()
    data.iloc[:,:] = standard_scaler.fit_transform(data.iloc[:,:])
    return data, standard_scaler

def L_cut_time_windows(data,fs,win_len,step):
    """This function is used to cut the time windows from the raw data

    :param data: dataFrame
    :param fs: float, Sampling frequency of data
    :param win_len: float, length of the window (in ms)
    :param step: float, length of the the step between windows (in ms) if step >= winlen it means that windows are not overlapped
    :return: list of dataframes containing one window each
    """

    # Initialization of some parameters.
    time_windows = []
    
    nbr_samples_tw = win_len * fs
    nbr_samples_overlap = step * fs
    

    current_index = 0
    last_index = len(data)
    estimated_index = 0


    while estimated_index < last_index:
        current_tw_emg = data.iloc[int(current_index) : int(current_index + nbr_samples_tw)]
        
        time_windows.append(current_tw_emg)
        
        current_index = current_index + nbr_samples_overlap
        
        # Until which index would next time windows go? 
        estimated_index = current_index + nbr_samples_tw
    
    return time_windows

def L_scale_labels(Labels,new_max_value):
    """ scale labels to obtain new_max_value as the maximum value

    :param Labels: dataframe with standard "app" structure
    :param new_max_value: float, maximum value to obtain
    :return: Labels scaled

    """
    for i in range(Labels.shape[1]):
        Labels.iloc[:,i] = Labels.iloc[:,i].values/abs(max(Labels.iloc[:,i].values))*new_max_value

    return Labels


def L_correct_Labels_6channels(Labels, LR):
    """
    This functions corrects the labels obtained from the virtual hand (change some close values that should have been the same) and works only for the 6 channels version

    :param Labels: dataframe with column names as {LR}_Hand_channel{i} and two last columns are subject_id and condition (standard "app" structure)
    :param LR: string in {'Left','Right'} corresponding to column name of Labels
    :return: Labels corrected
    """
    for i in range(3,7):
        Labels.loc[(Labels['{}_Hand_channel{}'.format(LR,i)] >= 19) & (Labels['{}_Hand_channel{}'.format(LR,i)] <= 21), 
                        '{}_Hand_channel{}'.format(LR,i)] = 21

    for i in range(1,3):
        Labels.loc[(Labels['{}_Hand_channel{}'.format(LR,i)] >= 9) & (Labels['{}_Hand_channel{}'.format(LR,i)] <= 10.5), 
                        '{}_Hand_channel{}'.format(LR,i)] = 10.5

    for i in range(2):
        Labels.iloc[:,i] = Labels.iloc[:,i].values * 69/39
        
    for i in range(1,3):
        Labels.loc[(Labels['{}_Hand_channel{}'.format(LR,i)] >= 18.5) & (Labels['{}_Hand_channel{}'.format(LR,i)] <= 21), 
                        '{}_Hand_channel{}'.format(LR,i)] = 21
    return Labels

###############################################################################################################
def H_Envelope(Tables, FSs, recs):
    """
    Apply filtering on data to obtain EMG envelope
    
    :param Tables:  List of dataframes containing data for each stream in standard "app" structure 
    :param FSs: List of sampling frequencies for each stream
    :param recs: List of selected recordings
    """
    
    for stream_idx in range(len(Tables)):  
        for rec in recs:
            #Window by window to mimic pseudo real time (Do it at receive time ?)
            # SEE THIS FOR RT : https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
            
            subj, cdt, nb = rec.split('_')
            cdt_nb = cdt + '_' + nb

            warnings.warn('The filtering will not work in RT')
            Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb),Tables[stream_idx].columns[:-2]] = L_Envelope(Tables[stream_idx].loc[(Tables[stream_idx]['subject_id'] == subj) & (Tables[stream_idx]['condition'] == cdt_nb),Tables[stream_idx].columns[:-2]],
                                                                                                                                            float(FSs[stream_idx])
                                                                                                                                            )
                                                                                                
    return Tables

def L_Envelope(data, fs): 
    """ Apply standard filters to obtain EMG envelope

    :param data: Pandas DataFrame with the EMG data to be filtered
    :param fs: float, Sampling frequency of data
    :return: envelope of data
    """
    
    data = data - np.mean(data)
    
    #50Hz notch filter
    b_notch, a_notch = sgn.iirnotch(w0 = 50, Q = 10, fs=fs)
    #Apply 
    data = sgn.lfilter(b_notch,a_notch, data, axis = 0)

    #bandpass filter 15-500 Hz
    sos_bandpass = sgn.iirfilter(N = 17, Wn = [15,500], rs=60, btype='band',
                        analog=False, ftype='cheby2', fs=fs, output='sos')
    #Apply
    data = sgn.sosfilt(sos_bandpass, data, axis = 0)
    
    #Rectify the signal
    data = np.abs(data - np.mean(data)) 
    
    #lowpass filter 2 Hz
    sos_bandpass = sgn.iirfilter(N = 10, Wn = [2], btype='low',
                        analog=False, ftype='butter', fs=fs, output='sos')
    #Apply
    data = sgn.sosfilt(sos_bandpass, data, axis = 0)
    return data
###############################################################################################################










###############################################################################################################
#Kalman filter for Offline & Real time
def Kalman_filter(prediction_buffer):
    """ Apply Kalman filter on predictions to smooth output
    
    :param prediction_buffer: Numpy array containing buffer on which to apply kalman filter (should be of shape (N_buffer,1) only one column)
    :return: Filtered prediction
    """

    # intial parameters
    n_iter = prediction_buffer.shape[0]
    sz = (n_iter,) # size of array
    
    z = prediction_buffer
    
    Q = 1e-5 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    #R = 0.1**2 # estimate of measurement variance, change to see effect
    R = np.var(prediction_buffer) # estimate of measurement variance, change to see effect
    
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
        
    return xhat[-1]

###############################################################################################################
# SHIFT LABELS (Not used yet)
# Labels after windowing
def L_shiftlabels_predict_future(labels,nb_win):
    """Shift labels to let the classifier predict the future
    
    :param labels: vector of labels to shift
    :param nb_win: number of windows to shift the labels
    :return: shifted labels
    """
    shift = np.tile(labels[-1,:],(nb_win,1))   
    shifted_labels = np.vstack((labels[nb_win:], shift))
    return shifted_labels

def L_shiftlabels_predict_past(labels,nb_win):
    """Shift labels to let the classifier predict the past
    
    :param labels: vector of labels to shift
    :param nb_win: number of windows to shift the labels
    :return: shifted labels
    """
    shift = np.tile(labels[0,:],(nb_win,1))   
    shifted_labels = np.vstack((shift, labels[:-nb_win]))
    return shifted_labels

def L_shift_labels(labels,nb_win,future):
    """Shift labels of nb_win windows to predict future if future is True or past if future is false
    
    :param labels: vector of labels to shift
    :param nb_win: number of windows to shift the labels
    :param future: Boolean, if true predict the future if false predict the past
    :return: shifted labels
    """
    if future is True:
        shifted_labels = L_shiftlabels_predict_future(labels,nb_win)
    else:
        shifted_labels = L_shiftlabels_predict_past(labels,nb_win)
    return shifted_labels
    