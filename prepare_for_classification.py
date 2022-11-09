#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:38:22 2022

@author: lau
"""

#%% IMPORTS

import mne
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

#%% PREPROCESSIN

def preprocess_sensor_space_data(subject, date, raw_path,
                                 event_id=dict(pospos=21,
                                               neupos=41,
                                               negneg=22,
                                               neuneg=42),
                                 h_freq=40,
                                 tmin=-0.200, tmax=0.500, baseline=(None, 0),
                                 reject=None, decim=1,
                                 return_epochs=False,
                                 ):
    n_recordings = 6
    epochs_list = list()
    for recording_index in range(n_recordings): # ## loop from 0 to 5
        fif_index = recording_index + 1 # files are not 0-indexed
        fif_fname = 'face_word_' + str(fif_index) 
        if subject == '0085': ## sonething went wrong with the first three rec.
            folder_name = '00' + str(fif_index + 3) + '.' + fif_fname
        else:
            folder_name = '00' + str(fif_index) + fif_fname
            
        full_path = join(raw_path, subject, date, 'MEG', folder_name,
                         'files', fif_fname + '.fif')
        raw = mne.io.read_raw(full_path, preload=True)
        raw.filter(l_freq=None, h_freq=h_freq)
        
        events = mne.find_events(raw, min_duration=0.002)
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline,
                            preload=True, decim=decim)
        epochs.pick_types(meg=True)
        if return_epochs:
            epochs_list.append(epochs)
        else:
            if recording_index == 0:
                X = epochs.get_data()
                y = epochs.events[:, 2]
            else:
                X = np.concatenate((X, epochs.get_data()), axis=0)
                y = np.concatenate((y, epochs.events[:, 2]))
    
    if return_epochs:
        return epochs_list
    else:
        return X, y

def preprocess_source_space_data(subject, date, raw_path, subjects_dir,
                                 epochs_list=None,
                              event_id=dict(pospos=21,
                                            neupos=41,
                                            negneg=22,
                                            neuneg=42),
                              h_freq=40,
                              tmin=-0.200, tmax=0.500, baseline=(None, 0),
                              reject=None, decim=1,
                              method='MNE', lambda2=1, pick_ori='normal',
                              label=None):
    if epochs_list is None:
        epochs_list = preprocess_sensor_space_data(subject, date, raw_path,
                                                   return_epochs=True)
    y = np.zeros(0)
    for epochs in epochs_list: # get y
        y = np.concatenate((y, epochs.events[:, 2]))
    
    if label is not None:
        label_path = join(subjects_dir, subject, 'label', label)
        label = mne.read_label(label_path)
    
    for epochs_index, epochs in enumerate(epochs_list): ## get X
        fwd_fname = 'face_word_' + str(epochs_index + 1) + '-oct-6-src-' + \
                    '5120-5120-5120-fwd.fif'
        fwd = mne.read_forward_solution(join(subjects_dir,
                                             subject, 'bem', fwd_fname))
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2,
                                                     method, label,
                                                     pick_ori=pick_ori)
        for stc_index, stc in enumerate(stcs):
            this_data = stc.data
            if epochs_index == 0 and stc_index == 0:
                n_trials = len(stcs)
                n_vertices, n_samples = this_data.shape
                this_X = np.zeros(shape=(n_trials, n_vertices, n_samples))
            this_X[stc_index, :, :] = this_data
            
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))
    return X, y

#%% RUNNING FUNCTIONS

X_sensor, y = preprocess_sensor_space_data('0085', '20221004_000000',
        raw_path='/home/lau/projects/undervisning_cs/raw/',
        decim=4) ##CHANGE TO YOUR PATHS

epochs_list = preprocess_sensor_space_data('0085', '20221004_000000',
        raw_path='/home/lau/projects/undervisning_cs/raw/',
        return_epochs=True, decim=4) ##CHANGE TO YOUR PATHS

X_source, y = preprocess_source_space_data('0085', '20221004_000000',
        raw_path='/home/lau/projects/undervisning_cs/raw/', 
        subjects_dir='/home/lau/projects/undervisning_cs/scratch/freesurfer',
        epochs_list=epochs_list) ##CHANGE TO YOUR PATHS

X_lateral_occipital_lh, y = preprocess_source_space_data('0085',
                                                      '20221004_000000',
        raw_path='/home/lau/projects/undervisning_cs/raw/', 
        subjects_dir='/home/lau/projects/undervisning_cs/scratch/freesurfer',
        label='lh.lateraloccipital.label', epochs_list=epochs_list)
        ##CHANGE TO YOUR PATHS
        
X_lateral_occipital_rh, y = preprocess_source_space_data('0085',
                                                      '20221004_000000',
        raw_path='/home/lau/projects/undervisning_cs/raw/',  
        subjects_dir='/home/lau/projects/undervisning_cs/scratch/freesurfer',
        label='rh.lateraloccipital.label', epochs_list=epochs_list)
        ##CHANGE TO YOUR PATHS


#%% COLLAPSE EVENTS

def collapse_events(y, new_value, old_values=list()):
    for old_value in old_values:
        y[y == old_value] = new_value
    return y


collapsed_y = collapse_events(y, 0, [21, 41])
collapsed_y = collapse_events(collapsed_y, 1, [22, 42])

#%% SIMPLE CLASSIFICATION

def simple_classication(X, y, penalty='none', C=1.0):

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    n_samples = X.shape[2]
    logr = LogisticRegression(penalty=penalty, C=C, solver='newton-cg')
    sc = StandardScaler() # especially necessary for sensor space as
                          ## magnetometers
                          # and gradiometers are on different scales 
                          ## (T and T/m)
    cv = StratifiedKFold()
    
    mean_scores = np.zeros(n_samples)
    
    for sample_index in range(n_samples):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)
        scores = cross_val_score(logr, this_X_std, y, cv=cv)
        mean_scores[sample_index] = np.mean(scores)
        print(sample_index)
        
    return mean_scores

#%% RUN FUNCTION

# X_lateral_occipital_both = np.concatenate((X_lateral_occipital_lh,
#                                            X_lateral_occipital_rh), axis=1)

# mean_scores_LO_both = simple_classication(X_lateral_occipital_both,
#                                   collapsed_y,
#                                   penalty='l2', C=1e-3)

mean_scores_sensor = simple_classication(X_sensor,
                                  collapsed_y,
                                  penalty='l2', C=1e-3)

mean_scores_LO_lh = simple_classication(X_lateral_occipital_lh,
                                  collapsed_y,
                                  penalty='l2', C=1e-3)
    
#%% PLOT

# times = epochs_list[0].times
# times = np.arange(-0.200, 0.501, 0.001)
    
def plot_classfication(times, mean_scores, title=None):

    plt.figure()
    plt.plot(times, mean_scores)
    plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
    plt.ylabel('Proportion classified correctly')
    plt.xlabel('Time (s)')
    if title is None:
        plt.title('Happy versus sad face')
    else:
        plt.title(title)
    plt.show()
    
plot_classfication(epochs_list[0].times, mean_scores_sensor)
plot_classfication(epochs_list[0].times, mean_scores_LO_lh)