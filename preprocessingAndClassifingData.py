import tensorflow as tf
import pandas as pd
import sklearn
import mne

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit,cross_val_score

from mne import Epochs,pick_types,events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws,read_raw_edf
from mne.decoding import CSP,SlidingEstimator,Scaler,Vectorizer
from mne.preprocessing import ICA
from mne.datasets import eegbci

import os
import numpy as np
import matplotlib.pyplot as plt

tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))

ssp_projectors = raw.info['projs']
raw.del_proj()

eeg_channels = pick_types(raw.info,meg=False,eeg=True,stim=False,eog=False,exclude='bads')

fig = raw.plot_psd(tmax=np.inf,fmax=80,average=True)

for ax in fig.axes[:2]:
    freqs = ax.lines[-1].get_xdata()
    psds =ax.lines[-1].get_ydata()
    for freq in (20,40,60,80):
        idx = np.searchsorted(freqs,freq)
        ax.arrow(x=freqs[idx],y=psds[idx], dx=0,dy=-12,color ='red',width = 0.1, head_width = 3, length_includes_head = True)
        plt.show()


#Getting rid of artifacts in the data

filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1.,h_freq=60)

ica = ICA(n_components=64,random_state=97)
ica.fit(filt_raw)


events, _ = events_from_annotations(raw,event_id=dict(T1=2,T2=3))

epochs = Epochs(raw,events,event_id,tmin,tmax,proj=True,picks=eeg_channels,baseline=None,preload=True)
label = epochs.events[:,-1] - 2

train_labels = pd.get_dummies(label)
data = epochs.to_data_frame()
df = pd.DataFrame(data)

df.head()

x = epochs.load_data()
train_labels = train_labels.values
print(train_labels.shape)

scores = []
epochs_data = epochs.get_data()
epochs_train = epochs.copy().crop(tmin=1.,tmax=10.)

epochs_data_train = epochs_train.get_data()

cv = ShuffleSplit(10,test_size=0.2,random_state = 42)
cv_split = cv.split(epochs_data_train)

lda = LinearDiscriminantAnalysis()

csp = CSP(n_components=4,reg=None,log=True,norm_trace = True)

clf = Pipeline([('CSP',csp),('LDA',lda)])

scores = cross_val_score(clf,epochs_data_train,label,cv=cv,n_jobs=1)

class_balance = np.mean(label == label[0])

class_balance = max(class_balance,1. - class_balance)
print("Classification accuracy: %f / Chance level:%f" % (np.mean(scores),class_balance))

csp.fit_transform(epochs_data,label)

csp.plot_patterns(epochs.info,ch_type = 'eeg',units='Patterns (AU)',size = 1.5)
#Looking at performance over time

sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = label[train_idx], label[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()

#Visualizing Connectivity using a circular graph

from mne.minimum_norm import apply_inverse_epochs,read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle, plot_sensors_connectivity

# We exclude the baseline period
fmin, fmax = 3., 9.
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
  # just keep MEG and no EOG now
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

plot_sensors_connectivity(epochs.info,con[:,:,0])

#Starting to take data and running a classifier

