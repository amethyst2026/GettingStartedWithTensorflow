from mne import concatenate_raws
from mne.decoding import Scaler,SlidingEstimator,Vectorizer,LinearModel,get_coef,CSP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,normalize
import pandas as pd
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
import seaborn as sns
import time
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
raw2 = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw2)  # set channel names
montage = make_standard_montage('standard_1005')
raw2.set_montage(montage)



raw2.rename_channels(lambda x: x.strip('.'))

ssp_projectors = raw2.info['projs']
raw2.del_proj()

eeg_channels = pick_types(raw2.info,meg=False,eeg=True,stim=False,eog=False,exclude='bads')

fig = raw2.plot_psd(tmax=np.inf,fmax=80,average=True)

for ax in fig.axes[:2]:
    freqs = ax.lines[-1].get_xdata()
    psds =ax.lines[-1].get_ydata()
    for freq in (20,40,60,80):
        idx = np.searchsorted(freqs,freq)
        ax.arrow(x=freqs[idx],y=psds[idx], dx=0,dy=-12,color ='red',width = 0.1, head_width = 3, length_includes_head = True)
        plt.show()


#Getting rid of artifacts in the data

filt_raw = raw2.copy()
filt_raw.load_data().filter(l_freq=1.,h_freq=60)

ica = ICA(n_components=64,random_state=97)
ica.fit(filt_raw)


events, _ = events_from_annotations(raw2,event_id=dict(T1=2,T2=3))

epochs = Epochs(raw2,events,event_id,tmin,tmax,proj=True,picks=eeg_channels,baseline=None,preload=True)
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

sfreq = raw2.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []


stdx = Scaler(x)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv1D,MaxPooling1D,BatchNormalization

def build_model():
    model = keras.Sequential()
    model.add(Conv1D(64,(3),input_shape = X.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv1D(64, (2)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(64, (2)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae','mse'])
    return model
model = build_model()

print(model.summary())
epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.fit(X,labels,batch_size=batch_size,epochs = 10)
    score = model.evaluate(X,labels,batch_size=batch_size)
    print(score)
    MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
    model.save(MODEL_NAME)
print("saved:")
print(MODEL_NAME)


