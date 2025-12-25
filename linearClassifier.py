from mne import Epochs,pick_types
from mne.channels import make_standard_montage
from mne.io import concatenate_raws,read_raw_edf
from mne.datasets import eegbci
from mne.decoding import Vectorizer,Scaler,get_coef,SlidingEstimator,LinearModel
from mne.minimum_norm import read_inverse_operator,apply_inverse
import mne

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# cue onset.
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

# Apply band-pass filter
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1] - 2

snr = 3.0

lambda2 = 1.0 / snr ** 2

aparc_label_name = 'bankssts-lh'
tmin,tmax = 0.080,0.120

evoked = mne.read_evokeds(raw_fnames,condition=0,baseline=(None,0))

inverse_operator = read_inverse_operator(raw_fnames)

