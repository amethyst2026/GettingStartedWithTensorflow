import tensorflow as tf
import mne

def prepare(filepath):
    raw_fname = filepath
    tmin,tmax = -0.200,4.00
    event_id = {'Auditory/left':1,'Visual/left':3}
    raw = mne.io.read_raw_fif(raw_fname,preload=True)

    raw.filter(2,20)
    events = mne.find_events(raw)
    raw.info['bads'] += ['MEG 2443','EEG 053']
    
    epochs = mne.Epochs(raw,events,event_id,tmin,tmax,proj=True,picks=('grad','eog'),baseline=(None,0),preload=True,reject=dict(grad=4000e-13,eog=150e-6),decim=10)

    epochs.pick_types(meg=True,exclude='bads')
    X = epochs.get_data()
    y = epochs.events[:,2]


model = tf.keras.models.load_model("C:\\Users\\dubst\\Documents\\new_models\\174.36-acc-64x3-batch-norm-10epoch-1598376016-loss-4.03.model")
prediction = model.predict([prepare("C:\\Users\\dubst\\mne_data\\MNE-sample-data\\MEG\\sample\\sample_audvis_raw.fif")])


