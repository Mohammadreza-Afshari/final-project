import tensorflow.keras as keras
import numpy as np
import librosa
import matplotlib.pyplot as plt


mapping = ['down' , 'off' , 'on' , 'no' , 'yes' , 'stop' , 'up' , 'right' , 'left' , 'go']

#load the model
model = keras.models.load_model('rnnmodel.h5')

def predict(filepath):
    signal, sr = librosa.load(filepath)
    signal = signal[:22050]   
    mfcc = librosa.feature.mfcc(y=signal,sr=22050,n_mfcc=13,n_fft=2048,hop_length=512).T
    plt.Figure(figsize=(10,20),dpi=400)
    plt.imshow(librosa.power_to_db(mfcc.T**2))
    
    #  (# of samples, # of segments, # of coefs, # of channels)
    mfcc = mfcc[np.newaxis, ... ]
    preds = model.predict(mfcc)
    idx = np.argmax(preds)
    print('predicted keyword is ',end='')
    print(mapping[idx])
    


predict('test/go.wav')
   

































