import librosa  
import os
import json
import numpy as np

data = {
        "map": ['down' , 'off' , 'on' , 'no' , 'yes' , 'stop' , 'up' , 'right' , 'left' , 'go'],        # what keyword each number represent 
        "labels": [],     # label of each audio file
        "MFCC": [],       # MFCC of each audio file 
        "file": []        # path of each audio file
        }
    
for i in range(len(data['map']) ):
    files = os.listdir('./raw data/' + data['map'][i])
    for f in files:
        filename = './raw data/' + data['map'][i] + '/' + f
        print(filename)
            
        signal , sr = librosa.load(filename)
            
        if len(signal) >= 22050:  # only add audio files that are longer than 1s 
            signal = signal[:22050]  # we only want 1s
            mfcc = librosa.feature.mfcc(signal, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048)
                
            data['labels'].append(i) # index of label in map
            data['MFCC'].append(mfcc.T.tolist())
            data['file'].append(filename)
                
        
with open('data.json','w') as f:
    json.dump(data,f,indent=4)
    
    
    
    