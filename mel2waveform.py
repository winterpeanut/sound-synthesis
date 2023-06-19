# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:44:47 2023

@author: P70077519
"""
import torch.nn as nn 
import torch
import numpy as np
import torchaudio
import h5py
import wave

method = 'Decomp'
roi = 'allroi'
subj = 3
cluster = 5
work_dir='D:\\EXP2\\Results\\SythesizedMel\\'

    
 # In[]  
class dB_to_Amplitude(nn.Module):
     def __call__(self, features):
         return(torch.from_numpy(np.power(10.0, features.numpy()/10.0)))

def get_waveform_from_logMel(features, n_fft=512, hop_length=256, sr = 16000):
     n_mels = features.shape[-2]
     inverse_transform = torch.nn.Sequential(
             dB_to_Amplitude(),
             torchaudio.transforms.InverseMelScale(n_stft=n_fft//2+1, n_mels=n_mels, sample_rate=sr),
             torchaudio.transforms.GriffinLim(n_fft=512, hop_length=256)
             )
     waveform = inverse_transform(torch.squeeze(features))
     return torch.unsqueeze(waveform,0)
 
for i in range(0,4):
    read_file = work_dir+  'randstarter700_'+str(r_lamb)+'rms_'+str(l_lamb)+'lv_subj'+\
        str(subj)+'_'+roi+'_'+method+'_cluster' + str(cluster)+'_6comps_'+ str(i+1) + '.hdf5'
    #betaname = 'subj1_all_roi_cluster4_6comps.h5'
    with h5py.File(read_file,'r') as f:
        setname = 'new_mel'
        newmel = f[setname][()]
    x = newmel   
    x = x.squeeze(0)
    x= x.astype('double')

    x=torch.tensor(x,dtype=torch.float32)
    waveform = get_waveform_from_logMel(x)
    waveform = waveform.numpy()

    out_dir_mel= 'D:\\EXP2\\Results\\SythesizedWaveform\\' 
    out_waveform = out_dir_mel+'randstarter700_'+str(r_lamb)+'rms_'+str(l_lamb)+'lv_subj'+str(subj)+\
        '_'+roi+'_'+method+'_cluster'+str(cluster)+'_6comps_'+ str(i+1) +'.wav'

    with wave.open(out_waveform,'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(waveform.tobytes())
    
    out_waveform_h5 = out_waveform.replace('wav','hdf5')
    hf=h5py.File(out_waveform_h5,'w')
    #for j in range(0,2):#range(len(out)):
    hf.create_dataset('new_sound',data=waveform)
    hf.close()