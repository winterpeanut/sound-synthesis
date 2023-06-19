# Code:
# Tensorflow implementation (Google research): https://github.com/tensorflow/models/tree/master/research/audioset/vggish
# Keras implementation with TF backend (DTaoo): 
# https://github.com/DTaoo/VGGish 


# In[]:
# import numpy as np
# import tensorflow.compat.v1 as tf


work_dir='D:\\EXP2\\AcoSemDNN_Behav_fMRI_Repo\\AcoSemDNN_Behav_fMRI_Repo'

import sys
# import scipy.io
import pandas as pd
import os
from copy import deepcopy #import copy as c
import h5py
import soundfile as sf
#import tqdm 

models_dir='D:\\python\\vggish'
dnn_dir= 'D:\\python\\vggish'
weights_dir=models_dir
sys.path.insert(1,dnn_dir)
import tensorflow as tf
from tensorflow.keras.models import Model

import numpy as np
import vggish_input_tensor
import vggish_keras
#os.chdir(work_dir)

import matplotlib.pyplot as plt

# In[]:
dataset='formisano'
#dataset='giordano'
#dataset='giordano2'

## In[prepare stim and output lists]:
stims_dir=work_dir+ '\\data\\' + dataset + '_stimuli\\'
stims_list=stims_dir+'stimuli_list.csv'
out_dir=work_dir+'\\data\\'+dataset+'_dnns\\vggish2\\'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

wavs_dir=stims_dir+'wav_16kHz\\'
dat=pd.read_csv(stims_list,header=None)
dat=dat.drop(labels=0,axis=0)
stim_fns=np.asarray(dat[0])

out_fns=deepcopy(stim_fns);
for i in range(len(stim_fns)):
    out_fns[i]=out_dir+out_fns[i].replace('wav','hdf5')



#weights pretrained model
#weights_fn = weights_dir+'\vggish_audioset_weights.h5'
checkpoint_path = 'vggish_weights.ckpt'
#pca_params_path = 'vggish_pca_params.npz'
#load model and weights
#from keras.models import Model
model = vggish_keras.get_vggish_keras()
#extracting layers name
layer_nams = [layer.name for layer in model.layers]


def loss_TV(x):
    hd = x[ 1:,:] - x[ :-1,:]
    
    loss = tf.reduce_mean(tf.square(hd))
    #loss = tf.reduce_mean(tf.square(hd))
    return loss

def rms(x):
    sq = tf.square(x)
    ms = tf.reduce_mean(sq)
    rms = tf.sqrt(ms)
    return rms
def jitter_image(img, max_pixels=6):
    sx, sy = np.random.randint(-max_pixels, max_pixels, size=2).tolist()
    img_shift = tf.roll(img,shift=[sx,sy],axis = [1,2])
    return img_shift

def feature_transform(x):
    x = tf.reduce_mean(x,axis=1);
    x = tf.reshape(x,[1,-1])
    #x = tf.transpose(x,[1,0])
    return x

def extract_features(model,x,op_layer):
    extractor = tf.keras.Model(inputs=model.input,outputs=model.get_layer(op_layer).output)
    features = extractor(x)
    return features



model.load_weights(checkpoint_path)



        
      
      
# In[]

oplayeri = 14;
oplayer = layer_nams[oplayeri]

iters = 800
#In[]
sr=16000;
l_lamb = 0.01
r_lamb = 1

dospeech = ''#'_dospeech'
pattern = 'stretch'
Betastr = 'realbeta'
Starter = 'randstarter'
method = 'Decomp'
roi = 'allroi'
subj = 1
cluster = [2,4,6]

features = np.zeros([4,len(cluster),128])

for j in range(1,5):
 for ci in range(0,len(cluster)): 
  out_dir_mel= 'D:\\EXP2\\Results\\DirectlyWaveform\\'+ pattern+'\\'
  out_file = out_dir_mel + pattern + '_'+Starter+'_'+ Betastr+'_'+str(iters)+'_'+str(r_lamb)+'rms_'+\
      str(l_lamb)+'lv_subj'+str(subj)+ '_'+roi+'_'+method+'_cluster' + str(cluster[ci])+'_6comps_'+ str(j) + '_waveform_SV10'+dospeech+'.hdf5'

  with h5py.File(out_file,'r') as f:
    setname = 'new_sound'
    tmpsound = f[setname][()]

  x = []
  x = tf.reshape(tmpsound,(16000,1))
  mel = vggish_input_tensor.waveform_to_examples(x,16000)
   
  features[j-1,ci,:] = extract_features(model,mel,oplayer)
  
out_dir_mel= 'D:\\EXP2\\Results\\DirectlyWaveform\\'+pattern+'\\ANNREPofSYN\\'
out_file = out_dir_mel + pattern + '_'+Starter+'_'+ Betastr+'_'+str(iters)+'_'+str(r_lamb)+'rms_'+\
        str(l_lamb)+'lv_subj'+str(subj)+ '_'+roi+'_'+method+'_6comps_waveform_SV10'+dospeech+'.hdf5'
    #out_file = out_dir+ 'opsound1.hdf5'

print('saving data')
   # if os.path.exists(out_file):
   #     os.remove(out_file)
hf=h5py.File(out_file,'w')
    #for j in range(0,2):#range(len(out)):
hf.create_dataset('ANNreps',data=features)
hf.close()

