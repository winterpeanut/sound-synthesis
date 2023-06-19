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
import vggish_input

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
#model.load_weights(checkpoint_path)
#extracting layers name
layer_nams = [layer.name for layer in model.layers]

# In[]:
#n_id = np.random.randint(0, 6, [5])  #inspect several neurons
# rep_num = 5  # reptitions of each neuron

# #n_id_torch = tf.tensor(n_id.tolist(), dtype=tf.int64).repeat_interleave(rep_num).view(-1, 1)
# init_images = np.random.uniform(0, 1, [ 1, 96, 64]).astype(np.float64)
# x = tf.convert_to_tensor(init_images,dtype=tf.float64)
#     #x = x.to(device)
# losses_tv = []
# losses_res = []
# losses = []



# def loss_TV(x):
#     wd = x[ :,1:, :] - x[ :,:-1, :]
#     hd = x[ :,:, 1:] - x[ :,:, :-1]
#     #loss = (wd ** 2).mean([1]).sum() + (hd ** 2).mean([2]).sum()
#     loss = tf.reduce_mean(tf.square(wd))+tf.reduce_mean(tf.square(hd))
#     #loss = tf.reduce_mean(tf.square(hd))
#     return loss

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





# def frame(data, window_length, hop_length):
#     data = tf.expand_dims(data, axis=-1)  # add an extra dimension for channels
#     num_samples = tf.shape(data)[0]
#     num_frames = 1 + tf.cast(tf.floor((num_samples - window_length) / hop_length), tf.int32)
#     shape = [num_frames, window_length, 1]
#     frames = tf.TensorArray(dtype=data.dtype, size=num_frames)
#     for i in tf.range(num_frames):
#         start = i * hop_length
#         end = start + window_length
#         frames = frames.write(i, tf.expand_dims(data[start:end], axis=0))
#     frames = tf.reshape(frames.stack(), shape)
#     return tf.squeeze(frames, axis=-1)




     
      
      

     
      
      
# In[]
lr = 1e-3 # learing rate for synthesis   
iters = 800 # iterations for synthesis
opsound = 99;
oplayeri = 8;
oplayer = layer_nams[oplayeri]
wav_file=wavs_dir+stim_fns[opsound]
print(wav_file)
wav_data, sr = sf.read(wav_file, dtype=np.int16)
assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
input_batch = vggish_input.waveform_to_examples(samples,sr)
input_batch2 = vggish_input_tensor.waveform_to_examples(samples,sr)

input_batch = tf.convert_to_tensor(input_batch)

#In[]
sr=16000;
l_lamb = 0.01
r_lamb = 1

dospeech = 'dospeech'
pattern = 'stretch'
Betastr = 'realbeta'
Starter = 'randstarter'
method = 'Decomp'
roi = 'allroi'
subj = 1
cluster = [2,5]

if Betastr == 'realbeta':
  betaname = 'subj'+str(subj)+'_'+roi+'_'+method+'_6comps_SV10'+dospeech+'.h5'
  model.load_weights(checkpoint_path)

elif Betastr == 'randbeta':
  betaname = 'subj'+str(subj)+'_'+roi+'_'+method+'_6comps_SV10_rand'+dospeech+'.h5'

with h5py.File(betaname,'r') as f:
    setname = 'beta'
    tmpbeta = f[setname][()]
tmpbeta = tmpbeta.transpose([1,0])
realbeta = tf.convert_to_tensor(tmpbeta,dtype=tf.float32) 

for ci in range(0,realbeta.shape[1]): 
 bestp = []
 bestx = []
 allcost = []
 allpred = []
 allrms = []
 alllv = []
 allx = []
 for j in range(1,5):
    x = []
    #init_images = np.random.uniform(0, 1, [ 1, 96, 64]).astype(np.float64)
    init_images = np.random.uniform(-1, 1, [ 16000, 1]).astype(np.float64)
    soundrms = rms(samples)
    randrms = rms(init_images)
    init_images = (soundrms/randrms)*init_images
    if Starter == 'randstarter':
       x = tf.convert_to_tensor(init_images,dtype=tf.float64)
    elif Starter =='soundstarter':
       x = tf.convert_to_tensor(samples)
       x = tf.reshape(x,(16000,1))
          
    # elif Betastr == 'randbeta':
    #    betarand = np.random.uniform(-1, 1, [ 4097, realbeta.shape[1]]).astype(np.float64)
    #    beta = betarand
    if Betastr == 'shufflebeta':
       beta = tf.random.shuffle(realbeta)
    else:
       beta = realbeta
    # out_dir_mel= 'D:\\EXP2\\Results\\DirectlyWaveform\\'
    # out_file = out_dir_mel + 'randstarter700_'+str(r_lamb)+'rms_'+str(l_lamb)+'lv_subj'+str(subj)+\
    #     '_'+roi+'_'+method+'_cluster' + str(cluster)+'_6comps_'+ str(j) + '_start.hdf5'
    # #out_file = out_dir+ 'opsound1.hdf5'

    # print('saving data')

    # hf=h5py.File(out_file,'w')
    # #for j in range(0,2):#range(len(out)):
    # hf.create_dataset('start_sound',data=x)
    # hf.close()
    
    best_pred = 0
    best_x = None
    best_cost=0
    costeach = np.zeros([iters,1])
    predeach = np.zeros([iters,1])
    rmseach = np.zeros([iters,1])
    lveach = np.zeros([iters,1])
    xeach = np.zeros([16000,iters])
    for i in range(iters):#range(len(out_fns)):
        xeach[:,i] = x.numpy().reshape((16000,))
        with tf.GradientTape() as tape:
            tape.watch(x)
        # # create a specific model that takes as input the input of vggish, and outputs the layers
        # extractor = Model(inputs=model.input, outputs=model.get_layer(oplayer).output)
        # features = extractor.predict(x[:,:,:,None],steps=1)  
           # x_jittered = jitter_image(x,1)
            mel = vggish_input_tensor.waveform_to_examples(x,16000)
        #starter_rms = rms(input_batch) 
            x_rms = rms(x)  
            lv = loss_TV(x)
        #sf = x_rms/starter_rms
        #x1 = tf.multiply(x, sf)
        #x_jittered = transform(x_jittered.repeat([1, 3, 1, 1]))
            features = extract_features(model,mel,oplayer)
        #features = tf.convert_to_tensor(features)
            
            if oplayeri < 12:
                fea = feature_transform(features)   
            else:
                fea = features    
            ones = tf.ones((tf.shape(fea)[0],1),dtype=fea.dtype)
            fea = tf.concat([ones,fea],axis=1)
            
            pred = tf.matmul(fea,beta)                            
            predtmp = tf.cast(-pred[0,ci],dtype=tf.float64)        
            predrms =  tf.add(predtmp, r_lamb*x_rms)
            cost =  tf.add(predrms, l_lamb*lv)
            
            print(f'cluster={cluster[ci]}, sound={j}, iters={i}, cost={cost.numpy()}, pred={pred[0,ci].numpy()}')

            if cost < best_cost:
                best_x = x
                best_cost = cost
                best_pred = pred
            
            grad = tape.gradient(cost, x)
            grad/= tf.math.reduce_std(grad,axis=[0,1],keepdims=True) + 1e-8
            x = x - lr*grad
            costeach[i]=cost.numpy() 
            predeach[i]=pred[0,ci].numpy() 
            rmseach[i]=x_rms.numpy() 
            lveach[i]=lv.numpy()
 #   print(layer)
    bestp.append(best_pred[0,ci].numpy())
    bestx.append(best_x.numpy())
    allcost.append(costeach)
    allpred.append(predeach)
    allrms.append(rmseach)
    alllv.append(lveach)
    allx.append(xeach)
   # out_file=out_fns[14]
    out_dir_mel= 'D:\\EXP2\\Results\\DirectlyWaveform\\stretch\\'
    out_file = out_dir_mel + pattern+'_'+Starter+'_'+ Betastr+'_'+str(iters)+'_'+str(r_lamb)+'rms_'+\
        str(l_lamb)+'lv_subj'+str(subj)+ '_'+roi+'_'+method+'_cluster' + str(cluster[ci])+'_6comps_'+ str(j) + '_waveform_SV10'+dospeech+'.hdf5'
    #out_file = out_dir+ 'opsound1.hdf5'

    print('saving data')
   # if os.path.exists(out_file):
   #     os.remove(out_file)
    
    hf=h5py.File(out_file,'w')
    #for j in range(0,2):#range(len(out)):
    hf.create_dataset('new_sound',data=best_x)
    hf.close()

 out_dir_mel= 'D:\\EXP2\\Results\\DirectlyWaveform\\Stretch\\logdata\\'
 out_file = out_dir_mel +pattern+ '_'+Starter+'_'+ Betastr+'_'+str(iters)+'_'+str(r_lamb)+'rms_'+str(l_lamb)+'lv_subj'+\
    str(subj)+ '_'+roi+'_'+method+'_cluster' + str(cluster[ci])+'_6comps__waveform_SV10'+dospeech+'.hdf5'
 hf=h5py.File(out_file,'w')
 hf.create_dataset('allcost',data=allcost)
 hf.create_dataset('allpred',data=allpred)
 hf.create_dataset('allrms',data=allrms)
 hf.create_dataset('alllv',data=alllv)
 hf.create_dataset('bestpred',data=bestp)
 hf.create_dataset('allwaveform',data=allx)

 hf.close()

print('all done')
# In[]
iteri = 0
plt.plot(allcost[iteri])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
plt.plot(-allpred[iteri])
plt.xlabel('Iteration')
plt.ylabel('Pred')
plt.show()
plt.plot(alllv[iteri])
plt.xlabel('Iteration')
plt.ylabel('loss_var')
plt.show()
plt.plot(allrms[iteri])
plt.xlabel('Iteration')
plt.ylabel('rms')
plt.show()
 