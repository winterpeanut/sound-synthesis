
import os
import pickle
import sys
import h5py
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import tensorflow.compat.v1 as tf
import numpy as np
sys.path.append('/Users/xinglidongsheng/ml/models-master/research/audioset/vggish')

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import tf_slim as slim
import vggish_params as params

#import extract_layers
# os.chdir('codes')
import models
import sklearn.metrics as metrics

#%matplotlib inline
 
#prepare data

# CHOOSE THE AUGMENTS IF NECESSARY
data_path = '/Users/xinglidongsheng/ml/models-master/research/audioset/vggish' # the path of the 'npc_v4_data.h5' file
batch_size = 2 # the batch size of the data loader
insp_layer = 3 # the middle layer extracted from alexnet, available in {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}

"""
The file npc_v4_data.h5 is structured in the following way:
images: a tensor of shape [num_images, width, height, num_colors]
neural data: a tensor of shape [num_repetitions, num_images, num_neurons]
target indices: a list containing the indices of target neurons in data tensor
For some animals the data is collected within two sessions that are indicated by session_x
"""

with h5py.File(os.path.join(data_path, 'mel.h5')) as hf:
    images_n = np.array(hf['mel'])
   # neural_n = np.array(hf['neural']['naturalistic']['monkey_m']['stretch']['session_2'])

neural_n = np.random.rand(10,10,32)

n_images = neural_n.shape[1]
n_neurons = neural_n.shape[2]
size_imags = images_n.shape[0]

reps = neural_n.shape[0]
rand_ind = np.arange(reps)
np.random.shuffle(rand_ind)

data_y_train = neural_n[:,:5].mean(0).astype(np.float32)
data_y_val_origin = neural_n[:, 5:].astype(np.float32)
data_y_val = data_y_val_origin.mean(0)


data_x = images_n[:, np.newaxis].astype(np.float32)
data_x = data_x / 255
data_x = np.tile(data_x, [1, 3, 1, 1])

data_x_train = data_x[:10]
data_x_val = data_x[10:]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    def __len__(self):
        return self.data_x.shape[0]

imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
imagenet_std =torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
#transform = lambda x : (x - imagenet_mean) / imagenet_std

dataset_train = Dataset(data_x_train, data_y_train)
dataset_val = Dataset(data_x_val, data_y_val)

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)

#Problem (a)

# CHOOSE THE AUGMENTS IF NECESSARY
lamd_s, lamd_d = [5e-3, 2e-3] # the coefficients of the losses. Try other coefficients!
epoches = 2 # total epochs for training the encoder
lr = 1e-3 # the learing rate for training the encoder

# alexnet = models.alexnet(pretrained=True)
# alexnet
# alexnet.eval()
# for param in alexnet.parameters():
#     param.requires_grad_(False)
# x = torch.from_numpy(data_x[0:1]).to(device)
# fmap = alexnet(x, layer=insp_layer)

def get_activations(input_tensor):
    # Load the model and PCA parameters
    vggish_slim.define_vggish_slim(training=False)
    sess = tf.Session()
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    # Get the activations of each layer
    activations = []
    with sess.as_default():
        for i, op in enumerate(sess.graph.get_operations()):
            if op.type == "Relu":
                layer_name = op.name.split("/")[-2]
                layer_activation = sess.run(op.outputs[0], feed_dict={features_tensor: input_tensor})
                activations.append((layer_name, layer_activation))
    return activations

flags = tf.app.flags
flags.DEFINE_string(
        'checkpoint', '/Users/xinglidongsheng/ml/models-master/research/audioset/vggish/vggish_model.ckpt',
        'Path to the VGGish checkpoint file.')
flags.DEFINE_string(
        'pca_params', '/Users/xinglidongsheng/ml/models-master/research/audioset/vggish/vggish_pca_params.npz',
        'Path to the VGGish PCA parameters file.')
FLAGS = flags.FLAGS

features = get_activations(images_n[0:2,:,:])
fmap = features[insp_layer][1]

neurons = data_y_train.shape[1]
sizes = fmap.shape[2:]
channels = fmap.shape[1]
#Downloading: "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth" to /home/hamza/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth
#100%|██████████| 233M/233M [01:07<00:00, 3.60MB/s] 

class conv_encoder(nn.Module):

    def __init__(self, neurons, sizes, channels):
        super(conv_encoder, self).__init__()
        # PUT YOUR CODES HERE
        self.W_s = nn.Parameter(torch.randn(size=(neurons,) + sizes))
        self.W_d = nn.Parameter(torch.randn(size = (neurons,channels,1,1)))
        self.W_b = nn.Parameter(torch.randn(size = (1,neurons)))


    def forward(self, x):
        # PUT YOUR CODES HERE
        out = torch.einsum('bchw , nhw -> bnchw',x,self.W_s) # dimension : N,n,C,h,w
        out = torch.stack(
            [F.conv2d(out[:,n,:,:,:],torch.unsqueeze(self.W_d[n],0)) for n in range(neurons)],dim=1) 
            #dimension:N,n,1,h,w
        out = torch.sum(out,dim=(2,3,4))
        out = out + self.W_b
        return out

def L_e(y,pred):
    return torch.mean(torch.sqrt(torch.sum((y-pred)**2,dim=1)))

def L_2(W_s,W_d,lamd_s=lamd_s,lamd_d=lamd_d):
    return lamd_s * torch.sum(W_s**2) + lamd_d * torch.sum(W_d**2)

K = torch.tensor([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]],dtype=torch.float).to(device)
def L_laplace(W_s,lamd_s=lamd_s):
    return lamd_s * torch.sum(F.conv2d(torch.unsqueeze(W_s,1),K.unsqueeze(0).unsqueeze(0))**2)


encoder = conv_encoder(neurons, sizes, channels).to(device)

def train_model(encoder, optimizer):
    losses = []
    encoder.train()
    for i,(x,y) in enumerate(loader_train):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        x = transform(x)
        fmap = alexnet(x,layer = insp_layer)
        out = encoder(fmap)
        l_e = L_e(y,out)
        l_2 = L_2(encoder.W_s,encoder.W_d)
        l_l = L_laplace(encoder.W_s)
#         print(f'L_e = {l_e} , L_2 = {l_2} , L_l = {l_l}')
        loss = L_e(y,out) + L_2(encoder.W_s,encoder.W_d) + L_laplace(encoder.W_s)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
#         print(f'iteration {i}, train loss: {losses[-1]}')
    
    return losses

def validate_model(encoder):
    encoder.eval()
    y_pred = []
    y_true = []
    losses = []
    for i,(x,y) in enumerate(loader_val):
        x = x.to(device)
        y = y.to(device)
        x = transform(x)
        fmap = alexnet(x,layer = insp_layer)
        out = encoder(fmap)
        y_pred.append(out)
        y_true.append(y)
        l_e = L_e(y,out)
        l_2 = L_2(encoder.W_s,encoder.W_d)
        l_l = L_laplace(encoder.W_s)
        print(f'L_e = {l_e} , L_2 = {l_2} , L_l = {l_l}')
        loss = L_e(y,out) + L_2(encoder.W_s,encoder.W_d) + L_laplace(encoder.W_s)
        losses.append(loss.item())
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    explained_variance = metrics.explained_variance_score(y_true = y_true.detach().cpu().numpy(),y_pred = y_pred.detach().cpu().numpy())
    return explained_variance,sum(losses)/len(losses)

"""
    You need to define the conv_encoder() class and train the encoder.
    The code of alexnet has been slightly modified from the torchvision, for convenience
    of extracting the middle layers.
    
    Example:
        >>> x = x.to(device) # x is a batch of images
        >>> x = transform(x)
        >>> fmap = alexnet(x, layer=insp_layer)
        >>> out= encoder(fmap)
        >>> ...
"""

'\n    You need to define the conv_encoder() class and train the encoder.\n    The code of alexnet has been slightly modified from the torchvision, for convenience\n    of extracting the middle layers.\n    \n    Example:\n        >>> x = x.to(device) # x is a batch of images\n        >>> x = transform(x)\n        >>> fmap = alexnet(x, layer=insp_layer)\n        >>> out= encoder(fmap)\n        >>> ...\n'

losses_train = []
losses_val = []
EVs = []

lr = 1e-3
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

for epoch in tqdm_notebook(range(epoches)):
    losses_train += train_model(encoder,optimizer)
    ev,loss = validate_model(encoder)
    EVs.append(ev)
    losses_val.append(loss)
    print(f'epoch {epoch}, EV = {ev}, val  loss = {loss} , train loss {sum(losses_train[-10:])/10}')
    
    
    
#Problem (b)
#b - Neural site response stretch
# CHOOSE THE AUGMENTS IF NECESSARY
n_id = np.random.randint(0, n_neurons, [5])  #inspect several neurons
rep_num = 5  # reptitions of each neuron
iters = 200 # iterations for synthesis
lr = 5e-3 # learing rate for synthesis
jitter = True # jitter the input for imitating the movement of the eyes

n_id_torch = torch.tensor(n_id.tolist(), dtype=torch.int64).to(device).repeat_interleave(rep_num).view(-1, 1)

def loss_TV(x):
    wd = x[:, :, 1:, :] - x[:, :, :-1, :]
    hd = x[:, :, :, 1:] - x[:, :, :, :-1]
    loss = (wd ** 2).mean([2]).sum() + (hd ** 2).mean([3]).sum()
    return loss

def jitter_image(img, max_pixels=19):
    sx, sy = np.random.randint(-max_pixels, max_pixels, size=2).tolist()
    img_shift = img.roll(sx, 3).roll(sy, 2)
    return img_shift

encoder.eval()
for param in encoder.parameters():
#     param.requires_grad_(False)
    param = param.detach()
    
init_images = np.random.uniform(0, 1, [len(n_id_torch), 1, 299, 299]).astype(np.float32)
x = torch.from_numpy(init_images)
x = x.to(device)

    
# PUT YOUR CODES HERE
""" 
    In each iteration, you can use jitter_image() function to jitter the input 
    image, and use transform predefined to apply standard imagnet preprocess
    
    Example:
        >>> x_jittered = jitter_image(x)
        >>> x_jittered = transform(x_jitter.repeat([1, 3, 1, 1]))
        ...
    
    In addition, it is better to normalize when updating the input image x
    
    Example:
        >>> grad = x.grad.detach()
        >>> grad /= grad.std([1, 2, 3], keepdim=True) + 1e-8
        >>> x = (x - lr * grad).clamp(0, 1).detach()    
"""
init_images = np.random.uniform(0, 1, [len(n_id_torch), 1, 299, 299]).astype(np.float32)
x = torch.from_numpy(init_images)
x = x.to(device)
losses_tv = []
losses_res = []
losses = []
best_l_res = 0
best_x = None
lr = 1e-1
lamb_tv = 1e-1
for i in tqdm_notebook(range(iters)):
    x = Variable(x,requires_grad = True)

    x_jittered = jitter_image(x,1)

    x_jittered = transform(x_jittered.repeat([1, 3, 1, 1]))

    fmap = alexnet(x_jittered,layer = insp_layer)

    out = encoder(fmap)
    l_res = -torch.sum(out.gather(dim=1,index=n_id_torch))
    l_tv = loss_TV(x)
    loss = l_res + lamb_tv * l_tv
    
    losses_tv.append(l_tv.item())
    losses_res.append(l_res.item())
    losses.append(loss.item())
    
    print(f'iteration {i}, L_tv = {losses_tv[-1]} , L_res = {losses_res[-1]} , total_loss = {losses[-1]}')
    if l_res < best_l_res:
        best_x = torch.tensor(x)
        best_l_res = l_res

    loss.backward()
    grad = x.grad.detach()

    grad /= grad.std([1, 2, 3], keepdim=True) + 1e-8
    x = (x - lr * grad).clamp(0, 1).detach()