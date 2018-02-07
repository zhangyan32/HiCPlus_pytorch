# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gzip
import model
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import sys
import torch.nn as nn
import utils
import math

use_gpu = 1

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


down_sample_ratio = 16
epochs = 10
HiC_max_value = 100



# This block is the actual training data used in the training. The training data is too large to put on Github, so only toy data is used. 
input_file = '/home/zhangyan/Desktop/chr21.10kb.matrix'
low_resolution_samples, index = utils.divide(input_file)

low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)

batch_size = low_resolution_samples.shape[0]

# Reshape the high-quality Hi-C sample as the target value of the training. 
sample_size = low_resolution_samples.shape[-1]
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2
output_length = sample_size - padding


print low_resolution_samples.shape

lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)

hires_loader = lowres_loader

model = model.Net(40, 28)
model.load_state_dict(torch.load('../model/pytorch_model_12000'))
if use_gpu:
    model = model.cuda()

_loss = nn.MSELoss()


running_loss = 0.0
running_loss_validate = 0.0
reg_loss = 0.0


for i, (v1, v2) in enumerate(zip(lowres_loader, hires_loader)):    
    _lowRes, _ = v1
    _highRes, _ = v2

    _lowRes = Variable(_lowRes).float()
    _highRes = Variable(_highRes).float()

    
    if use_gpu:
        _lowRes = _lowRes.cuda()
        _highRes = _highRes.cuda()
    y_prediction = model(_lowRes)

    
print '-------', i, running_loss, strftime("%Y-%m-%d %H:%M:%S", gmtime())


y_predict = y_prediction.data.cpu().numpy()


print y_predict.shape

# recombine samples

length = int(y_predict.shape[2])
y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))


chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

chrN = 21

length = chrs_length[chrN-1]/10000

prediction_1 = np.zeros((length, length))


print 'predicted sample: ', y_predict.shape, '; index shape is: ', index.shape
#print index
for i in range(0, y_predict.shape[0]):          
    if (int(index[i][1]) != chrN):
        continue
#print index[i]
x = int(index[i][2])
y = int(index[i][3])
#print np.count_nonzero(y_predict[i])
prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]

np.save(input_file + 'enhanced.npy', prediction_1)





