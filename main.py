import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import spikingjelly.clock_driven.ann2snn.examples.utils as utils
import torch
import torch.nn as nn
from spikingjelly.clock_driven.ann2snn import classify_simulator, parser

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler



## Define the Dataset and the Dataloader

## Customized dataset
class AESDataset(Dataset):
    def __init__(self, data_path, label_path,max_values,min_values):
        # Transforms
        self.to_tensor = transforms.ToTensor()
    # Read the csv file
        self.data = pd.read_csv(data_path, header=None)
        self.label = pd.read_csv(label_path, header=None)
        self.data = self.data.to_numpy() 

        # print(self.data.shape[0])
        for i in range(self.data.shape[0]):
            # max_val = self.data[i][:].max()
            # min_val = self.data[i][:].min()
            # print(i)
            # print(self.data[i][0])
            max_val = max_values[i]
            min_val = min_values[i]
            self.data[i][:] = (self.data[i][:] - min_val) / (max_val - min_val)
        self.label = self.label.to_numpy()
        self.data_len = self.data.shape[1]

    def __getitem__(self, index):
        # Get image name from the pandas df
        return (self.data[:, index].astype(np.float32), self.label[:, index][0].astype(np.long))

    def __len__(self):
        return self.data_len

## Loading the data
    
def get_loader(batch_size,max_values,min_values):
    validation_split = .2
    shuffle_dataset = True
    random_seed = 50 
    dataset = AESDataset('data_training.csv', 'output.csv', max_values,min_values)
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=train_sampler, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              sampler=test_sampler, batch_size=batch_size)
    return train_loader, test_loader


## Define the ANN model (fully-connected). The inputs are 8 values and the output is 11 values (0 core -> 10 cores)
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, 5),
            nn.ReLU(),
            nn.Linear(5, 11)
        )
    def forward(self, x):
        x = self.network(x)
        return x


## Setting of the spikingjelly (log dir, training hyperparameters)

log_dir = "./aes_fc"
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)
train_device = "cpu"
parser_device = "cpu"
simulator_device = parser_device
batch_size = 64
learning_rate = 1e-1
T = 20
train_epoch = 100
model_name = "aes_fc"
load = False
if log_dir == None:
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = model_name + '-' + current_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, model_name + '.pkl')):
        print('%s has no model to load.' % (log_dir))
        load = False
    else:
        load = True


## Traing the ANN
# initialize data loader
max_values = [100,100,100,100,100,100,100,8]
min_values = [0,0,0,0,-100,-100,-100,-8]


train_data_loader, test_data_loader = get_loader(batch_size,max_values,min_values)
ann = ANN().to(train_device)
loss_function = nn.CrossEntropyLoss()
load=False
if not load:
    # print("a")
    optimizer = torch.optim.Adam(
        ann.parameters(), lr=learning_rate, weight_decay=5e-4)
    best_acc = 0.0
    for epoch in range(train_epoch):
        utils.train_ann(net=ann,
                        device=train_device,
                        data_loader=train_data_loader,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        epoch=epoch
                        )
        acc = utils.val_ann(net=ann,
                            device=train_device,
                            data_loader=test_data_loader,
                            loss_function=loss_function,
                            epoch=epoch
                            )
        if best_acc <= acc:
            utils.save_model(ann, log_dir, model_name + '.pkl')
ann = torch.load(os.path.join(log_dir, model_name + '.pkl'))
print('validating best model...')
ann_acc = utils.val_ann(net=ann,
                        device=train_device,
                        data_loader=test_data_loader,
                        loss_function=loss_function
                        )
print(ann_acc)

## Convert ANN to SNN

percentage = 0.004  # load 0.004 of the data
norm_data_list = []
for idx, (imgs, targets) in enumerate(train_data_loader):
    norm_data_list.append(imgs)
    if idx == int(len(train_data_loader) * percentage) - 1:
        break
norm_data = torch.cat(norm_data_list)
print('use %d imgs to parse' % (norm_data.size(0)))

# 调用parser，使用kernel为onnx
# Call parser, use onnx kernel
onnxparser = parser(name=model_name,
                    log_dir=log_dir + '/parser',
                    kernel='onnx')
#     ann = pytorch_kernel.layer_reduction(ann)
snn = onnxparser.parse(ann, norm_data.to(parser_device))

# Save SNN model
torch.save(snn, os.path.join(log_dir, 'snn-' + model_name + '.pkl'))

## Run the SNN simulator

fig = plt.figure('simulator')
    # define simulator for classification task
sim = classify_simulator(snn,
                             log_dir=log_dir + '/simulator',
                             device=simulator_device,
                             canvas=fig
                             )
    # Simulate SNN
sim.simulate(test_data_loader,
                 T=T,
                 online_drawer=True,
                 ann_acc=ann_acc,
                 fig_name=model_name,
                 step_max=True
                 )

## Save to "graph.pdf"

fig.savefig("graph.pdf")


## Quantization and running the testing RTL model

def fixed_point_quantize(x, wl, fl):
    """ Quantize a single precision Floating Point into low-precision Fixed
    Point

    Args:
        - :param: `x` (torch.Tensor) :  the single precision number to be
          quantized
        - :param: `wl` (int) : word length of the fixed point number being
          simulated
        - :param: `fl` (int) : fractional length of the fixed point number
          being simulated

    Returns:
        - a quantized low-precision block floating point number (torch.Tensor)
          """
    assert isinstance(x, torch.Tensor)
    # Clamp the tensor to range of [-2**(wl-1)/2**fl,2**(wl-1)/2**(wl-1)/2**fl]
    min_val, max_val = -2**(wl-1)/2**fl, (2**(wl-1)-1)/2**fl
    out = torch.clamp(x, min_val, max_val)
    # quantize process
    out = torch.round(out*2**fl)/2**fl
    # assert rounding in ["stochastic", "nearest"]
    return out


# from spikingjelly.clock_driven import neuron
snn = torch.load('./aes_fc/snn-aes_fc.pkl')
batch_size = 439 #set the batch_size to whole testset size
max_values = [100,100,100,100,100,100,100,8]
min_values = [0,0,0,0,-100,-100,-100,-8]
train_loader, test_loader = get_loader(batch_size,max_values,min_values)
test_data, test_label = next(iter(test_loader))
is_quantize = True

T = 30 #number of timestepts
n_fractional = 6

for module in snn.module_list:
    if (isinstance(module, nn.Linear) and is_quantize):
        module.weight.data = fixed_point_quantize(module.weight.data,8,n_fractional) #Quantize to n bit fractional
        module.bias.data = fixed_point_quantize(module.bias.data,16,2*n_fractional) #Quantize to n bit fractional
layer_keys = ["Flatten", "FC1", "IF1", "FC2"]
module_dict = dict(zip(layer_keys, snn.module_list))
if is_quantize:  
  test_data = fixed_point_quantize(test_data,8,n_fractional)
out = {}
membrane_potential = []
spike_list = []
for i in range(T):
    temp = test_data
    for key in module_dict.keys():
        out[key] = module_dict[key](temp)
        if key=="IF1":
            membrane_potential.append(module_dict[key].v)
            # print(module_dict[key].v[0])
            spike_list.append(out[key])
        temp = out[key]
    if i == 0:
        counter = out["FC2"]
    else:
        # print(out["FC2"])
        counter += out["FC2"]
print(counter)
correct = (counter.max(1)[1]==test_label).sum()
diff = (counter.max(1)[1]-test_label).max()
print(diff)
print("Correct= {:.2f}%".format(correct/batch_size*100))



## Quantization and save to file

from spikingjelly.clock_driven import neuron

snn = torch.load('./aes_fc/snn-aes_fc.pkl')
batch_size = 439 #set the batch_size to whole testset size
max_values = [100,100,100,100,100,100,100,8]
min_values = [0,0,0,0,-100,-100,-100,-8]
train_loader, test_loader = get_loader(batch_size,max_values,min_values)
test_data, test_label = next(iter(test_loader))


snn = torch.load('./aes_fc/snn-aes_fc.pkl')
is_quantize = True
T = 30 #number of timestepts
n_fractional = 6

for name, module in snn.named_modules():
    if (isinstance(module, nn.Linear) and is_quantize):
        module.weight.data = fixed_point_quantize(module.weight.data,8,n_fractional)
        if "3" in name: #Quantize to n bit fractional
           module.bias.data = fixed_point_quantize(module.bias.data,8,n_fractional)
        else:
           module.bias.data = fixed_point_quantize(module.bias.data,16,2*n_fractional)
if is_quantize:  
  test_data_new = fixed_point_quantize(test_data,8,n_fractional)
layer_keys = ["Flatten", "FC1", "IF1", "FC2"]
module_dict = dict(zip(layer_keys, snn.module_list))
out = {}
membrane_potential = []
spike_list = []
for i in range(T):
    temp = test_data_new
    for key in module_dict.keys():
        out[key] = module_dict[key](temp)
        if key == "IF1":
            membrane_potential.append(module_dict[key].v)
            spike_list.append(out[key])
            temp = out[key]
        else:
            temp =out[key]
    if i == 0:
        counter = out["FC2"]
    else:
        counter += out["FC2"]
correct = (counter.max(1)[1]==test_label).sum()
diff = (counter.max(1)[1]-test_label).max()
print(diff)
print("Max difference between SNN predictions and labels= "+ str(diff.item()))
diff = (counter.max(1)[1]-test_label).min()
print("Min difference between SNN predictions and labels= "+ str(diff.item()))
print(counter.max(1)[1])
print(counter)
print("Correct= {:.2f}%".format(correct/batch_size*100))


snn = torch.load('./aes_fc/snn-aes_fc.pkl')
is_quantize = True
T = 30 #number of timestepts
n_fractional = 6

for name, module in snn.named_modules():
    if (isinstance(module, nn.Linear) and is_quantize):
        module.weight.data = fixed_point_quantize(module.weight.data,8,n_fractional)*2**n_fractional
        if "3" in name: #Quantize to n bit fractional
           module.bias.data = fixed_point_quantize(module.bias.data,8,n_fractional)*2**n_fractional
        else:
           module.bias.data = fixed_point_quantize(module.bias.data,16,2*n_fractional)*2**(2*n_fractional)
    if (isinstance(module,neuron.IFNode)):
        module.v_threshold = module.v_threshold*2**(2*n_fractional)

if is_quantize:  
  test_data_new = fixed_point_quantize(test_data,8,n_fractional)*2**n_fractional
layer_keys = ["Flatten", "FC1", "IF1", "FC2"]
module_dict = dict(zip(layer_keys, snn.module_list))
out = {}
membrane_potential = []
spike_list = []
for i in range(T):
    temp = test_data_new
    for key in module_dict.keys():
        out[key] = module_dict[key](temp)
        if key == "IF1":
            membrane_potential.append(module_dict[key].v)
            # print(module_dict[key].v[0])
            spike_list.append(out[key])
            temp = out[key]
        else:
            temp =out[key]
    if i == 0:
        counter = out["FC2"]
    else:
        counter += out["FC2"]
        # print(out["FC2"])
#print(module_dict['IF1'].v)
correct = (counter.max(1)[1]==test_label).sum()
print(counter)
#print(module_dict['FC1'].weight.data)
#print(spike_list[5][0])
#print(membrane_potential[29][0])
print("Correct= {:.2f}%".format(correct/batch_size*100))

for key in module_dict.keys():
  if (isinstance(module_dict[key],nn.Linear)):
    filename_weight = key + "_weight.txt"
    filename_bias = key +"_bias.txt"
    weight = module_dict[key].weight.data.numpy()
    weight = np.reshape(weight,(-1,1))
    #print(weight)
    np.savetxt(filename_weight,weight,fmt="%d")
    bias = module_dict[key].bias.data.numpy()
    bias = np.reshape(bias,(-1,1))
    np.savetxt(filename_bias,bias,fmt="%d")
np.savetxt("input_img.txt",test_data_new,fmt="%d")
np.savetxt("input_label.txt",test_label,fmt="%d")
np.savetxt("v.txt",counter.detach().cpu().numpy(),fmt="%d")





## Test with a single input

def snn_single_input(snn, x, max_values, min_values,T):
    """ Run inference on a single input

    Args:
        - :param: `snn` (SNN model) :  the trained SNN model
        - :param: `x` (numpy.ndarray) : numpy array with size of [8,]
           for example: x = np.array([64, 64, 64, 64, 0 , 0 ,0 , 32])
        - :param: `max_values` (list) : list of max values
        - :param: `min_values` (list) : list of min_values
        - :param: `T` (int) : number of time steps
    Returns:
        - Guessed number of cores
          """
    x = x.astype(float)
    for i in range(x.shape[0]):
      max_val = max_values[i]
      min_val = min_values[i]
      x[i] = (x[i]-min_val)/(max_val-min_val)
    x = np.expand_dims(x,axis=0)
    x = torch.from_numpy(x).float()
    for i in range(T):
      if i==0:
        counter=snn(x)
      else:
        counter+=snn(x)
    return counter.max(1)[1]



snn = torch.load('./aes_fc/snn-aes_fc.pkl') # reload the model from pkl file
max_values = [100,100,100,100,100,100,100,8] # set max values 
min_values = [0,0,0,0,-100,-100,-100,-8] # set min values
T = 30 
x = np.array([66,61,55,50,5,6,5,-4]) # input for prediction
result = snn_single_input(snn,x,max_values,min_values,T) # run the prediction
print(result) # -- Print


CSVData = open("data_testing-Full_random.csv") # read file
test_random_val = np.genfromtxt(CSVData, delimiter=",").T # transpose

indx = 1 # testing index
x= test_random_val[indx][0:8]
print(x)

result = snn_single_input(snn,x,max_values,min_values,T) # run the prediction
print(result)

import csv
n_data_rand = test_random_val.shape[0] # get the shape of data
predict_result = -999*np.ones(n_data_rand)
for indx in range (n_data_rand):
  x= test_random_val[indx][0:8]
  predict_result[indx] = snn_single_input(snn,x,max_values,min_values,T) # run the prediction
with open('predict_result.csv', 'w') as f:
  writer = csv.writer(f)  
  writer.writerow(predict_result)
print(predict_result)



