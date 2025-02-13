import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (global_add_pool, global_mean_pool, global_max_pool,GCNConv)
import numpy as np
from libs.spect_conv import SpectConv,ML3Layer, GMNLayer
from libs.utils import GraphCountDataset,SpectralDesign,get_n_params
import scipy.io as sio
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau



class GcnNet(nn.Module):
    
    def __init__(self, num_layer = 5, node_input_dim = 1, output_dim = 1, neuron_dim = [16,16,16,16,16] , final_neuron = 16, readout_type  = "sum" ):
        
        super(GcnNet, self).__init__()
        
        self.num_layer = num_layer
        self.node_input_dim = node_input_dim
        self.outpout_dim = output_dim
        self.neuron_dim = neuron_dim
        self.final_neuron = final_neuron
        self.readout_type = readout_type
        self.conv = []
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layer must be greater than 1")
            
        if self.num_layer != self.neuron_dim.shape[0]:
            raise ValueError("Number of GNN layer must match length of neuron_dim.\n num_layer = {}, neuron_dim length = {}".format(self.num_layer,self.neuron_dim.shape[0]))
        
        if self.readout_type == "sum":
            self.readout = global_add_pool
        elif self.readout_type == "mean":
            self.readout = global_mean_pool
        elif self.readout_type == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid readout type")
        
        
        for i in range(self.num_layer):
            if i == 0:
                self.conv.append(GCNConv(self.node_input_dim, self.neuron[i], cached=False))
            else:
                self.conv.append(GCNConv(self.neuron[i-1], self.neuron[i], cached=False))
        self.fc1 = torch.nn.Linear(self.neuron[self.num_layer-1], self.final_neuron)
        self.fc2 = torch.nn.Linear(self.final_neuron, self.output_dim)
        
        

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index  

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = self.readout(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 


class GNNML3(nn.Module):
    def __init__(self,  node_input_dim, edge_input_dim, nfreq, output_dim,
                 num_layer = 5, neuron_dim = [16,16,16,16,16], 
                 neuron_dim2 = [16,16,16,16,16], edge_dim = 16 ,final_neuron = 16,
                 readout_type  = "sum" ):
        
        super(GNNML3, self).__init__()
        
        self.num_layer = num_layer
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.nfreq = nfreq
        self.outpout_dim = output_dim
        self.neuron_dim = neuron_dim
        self.neuron_dim2 = neuron_dim2
        self.edge_dim = edge_dim
        self.final_neuron = final_neuron
        self.readout_type = readout_type
        self.conv = []
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layer must be greater than 1")
            
        if self.num_layer != self.neuron_dim.shape[0]:
            raise ValueError("Number of GNN layer must match length of neuron_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,self.neuron_dim.shape[0]))
            
        if self.num_layer != self.neuron_dim2.shape[0]:
            raise ValueError("Number of GNN layer must match length of neuron_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,self.neuron_dim2.shape[0]))
        
        if self.readout_type == "sum":
            self.readout = global_add_pool
        elif self.readout_type == "mean":
            self.readout = global_mean_pool
        elif self.readout_type == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid readout type")
            
        for i in range(self.num_layer):
            if i == 0:
                self.conv.append(ML3Layer(nedgeinput=self.nfreq, nedgeinput2= self.edge_input_dim,
                                          nedgeoutput = self.edge_dim , ninp= self.node_input_dim,
                                          nout1= self.neuron_dim[0], nout2= self.neuron_dim2[0]))
            else:
                nin = self.neuron_dim[i-1] + self.neuron_dim2[i-1]
                self.conv.append(ML3Layer(nedgeinput=self.nfreq, nedgeinput2= self.edge_input_dim,
                                          nedgeoutput = self.edge_dim , ninp= nin,
                                          nout1= self.neuron_dim[i], nout2= self.neuron_dim2[i]))
        nin = self.neuron_dim[self.num_layer-1] + self.neuron_dim2[self.num_layer-1]
        self.fc1 = torch.nn.Linear(nin, self.final_neuron)
        self.fc2 = torch.nn.Linear(self.final_neuron, self.output_dim)
        
        
        
        
        