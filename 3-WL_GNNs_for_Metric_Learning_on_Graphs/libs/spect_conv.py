from typing import Optional
from torch_geometric.typing import OptTensor
import math
import numpy as np
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import time

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class SpectConv(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, device, K=1,bias=True):
        super(SpectConv, self).__init__()

        assert K > 0       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapetensor = torch.zeros((K,1)).to(device)
        

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
       
    def forward(self, h, X,edge_index,batch_node):
        """"""
        
        zer = torch.unsqueeze(batch_node*0.,0)

        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
        resx[:,edge_index[0],edge_index[1]] = X.T
        res = torch.matmul(resx,h)
        res = torch.matmul(res,self.weight).sum(0)           

        if self.bias is not None:
            res += self.bias

        return res

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))




class GMNLayer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,ninp,nout1,device):
        super(GMNLayer, self).__init__()

        self.nedgeinput = nedgeinput
        self.nout1 = nout1
        self.ninp = ninp
        self.shapetensor = torch.zeros(nedgeinput,1).to(device)
        
        
    
        # self.fc1_1 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.fc1_2 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.fc1_3 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.fc1_4 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.fc1_5 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.fc1_6 = torch.nn.Linear(3*nedgeinput + max(nedgeinput,ninp)  ,8*nedgeinput,bias=False)
        self.fc1_7 = torch.nn.Linear(8*nedgeinput ,nedgeoutput,bias=False)
        
        self.fcnode = torch.nn.Linear(ninp,max(nedgeinput,ninp),bias = False)
            

        
        self.conv1 = SpectConv(ninp,nout1, device, K=nedgeoutput,bias = True)


    
    # def matmulopti2(self,X,Y,batch_node,edge_index):

    #     zer = torch.unsqueeze(batch_node*0.,0)

    #     resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)

    #     resy = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
       
    #     resx[:,edge_index[0],edge_index[1]] = X.T
        
    #     resy[:,edge_index[0],edge_index[1]] = Y.T
        
    #     res = torch.matmul(resx,resy)
        
    #     return res[:,edge_index[0],edge_index[1]].T
    
    def matmulopti2(self,X,Y,batch_node,edge_index):
        
        zer = torch.unsqueeze(batch_node*0.,0).detach()

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()

        resy = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()
        # resy = torch.clone(resx)
       
        resx[:,edge_index[0],edge_index[1]] = X.T
        
        resy[:,edge_index[0],edge_index[1]] = Y.T
        
        res = torch.matmul(resx,resy)
        
        return res[:,edge_index[0],edge_index[1]].T
            
            
     
     
    
    def matmulopti3(self,X,Y,edge_index,batch_edge_size):
        enditr = X.shape[1]
        N = int(X.shape[0]*enditr)
        m = edge_index[0].max()+1
        e_tmp =torch.repeat_interleave(self.shapetensor*m,edge_index.shape[1])
        
        e_ind = torch.vstack((edge_index[0].repeat(1,enditr)+e_tmp,edge_index[1].repeat(1,enditr)+e_tmp))
        e_ind2 = torch.vstack((batch_edge_size.repeat(1,enditr),torch.repeat_interleave(self.shapetensor, edge_index.shape[1])))
        
        tp = torch.sparse_coo_tensor(e_ind,X.T.reshape((N)))
        
        tmp = torch.sparse_coo_tensor(e_ind2,torch.sparse.mm(torch.sparse_coo_tensor(e_ind,X.T.reshape((N))),torch.sparse_coo_tensor(e_ind,Y.T.reshape((N)))).coalesce().values())
        
        
        
        return tmp.to_dense()
    
    
        
        
    
    def diag(self,h,edge_index):
        res2= torch.diag_embed(h.T)
        return   res2[:,edge_index[0],edge_index[1]].T 
    
    # def diag2(self,h,index_diag,batch_diag_size):
    #     enditr = h.shape[1]
    #     m = index_diag[0].max()+1
    #     e_tmp =torch.repeat_interleave(self.shapetensor*m,index_diag.shape[1])
    #     e_ind = torch.vstack((index_diag[0].repeat(1,enditr)+e_tmp,index_diag[1].repeat(1,enditr)+e_tmp))
    #     print(e_ind.shape,h.shape)
    #     res2= torch.sparse_coo_tensor(e_ind, h.T.reshape((h.shape[0]*h.shape[1])))
    #     e_ind2 = torch.vstack((batch_diag_size.repeat(1,h.shape[1]),torch.repeat_interleave(self.shapetensor, index_diag.shape[1])))
    #     print(e_ind2)
    #     return   (torch.sparse_coo_tensor(e_ind2,res2.coalesce().values())).to_dense()
            
    

    # def forward(self, x,edge_index,SP,batch_edge_size,nb_nodes,index_diag,batch_diag_size):
    def forward(self, x,edge_index,SP,batch_node):
        
       
        """tmp_diag = self.diag( (self.fcnode(x)/self.nedgeinput),edge_index,)
        # tmp_diag = self.diag( x,edge_index)
        # tmp_diag = self.diag2( (self.fcnode(x)/self.nedgeinput),index_diag,batch_diag_size)

        tmp_matmul = self.matmulopti2(  (self.fc1_4(SP)/self.nedgeinput),  (self.fc1_5(SP)/self.nedgeinput),batch_node, edge_index)

        
        # tmp_matmul = self.matmulopti2(  (self.fc1_4(SP)/self.nedgeinput),  (self.fc1_5(SP)/self.nedgeinput),nb_nodes)
        
        
        tmp=torch.cat([  (SP),(self.fc1_2(SP)/self.nedgeinput)*  (self.fc1_3(SP)/self.nedgeinput),tmp_diag,tmp_matmul],1)
        # tmp=torch.cat([  (SP), ],1)
        # tmp=torch.cat([SP,  (self.fc1_2(SP))* (self.fc1_3(SP))],1)
        edge_attr = self.fc1_7(F.relu((self.fc1_6(tmp))))

        
        x=(self.conv1(x,edge_attr,edge_index,batch_node))/self.ninp"""
        
        tmp_diag = self.diag( (self.fcnode(x)),edge_index)
        # tmp_diag = self.diag( x,edge_index)
        # tmp_diag = self.diag2( (self.fcnode(x)/self.nedgeinput),index_diag,batch_diag_size)

        tmp_matmul = self.matmulopti2(  (self.fc1_4(SP)), (self.fc1_5(SP)),batch_node, edge_index)

        
        # tmp_matmul = self.matmulopti2(  (self.fc1_4(SP)/self.nedgeinput),  (self.fc1_5(SP)/self.nedgeinput),nb_nodes)
        
        
        tmp=torch.cat([  (SP),(self.fc1_2(SP))*  (self.fc1_3(SP)),tmp_diag,tmp_matmul],1)
        # tmp=torch.cat([  (SP), ],1)
        # tmp=torch.cat([SP,  (self.fc1_2(SP))* (self.fc1_3(SP))],1)
        edge_attr = self.fc1_7(F.relu((self.fc1_6(tmp))))

        
        x=(self.conv1(x,edge_attr,edge_index,batch_node))
        return x ,  edge_attr
    
class PPGNLayer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,learnedge= True):
        super(PPGNLayer, self).__init__()

        self.learnedge=learnedge

        self.nedgeinput = nedgeinput
        
        
        
        if self.learnedge:

            self.fc1_4 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
            self.fc1_5 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
            self.fc1_6 = torch.nn.Linear(2*nedgeinput ,nedgeoutput,bias=False)
            

            
       
    
    def matmul(self,X,Y,edge_index,shapemat,shapenode):
        device = X.device
        ind = 0
        indic = 0
        res2  = torch.zeros((X.shape[1],X.shape[0])).to(device)
        for i in range(len(shapemat)):
            sh = shapemat[i]
            nh = shapenode[i]
            ind1 = torch.where(edge_index[0]>=indic)
            
            ind2 = torch.where(edge_index[0][ind1]< nh+indic)
            ind1 = (ind1[0][ind2],)
            
            index = (edge_index[0][ind1]-indic,edge_index[1][ind1]-indic)
            x_tmp = X[ind:(ind+sh),:]
            y_tmp = Y[ind:(ind+sh),:]
            x = torch.zeros(x_tmp.shape[1],nh,nh).to(device)
            y = torch.zeros(x_tmp.shape[1],nh,nh).to(device)
            
            x[:,index[0],index[1]] = x_tmp.T
            y[:,index[0],index[1]] = y_tmp.T
            res2[:,ind:(ind+sh)]+= torch.bmm(y,x)[:,index[0],index[1]]
            ind += sh
            indic += nh
        
        return res2.T
    
    def matmulopti(self,X,Y,edge_index):
        enditr = X.shape[1]
        res = []
        for i in range(enditr):
            x = torch.sparse_coo_tensor(edge_index,X[:,i])
            y = torch.sparse_coo_tensor(edge_index,Y[:,i])
            tmp = torch.sparse.mm(x,y).to_dense()
            tmp =  tmp[edge_index[0],edge_index[1]]
            res.append( tmp.reshape((tmp.shape[0],1)))
        
        return torch.cat(res,1)
    
    def indreshape(self,ind,m):
        device = ind.device
        l = torch.arange(ind.shape[0]).to(device)
        res1 = []
        res2 = []
        for i in range(m):
            res1.append(l*0+i)
            res2.append(l)
        res1 = torch.cat(res1)
        res2 = torch.cat(res2)
        return torch.vstack((res2,res1))
     
    
    def matmulopti2(self,X,Y,edge_index):
        enditr = X.shape[1]
        N = int(X.shape[0]*enditr)
        e0 = [edge_index[0]]
        e1 = [edge_index[1]]
        m = edge_index[0].max()+1
        for i in range(enditr-1):
            e0.append(edge_index[0]+(i+1)*m)
            e1.append(edge_index[1]+(i+1)*m)
        e0 = torch.cat(e0)
        e1 = torch.cat(e1)
        e_ind = torch.vstack([e0,e1])
        e_ind2  = self.indreshape(edge_index[0],enditr)
        tmp = torch.sparse_coo_tensor(e_ind2,torch.sparse.mm(torch.sparse_coo_tensor(e_ind,X.T.reshape((N))),torch.sparse_coo_tensor(e_ind,Y.T.reshape((N)))).coalesce().values())
        
        return tmp.to_dense()
    

            
    

    def forward(self,edge_index,SP):
        
        
        
        
        if self.learnedge:

            tmp_matmul = self.matmulopti2(  (self.fc1_4(SP)/self.nedgeinput),  (self.fc1_5(SP)/self.nedgeinput),edge_index)
            tmp=torch.cat([  (SP), tmp_matmul],1)
            # tmp=torch.cat([SP,  (self.fc1_2(SP))* (self.fc1_3(SP))],1)
            edge_attr = F.relu(self.fc1_6(tmp))

        return  edge_attr
