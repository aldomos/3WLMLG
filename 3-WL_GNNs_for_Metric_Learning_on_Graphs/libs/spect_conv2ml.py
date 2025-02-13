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

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class SpectConv(MessagePassing):
    r"""
    """
    def __init__(self, in_channels, out_channels, K=1, selfconn=True, depthwise=False,bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpectConv, self).__init__(**kwargs)

        assert K > 0       

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise=depthwise
        #self.selfmult=selfmult
        
        self.selfconn=selfconn 
        
        
        if self.selfconn:
            K=K+1

        if self.depthwise:            
            self.DSweight = Parameter(torch.Tensor(K,in_channels))            
            self.nsup=K
            K=1
        

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        if self.depthwise:
            zeros(self.DSweight)

    def forward(self, x,edge_index, edge_attr, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
    

        Tx_0 = x  
        out=0
        if not self.depthwise:
            enditr=self.weight.size(0)
            if self.selfconn:
                out = torch.matmul(Tx_0, self.weight[-1]) 
                enditr-=1 

            for i in range(0,enditr):
                h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,i], size=None) 
                # if self.selfmult:
                #     h=h*Tx_0                
                out = out+ torch.matmul(h, self.weight[i])
        else:
            enditr=self.nsup
            if self.selfconn:
                out = Tx_0* self.DSweight[-1] 
                enditr-=1 

            out= out + (1+self.DSweight[0:1,:])*self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,0], size=None)
            for i in range(1,enditr):
                out= out + self.DSweight[i:i+1,:]*self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,i], size=None) 

            out = torch.matmul(out, self.weight[0])                   

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))

class SpectConCatConv(MessagePassing):
    r"""
    """
    def __init__(self, in_channels, out_channels, K, selfconn=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpectConCatConv, self).__init__(**kwargs)

        assert K > 0       

        self.in_channels = in_channels
        self.out_channels = out_channels       
        
        self.selfconn=selfconn 
        
        
        if self.selfconn:
            K=K+1       
        

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(K*out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)        

    def forward(self, x, edge_index,edge_attr, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        
        Tx_0 = x  
        out=[]
        
        enditr=self.weight.size(0)
        if self.selfconn:
            out.append(torch.matmul(Tx_0, self.weight[-1])) 
            enditr-=1 

        for i in range(0,enditr):
            h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,i], size=None)                           
            out.append(torch.matmul(h, self.weight[i]))

        out=torch.cat(out,1)               

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))


class EdgeEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()
        
        self.fc1 = torch.nn.Linear(emb_dim[0], emb_dim[1])
        self.fc2 = torch.nn.Linear(emb_dim[1], emb_dim[2])

    def forward(self, edge_attr):
        x = F.relu(self.fc1(edge_attr))
        x = F.relu(self.fc2(x))
        return x


class GMNLayer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,ninp,nout1,nout2,learnedge= True):
        super(GMNLayer, self).__init__()

        self.learnedge=learnedge
        self.nout2=nout2
        self.nedgeinput = nedgeinput
        self.nout1 = nout1
        self.nout1 = nout2
        
        
        
        if self.learnedge:
            self.fc1_11 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_12 = torch.nn.Linear(2*nedgeinput, nedgeinput,bias=False)
            self.fc1_21 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_22 = torch.nn.Linear(2*nedgeinput, nedgeinput,bias=False)
            self.fc1_21 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_22 = torch.nn.Linear(2*nedgeinput, nedgeinput,bias=False)
            self.fc1_31 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_32 = torch.nn.Linear(2*nedgeinput, nedgeinput,bias=False)
            self.fc1_41 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_42 = torch.nn.Linear(2*nedgeinput, nedgeinput,bias=False)
            self.fc1_61 = torch.nn.Linear(4*nedgeinput ,8*nedgeoutput,bias=False)
            self.fc1_62 = torch.nn.Linear(8*nedgeinput ,4*nedgeoutput,bias=False)
            
            self.fcnode1 = torch.nn.Linear(ninp,2*ninp,bias = False)
            self.fcnode2 = torch.nn.Linear(2*ninp,nedgeinput,bias = False)
            # self.fcnode2 = torch.nn.Linear(ninp,nedgeinput,bias = False)
            
        else:
            nedgeoutput=nedgeinput
        
        self.conv1 = SpectConv(ninp,nout1, nedgeoutput,selfconn=False,bias = False)

        if nout2>0:
            self.fc11 = torch.nn.Linear(ninp, nout2) 
            self.fc12 = torch.nn.Linear(ninp, nout2)
    
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
        x = torch.sparse_coo_tensor(e_ind,X.T.reshape((N)))
        y = torch.sparse_coo_tensor(e_ind,Y.T.reshape((N)))
        tmp = torch.sparse.mm(x,y)
        e_ind  = self.indreshape(edge_index[0],enditr)
        tmp = torch.sparse_coo_tensor(e_ind,tmp.coalesce().values())
        
        return tmp.to_dense()
    
    def denormalize(self,X,shapenode,norm):
        device = X.device
        res = torch.zeros(X.shape).to(device)
        n = shapenode.shape[0]
        nor = torch.tensor(norm).to(device)
        ind = 0
        for i in range(n):
            m = int((shapenode[i]*shapenode[i]).item())
            ind+= m
            res[ind:ind+m,:] = nor[i]
            
        return X*res
    
    
    def vector_to_matrix(self,h,k,edge_index):
        res = h@k.T
        return res[:,edge_index[0],edge_index[1]].T
        
        
        
    
    def diag(self,h,edge_index):
        res2= torch.diag_embed(h.T)
        return   res2[:,edge_index[0],edge_index[1]].T 
            
    

    def forward(self, x,edge_index,SP):
        
        
        
        
        if self.learnedge:
            tmp_diag = self.diag( self.fcnode2(F.relu(self.fcnode1(x)/2/self.ninp))/self.nedgeinput,edge_index)
            # tmp_vect = self.vector_to_matrix(self.fcnode(x), self.fcnode2(x), edge_index)
            tmp_matmul = self.matmulopti2(  self.fc1_42(F.relu(self.fc1_41(SP)/self.nedgeinput)/2/self.nedgeinput),  self.fc1_52(F.relu(self.fc1_51(SP)/self.nedgeinput)/2/self.nedgeinput),edge_index)
            tmp=torch.cat([  (SP),  self.fc1_22(F.relu(self.fc1_21(SP)/self.nedgeinput)/2/self.nedgeinput)*   self.fc1_32(F.relu(self.fc1_31(SP)/self.nedgeinput)/2/self.nedgeinput),tmp_matmul,tmp_diag],1)
            # tmp=torch.cat([SP,  (self.fc1_2(SP))* (self.fc1_3(SP))],1)
            edge_attr = F.relu(self.fc1_62(F.relu(self.fc1_61(tmp))))

        if self.nout2>0:            
            x=torch.cat([ (self.conv1(x, edge_index,edge_attr)/self.nout1),   (self.fc11(x)/self.nout2)*  (self.fc12(x)/self.nout2)],1)
        else:
            x=(self.conv1(x, edge_index,edge_attr))
        return x ,  edge_attr