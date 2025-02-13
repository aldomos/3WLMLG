from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from time import time




def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)





class Conv_agg(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, device, K=1,bias=True):
        super(Conv_agg, self).__init__()

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

        zer = torch.unsqueeze(batch_node*0.,0)

        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
        resx[:,edge_index[0],edge_index[1]] = X.T
        res = torch.matmul(resx,h)
        res = torch.matmul(res,self.weight).sum(0)           

        if self.bias is not None:
            res += self.bias

        return res
    
    
    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))


class G2N2Layer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,nnodeinput,nnodeoutput,device):
        super(G2N2Layer, self).__init__()
        self.nedgeinput = nedgeinput
        self.nnodeinput = nnodeinput
        self.nnodeoutput = nnodeoutput
        self.shapetensor = torch.zeros(nedgeinput,1).to(device)
         
        
        

        self.L1 = torch.nn.Sequential(torch.nn.Linear(nedgeinput, nedgeinput,bias=False),
                                      # torch.nn.BatchNorm1d(nedgeinput)
                                      )
        
        self.L2 = torch.nn.Sequential(torch.nn.Linear(nedgeinput, nedgeinput,bias=False),
                                      # torch.nn.BatchNorm1d(nedgeinput)
                                      )

        self.L3 = torch.nn.Sequential(torch.nn.Linear(nedgeinput, nedgeinput,bias=False),
                                      # torch.nn.BatchNorm1d(nedgeinput)
                                      )

        self.L4 = torch.nn.Sequential(torch.nn.Linear(nedgeinput, nedgeinput,bias=False),
                                      # torch.nn.BatchNorm1d(nedgeinput)
                                      )

        self.L5 = torch.nn.Sequential(torch.nn.Linear(nnodeinput,max(nnodeinput,nedgeinput),bias = False),
                                      # torch.nn.BatchNorm1d(max(nnodeinput,nedgeinput))
                                      )

        
        # self.mlp1 = torch.nn.Linear(2*nedgeinput + max(nnodeinput,nedgeinput) ,nedgeoutput,bias=False)
        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3*nedgeinput + max(nnodeinput,nedgeinput) ,4*nedgeoutput,bias=False),
                                                       torch.nn.ReLU(),torch.nn.Linear(4*nedgeoutput ,nedgeoutput,bias=False)) # attention nb couches MLP
            
        
        self.agg = Conv_agg(nnodeinput,nnodeoutput, device, K=nedgeoutput,bias = False)
        self.norm1 = torch.nn.LayerNorm(nedgeoutput,eps = 1e-5)
        #self.norm1 = torch.nn.BatchNorm1d(nedgeoutput,eps = 1e-5,track_running_stats=False)
        # self.normhadam = torch.nn.BatchNorm1d(nedgeinput,eps = 1e-5,track_running_stats=False)
        # self.normmatmul = torch.nn.BatchNorm1d(nedgeinput,eps = 1e-5,track_running_stats=False)
        #self.norm2 = torch.nn.BatchNorm1d(nnodeoutput,eps = 1e-5,track_running_stats=False)
        self.norm2 = torch.nn.LayerNorm(nnodeoutput,eps = 1e-5)

    
     
    
    def matmul(self,X,Y,batch_node,edge_index):
        
        zer = torch.unsqueeze(batch_node*0.,0).detach()

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()

        resy = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()
       
       
        resx[:,edge_index[0],edge_index[1]] = X.T
        
        resy[:,edge_index[0],edge_index[1]] = Y.T
        
        res = torch.matmul(resx,resy)
        
        return res[:,edge_index[0],edge_index[1]].T
    
    # def matmul(self,X,Y,batch_node,edge_index):
        
    #     n = batch_node.shape[0]//2
    #     end = X.shape[1]
    #     res = torch.matmul(X[:(n*n),:].T.reshape(end,n,n),Y[:(n*n),:].T.reshape(end,n,n))
    #     res2 = torch.matmul(X[(n*n):,:].T.reshape(end,n,n),Y[(n*n):,:].T.reshape(end,n,n))
    #     return torch.cat([res.reshape(end,n*n).T,res2.reshape(end,n*n).T],0)
    
       
    
    def diag(self,h,edge_index):
        res2= torch.diag_embed(h.T)
        return   res2[:,edge_index[0],edge_index[1]].T
            
    

    def forward(self, x,edge_index,C,batch_node):
        
        tmp_diag = self.diag( (self.L5(x)),edge_index)
        tmp_matmul = self.matmul(  (self.L3(C)),  (self.L4(C)),batch_node, edge_index)
        tmp=torch.cat([  (C),(self.L1(C))*  (self.L2(C)),tmp_diag,tmp_matmul],1)
        Cout = self.mlp(tmp)
        # Cout = (self.mlp1(tmp))
        
        xout=(self.agg(x, Cout, edge_index, batch_node))
        
        
        # tmp_diag = self.diag( (self.L5(x)/self.nnodeinput),edge_index)
        # tmp_matmul = self.matmul(  self.L32(torch.relu(self.L3(C)/self.nedgeinput)),  self.L42(torch.relu(self.L4(C)/self.nedgeinput)),batch_node, edge_index)
        # tmp=torch.cat([  (C),self.L12(torch.relu(self.L1(C)/self.nedgeinput))*  self.L22(torch.relu(self.L2(C)/self.nedgeinput)),tmp_diag,tmp_matmul],1)
        # Cout = self.mlp2(torch.relu((self.mlp1(tmp))))
        # Cout = (self.mlp1(tmp))
        
        # xout=(self.agg(x, Cout, edge_index, batch_node))/self.nnodeinput
        # print(Cout, self.norm1(Cout))
        return self.norm2(xout) ,  self.norm1(Cout) #xout,Cout
    
