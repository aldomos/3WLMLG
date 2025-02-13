import random
import numpy as np
import torch
import torch.nn.functional as F
import ot
from external.PairNorm.layers import PairNorm
from torch.nn.functional import normalize
import torch_geometric as tg
from torch.autograd import Variable
from torch.autograd import Function
import layer_G2N2
import modules_ppgn
from easydict import EasyDict

class siamese_GNN(torch.nn.Module):
    def __init__(self, args, number_of_labels):

        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def setup_layers(self):
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None
        self.edges_features_layer = torch.nn.Linear(3,64) #attention size 
          
        self.gnn_layers = torch.nn.ModuleList([])
        
        num_ftrs = self.number_labels #+self.args.rw_k # dim + rw dim
        self.num_gnn_layers = len(self.args.gnn_size)
        hidden_size_node = 1 #self.args.gnn_size[0] 
        hidden_size_edge = 2 #features dim + A + I
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model=='ppgn':
            config_dict = {}
            config_dict["architecture"]={}
            config_dict["architecture"]["depth_of_mlp"] = 2
            config = EasyDict(config_dict)
            for i in range(self.num_gnn_layers):
                self.gnn_layers.append(modules_ppgn.RegularBlock(config,hidden_size_edge,self.args.gnn_size[i]))
                hidden_size_edge = self.args.gnn_size[i]
        else :
            for i in range(self.num_gnn_layers):
                self.gnn_layers.append(layer_G2N2.G2N2Layer(nedgeinput=hidden_size_edge,nedgeoutput=self.args.gnn_size[i],
                nnodeinput=hidden_size_node,nnodeoutput=self.args.gnn_size[i],device = device))
                hidden_size_node = self.args.gnn_size[i]
                hidden_size_edge = hidden_size_node
                
        self.node_out_emb = torch.nn.Sequential(
            torch.nn.Linear(self.args.gnn_size[0]*(self.num_gnn_layers), 2*self.args.gnn_size[0]*(self.num_gnn_layers)),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.args.gnn_size[0]*(self.num_gnn_layers), self.args.gnn_size[0]*(self.num_gnn_layers))#self.args.gnn_size[0]*(self.num_gnn_layers)
        )
        if args.readout:
            self.edge_out_emb = torch.nn.Sequential(
                torch.nn.Linear(2*self.args.gnn_size[0]*(self.num_gnn_layers), 4*self.args.gnn_size[0]*(self.num_gnn_layers)),
                torch.nn.ReLU(),
                torch.nn.Linear(4*self.args.gnn_size[0]*(self.num_gnn_layers), 2*self.args.gnn_size[0]*(self.num_gnn_layers))#size modified self.args.gnn_size[0]*(self.num_gnn_layers)
            )
        else:
            self.edge_out_emb = torch.nn.Sequential(
                torch.nn.Linear(self.args.gnn_size[0]*(self.num_gnn_layers), 2*self.args.gnn_size[0]*(self.num_gnn_layers)),
                torch.nn.ReLU(),
                torch.nn.Linear(2*self.args.gnn_size[0]*(self.num_gnn_layers), self.args.gnn_size[0]*(self.num_gnn_layers))#size modified self.args.gnn_size[0]*(self.num_gnn_layers)
            )

    def graph_3Wl_embedding(self,features, edge_index,edges_features,node_index):


        node_feature_tensor = []
        edge_feature_tensor = []

        for i in range(self.num_gnn_layers-1):

            if args.model == 'ppgn':
                edges_features = self.gnn_layers[i](edges_features)
                edges_features = torch.nn.functional.relu(edges_features)
                edge_feature_tensor.append(edges_features)
            else :
                features,edges_features = self.gnn_layers[i](features, edge_index,edges_features,node_index)
                features = torch.nn.functional.relu(features)
                edges_features = torch.nn.functional.relu(edges_features)
                node_feature_tensor.append(features)
                edge_feature_tensor.append(edges_features)
        
        if args.model == 'ppgn':
            edges_features = self.gnn_layers[-1](edges_features)
            edges_features = torch.nn.functional.relu(edges_features)
            edge_feature_tensor.append(edges_features)
            edge_feature_tensor = [torch.cat(edge_feature_tensor,1)]
        else :
            features,edges_features = self.gnn_layers[-1](features, edge_index,edges_features,node_index)
            features = torch.nn.functional.relu(features)
            edges_features = torch.nn.functional.relu(edges_features)
            node_feature_tensor.append(features)
            edge_feature_tensor.append(edges_features)
            node_feature_tensor = [torch.cat(node_feature_tensor,1)]
            edge_feature_tensor = [torch.cat(edge_feature_tensor,1)]
        
        #readout = tg.nn.global_add_pool(abstract_feature_matrices[0],batch)
        return node_feature_tensor,edge_feature_tensor
        

    def diag_offdiag_maxpool(self,input):
        N = input.shape[-1]

        max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

        # with torch.no_grad():
        max_val = torch.max(max_diag)
        min_val = torch.max(-1 * input)
        val = torch.abs(torch.add(max_val, min_val))

        min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

        max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

        return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S
        
    def forward(self, batch, return_matching=True, return_embedding = False):
        if args.model == 'ppgn':
            e_features1 = batch["ef1"].cuda()
            e_features2 = batch["ef2"].cuda()
            edge_feature_tensor_1 = self.graph_3Wl_embedding(None,None,e_features1,None)
            edge_feature_tensor_2  = self.graph_3Wl_embedding(None,None,e_features2,None)
            node_feature_tensor_1 = None
            node_feature_tensor_2 = None
            if args.readout:
                emb_g1 = self.diag_offdiag_maxpool(edge_feature_tensor_1[0])
                emb_g2 = self.diag_offdiag_maxpool(edge_feature_tensor_2[0])
                emb_g1 = self.edge_out_emb(emb_g1)
                emb_g2 = self.edge_out_emb(emb_g2)
                return node_feature_tensor_1,emb_g1,node_feature_tensor_2,emb_g2
            else:
                edge_feature_tensor_1 = torch.permute(edge_feature_tensor_1[0],(0,2,3,1))
                edge_feature_tensor_2 = torch.permute(edge_feature_tensor_2[0],(0,2,3,1))
                edge_feature_tensor_1 = self.edge_out_emb(edge_feature_tensor_1)
                edge_feature_tensor_2 = self.edge_out_emb(edge_feature_tensor_2)
                return node_feature_tensor_1,edge_feature_tensor_1,node_feature_tensor_2,edge_feature_tensor_2   
       
        else : 
            edge_index_1 = batch["eid1"].to(device='cuda', dtype=torch.int64)
            edge_index_2 = batch["eid2"].to(device='cuda', dtype=torch.int64)
            node_index_1 = batch["nid1"].to(device='cuda', dtype=torch.int64)
            node_index_2 = batch["nid2"].to(device='cuda', dtype=torch.int64)
        
            features_1 = batch["nf1"].cuda()
            features_2 = batch["nf2"].cuda()
            e_features1 = batch["ef1"].cuda()
            e_features2 = batch["ef2"].cuda()

            node_feature_tensor_1,edge_feature_tensor_1 = self.graph_3Wl_embedding(features_1,edge_index_1,e_features1,node_index_1)
            node_feature_tensor_2,edge_feature_tensor_2  = self.graph_3Wl_embedding(features_2,edge_index_2,e_features2,node_index_2)
            node_feature_tensor_1 = self.node_out_emb(node_feature_tensor_1[0])
            node_feature_tensor_2 = self.node_out_emb(node_feature_tensor_2[0])
            edge_feature_tensor_1 = self.edge_out_emb(edge_feature_tensor_1[0])
            edge_feature_tensor_2 = self.edge_out_emb(edge_feature_tensor_2[0])
            return node_feature_tensor_1,edge_feature_tensor_1,node_feature_tensor_2,edge_feature_tensor_2     
       
