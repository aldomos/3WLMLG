import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import torch.nn.functional as Func
import numpy as np
import scipy.io as sio
import libs.utils_matlang as mat
import libs.countsub as cs
import pandas as pd
import gzip
import pickle
import os
import libs.graphs as graph
from rdkit.Chem import AllChem
import networkx as nx

import scipy.spatial.distance as dist


def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class GraphCounttrianglenodeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCounttrianglenodeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["triangleset.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        # Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            A4=A3.dot(a)
            A5=A4.dot(a)
            one = mat.one(a)
            I = mat.diag(one)
            J = one@one.T - I
            tri=torch.tensor(A3*I/2).sum(1)
            tri = tri.reshape((tri.shape[0],1))
            tailedtri=torch.tensor((np.diag(A3)/2)*(a.sum(0)-2))
            tailedtri = tailedtri.reshape((tailedtri.shape[0],1))
            cyc4=1/2*torch.tensor(A4*I-A2*J-A2*A2*I).sum(1)
            cyc4 = cyc4.reshape((cyc4.shape[0],1))
            trisquare=torch.tensor(1/4*(A2*a*(A2-(A2>0))).sum(1))
            trisquare = trisquare.reshape((trisquare.shape[0],1))
            cyc5=torch.tensor(cs.fivecyclenode(a))
            cyc5 = cyc5.reshape((cyc5.shape[0],1))
           
        
    
            expy=torch.cat([tri,tailedtri,cyc4,trisquare,cyc5],1)

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            edge_attr = None
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphCounttriangleDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCounttriangleDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["triangleset.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        # Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            A4=A3.dot(a)
            A5=A4.dot(a)
            one = mat.one(a)
            I = mat.diag(one)
            J = one@one.T - I
            tri=np.trace(A3)/6
            tailedtri=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(A4*I-A2*J-A2*A2*I).sum()
            trisquare=1/4*(A2*a*(A2-(A2>0))).sum()
            cyc5=cs.fivecycle(a)
           
        
    
            expy=torch.tensor([[tri,tailedtri,cyc4,trisquare,cyc5]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            edge_attr = None
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            A4=A3.dot(a)
            A5=A4.dot(a)
            one = mat.one(a)
            I = mat.diag(one)
            J = one@one.T - I
            tri=np.trace(A3)/6
            tailedtri=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(A4*I-A2*J-A2*A2*I).sum()
            trisquare=1/4*(A2*a*(A2-(A2>0))).sum()
            cyc5=cs.fivecycle(a)
           
        
    
            expy=torch.tensor([[tri,tailedtri,cyc4,trisquare,cyc5]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            edge_attr = None
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class GraphCountnodeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountnodeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        # Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            A4=A3.dot(a)
            A5=A4.dot(a)
            one = mat.one(a)
            I = mat.diag(one)
            J = one@one.T - I
            tri=torch.tensor(A3*I/2).sum(1)
            tri = tri.reshape((tri.shape[0],1))
            tailedtri=torch.tensor((np.diag(A3)/2)*(a.sum(0)-2))
            tailedtri = tailedtri.reshape((tailedtri.shape[0],1))
            cyc4=1/2*torch.tensor(A4*I-A2*J-A2*A2*I).sum(1)
            cyc4 = cyc4.reshape((cyc4.shape[0],1))
            trisquare=torch.tensor(1/4*(A2*a*(A2-(A2>0))).sum(1))
            trisquare = trisquare.reshape((trisquare.shape[0],1))
            cyc5=torch.tensor(cs.fivecyclenode(a))
            cyc5 = cyc5.reshape((cyc5.shape[0],1))
           
        
    
            expy=torch.cat([tri,tailedtri,cyc4,trisquare,cyc5],1)

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            edge_attr = None
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class HivDataset_to_count(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HivDataset_to_count, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["edge.csv.gz","edge-feat.csv.gz","graph-label.csv.gz","node-feat.csv.gz","num-edge-list.csv.gz","num-node-list.csv.gz"]

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        
        data_list =  []
        
        df_num_node = pd.read_csv(self.raw_paths[5], compression='gzip', header = None)
        df_num_edge = pd.read_csv(self.raw_paths[4], compression='gzip', header = None)
        # df_node_feat = pd.read_csv(self.raw_paths[3], compression='gzip', header = None)

        # df_edge_feat = pd.read_csv(self.raw_paths[1], compression='gzip', header = None)
        df_edge = pd.read_csv(self.raw_paths[0], compression='gzip', header = None)
        
        loc_node = 0
        loc_edge = 0
        total = 0
        for i in range(len(df_num_node)):
            nod = np.array(df_num_node.iloc[[i]])[0][0]
            edg = np.array(df_num_edge.iloc[[i]])[0][0]
            E = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),0])
            F = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),1])
            # test = torch.Tensor(np.array(df_node_feat.iloc[range(loc_node,loc_node+nod)])).type(torch.int32)
            edge_index = torch.Tensor(np.vstack((E,F))).type(torch.int64)
            x = torch.ones((nod,1))
            edge_attr = None
            
            
            A=np.zeros((nod,nod),dtype=np.float32)
            A[E,F]=1
            if np.linalg.norm(A-A.T)>0:
                A = A + A.T
            # if(A[0,:].sum())< 1:
            #     total += 1
            #     print(edge_index,edg)
            a=A
            A2=a.dot(a)
            A3=A2.dot(a)
            A4=A3.dot(a)
            A5=A4.dot(a)
            one = mat.one(a)
            I = mat.diag(one)
            J = one@one.T - I
            tri=np.trace(A3)/6
            tailedtri=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(A4*I-A2*J-A2*A2*I).sum()
            trisquare=1/4*(A2*a*(A2-(A2>0))).sum()
            cyc5=cs.fivecycle(a)
            
    
            expy=torch.tensor([[tri,tailedtri,cyc4,trisquare,cyc5]])
            
            
            loc_node += nod
            loc_edge += edg
            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
        
       
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])     


class HivDataset_to_countnode(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HivDataset_to_countnode, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["edge.csv.gz","edge-feat.csv.gz","graph-label.csv.gz","node-feat.csv.gz","num-edge-list.csv.gz","num-node-list.csv.gz"]

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        
        data_list =  []
        
        df_num_node = pd.read_csv(self.raw_paths[5], compression='gzip', header = None)
        df_num_edge = pd.read_csv(self.raw_paths[4], compression='gzip', header = None)
        # df_node_feat = pd.read_csv(self.raw_paths[3], compression='gzip', header = None)

        # df_edge_feat = pd.read_csv(self.raw_paths[1], compression='gzip', header = None)
        df_edge = pd.read_csv(self.raw_paths[0], compression='gzip', header = None)
        
        loc_node = 0
        loc_edge = 0
        total = 0
        for i in range(len(df_num_node)):
            nod = np.array(df_num_node.iloc[[i]])[0][0]
            edg = np.array(df_num_edge.iloc[[i]])[0][0]
            E = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),0])
            F = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),1])
            # test = torch.Tensor(np.array(df_node_feat.iloc[range(loc_node,loc_node+nod)])).type(torch.int32)
            edge_index = torch.Tensor(np.vstack((E,F))).type(torch.int64)
            x = torch.ones((nod,1))
            edge_attr = None
            
            
            A=np.zeros((nod,nod),dtype=np.float32)
            A[E,F]=1
            if np.linalg.norm(A-A.T)>0:
                A = A + A.T
            # if(A[0,:].sum())< 1:
            #     total += 1
            #     print(edge_index,edg)
            a=A
            A2=a.dot(a)
            A3=A2.dot(a)
            A4=A3.dot(a)
            A5=A4.dot(a)
            one = mat.one(a)
            I = mat.diag(one)
            J = one@one.T - I
            tri=torch.tensor(A3*I/2).sum(1)
            tri = tri.reshape((tri.shape[0],1))
            tailedtri=torch.tensor((np.diag(A3)/2)*(a.sum(0)-2))
            tailedtri = tailedtri.reshape((tailedtri.shape[0],1))
            cyc4=1/2*torch.tensor(A4*I-A2*J-A2*A2*I).sum(1)
            cyc4 = cyc4.reshape((cyc4.shape[0],1))
            trisquare=torch.tensor(1/4*(A2*a*(A2-(A2>0))).sum(1))
            trisquare = trisquare.reshape((trisquare.shape[0],1))
            cyc5=torch.tensor(cs.fivecyclenode(a))
            cyc5 = cyc5.reshape((cyc5.shape[0],1))
           
        
    
            expy=torch.cat([tri,tailedtri,cyc4,trisquare,cyc5],1)
            
            
            
            loc_node += nod
            loc_edge += edg
            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
        
       
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])    


class HivDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HivDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["edge.csv.gz","edge-feat.csv.gz","graph-label.csv.gz","node-feat.csv.gz","num-edge-list.csv.gz","num-node-list.csv.gz"]

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        
        data_list =  []
        
        df_num_node = pd.read_csv(self.raw_paths[5], compression='gzip', header = None)
        df_num_edge = pd.read_csv(self.raw_paths[4], compression='gzip', header = None)
        df_node_feat = pd.read_csv(self.raw_paths[3], compression='gzip', header = None)
        df_y = pd.read_csv(self.raw_paths[2], compression='gzip', header = None)
        df_edge_feat = pd.read_csv(self.raw_paths[1], compression='gzip', header = None)
        df_edge = pd.read_csv(self.raw_paths[0], compression='gzip', header = None)
        
        loc_node = 0
        loc_edge = 0

        for i in range(len(df_num_node)):
            nod = np.array(df_num_node.iloc[[i]])[0][0]
            edg = np.array(df_num_edge.iloc[[i]])[0][0]
            E = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),0])
            F = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),1])
            y = torch.tensor(np.array(df_y.iloc[[i]])).type(torch.float32)
            edge_index = torch.Tensor(np.vstack((E,F))).type(torch.int64)
            x = torch.Tensor(np.array(df_node_feat.iloc[range(loc_node,loc_node+nod)])).type(torch.float32)
            
            edge_attr = torch.Tensor(np.array(df_edge_feat.iloc[range(loc_edge,loc_edge+edg)]))
            edge_attr = torch.cat([Func.one_hot(edge_attr[:,1].type(torch.int64),4).type(torch.float32),edge_attr[:,1:]],1)

    
            
            
                
            
    
                
            data_list.append(Data(edge_index=edge_index, x=x, y=y, edge_attr = edge_attr))
            
                
            loc_node += nod
            loc_edge += edg
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
        
        
class PcbaDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PcbaDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["edge.csv.gz","edge-feat.csv.gz","graph-label.csv.gz","node-feat.csv.gz","num-edge-list.csv.gz","num-node-list.csv.gz"]

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        
        data_list =  []
        
        df_num_node = pd.read_csv(self.raw_paths[5], compression='gzip', header = None)
        df_num_edge = pd.read_csv(self.raw_paths[4], compression='gzip', header = None)
        df_node_feat = pd.read_csv(self.raw_paths[3], compression='gzip', header = None)
        df_y = pd.read_csv(self.raw_paths[2], compression='gzip', header = None)
        df_edge_feat = pd.read_csv(self.raw_paths[1], compression='gzip', header = None)
        df_edge = pd.read_csv(self.raw_paths[0], compression='gzip', header = None)
        
        loc_node = 0
        loc_edge = 0
        for i in range(len(df_num_node)):
            nod = np.array(df_num_node.iloc[[i]])[0][0]
            edg = np.array(df_num_edge.iloc[[i]])[0][0]
            E = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),0])
            F = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),1])
            y = torch.tensor(np.array(df_y.iloc[[i]])).type(torch.float32)
            edge_index = torch.Tensor(np.vstack((E,F))).type(torch.int64)
            x = torch.Tensor(np.array(df_node_feat.iloc[range(loc_node,loc_node+nod)]))
            
            edge_attr = torch.Tensor(np.array(df_edge_feat.iloc[range(loc_edge,loc_edge+edg)]))
            
            
            loc_node += nod
            loc_edge += edg
            
            data_list.append(Data(edge_index=edge_index, x=x, y=y, edge_attr = edge_attr))
        
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(data_list[0])
                
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Zinc12KDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Zinc12KDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Zinc.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) 
        # list of adjacency matrix
        F=a['F'][0]
        A=a['E'][0]
        Y=a['Y']
        
        ntype=21

        data_list = []
        for i in range(len(A)):
            edge_attr = None
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.zeros(A[i].shape[0],ntype)
            deg=(A[i]>0).sum(1)
            for j in range(F[i][0].shape[0]):
                # put atom code
                x[j,F[i][0][j]]=1
                
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y,edge_attr =edge_attr))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        with open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb") as f:
            data_list = pickle.load(f)
        data_list2 = []
        for data in data_list:
            data_list2.append(Data(**data.__dict__))


        if self.pre_filter is not None:
            data_list = [data for data in data_list2 if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list2]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  
        
class TwoDGrid30(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwoDGrid30, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["TwoDGrid30.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        
        Y=a['Y'] 
        
        M = a['M']
        F=F.astype(np.float32)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.reshape(torch.tensor(F[i]),(F[i].shape[0],1))
            # x = torch.cat([x,torch.ones(x.shape)],axis = 1)
            y=torch.tensor(Y[i])
            mask = torch.tensor(M[i])
            edge_attr = None
            data_list.append(Data(edge_index=edge_index, x=x, y=y,mask = mask, edge_attr=edge_attr))
        
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


        
class SpectralDesign(object):   

    def __init__(self,recfield=-1,power=1,hadam=1,neighborshape=1,hadamneighbor = 1,operator = "adj",QM9 = False, PPGN = False):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield=recfield  
        self.power = power
        self.hadam = hadam
        self.neighborshape = neighborshape
        self.hadamneighbor = hadamneighbor
        # use laplacian or adjacency for spectrum
        self.operator = operator
        self.QM9 = QM9
        self.PPGN =PPGN
        
     

    

    def __call__(self, data):
        if data.x is not None:
            n =data.x.shape[0]
        else:
            n = data.num_nodes
            data.x = torch.ones((n,1))
        data.x = data.x.type(torch.float)
        # nf = 1
        # if len(data.x.shape)>1:
        #     nf=data.x.shape[1]
        # for i in range(nf):
        #     norm = torch.linalg.norm(data.x[:,i])
        #     if norm > 0:
        #         data.x[:,i] = data.x[:,i]/norm
        if data.edge_attr is not None:
            if len(data.edge_attr.shape)>1:
                nfeat = data.edge_attr.shape[1]
            else:
                nfeat = 1
                data.edge_attr = data.edge_attr.reshape((data.edge_attr.shape[0],1))
            # data.edge_attr = data.edge_attr

        # data.x=data.x.type(torch.float32)
        
        if self.QM9:
            distance_mat = np.zeros((1,n,n))
            distance_mat[0,:,:] = dist.squareform(dist.pdist(data.pos))
        
        
        
               
        nsup=self.power+1+self.hadam-1+2*(self.neighborshape-1)+self.hadamneighbor-1
        
            
        A=np.zeros((n,n),dtype=np.float32)
        SP=np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0],data.edge_index[1]]=1
        if np.linalg.norm(A-A.T)>0:
            A = A + A.T

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield==0:
            M=A
        else:
            M=(A+np.eye(n))
            for i in range(1,self.recfield):
                M=M.dot(M)
        if self.recfield == -1:
            M = np.ones((n,n))
            
                               
        M=(M>0)

        # norm = np.linalg.norm(A)
        # if norm >0:
        #     A=A/norm
       
        # normalized Laplacian matrix.
        
    
        if self.operator == "lap":        
            A = graph.Laplaciannormal(A)
        if self.operator == "norm":
            A = graph.normalize(A)
            
        elif self.operator == "gcn":
            A = graph.gcnoperator(A)
        
        if self.operator == "cheb":
            A = graph.Laplaciannormal(A)
            V,U = np.linalg.eigh(A)
            vmax = V.max()
            A = M * (2*A/vmax - np.eye(n))
                       
        
            SP[0,:,:] = M* np.eye(n)
            SP[1,:,:] = M * (2*A/vmax - np.eye(n))
                 
                          
            for i in range(2,nsup):
                SP[i,:,:]=M* (2*SP[2,:,:]@SP[i-1,:,:]-SP[i-2,:,:])
        else:
            for i in range(self.power+1):
                res = M* np.linalg.matrix_power(A,i)
                
                # SP[i,:,:]=res/np.linalg.norm(res)
                SP[i,:,:]=res
            if self.hadam>1:
                for i in range(self.hadam-1):
                    I = np.eye(A.shape[0])
                    SP[i+self.power+1,:,:] = M*np.linalg.matrix_power(A,i+2)*I
            if self.neighborshape>1:
                index = self.power+1+self.hadam-1
                for i in range((self.neighborshape-1)):
                    I = np.eye(A.shape[0])
                    tmp = np.linalg.matrix_power(A,i+2)*I
                    SP[index,:,:] = M*(A@tmp+tmp@A)
                    SP[index+1,:,:] = M*(A@tmp@A)
                    index += 2
            if self.hadamneighbor>1:
                for i in range(self.hadamneighbor-1):
                    SP[i+self.power+1+self.hadam-1+2*(self.neighborshape-1),:,:] = M*A*np.linalg.matrix_power(A,i+2)
        
        
           
        # set convolution support weigths as an edge feature
        E=np.where(M>0)
        
        data.batch_edge = torch.zeros(n*n,dtype=torch.int64)
        data.edge_index2=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        if data.edge_attr is not None:
            C = np.zeros((nfeat,n,n))
            for i in range(nfeat):
                C[i,data.edge_index[0],data.edge_index[1]] = data.edge_attr[:,i]
                res = C[i,:,:]
                if np.linalg.norm(res- res.T)>0:
                    res = res+ res.T
                if self.operator == 'norm':
                    res = graph.normalize(res)
                C[i,:,:] = res
            if self.QM9:
                data.edge_attr = torch.cat([torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(C[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(distance_mat[:,E[0],E[1]].T).type(torch.float32)],1)
            else:
                data.edge_attr = torch.cat([torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(C[:,E[0],E[1]].T).type(torch.float32)],1)
            if self.PPGN:
                data.edge_attr = torch.cat([data.edge_attr,torch.diag_embed(data.x.T)[:,E[0],E[1]].T],1)
        else:
            data.edge_attr = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)
            if self.PPGN:
                data.edge_attr = torch.cat([data.edge_attr,torch.diag_embed(data.x.T)[:,E[0],E[1]].T],1)
                
        return data