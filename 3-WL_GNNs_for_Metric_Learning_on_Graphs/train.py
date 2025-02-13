import os
from os import path
import json
import random
from datetime import datetime, timedelta, time
import numpy as np
import torch
import dgl
from tqdm import tqdm
from models import siamese_GNN
from utils import tab_printer, calculate_sigmoid_loss
from utils import Metric
from libs.utils import SpectralDesign
import ot


def process_pair_v2(data, global_labels,global_edges_labels,args):
    if args.model == 'g2n2' :
        edges_1 = data["graph_1"] 
        edges_2 = data["graph_2"] 
        edges_1 = np.array(edges_1, dtype=np.int64)
        edges_2 = np.array(edges_2, dtype=np.int64)
        G_1 = dgl.graph((edges_1[:,0], edges_1[:,1]))
        G_2 = dgl.graph((edges_2[:,0], edges_2[:,1]))

        n_1 = G_1.num_nodes()
        n_2 = G_2.num_nodes()    
        data["e_lab1"] = data["e_lab1"] 
        data["e_lab2"] = data["e_lab2"]
        
        data["edge_index_1"] = torch.ones((n_1,n_1))
        data["edge_index_1"]=data["edge_index_1"].type(torch.long)
        E1 = torch.where(data["edge_index_1"]>0)
        data["edge_index_1"] = torch.vstack((E1[0],E1[1]))
        data["edge_index_2"] = torch.ones((n_2,n_2))
        data["edge_index_2"]=data["edge_index_2"].type(torch.long)
        E2 = torch.where(data["edge_index_2"]>0)
        data["edge_index_2"] = torch.vstack((E2[0],E2[1]))
        
        features_1, features_2 = [], []

        for n in data["labels_1"]:
            features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        
        G_1.ndata['features'] = torch.FloatTensor(np.array(features_1))
        G_2.ndata['features'] = torch.FloatTensor(np.array(features_2))
        
        A_1 = torch.tensor(G_1.adj(scipy_fmt='coo').todense())

        A_2 = torch.tensor(G_2.adj(scipy_fmt='coo').todense())

        I_1 = torch.eye(G_1.num_nodes())
        I_2 = torch.eye(G_2.num_nodes())

        e_features1 = torch.cat((torch.unsqueeze(I_1,2),torch.unsqueeze(A_1,2)),2)
        e_features2 = torch.cat((torch.unsqueeze(I_2,2),torch.unsqueeze(A_2,2)),2)
        
        
        data['e_features1'] = e_features1[E1[0],E1[1],:]
        data['e_features2'] = e_features2[E2[0],E2[1],:]

        data['G_1'] = G_1
        data['G_2'] = G_2
        
        

        data["target"] = torch.from_numpy(np.array(data["ged"])).float()

    elif args.model == 'ppgn' :
        edges_1 = data["graph_1"] 
        edges_2 = data["graph_2"] 
        edges_1 = np.array(edges_1, dtype=np.int64)
        edges_2 = np.array(edges_2, dtype=np.int64)
        G_1 = dgl.graph((edges_1[:,0], edges_1[:,1]))
        G_2 = dgl.graph((edges_2[:,0], edges_2[:,1]))
        n_1 = G_1.num_nodes()
        n_2 = G_2.num_nodes()    
        data["e_lab1"] = data["e_lab1"] 
        data["e_lab2"] = data["e_lab2"]
        
        data["edge_index_1"] = torch.ones((n_1,n_1))
        data["edge_index_1"]=data["edge_index_1"].type(torch.long)
        E1 = torch.where(data["edge_index_1"]>0)
        data["edge_index_1"] = torch.vstack((E1[0],E1[1]))
        data["edge_index_2"] = torch.ones((n_2,n_2))
        data["edge_index_2"]=data["edge_index_2"].type(torch.long)
        E2 = torch.where(data["edge_index_2"]>0)
        data["edge_index_2"] = torch.vstack((E2[0],E2[1]))
        
        features_1, features_2 = [], []

        for n in data["labels_1"]:
            features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        
        G_1.ndata['features'] = torch.FloatTensor(np.array(features_1))
        G_2.ndata['features'] = torch.FloatTensor(np.array(features_2))


        
        A_1 = torch.tensor(G_1.adj(scipy_fmt='coo').todense())

        A_2 = torch.tensor(G_2.adj(scipy_fmt='coo').todense())

        I_1 = torch.zeros(G_1.num_nodes(),G_1.num_nodes(),G_1.ndata['features'].shape[1])
        I_2 = torch.zeros(G_2.num_nodes(),G_2.num_nodes(),G_2.ndata['features'].shape[1])
        for i in range(I_1.shape[0]):
            I_1[i,i,:] = G_1.ndata['features'][i]
        for i in range(I_2.shape[0]):
            I_2[i,i,:] = G_2.ndata['features'][i]
        e_features1 = torch.cat((I_1,torch.unsqueeze(A_1,2)),2)
        e_features2 = torch.cat((I_2,torch.unsqueeze(A_2,2)),2)

        data['e_features1'] = e_features1
        data['e_features2'] = e_features2

        data['G_1'] = G_1
        data['G_2'] = G_2

        data["target"] = torch.from_numpy(np.array(data["ged"])).float()
        
    else:
        print("error choose a model, ppgn or g2n2")
    return data

     

class Trainer(object):
    def __init__(self, args, training_pairs, validation_pairs, testing_pairs, device, model_class=siamese_GNN):

        self.args = args
        self.model_class = model_class
        
        self.training_pairs = training_pairs
        self.validation_pairs = validation_pairs
        self.testing_pairs = testing_pairs

        self.initial_label_enumeration()

        self.epoch = 0
        self.best_val_mse = None
        self.val_mses = []
        self.best_val_metric = None
        self.val_metrics = []
        self.losses = []
        self.early_stop = False
        self.counter = 0
        self.epoch_times = []

        
        self.device = device
        self.setup_model()
        self.initialize_model()
        self.mse_val = []
        
    def setup_model(self):
        self.model = self.model_class(self.args, self.number_of_labels)
        print(self.model)
        
        self.model = self.model.to(self.device)

        self.best_model = self.model_class(self.args, self.number_of_labels)
        
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                          lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
        self.lrs = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0000001, max_lr=self.args.learning_rate, step_size_up=2000, step_size_down=2000, cycle_momentum=False)
        
    def initialize_model(self):
        if path.exists(self.args.exp_dir):
            checkpoint_files = sorted([d for d in os.listdir(self.args.exp_dir) if 'checkpoint_' in d])
            #checkpoint_files = []
            if len(checkpoint_files) > 0: #there exist some model, load them
                checkpoint_path = path.join(self.args.exp_dir, checkpoint_files[-1]) # 0!!!!
                if self.args.verbose >= 1:
                    print('Loading existing checkpoint: {}'.format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
                self.best_val_mse = checkpoint['best_val_mse']
                self.val_mses = checkpoint['val_mses']
                if 'best_val_metric' in checkpoint:
                    self.best_val_metric = checkpoint['best_val_metric']
                    self.val_metrics = checkpoint['val_metrics']
                self.losses = checkpoint['losses']
                self.early_stop = checkpoint['early_stop']
                self.epoch = checkpoint['epoch'] + 1
                self.counter = checkpoint['counter']
                self.epoch_times = checkpoint['epoch_times']
                if self.args.verbose >= 2:
                    print('Starting from epoch {}'.format(self.epoch))
                    
    def save_checkpoint(self, is_best_model=False):
        if not is_best_model:
            save_path = path.join(self.args.exp_dir, 'checkpoint_{:04d}.pt'.format(self.epoch))
            if self.args.verbose >= 3:
                print('Save checkpoint: {}'.format(save_path))
            torch.save({
                'epoch': self.epoch,
                'counter': self.counter,
                'best_val_mse': self.best_val_mse,
                'val_mses': self.val_mses, 
                'best_val_metric': self.best_val_metric,
                'val_metrics': self.val_metrics, 
                'losses': self.losses,
                'early_stop': self.early_stop,
                'model': self.model.state_dict(),
                'epoch_times': self.epoch_times,
                'optimizer': self.optimizer.state_dict()
            }, save_path)
            
        else: 
            save_path = path.join(self.args.exp_dir, 'best_checkpoint.pt')
            if self.args.verbose >= 3:
                print('Save best checkpoint: {}'.format(save_path))
            torch.save({
                'epoch': self.epoch,
                'best_val_mse': self.best_val_mse,
                'best_val_metric': self.best_val_metric,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, save_path)
        


    def create_batches(self,pair_set,shuffle = False):
        if shuffle:
            random.shuffle(pair_set)
        batches = []
        for graph in range(0, len(pair_set), self.args.batch_size):
            batches.append(pair_set[graph:graph+self.args.batch_size])
        return batches
    
    def collate(self, batch):
        if args.model =='ppgn':
            batched_edge_feat_G1 = []
            batched_edge_feat_G2 = []
            batched_GED = []
            max_g1 = max([data['G_1'].ndata['features'].shape[0] for data in batch])
            max_g2 = max([data['G_2'].ndata['features'].shape[0] for data in batch])
            maximum_batch_node = max((max_g1,max_g2))
            for data in batch:
                shape = data['e_features1'].shape
                batch_efeat_G1 = torch.zeros((shape[-1],maximum_batch_node,maximum_batch_node))

                batch_efeat_G1[:,:shape[0],:shape[1]] = torch.permute(data['e_features1'],(2,0,1))

                batched_edge_feat_G1.append(torch.unsqueeze(batch_efeat_G1,0))
                
                shape = data['e_features2'].shape
                batch_efeat_G2 = torch.zeros((shape[-1],maximum_batch_node,maximum_batch_node))
                batch_efeat_G2[:,:shape[0],:shape[1]] = torch.permute(data['e_features2'],(2,0,1))
                batched_edge_feat_G2.append(torch.unsqueeze(batch_efeat_G2,0))
                
                batched_GED.append(data["target"])

            batch = {}
            batch["ef1"] = torch.cat(batched_edge_feat_G1,0)
            batch["ef2"] = torch.cat(batched_edge_feat_G2,0)
            batch["GED"] = torch.tensor(batched_GED)
        else :
            batched_node_feat_G1 = []
            batched_node_idx_G1 = []
            batched_edge_feat_G1 = []
            batched_edge_idx_G1 = []
            batched_node_feat_G2 = []
            batched_node_idx_G2 = []
            batched_edge_feat_G2 = []
            batched_edge_idx_G2 = []
            batched_GED = []
            i = 0
            nb_node_1 = 0
            nb_node_2 = 0
            max_node_1 = 0
            max_node_2 = 0
            for data in batch:
                batched_edge_idx_G1.append(data["edge_index_1"]+max_node_1)
                nb_node_1 = data['G_1'].ndata['features'].shape[0]
                max_node_1 += nb_node_1
                batched_node_feat_G1.append(data['G_1'].ndata['features'])
                batched_node_idx_G1.append(i*torch.ones(nb_node_1))
                batched_edge_feat_G1.append(data['e_features1'])
                
                batched_edge_idx_G2.append(data["edge_index_2"]+max_node_2)
                nb_node_2 = data['G_2'].ndata['features'].shape[0]
                max_node_2 += nb_node_2
                batched_node_feat_G2.append(data['G_2'].ndata['features'])
                batched_node_idx_G2.append(i*torch.ones(nb_node_2))
                batched_edge_feat_G2.append(data['e_features2'])
                
                batched_GED.append(data["target"])
                i+=1
            batch = {}
            batch["nf1"] = torch.cat(batched_node_feat_G1,0)
            batch["nid1"] = torch.cat(batched_node_idx_G1,0)
            batch["ef1"] = torch.cat(batched_edge_feat_G1,0)
            batch["eid1"] = torch.cat(batched_edge_idx_G1,1)
            batch["nf2"] = torch.cat(batched_node_feat_G2,0)
            batch["nid2"] = torch.cat(batched_node_idx_G2,0)
            batch["ef2"] = torch.cat(batched_edge_feat_G2,0)
            batch["eid2"] = torch.cat(batched_edge_idx_G2,1)
            batch["GED"] = torch.tensor(batched_GED)
        return batch
    
    def decollate(self,nt1,et1,nt2,et2,i,nid1,nid2,idx_edges_1,idx_edges_2):
        nt1i = nt1[torch.where(nid1==i)[0]]
        nt2i = nt2[torch.where(nid2==i)[0]]
        nb_egde_1 = len(nt1i)**2
        nb_edge_2 = len(nt2i)**2
        et1i = et1[idx_edges_1:idx_edges_1+nb_egde_1 ]
        et2i = et2[idx_edges_2:idx_edges_2+nb_edge_2 ]
        idx_edges_1 += nb_egde_1
        idx_edges_2 += nb_edge_2
        return  nt1i,nt2i,et1i,et2i,idx_edges_1,idx_edges_2
    
    def decollate_ppgn(self,et1,et2,i):
        et1i = torch.squeeze(et1[i,:,:,:],0)
        et2i = torch.squeeze(et2[i,:,:,:],0)
        return et1i,et2i

    def process_PFGW(self,nt1,et1,nt2,et2,data):
        emd = False
        if args.model == 'ppgn' :
            if emd == True :
                emb1 = torch.sum(et1,1)
                emb2 = torch.sum(et2,1)
                s1 = emb1.shape
                s2 = emb2.shape
                nb_node_g1 = data['G_1'].ndata['features'].shape[0]
                nb_node_g2 = data['G_2'].ndata['features'].shape[0]
                max_num_node = max((nb_node_g1,nb_node_g2))
                if nb_node_g1<max_num_node :
                    cG1=torch.zeros([max_num_node,s1[1]]).to(device='cuda')
                    cG1[:nb_node_g1,:] = emb1[:nb_node_g1,:]
                    cG2 = emb2[:nb_node_g2,:]
                else :
                    cG2=torch.zeros([max_num_node,s2[1]]).to(device='cuda')
                    cG2[:nb_node_g2,:] = emb2[:nb_node_g2,:]
                    cG1 = emb1[:nb_node_g1,:]

                a=(torch.ones(max_num_node)*(1/max_num_node)).to(device='cuda')
                C = ot.dist(cG1,cG2)
                pt_epsi = ot.emd(a,a,C, numThreads='max')
                pt_epsi = pt_epsi*max_num_node
                loss = torch.sum(pt_epsi*C)
                loss,matching = loss,(pt_epsi,loss)
                reg = 0
        else :
            shapeG1 = nt1.shape
            shapeG2 = nt2.shape
            max_num_node = max((shapeG1[0],shapeG2[0]))
            if shapeG1[0]<max_num_node :
                cG1=torch.zeros([max_num_node,shapeG1[1]]).to(device='cuda')
                cG1[:shapeG1[0],:] = nt1
                cG2 = nt2
                sG1 = torch.zeros([max_num_node,max_num_node,et1.shape[1]]).to(device='cuda')
                sG1[:shapeG1[0],:shapeG1[0],:] = torch.reshape(et1,(shapeG1[0],shapeG1[0],-1))
                sG2 = torch.reshape(et2,(shapeG2[0],shapeG2[0],-1))
            else :
                cG2=torch.zeros([max_num_node,shapeG2[1]]).to(device='cuda')
                cG2[:shapeG2[0],:shapeG2[1]] = nt2
                cG1 = nt1
                sG2 = torch.zeros([max_num_node,max_num_node,et2.shape[1]]).to(device='cuda')
                sG2[:shapeG2[0],:shapeG2[0],:] = torch.reshape(et2,(shapeG2[0],shapeG2[0],-1))
                sG1 = torch.reshape(et1,(shapeG1[0],shapeG1[0],-1))
            if emd == True :
                emb1 = cG1 + torch.sum(sG1,1)
                emb2 = cG2 + torch.sum(sG2,1)
                a=(torch.ones(max_num_node)*(1/max_num_node)).to(device='cuda')
                C = ot.dist(emb1,emb2)
                pt_epsi = ot.emd(a,a,C, numThreads='max')
                pt_epsi = pt_epsi*max_num_node
                loss = torch.sum(pt_epsi*C)
                loss,matching = loss,(pt_epsi,loss)
                reg = 0
                
        return_matching=True
        return_embedding = False

        if return_embedding:
            matching_cost = loss
            matches = matching
            score = matching_cost
            score_logits = matching_cost
            return score, score_logits, matches,self.emb
        if return_matching:

            matching_cost = loss
            matches = matching
        else:
            matching_cost = loss
        
        if 'matching_type' in self.args:
            if self.args.matching_type == 'last': 
                score_logits = matching_cost[-1]
                score = torch.sigmoid(score_logits)
        else:
            score = matching_cost
            score_logits = reg
        if return_matching:
            return score,score_logits, matches  
        else:
            return score, score_logits


    def process_batch(self, batch,collated_batch):
        self.optimizer.zero_grad()
        losses = torch.tensor(0.0).to(self.device)
        if args.model == 'ppgn':
            if args.readout:
                nt1,emb_g1,nt2,emb_g2 = self.model(collated_batch)
                predictions = torch.norm(emb_g1-emb_g2,p=2,dim=1)
                targets = torch.tensor([batch[i]["target"]for i in range(len(batch))]).to(self.device).squeeze(0)
                losses = losses + torch.nn.functional.l1_loss(targets, predictions)
                #losses = losses + torch.nn.functional.mse_loss(target, prediction)
                losses.backward()
                self.optimizer.step()
                self.lrs.step()
                loss = losses.item()
            else:
                nt1,et1,nt2,et2 = self.model(collated_batch)
                for i in range(len(batch)):
                    data = batch[i]
                    et1i,et2i= self.decollate_ppgn(et1,et2,i)
                    prediction, reg, matches = self.process_PFGW(None,et1i,None,et2i,data)
                    target = data["target"].to(self.device).squeeze(0)
                    #losses = losses + torch.nn.functional.mse_loss(target, prediction)
                    losses = losses + torch.nn.functional.l1_loss(target, prediction)
                losses = losses/len(batch)
                losses.backward()
                self.optimizer.step()
                self.lrs.step()
                loss = losses.item()
        else :
            nt1,et1,nt2,et2 = self.model(collated_batch)
            nid1 = collated_batch["nid1"]
            nid2 = collated_batch["nid2"]
            idx_edges_1 = 0
            idx_edges_2 = 0
            for i in range(len(batch)):
                data = batch[i]
                nt1i,nt2i,et1i,et2i,idx_edges_1,idx_edges_2 = self.decollate(nt1,et1,nt2,et2,i,nid1,nid2,idx_edges_1,idx_edges_2)
                prediction, reg, matches = self.process_PFGW(nt1i,et1i,nt2i,et2i,data)
                target = data["target"].to(self.device).squeeze(0)
                losses = losses + torch.nn.functional.l1_loss(target, prediction)
            losses = losses/len(batch)
            losses.backward()
            self.optimizer.step()
            self.lrs.step()
            
            loss = losses.item()
        return loss   
    
    def track_state(self, val_mse, is_final_epoch=False):
        if self.best_val_mse is None:
            self.best_val_mse = val_mse
            self.val_mses.append(val_mse)
            self.save_checkpoint(is_best_model=True)
        else:
            min_mse = min(self.val_mses[-self.args.patience:])
            if val_mse > min_mse + self.args.delta: 
                self.val_mses.append(val_mse)

                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.args.patience}')
                if self.counter >= self.args.patience:
                    self.early_stop = True
            else: #improve from last min
                if self.args.verbose >= 1:
                    print(f'Validation MSE decreased ({min_mse:.6f} --> {val_mse:.6f}).')
                self.val_mses.append(val_mse)
                self.counter = 0

            if val_mse < self.best_val_mse: 
                if self.args.verbose >= 2:
                    print(f'Best MSE ({self.best_val_mse:.6f} --> {val_mse:.6f}).  Will save best model.')
                self.best_val_mse = val_mse
                self.save_checkpoint(is_best_model=True)
            
        if self.early_stop or is_final_epoch or self.epoch % self.args.save_frequency == 0:
            self.save_checkpoint()

    def track_metric_state(self, val_metric, is_final_epoch=False):
        if self.best_val_metric is None:
            self.best_val_metric = val_metric
            self.val_metrics.append(val_metric)
            self.save_checkpoint(is_best_model=True)
        else:
            min_metric = min(self.val_metrics[-self.args.patience:])
            if val_metric > min_metric + self.args.delta: 
                print("val and min metric",val_metric,min_metric)
                self.val_metrics.append(val_metric)

                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.args.patience}')
                if self.counter >= self.args.patience:
                    self.early_stop = True
            else:
                if self.args.verbose >= 1:
                    print(f'Validation Metric decreased ({min_metric:.6f} --> {val_metric:.6f}).')
                self.val_metrics.append(val_metric)
                self.counter = 0

            if val_metric < self.best_val_metric: 
                if self.args.verbose >= 2:
                    print(f'Best Metric ({self.best_val_metric:.6f} --> {val_metric:.6f}).  Will save best model.')
                self.best_val_metric = val_metric
                self.save_checkpoint(is_best_model=True)
            
        if self.early_stop or is_final_epoch or self.epoch % self.args.save_frequency == 0:
            self.save_checkpoint()            
    
    def fit(self):
        print("\nModel training.\n")
        
        self.model.train()
        epochs = tqdm(range(self.epoch, self.args.epochs), leave=True, desc="Epoch")
        
        iters_per_stat = self.args.iters_per_stat if 'iters_per_stat' in self.args else 100
        metric = Metric(self.validation_pairs);
        evaluation_frequency = self.args.evaluation_frequency if 'evaluation_frequency' in self.args else 1
        for self.epoch in epochs:
            if self.early_stop:
                print('Early stopping!')
                break

            epoch_start_time = datetime.now()
            
            self.model.train()
            batches = self.create_batches(self.training_pairs_dgl,shuffle = True)
            self.loss_sum = 0.0

            for index, batch in tqdm(enumerate(batches), total=len(batches), mininterval=1):
                collated_batch = self.collate(batch)
                loss_score = self.process_batch(batch,collated_batch)
                self.loss_sum += loss_score 
                loss = self.loss_sum/(index+1)

                if index % iters_per_stat == 0:
                    print("[Epoch %04d][Iter %d/%d] (Loss=%.05f)" % (self.epoch, index, len(batches), round(loss, 5)))


            epoch_duration = (datetime.now() - epoch_start_time)
            self.epoch_times.append(epoch_duration.total_seconds())
            print('[Epoch {}]: Finish in {:.2f} sec ({:.2f} min).'.format(
                    self.epoch, epoch_duration.total_seconds(), epoch_duration.total_seconds() / 60))
            self.losses.append(loss)

            batches = self.create_batches(self.validation_pairs_dgl)
            validation_mse = 0.0
            predictions = []
            for batch in tqdm(batches) :
                collated_batch = self.collate(batch)
                valid,pred = self.score(batch,collated_batch)
                predictions.append(pred)
                validation_mse += valid
            validation_mse = validation_mse / len(batches)
            baseline_mse = self.baseline_score(test_pairs=self.validation_pairs_dgl);
            print("Validation MSE: {:.05f}, Baseline: {:.05f}".format(validation_mse, baseline_mse));
            self.mse_val.append(validation_mse)
            is_final_epoch=(self.epoch+1 == self.args.epochs)
            predictions = np.concatenate(predictions,0)
            if 'early_stopping_metric' in self.args:
                if self.epoch % evaluation_frequency == 0 or is_final_epoch:
                    print(f'Evaluating using metric: {self.args.early_stopping_metric}')
                    eval_start_time = datetime.now()
                    
                    validation_predictions = predictions
                    print("valeurs des predictions de la validation dans le train.fit", validation_predictions[0:10])

                    valid_mae = metric.mae(validation_predictions, unnormalized=False)
                    valid_mse = metric.mse(validation_predictions, unnormalized=False)

                    if self.args.early_stopping_metric == 'mae':
                        validation_metric = valid_mae
                    elif self.args.early_stopping_metric == 'mse':
                        validation_metric = valid_mse
                    else:
                        raise Exception(f'not supported metric: {self.args.early_stopping_metric}')
                    
                    self.track_metric_state(validation_metric, is_final_epoch=is_final_epoch)
                    print(f'Evaluation: {validation_metric:.05f} (finishes in {(datetime.now() - eval_start_time).total_seconds()})')
                
            else:
                self.track_state(validation_mse, is_final_epoch=is_final_epoch)
            print("learning rate : ",self.lrs.get_last_lr())
        print("Final Validation MSE: ", validation_mse)

        
    def score(self,batch,collated_batch):

        self.model.eval() 
        with torch.no_grad() :
            scores = []
            predictions = []
            if args.model == 'ppgn' :
                if args.readout:
                    nt1,emb_g1,nt2,emb_g2 = self.model(collated_batch)
                    targets = [batch[i]["target"]for i in range(len(batch))]
                    predictions = torch.norm(emb_g1-emb_g2,p=2,dim=1)
                    predictions = predictions.detach().cpu().numpy()
                    scores = calculate_sigmoid_loss(predictions, targets)
                else :
                    nt1,et1,nt2,et2 = self.model(collated_batch) 
                    for i in range(len(batch)):
                        data = batch[i]
                        et1i,et2i= self.decollate_ppgn(et1,et2,i)
                        prediction, reg, matches = self.process_PFGW(None,et1i,None,et2i,data)
                        match_loss = matches[-1].detach().cpu().numpy()
                        predictions.append(match_loss)
                        scores.append(calculate_sigmoid_loss(match_loss, data))
                    predictions = np.array(predictions)
            else :
                nt1,et1,nt2,et2 = self.model(collated_batch)
                nid1 = collated_batch["nid1"]
                nid2 = collated_batch["nid2"]
                idx_edges_1 = 0
                idx_edges_2 = 0
                for i in range(len(batch)):
                    data = batch[i]
                    nt1i,nt2i,et1i,et2i,idx_edges_1,idx_edges_2 = self.decollate(nt1,et1,nt2,et2,i,nid1,nid2,idx_edges_1,idx_edges_2)
                    prediction, reg, matches = self.process_PFGW(nt1i,et1i,nt2i,et2i,data)
                    match_loss = matches[-1].detach().cpu().numpy()
                    predictions.append(match_loss)
                    scores.append(calculate_sigmoid_loss(match_loss, data))
                predictions = np.array(predictions)
        return np.mean(scores),predictions


    def load_best_model(self, load_path=None):
        if self.best_model is None:
            self.best_model = self.model.clone()
            
        if load_path is None:
            load_path = path.join(self.args.exp_dir, 'best_checkpoint.pt' ) #
            print('Load best model from {}'.format(load_path))
        checkpoint = torch.load(load_path)

        self.best_model.load_state_dict(checkpoint['model'])
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        
    def load_model(self, epoch=None, load_path=None):
        if load_path is None:
            load_path = path.join(self.args.exp_dir, 'checkpoint_{:04d}.pt'.format(epoch))
        if self.args.verbose >= 2:
            print('Load model from {}'.format(load_path))
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.model = self.best_model.to(self.device)
        self.model.eval()           
    
    def predict_best_model(self, test_pairs, load_path=None):
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        with torch.no_grad() :
            predictions = []
            matches = []
            QAP_GED = []
            batches = self.create_batches(test_pairs)
            self.loss_sum = 0.0
            if args.model == 'ppgn':
                if args.readout :
                    targets = []
                    for batch in tqdm(batches):
                        collated_batch = self.collate(batch)
                        nt1,emb_g1,nt2,emb_g2 = self.model(collated_batch)
                        targets.append([batch[i]["target"].detach().cpu().numpy() for i in range(len(batch))])
                        predictions.append(torch.norm(emb_g1-emb_g2,p=2,dim=1).detach().cpu().numpy())
                        
                    targets = np.concatenate(targets,0)
                    predictions = np.concatenate(predictions,0)
                    return predictions,predictions,targets
                else :
                    for batch in tqdm(batches):
                        collated_batch = self.collate(batch)
                        nt1,et1,nt2,et2 = self.best_model(collated_batch)
                        for i in range(len(batch)):
                            data = batch[i]
                            et1i,et2i= self.decollate_ppgn(et1,et2,i)
                            prediction, reg, match = self.process_PFGW(None,et1i,None,et2i,data)
                            predictions.append(prediction.detach().cpu().numpy())
                            matches.append(match[0].detach().cpu().numpy())
                            QAP_GED.append(match[-1].detach().cpu().numpy())
                    predictions = np.array(predictions)    
                    QAP_GED = np.array(QAP_GED)
            else:
                for batch in tqdm(batches):
                    collated_batch = self.collate(batch)
                    nt1,et1,nt2,et2 = self.best_model(collated_batch)
                    nid1 = collated_batch["nid1"]
                    nid2 = collated_batch["nid2"]
                    idx_edges_1 = 0
                    idx_edges_2 = 0
                    for i in range(len(batch)):
                        data = batch[i]
                        nt1i,nt2i,et1i,et2i,idx_edges_1,idx_edges_2 = self.decollate(nt1,et1,nt2,et2,i,nid1,nid2,idx_edges_1,idx_edges_2)
                        prediction, reg, match = self.process_PFGW(nt1i,et1i,nt2i,et2i,data)
                        predictions.append(prediction.detach().cpu().numpy())
                        matches.append(match[0].detach().cpu().numpy())
                        QAP_GED.append(match[-1].detach().cpu().numpy())
                predictions = np.array(predictions)    
                QAP_GED = np.array(QAP_GED)
        return predictions,matches,QAP_GED

        

    
    def baseline_score(self, test_pairs):
        self.model.eval()
        scores = []
        average_ged = np.mean([data["target"].detach().numpy()for data in test_pairs]);
        base_error = np.mean([(data["target"].detach().numpy()-average_ged)**2 for data in test_pairs])
        return base_error
        
    def initial_label_enumeration(self):

        print("\nEnumerating unique labels.\n")
        graph_pairs = self.training_pairs + self.testing_pairs + self.validation_pairs
        self.global_labels = set()
        self.global_edges_labels = set()
        for data in graph_pairs:
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
            self.global_edges_labels = self.global_edges_labels.union(set(data["e_lab1"]))
            self.global_edges_labels = self.global_edges_labels.union(set(data["e_lab2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.global_edges_labels = list(self.global_edges_labels)
        self.global_edges_labels = {val:index  for index, val in enumerate(self.global_edges_labels)}
        self.number_of_labels = len(self.global_labels)
        self.number_of_edges_labels = len(self.global_edges_labels)
        

        print('Scale GED by original method (exponential)')
        self.training_pairs_dgl = [process_pair_v2(graph_pair, self.global_labels,self.global_edges_labels,self.args) for graph_pair in tqdm(self.training_pairs, mininterval=2)]
        self.testing_pairs_dgl = [process_pair_v2(graph_pair, self.global_labels,self.global_edges_labels,self.args) for graph_pair in tqdm(self.testing_pairs, mininterval=2)]
        self.validation_pairs_dgl = [process_pair_v2(graph_pair, self.global_labels,self.global_edges_labels,self.args) for graph_pair in tqdm(self.validation_pairs, mininterval=2)] 
    
