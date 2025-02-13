import os
from os import path
from argparse import Namespace

from utils import tab_printer, calculate_sigmoid_loss

from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, math

import json
import glob
import torch
import numpy as np
import random

from train import Trainer
from utils import Metric, summarize_results
from models import siamese_GNN
import matplotlib.pyplot as plt
import argparse
import extract_pairs
from pandas import DataFrame
import pandas as pd
import pickle



def parameter_parser():

	parser = argparse.ArgumentParser(description="Run DeepPFGW.")

	parser.add_argument("--dataset",
						nargs="?",
						default="/data/graph_search/PTC_ged/",
						help="Folder with graphs, should contain a subfolder for each fold.")

	parser.add_argument("--epochs",
						type=int,
						default=200,
						help="Number of training epochs. Default is 5.")

	parser.add_argument("--batch_size",
						type=int,
						default=32,
						help="Number of graph pairs in a mini-batch. Default is 32.")

	parser.add_argument("--gnn_size",
						type=int,
						default=[8,8,8,8],
						nargs='+',
						help="List of neurons for the convolution layers. Default is 128 64 32.")

	parser.add_argument("--dropout",
						type=float,
						default=0,
						help="Dropout probability. Default is 0.5.")

	parser.add_argument("--learning-rate",
						type=float,
						default=0.0001,
						help="Learning rate. Default is 0.001.")

	parser.add_argument("--weight-decay",
						type=float,
						default=5*10**-4,
						help="Adam weight decay. Default is 5*10^-4.")

	parser.add_argument("--folds",
						type=int,
						default=1,
						help="Number of folds. Default is 5.")

	parser.add_argument("--patience",
						type=int,
						default=5,
						help="Patience for early stopping. Default is 1.")
						
	
	parser.add_argument("--use_pairnorm", action="store_true", default = True)
	parser.add_argument("--delta", type=float, default = 0.0001)
	parser.add_argument("--basedir", type=str)
	parser.add_argument("--exp_label", type=str)

	parser.add_argument("--verbose", type=int, default = 2)
	parser.add_argument("--save_frequency", type=int, default = 1)
	
	parser.add_argument("--distance_type", type=str, default = 'batch_cosine')
	parser.add_argument("--early_stopping_metric", type=str, default = 'mse')
	parser.add_argument("--evaluation_frequency", type=int, default = 1)
	parser.add_argument("--patient", type=int, default = 3)
	parser.add_argument("--model", type=str, default = 'g2n2', help = "g2n2 or ppgn")
	parser.add_argument("--readout", default = False, help = "Graph level readout")
	

	return parser.parse_args()

def main():
	args = parameter_parser()

	args.exp_label = f"{args.batch_size}_{args.learning_rate}"
	
	tab_printer(args)
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	args.exp_dir = path.join(args.basedir, args.exp_label)
	if path.exists(args.exp_dir):
		print('Experiment dir exists...')
	else:
		print('Creating experiment dir: {}'.format(args.exp_dir))
		os.makedirs(args.exp_dir)
	
	#Linux/aids/IMDB
	
	dataset_name = args.dataset.split('_')[0]
	train_folder = dataset_name+'_train/'
	test_folder = dataset_name+'_test/'
	train_location = glob.glob(os.path.join("CVfolder/", train_folder, "*.json"))
	test_location = glob.glob(os.path.join("CVfolder/", test_folder, "*.json"))
	data = json.load(open(train_location[0]))
	test_data = json.load(open(test_location[0]))
	df = pd.read_csv(args.dataset, delimiter = ",")
	list_key = list(data['nom'].values())
	random.shuffle(list_key)

	val_key = list_key[0:300]
	train_key = list_key[300:]
	
	training_fold = extract_pairs.split_dic(data,train_key)
	validation_fold = extract_pairs.split_dic(data,val_key)
	testing_fold = test_data

	training_set =  df[df.Graph1.isin(train_key)&df.Graph2.isin(train_key)]
	validation_set = df[df.Graph1.isin(train_key)&df.Graph2.isin(val_key)]
	testing_set = df[df.Graph1.isin(train_key)&df.Graph2.isin(list(testing_fold['nom'].values()))]
	l = [training_fold,validation_fold]
	validation_pair_set = extract_pairs.merge_dict(l)
	l = [training_fold,testing_fold]
	testing_pair_set = extract_pairs.merge_dict(l)

	training_pairs = extract_pairs.set_to_dict(training_set, training_fold,args.dataset)
	random.shuffle(training_pairs)
	validation_pairs = extract_pairs.set_to_dict(validation_set, validation_pair_set,args.dataset)
	random.shuffle(validation_pairs)
	testing_pairs = extract_pairs.set_to_dict(testing_set, testing_pair_set,args.dataset)
	random.shuffle(testing_pairs)

	test_results = []
	train_loss = []
	val_mse = []

	trainer = Trainer(args, training_pairs, validation_pairs, testing_pairs, device, 
				   model_class=siamese_GNN)
	trainer.fit()

	metric = Metric(trainer.testing_pairs)
	test_predictions, test_matches,test_QAP_GED = trainer.predict_best_model(trainer.testing_pairs)
	
	train_loss.append(trainer.losses)

	val_mse.append(trainer.mse_val)
	
	#mse 
	test_mse = metric.mse(test_predictions, unnormalized=False)
	#mae
	test_mae = metric.mae(test_predictions, unnormalized=False)

	test_results.append(Namespace(mse=test_mse, mae=test_mae)) 

	print(f'[Fold Test]: mse {test_mse:.05f}, mae {test_mae:.05f}')
        
        #saving results
	csvdict = {}
	csvdict['Graph1'] = []
	csvdict['Graph2'] = []
	csvdict['GED'] = []
	csvdict['Pred'] = []
	csvdict['match'] = []
	for i in range(len(trainer.testing_pairs)):
		csvdict['Graph1'].append(trainer.testing_pairs[i]["id_1"])
		csvdict['Graph2'].append(trainer.testing_pairs[i]["id_2"])
		csvdict['GED'].append(trainer.testing_pairs[i]["ged"])
		csvdict['Pred'].append(test_predictions[i])
		csvdict['match'].append(np.array(test_matches[i]))
	df_res = DataFrame(csvdict, columns= ['Graph1', 'Graph2', 'GED', 'Pred','match'])

	nom = 'CSV_res/'+dataset_name+'_nbcouches_'+str(args.gnn_size)+'_lr_'+str(args.learning_rate)+ '_drop_'+str(args.dropout)+ '_mse_'+str(round(test_mse,5))+'mae_'+str(round(test_mae,5))+'.csv'
	df_res.to_csv (nom, index = None, header=True, encoding='utf-8', sep=';')

	print("val_mse" ,val_mse)	

	return test_results
	
if __name__ == "__main__":
	results = main()  

