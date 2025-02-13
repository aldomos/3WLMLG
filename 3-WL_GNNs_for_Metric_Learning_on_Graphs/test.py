#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:07:59 2022

@author: aldomoscatelli
"""

import extract_pairs

path = "../symbolic datasets/result_symbolic_mao_romain-f24threads_cost1.csv"
split = [0.6,0.8]
dl = extract_pairs.data_loader(path,split)
dl.pairs()
dl.pairs_to_json()