# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from data_utils import ABSADataReader, build_tokenizer, build_embedding_matrix
from models import OTE
import pandas as pd

def convert_triplet_format_in_span_format(triplet):
  lab_to_index = {'N':0, 'NEU':1, 'NEG':2, 'POS':3}
  n_triplet = []
  for tri in triplet:
    tri[0]
    tri[1]
    tri[2]
    s = (tri[0][0], tri[0][-1], tri[1][0], tri[1][-1], lab_to_index[tri[2]])
    n_triplet.append(s)
  
  return n_triplet

def _metrics(targets, outputs):
    TP, FP, FN = 0, 0, 0
    n_sample = len(targets)
    assert n_sample == len(outputs)
    for i in range(n_sample):
        n_hit = 0
        n_output = len(outputs[i])
        n_target = len(targets[i])
        for t in outputs[i]:
            if t in targets[i]:
                n_hit += 1
        TP += n_hit
        FP += (n_output - n_hit)
        FN += (n_target - n_hit)
    precision = float(TP) / float(TP + FP + 1e-5)
    recall = float(TP) / float(TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return [precision, recall, f1]

def load_large_sentences(dataset, cluster, _min = 22, _max = 512):
    data = pd.read_csv('cross_validation/data/'+dataset+'/'+cluster+'/test_triplets.txt', sep='####', header=None, engine='python')

    sentences = []
    triplets = []
    for text, triplet in zip(data[0], data[1]):
        arr = len(text.split())
        if arr >= _min and arr <= _max:
            sentences.append(text)
            triplets.append(eval(triplet))
    
    return sentences, triplets


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
       
        absa_data_reader = ABSADataReader(data_dir=opt.data_dir)
        self.tokenizer = build_tokenizer(data_dir=opt.data_dir)
        embedding_matrix = build_embedding_matrix(opt.data_dir, self.tokenizer.word2idx, opt.embed_dim, opt.dataset, opt.glove_fname)
        self.idx2tag, self.idx2polarity = absa_data_reader.reverse_tag_map, absa_data_reader.reverse_polarity_map
        self.model = opt.model_class(embedding_matrix, opt, self.idx2tag, self.idx2polarity).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text):
        text_indices = self.tokenizer.text_to_sequence(text)
        text_mask = [1] * len(text_indices)
        t_sample_batched = {
            'text_indices': torch.tensor([text_indices]),
            'text_mask': torch.tensor([text_mask], dtype=torch.uint8),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)
        
        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min', default=22, type=int)
    parser.add_argument('--max', default=512, type=int)
    opt = parser.parse_args()

    datasets = ['rest14', 'rest16', 'rehol']

    for dataset in datasets:

        if dataset in ['rest14', 'rest16']:
            opt.lang = 'en'
        elif dataset == 'rehol':
            opt.lang = 'pt'

        data_dirs = {
            'rest14_c_2': 'cross_validation/data/rest14/c_2',
            'rest16_c_2': 'cross_validation/data/rest16/c_2',
            'rehol_c_2': 'cross_validation/data/rehol/c_2',
        }

        glove_files = {
            'en': 'glove.300d.txt',
            'pt': 'glove.300d_pt.txt',
            'es': 'glove.300d_es.txt',
        }

        cluster = 'c_2'
        opt.model_name = 'ote'
        opt.dataset = dataset+'_'+cluster
        opt.eval_cols = ['ap_spans', 'op_spans','triplets']
        opt.model_class = OTE
        opt.input_cols = ['text_indices', 'text_mask']
        opt.target_cols = ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask']
        opt.state_dict_path = 'state_dict/ote_'+opt.dataset+'.pkl'
        opt.embed_dim = 300
        opt.hidden_dim = 300
        opt.polarities_dim = 4
        opt.batch_size = 32
        opt.data_dir = data_dirs[opt.dataset]
        opt.glove_fname = glove_files[opt.lang]
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        inf = Inferer(opt)

        texts, triplets_real = load_large_sentences(dataset, cluster, opt.min, opt.max)
        all_outputs = []
        all_targets = []
        count_triplets = 0

        for i, (text, tri) in enumerate(zip(texts, triplets_real)):
            
            triplets = inf.evaluate(text)[2][0]
            words = text.split()
            polarity_map = {0:'N', 1:'NEU', 2:'NEG', 3:'POS'}
            count_triplets += len(tri)

            target = convert_triplet_format_in_span_format(tri)
            all_outputs.append(triplets)
            all_targets.append(target)

        
        print("In dataset", opt.dataset)
        print('No. sentences',len(texts),'No. Triplets', count_triplets)
        print('Avegare triplets per sentence', count_triplets/len(texts))
        print(_metrics(all_targets, all_outputs))
        print('\n')

