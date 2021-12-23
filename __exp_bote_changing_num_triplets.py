# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from models import BOTE
from data_utils import ABSADataReaderBERT, BertTokenizer
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

def load_large_sentences(dataset, cluster, _min = 22, _max = 512, num_triplets = 4):
    data = pd.read_csv('cross_validation/data/'+dataset+'/'+cluster+'/test_triplets.txt', sep='####', header=None, engine='python')

    fp = open('cross_validation/data/'+dataset+'/'+cluster+'/test_triplets.txt.graph', 'rb')
    idx2gragh = pickle.load(fp)
    fp.close()

    sentences = []
    triplets = []
    idx2gragh_ = []
    for i, (text, triplet) in enumerate(zip(data[0], data[1])):
        arr = len(text.split())
        if arr >= _min and arr <= _max and len(eval(triplet)) == num_triplets:
            sentences.append(text)
            triplets.append(eval(triplet))
            idx2gragh_.append(idx2gragh[i])
            if len(sentences) == 20: break
    
    return sentences, triplets, idx2gragh_


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = BertTokenizer(opt.bert_model, opt.case, opt.spacy_lang, opt.lang)
        absa_data_reader = ABSADataReaderBERT(data_dir=opt.data_dir)
        self.idx2tag, self.idx2polarity = absa_data_reader.reverse_tag_map, absa_data_reader.reverse_polarity_map
        self.model = opt.model_class([], opt, self.idx2tag, self.idx2polarity).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, idx2gragh):
        text_indices, text_indices_bert, position_bert_in_naive = self.tokenizer.text_to_sequence(text)
        postag_indices = self.tokenizer.text_to_sequence_postags(text)
        text_mask = [1] * len(text_indices)
        text_mask_bert = [1] * len(text_indices_bert)
        t_sample_batched = {
            'text_indices': torch.tensor([text_indices]),
            'text_mask': torch.tensor([text_mask], dtype=torch.uint8),
            'text_indices_bert': torch.tensor([text_indices_bert]),
            'text_mask_bert': torch.tensor([text_mask_bert], dtype=torch.uint8),
            'position_bert_in_naive': torch.tensor([position_bert_in_naive]),
            'postag_indices': torch.tensor([postag_indices]),
            'dependency_graph': torch.tensor([idx2gragh]),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)
        
        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_triplets', default=4, type=int)
    opt = parser.parse_args()

    datasets = ['restES']#['rest14', 'rest16', 'rehol', 'restES']

    for num_triplets in [1,2,3,4]:
        print(80*'#')
        print("Considering", str(num_triplets), 'triplets per sentence')

        for dataset in datasets:

            if dataset in ['rest14', 'rest16']:
                opt.bert_model = 'bert-base-uncased'
                opt.case = 'uncased'
                opt.lang = 'en'
            elif dataset in ['rehol']:
                opt.bert_model = 'neuralmind/bert-base-portuguese-cased'
                opt.case = 'cased'
                opt.lang = 'pt'
            elif dataset in ['restES']:
                opt.bert_model = 'dccuchile/bert-base-spanish-wwm-cased'
                opt.case = 'cased'
                opt.lang = 'es'

            data_dirs = {
                'rest14_c_0': 'cross_validation/data/rest14/c_0',
                'rest16_c_0': 'cross_validation/data/rest16/c_0',
                'rehol_c_0': 'cross_validation/data/rehol/c_0',
                'restES_c_0': 'cross_validation/data/restES/c_0',
            }

            spacy_languages = {
                'en': 'en_core_web_md',
                'pt': 'pt_core_news_sm',
                'es': 'es_core_news_md',
            }

            cluster = 'c_0'
            opt.model_name = 'bote'
            opt.dataset = dataset+'_'+cluster
            opt.bert_layer_index = 10
            opt.eval_cols = ['ap_spans', 'op_spans','triplets']
            opt.model_class = BOTE
            opt.input_cols = ['text_indices', 'text_mask', 'text_indices_bert', 'text_mask_bert', 'position_bert_in_naive', 'postag_indices', 'dependency_graph']
            opt.target_cols = ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask']
            opt.state_dict_path = 'state_dict/bote_'+opt.dataset+'.pkl'
            opt.embed_dim = 768
            opt.polarities_dim = 4
            opt.batch_size = 32
            opt.data_dir = data_dirs[opt.dataset]
            opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            opt.spacy_lang = spacy_languages[opt.lang]

            inf = Inferer(opt)

            texts, triplets_real, idx2gragh = load_large_sentences(dataset, cluster, 12, 30, num_triplets)
            all_outputs = []
            all_targets = []
            count_triplets = 0

            for i, (text, tri) in enumerate(zip(texts, triplets_real)):
                
                triplets = inf.evaluate(text, idx2gragh[i])[2][0]
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

