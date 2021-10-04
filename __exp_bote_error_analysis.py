# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from models import BOTE
from data_utils import ABSADataReaderBERT, BertTokenizer
import pandas as pd
from collections import Counter
import numpy as np

def is_aspect_overlapped(aspects):
  for ap in aspects:
    if aspects.count(ap) > 1:
      return True
  
  return False

def is_opinion_overlapped(opinions):
  for op in opinions:
    if opinions.count(op) > 1:
      return True
  
  return False


def get_overlapped_senteces(t):

    aspects = []
    opinions = []
    aspect_overlapped = False
    opinion_overlapped = False
    opinion_aspect_overlapped = aspect_overlapped*opinion_overlapped

    triplets = eval(t)

    for triplet in triplets:
        aspects.append(triplet[0])
        opinions.append(triplet[1])

    aspect_overlapped = is_aspect_overlapped(aspects)
    opinion_overlapped = is_opinion_overlapped(opinions)
    opinion_aspect_overlapped = aspect_overlapped*opinion_overlapped

    if aspect_overlapped and not opinion_aspect_overlapped:
        return 'ap'
    elif opinion_overlapped and not opinion_aspect_overlapped:
        return 'op'
    elif opinion_aspect_overlapped:
        return 'ap_op'
    elif not aspect_overlapped and not opinion_overlapped:
        return 'not'

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

def load_sentences(dataset, cluster, _min = 2, _max = 512):
    data = pd.read_csv('cross_validation/data/'+dataset+'/'+cluster+'/test_triplets.txt', sep='####', header=None, engine='python')

    fp = open('cross_validation/data/'+dataset+'/'+cluster+'/test_triplets.txt.graph', 'rb')
    idx2gragh = pickle.load(fp)
    fp.close()

    sentences = []
    triplets = []
    idx2gragh_ = []
    overlapped = []
    for i, (text, triplet) in enumerate(zip(data[0], data[1])):
        arr = len(text.split())
        if arr >= _min and arr <= _max:
            sentences.append(text)
            triplets.append(eval(triplet))
            idx2gragh_.append(idx2gragh[i])
            overlapped.append(get_overlapped_senteces(triplet))
    
    return sentences, triplets, idx2gragh_, overlapped


def get_false_negatives_from_sentence(target, output, term_type, text):
  if term_type == 'ap':
    ini = 0
    end = 2
  elif term_type == 'op':
    ini = 2
    end = 4
  elif term_type == 'sent':
    ini = 4
    end = 5
  
  num_false_negative = 0
  for term_target in target:
    false_negative = True
    for term_output in output:
      if term_target[ini:end] == term_output[ini:end]:
        false_negative = False
        break
    
    if false_negative:
      num_false_negative += 1
      #if term_type == 'ap':
        #print('printing false negatives')
        #print(text)
        #print(target, "####", output)

  return num_false_negative


def get_false_positives_from_sentence(target, output, term_type, text):
  if term_type == 'ap':
    ini = 0
    end = 2
  elif term_type == 'op':
    ini = 2
    end = 4
  elif term_type == 'sent':
    ini = 4
    end = 5

  num_false_positive = 0
  for term_output in output:
    false_positive = True
    for term_target in target:
      if term_target[ini:end] == term_output[ini:end]:
        false_positive = False
        break
    
    if false_positive:
      num_false_positive += 1
      #if term_type == 'ap':
        #print('printing false positives')
        #print(text)
        #print(target, "####", output)

  
  return num_false_positive


def total_FN_FP(targets, outputs, overlapped, term_type, texts):
  FN_total = {'ap': 0, 'op': 0, 'ap_op': 0, 'not': 0, 'over': 0}
  FP_total = {'ap': 0, 'op': 0, 'ap_op': 0, 'not': 0, 'over': 0}

  for target, output, overlap, text in zip(targets, outputs, overlapped, texts):
    FN = get_false_negatives_from_sentence(target, output, term_type, text)
    FP = get_false_positives_from_sentence(target, output, term_type, text)
    FN_total[overlap] += FN
    FP_total[overlap] += FP

    if overlap in ['ap', 'op', 'ap_op']:
      FN_total['over'] += FN
      FP_total['over'] += FP


  return FN_total, FP_total

def get_proportion_from_dic(dict_):
    values = np.array(list(dict_.values()))
    values_p = np.round(list(values/np.sum(values)), 4)
    keys = dict_.keys()
    dictionary = dict(zip(keys, values_p))

    return dictionary
    

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
    parser.add_argument('--min', default=0, type=int)
    parser.add_argument('--max', default=512, type=int)
    opt = parser.parse_args()

    datasets = ['restES']#['rest14', 'lap14', 'rehol', 'reli', 'rest15', 'rest16', 'restES']

    for dataset in datasets:

        if dataset in ['rest14', 'rest16', 'lap14', 'rest15']:
            opt.bert_model = 'bert-base-uncased'
            opt.case = 'uncased'
            opt.lang = 'en'
        elif dataset in ['rehol', 'reli']:
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
            'rest15_c_0': 'cross_validation/data/rest15/c_0',
            'rehol_c_0': 'cross_validation/data/rehol/c_0',
            'reli_c_0': 'cross_validation/data/reli/c_0',
            'lap14_c_0': 'cross_validation/data/lap14/c_0',
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

        texts, triplets_real, idx2gragh, overlapped = load_sentences(dataset, cluster, opt.min, opt.max)

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

        FNx_ap, FPx_ap = total_FN_FP(all_targets, all_outputs, overlapped, 'ap', texts)
        FNx_op, FPx_op = total_FN_FP(all_targets, all_outputs, overlapped, 'op', texts)
        FNx_sent, FPx_sent = total_FN_FP(all_targets, all_outputs, overlapped, 'sent', texts)


        """for i, (tar, out, text, over) in enumerate(zip(all_targets, all_outputs, texts, overlapped)):
            print(text)
            print(over)
            print(tar)
            print(out)
            print('\n')


            if i == 5: break"""

        over_distribution = Counter(overlapped)
        rel_dist_over = get_proportion_from_dic(over_distribution)

        print("Abs", over_distribution, "Rel", rel_dist_over)
        print(rel_dist_over)
        print("False Negatives (aspect):", FNx_ap, " Rel:", get_proportion_from_dic(FNx_ap))
        print("False Positives (aspect):", FPx_ap, " Rel:", get_proportion_from_dic(FPx_ap))
        print('\n')
        print("False Negatives (opinion):", FNx_op, " Rel:", get_proportion_from_dic(FNx_op))
        print("False Positives (opinion):", FPx_op, " Rel:", get_proportion_from_dic(FPx_op))
        print('\n')
        print("False Negatives (sentiment):", FNx_sent, " Rel:", get_proportion_from_dic(FNx_sent))
        print("False Positives (sentiment):", FPx_sent, " Rel:", get_proportion_from_dic(FPx_sent))

