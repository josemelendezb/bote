# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from models import BOTE
from data_utils import ABSADataReaderBERT, BertTokenizer

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

    def evaluate(self, text):
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
            'dependency_graph': torch.tensor([]),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)
        
        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred]


if __name__ == '__main__':
    dataset = 'rehol_c_0'
    # set your trained models here
    model_state_dict_paths = {
        'bote': 'state_dict/bote_'+dataset+'.pkl',
    }
    model_classes = {
        'bote': BOTE,
        
    }
    input_colses = {
        'bote': ['text_indices', 'text_mask', 'text_indices_bert', 'text_mask_bert', 'position_bert_in_naive', 'postag_indices', 'dependency_graph'],
    }
    target_colses = {
        'bote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
    }

    glove_files = {
        'en': 'glove.300d.txt',
        'pt': 'glove.300d_pt.txt',
        'es': 'glove.300d_es.txt',
    }

    data_dirs = {
        'rest14_c_0': 'cross_validation/data/rest14/c_0',
        'reli_c_0': 'cross_validation/data/reli/c_0',
        'rehol_c_0': 'cross_validation/data/rehol/c_0',
    }

    spacy_languages = {
        'en': 'en_core_web_md',
        'pt': 'pt_core_news_sm',
        'es': 'es_core_news_md',
    }

    class Option(object): pass
    opt = Option()
    opt.bert_model = 'neuralmind/bert-base-portuguese-cased'
    opt.case = 'cased'
    opt.bert_layer_index = 10
    opt.lang = 'pt'

    opt.dataset = dataset
    opt.model_name = 'bote'
    opt.eval_cols = ['ap_spans', 'op_spans','triplets']
    opt.model_class = model_classes[opt.model_name]
    opt.input_cols = input_colses[opt.model_name]
    opt.target_cols = target_colses[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 768
    opt.polarities_dim = 4
    opt.batch_size = 32
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.spacy_lang = spacy_languages[opt.lang]

    inf = Inferer(opt)

    texts = ['Prós : Colchão razoável , chuveiro regular e boa limpeza geral .']
    texts = []
    targets = []
    filename = os.path.join(opt.data_dir+'/test_triplets.txt')
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp:
            text = line.strip().split('####')[0]
            target = line.strip().split('####')[1]
            
            texts.append(text)
            targets.append(target)



    for i, (text, target) in enumerate(zip(texts, targets)):
        print(text)
        triplets = inf.evaluate(text)[2][0]
        words = text.split()
        polarity_map = {0:'N', 1:'NEU', 2:'NEG', 3:'POS'}
        #print(triplets)
        for triplet in triplets:
            ap_beg, ap_end, op_beg, op_end, p = triplet
            ap = ' '.join(words[ap_beg:ap_end+1])
            op = ' '.join(words[op_beg:op_end+1])
            polarity = polarity_map[p]
            print(f'{ap}, {op}, {polarity}')
        
        print("Triplets obtidos")
        print(triplets)
        print("Triplets reais")
        print(target)
        print('\n')

        if i == 20: break

