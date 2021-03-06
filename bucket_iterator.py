# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['text_indices']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    @staticmethod
    def pad_data(batch_data):
        batch_text_indices = []
        batch_text_mask = []
        batch_ap_indices = []
        batch_op_indices = []
        batch_triplet_indices = []
        batch_ap_spans = []
        batch_op_spans = []
        batch_triplets = []
        max_len = max([len(t['text_indices']) for t in batch_data])
        for item in batch_data:
            text_indices, ap_indices, op_indices, triplet_indices, ap_spans, op_spans, triplets = \
                item['text_indices'], item['ap_indices'], item['op_indices'], item['triplet_indices'], \
                item['ap_spans'], item['op_spans'], item['triplets']
            # 0-padding because 0 stands for 'O'
            text_padding = [0] * (max_len - len(text_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_text_mask.append([1] * len(text_indices) + text_padding)
            batch_ap_indices.append(ap_indices + text_padding)
            batch_op_indices.append(op_indices + text_padding)
            batch_triplet_indices.append(numpy.pad(triplet_indices, ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_ap_spans.append(ap_spans)
            batch_op_spans.append(op_spans)
            batch_triplets.append(triplets)
        return { 
                'text_indices': torch.tensor(batch_text_indices), 
                'text_mask': torch.tensor(batch_text_mask, dtype=torch.bool),
                'ap_indices': torch.tensor(batch_ap_indices), 
                'op_indices': torch.tensor(batch_op_indices), 
                'triplet_indices': torch.tensor(batch_triplet_indices), 
                'ap_spans': batch_ap_spans,
                'op_spans': batch_op_spans,
                'triplets': batch_triplets,
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]



class BucketIteratorBert(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['text_indices']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    @staticmethod
    def pad_data(batch_data):

        batch_text_indices = []
        batch_text_mask = []
        batch_ap_indices = []
        batch_op_indices = []
        batch_triplet_indices = []
        batch_ap_spans = []
        batch_op_spans = []
        batch_triplets = []
        batch_text_indices_bert = []
        batch_position_bert_in_naive = []
        batch_text_mask_bert = []
        batch_postag_indices = []
        batch_dependency_graph = []
        
        max_len = max([len(t['text_indices']) for t in batch_data])
        max_len_bert = max([len(t['text_indices_bert']) for t in batch_data])
        #j = 0
        for item in batch_data:
            text_indices, ap_indices, op_indices, triplet_indices, ap_spans, op_spans, triplets, text_indices_bert, position_bert_in_naive, dependency_graph, postag_indices  = \
                item['text_indices'], item['ap_indices'], item['op_indices'], item['triplet_indices'], item['ap_spans'], item['op_spans'], \
                item['triplets'], item['text_indices_bert'], item['position_bert_in_naive'], item['dependency_graph'], item['postag_indices']
            
            # 0-padding because 0 stands for 'O'
            text_padding = [0] * (max_len - len(text_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_text_mask.append([1] * len(text_indices) + text_padding)
            batch_ap_indices.append(ap_indices + text_padding)
            batch_op_indices.append(op_indices + text_padding)
            batch_triplet_indices.append(numpy.pad(triplet_indices, ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_ap_spans.append(ap_spans)
            batch_op_spans.append(op_spans)
            batch_triplets.append(triplets)
            batch_postag_indices.append(postag_indices + text_padding)
            
            text_padding_bert = [0] * (max_len_bert - len(text_indices_bert))
            padding_position = [[-1,-1]] * (max_len - len(text_indices))
            batch_text_indices_bert.append(text_indices_bert + text_padding_bert)
            batch_text_mask_bert.append([1] * len(text_indices_bert) + text_padding_bert)
            batch_position_bert_in_naive.append(list(position_bert_in_naive) + padding_position)

            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            
        return { 
                'text_indices': torch.tensor(batch_text_indices), 
                'text_mask': torch.tensor(batch_text_mask, dtype=torch.bool),
                'ap_indices': torch.tensor(batch_ap_indices), 
                'op_indices': torch.tensor(batch_op_indices), 
                'triplet_indices': torch.tensor(batch_triplet_indices), 
                'ap_spans': batch_ap_spans,
                'op_spans': batch_op_spans,
                'triplets': batch_triplets,
                'text_indices_bert': torch.tensor(batch_text_indices_bert),
                'text_mask_bert': torch.tensor(batch_text_mask_bert, dtype=torch.bool),
                'position_bert_in_naive': torch.tensor(batch_position_bert_in_naive),
                'postag_indices': torch.tensor(batch_postag_indices),
                'dependency_graph': torch.tensor(batch_dependency_graph),
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]