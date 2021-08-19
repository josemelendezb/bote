# -*- coding: utf-8 -*-

import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
import math

class BOTE():
    def __init__(self, bert_model, device):
        
        self.device = device
        self.bert = AutoModel.from_pretrained(bert_model).to(self.device)

        
    def set_bert_vectors_to_naive_bert_vectors(self, batch_bert_vectors, batch_position_bert_in_naive, batch_text_indices_bert, batch_text_mask_bert):
        batch_vectors_subwords = []
        for k, (vectors, position, text_indices, text_mask) in enumerate(zip(batch_bert_vectors, batch_position_bert_in_naive, batch_text_indices_bert, batch_text_mask_bert)):
            
            text_indices = text_indices.to(self.device)
            text_mask = text_mask.to(self.device)
            vectors_subwords = torch.tensor([]).to(self.device)
            
            i = 0
            while i < len(position):
                pair = position[i]
                
                if pair[0] != -1 and pair[1] != -1:
                    mean_vectors_subwords = torch.mean(vectors[pair[0]:pair[1]+1], 0).to(self.device)
                    vectors_subwords = torch.cat((vectors_subwords, torch.reshape(mean_vectors_subwords, (1, mean_vectors_subwords.shape[0]))))
                    i += 1
                else:
                    #Complete pad tokens using the former bert pad token from bert_layer
                    first_pad_token = position[i-1][1] + 1 #first pad token
                    first_pad_token = int(first_pad_token)
                    fill_padding = len(position) - len(vectors_subwords)
                    pad_bert_vectors = vectors[first_pad_token: first_pad_token + fill_padding]
                    
                    if len(pad_bert_vectors) < fill_padding:
                        add_n_pads = fill_padding - len(pad_bert_vectors)
                        repeat_mean_tensor = self.generate_pad_vectors(text_indices, text_mask, add_n_pads, fill_padding, vectors_subwords, 'new_pads')
                        #repeat_mean_tensor = torch.zeros(add_n_pads,self.opt.embed_dim).to(self.device)
                        vectors_subwords = torch.cat((vectors_subwords, pad_bert_vectors))
                        vectors_subwords = torch.cat((vectors_subwords, repeat_mean_tensor))
                    else:
                        vectors_subwords = torch.cat((vectors_subwords, pad_bert_vectors))
                        
                    break

            batch_vectors_subwords.append(vectors_subwords)
        
        batch_vectors_subwords = torch.stack(batch_vectors_subwords)
        return batch_vectors_subwords
        
    def generate_pad_vectors(self, text_indices_bert, text_mask_bert, add_n_pads, fill_padding, vectors_subwords, mode = 'new_pads'):
        text_indices_bert = text_indices_bert.to(self.device)
        if mode == 'mean':
            mean_tensor = torch.mean(vectors_subwords,dim = 0)
            mean_tensor = torch.reshape(mean_tensor, (1, mean_tensor.shape[0]))
            repeat_mean_tensor = torch.cat(add_n_pads*[mean_tensor])
            return repeat_mean_tensor
        else:
            pads = torch.zeros(add_n_pads).to(self.device)
            text_indices_bert = torch.cat((text_indices_bert, pads)).int()
            text_mask_bert = torch.cat((text_mask_bert, pads.bool()))
            
            text_indices_bert = torch.reshape(text_indices_bert, (1, text_indices_bert.shape[0]))
            text_mask_bert = torch.reshape(text_mask_bert, (1, text_mask_bert.shape[0]))
            bert_layer = self.bert(input_ids = text_indices_bert, attention_mask = text_mask_bert).last_hidden_state
            
            return bert_layer[0, -add_n_pads:]

    
    def get_vectors_measure(self, text_indices_bert, text_mask_bert, position_bert_in_naive, bert_layer_index):
        
        text_indices_bert = torch.tensor([text_indices_bert]).to(self.device)
        text_mask_bert = torch.tensor([text_mask_bert]).to(self.device)
        position_bert_in_naive = torch.tensor([position_bert_in_naive]).to(self.device)

        with torch.no_grad():
            bert_layer = self.bert(input_ids = text_indices_bert, attention_mask = text_mask_bert, output_hidden_states = True).hidden_states[bert_layer_index]

        bert_layer = self.set_bert_vectors_to_naive_bert_vectors(bert_layer, position_bert_in_naive, text_indices_bert, text_mask_bert)

        return bert_layer


from data_utils import BertTokenizer

texts = ['estou gostando dessa casa']
pred = [(20, 22, 24, 24, 1)]
real = [([20, 21, 22], [24], 'NEU')]



tokenizer = BertTokenizer('neuralmind/bert-base-portuguese-cased', 'cased', 'pt_core_news_sm', 'pt')
model = BOTE('neuralmind/bert-base-portuguese-cased', 'cuda')

for text in texts:

    text_indices, text_indices_bert, position_bert_in_naive = tokenizer.text_to_sequence(text)
    text_mask_bert = [1] * len(text_indices_bert)
    #print(text_indices, text_indices_bert, position_bert_in_naive)

    word_vectors = model.get_vectors_measure(text_indices_bert, text_mask_bert, position_bert_in_naive, 10)
    print(word_vectors)

    