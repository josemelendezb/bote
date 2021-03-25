# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tag_utils import bio2bieos, bieos2span, find_span_with_end

class Biaffine(nn.Module):
    def __init__(self, opt, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.opt = opt
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.opt.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.opt.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

class BOTE(nn.Module):
    def __init__(self, embedding_matrix, opt, idx2tag, idx2polarity):
        super(BOTE, self).__init__()
        self.opt = opt
        self.idx2tag = idx2tag
        self.tag_dim = len(self.idx2tag)
        self.idx2polarity = idx2polarity
        reduc_dim = 400
        
        #num_filters=256
        
        self.bert = AutoModel.from_pretrained(opt.bert_model)
        self.bert_dropout = nn.Dropout(0.5)
        self.reduc = nn.Linear(opt.embed_dim, reduc_dim)
        self.ap_fc = nn.Linear(reduc_dim, 100)
        self.op_fc = nn.Linear(reduc_dim, 100)
        self.triplet_biaffine = Biaffine(opt, 100, 100, opt.polarities_dim, bias=(True, False))
        self.ap_tag_fc = nn.Linear(100, self.tag_dim)
        self.op_tag_fc = nn.Linear(100, self.tag_dim)

        #self.conv1d = nn.Conv1d(in_channels=opt.embed_dim, out_channels=num_filters, kernel_size=1)
        
        
        for param in self.bert.base_model.parameters():
            param.requires_grad = False

    def calc_loss(self, outputs, targets):
        ap_out, op_out, triplet_out = outputs
        ap_tag, op_tag, triplet, mask = targets
        # tag loss
        ap_tag_loss = F.cross_entropy(ap_out.flatten(0, 1), ap_tag.flatten(0, 1), reduction='none')
        ap_tag_loss = ap_tag_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()
        op_tag_loss = F.cross_entropy(op_out.flatten(0, 1), op_tag.flatten(0, 1), reduction='none')
        op_tag_loss = op_tag_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()
        tag_loss = ap_tag_loss + op_tag_loss
        # sentiment loss
        mat_mask = mask.unsqueeze(2)*mask.unsqueeze(1)
        sentiment_loss = F.cross_entropy(triplet_out.view(-1, self.opt.polarities_dim), triplet.view(-1), reduction='none')
        sentiment_loss = sentiment_loss.masked_select(mat_mask.view(-1)).sum() / mat_mask.sum()
        return tag_loss + sentiment_loss
    
    def set_bert_vectors_to_naive_bert_vectors(self, batch_bert_vectors, batch_position_bert_in_naive, batch_text_indices_bert, batch_text_mask_bert):
        batch_vectors_subwords = []
        for k, (vectors, position, text_indices, text_mask) in enumerate(zip(batch_bert_vectors, batch_position_bert_in_naive, batch_text_indices_bert, batch_text_mask_bert)):
            
            text_indices = text_indices.to(self.opt.device)
            text_mask = text_mask.to(self.opt.device)
            vectors_subwords = torch.tensor([]).to(self.opt.device)
            
            i = 0
            while i < len(position):
                pair = position[i]
                
                if pair[0] != -1 and pair[1] != -1:
                    mean_vectors_subwords = torch.mean(vectors[pair[0]:pair[1]+1], 0).to(self.opt.device)
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
                        #repeat_mean_tensor = torch.zeros(add_n_pads,self.opt.embed_dim).to(self.opt.device)
                        vectors_subwords = torch.cat((vectors_subwords, pad_bert_vectors))
                        vectors_subwords = torch.cat((vectors_subwords, repeat_mean_tensor))
                    else:
                        vectors_subwords = torch.cat((vectors_subwords, pad_bert_vectors))
                        
                    break

            batch_vectors_subwords.append(vectors_subwords)
        
        batch_vectors_subwords = torch.stack(batch_vectors_subwords)
        return batch_vectors_subwords
        
    def generate_pad_vectors(self, text_indices_bert, text_mask_bert, add_n_pads, fill_padding, vectors_subwords, mode = 'new_pads'):
        text_indices_bert = text_indices_bert.to(self.opt.device)
        if mode == 'mean':
            mean_tensor = torch.mean(vectors_subwords,dim = 0)
            mean_tensor = torch.reshape(mean_tensor, (1, mean_tensor.shape[0]))
            repeat_mean_tensor = torch.cat(add_n_pads*[mean_tensor])
            return repeat_mean_tensor
        else:
            pads = torch.zeros(add_n_pads).to(self.opt.device)
            text_indices_bert = torch.cat((text_indices_bert, pads)).int()
            text_mask_bert = torch.cat((text_mask_bert, pads.bool()))
            
            text_indices_bert = torch.reshape(text_indices_bert, (1, text_indices_bert.shape[0]))
            text_mask_bert = torch.reshape(text_mask_bert, (1, text_mask_bert.shape[0]))
            bert_layer = self.bert(input_ids = text_indices_bert, attention_mask = text_mask_bert).last_hidden_state
            
            return bert_layer[0, -add_n_pads:]
        
        
    def forward(self, inputs):
        text_indices, text_mask, text_indices_bert, text_mask_bert, position_bert_in_naive = inputs
        
        bert_layer = self.bert(input_ids = text_indices_bert, attention_mask = text_mask_bert).last_hidden_state
        #bert_layer = self.bert(input_ids = text_indices, attention_mask = text_mask).last_hidden_state
        bert_layer = self.set_bert_vectors_to_naive_bert_vectors(bert_layer, position_bert_in_naive, text_indices_bert, text_mask_bert)

        #x_reshaped = bert_layer.permute(0, 2, 1)
        #x_conv_list = F.relu(self.conv1d(x_reshaped))
        #bert_layer = x_conv_list.permute(0,2,1)
        bert_layer = F.relu(self.reduc(bert_layer))
        bert_layer = self.bert_dropout(bert_layer)
        ap_rep = F.relu(self.ap_fc(bert_layer))
        op_rep = F.relu(self.op_fc(bert_layer))
        ap_node = F.relu(self.ap_fc(bert_layer))
        op_node = F.relu(self.op_fc(bert_layer))
        ap_out = self.ap_tag_fc(ap_rep)
        op_out = self.op_tag_fc(op_rep)

        triplet_out = self.triplet_biaffine(ap_node, op_node)
        
        return [ap_out, op_out, triplet_out]

    def inference(self, inputs):
        text_indices, text_mask, text_indices_bert, text_mask_bert, position_bert_in_naive = inputs
        text_len = torch.sum(text_mask, dim=-1)
        bert_layer = self.bert(input_ids = text_indices_bert, attention_mask = text_mask_bert).last_hidden_state
        #bert_layer = self.bert(input_ids = text_indices, attention_mask = text_mask).last_hidden_state
        bert_layer = self.set_bert_vectors_to_naive_bert_vectors(bert_layer, position_bert_in_naive, text_indices_bert, text_mask_bert)
        
        #x_reshaped = bert_layer.permute(0, 2, 1)
        #x_conv_list = F.relu(self.conv1d(x_reshaped))
        #bert_layer = x_conv_list.permute(0,2,1)
        bert_layer = F.relu(self.reduc(bert_layer))
        ap_rep = F.relu(self.ap_fc(bert_layer))
        op_rep = F.relu(self.op_fc(bert_layer))
        ap_node = F.relu(self.ap_fc(bert_layer))
        op_node = F.relu(self.op_fc(bert_layer))
        ap_out = self.ap_tag_fc(ap_rep)
        op_out = self.op_tag_fc(op_rep)

        triplet_out = self.triplet_biaffine(ap_node, op_node)
        
        batch_size = text_len.size(0)
        ap_tags = [[] for _ in range(batch_size)]
        op_tags = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            for i in range(text_len[b]):
                ap_tags[b].append(ap_out[b, i, :].argmax(0).item())
        for b in range(batch_size):
            for i in range(text_len[b]):
                op_tags[b].append(op_out[b, i, :].argmax(0).item())

        text_indices = text_indices.cpu().numpy().tolist()
        ap_spans = self.aspect_decode(text_indices, ap_tags, self.idx2tag)
        op_spans = self.opinion_decode(text_indices, op_tags, self.idx2tag)
        mat_mask = (text_mask.unsqueeze(2)*text_mask.unsqueeze(1)).unsqueeze(3).expand(
                              -1, -1, -1, self.opt.polarities_dim)  # batch x seq x seq x polarity
        triplet_indices = torch.zeros_like(triplet_out).to(self.opt.device)
        triplet_indices = triplet_indices.scatter_(3, triplet_out.argmax(dim=3, keepdim=True), 1) * mat_mask.float()
        triplet_indices = torch.nonzero(triplet_indices).cpu().numpy().tolist()
        triplets = self.sentiment_decode(text_indices, ap_tags, op_tags, triplet_indices, self.idx2tag, self.idx2polarity)
        
        return [ap_spans, op_spans, triplets]

    @staticmethod
    def aspect_decode(text_indices, tags, idx2tag):
        #text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span(bio2bieos(_tag_seq), tp='')
        return result

    @staticmethod
    def opinion_decode(text_indices, tags, idx2tag):
        #text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span(bio2bieos(_tag_seq), tp='')
        return result
                
    @staticmethod
    def sentiment_decode(text_indices, ap_tags, op_tags, triplet_indices, idx2tag, idx2polarity):
        #text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(ap_tags)
        result = [[] for _ in range(batch_size)]
        for i in range(len(triplet_indices)):
            b, ap_i, op_i, po = triplet_indices[i]
            if po == 0:
                continue
            _ap_tags = list(map(lambda x: idx2tag[x], ap_tags[b]))
            _op_tags = list(map(lambda x: idx2tag[x], op_tags[b]))
            ap_beg, ap_end = find_span_with_end(ap_i, text_indices[b], _ap_tags, tp='')
            op_beg, op_end = find_span_with_end(op_i, text_indices[b], _op_tags, tp='')
            triplet = (ap_beg, ap_end, op_beg, op_end, po)
            result[b].append(triplet)
        return result