# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import json
import torch
import torch.nn as nn
import numpy as np
from bucket_iterator import BucketIterator, BucketIteratorBert
from data_utils import ABSADataReader, ABSADataReaderBERT, BertTokenizer, build_tokenizer, build_embedding_matrix
from models import CMLA, HAST, OTE, BOTE


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        if opt.model == 'bote':
            absa_data_reader = ABSADataReaderBERT(data_dir=opt.data_dir)
            tokenizer = BertTokenizer(opt.bert_model, opt.case, opt.spacy_lang, opt.lang)
            embedding_matrix = []
            self.train_data_loader = BucketIteratorBert(data=absa_data_reader.get_train(tokenizer), batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = BucketIteratorBert(data=absa_data_reader.get_dev(tokenizer), batch_size=opt.batch_size, shuffle=False)
            self.test_data_loader = BucketIteratorBert(data=absa_data_reader.get_test(tokenizer), batch_size=opt.batch_size, shuffle=False)
        else:
            absa_data_reader = ABSADataReader(data_dir=opt.data_dir)
            tokenizer = build_tokenizer(data_dir=opt.data_dir)
            embedding_matrix = build_embedding_matrix(opt.data_dir, tokenizer.word2idx, opt.embed_dim, opt.dataset, opt.glove_fname)
            self.train_data_loader = BucketIterator(data=absa_data_reader.get_train(tokenizer), batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = BucketIterator(data=absa_data_reader.get_dev(tokenizer), batch_size=opt.batch_size, shuffle=False)
            self.test_data_loader = BucketIterator(data=absa_data_reader.get_test(tokenizer), batch_size=opt.batch_size, shuffle=False)
            
        self.idx2tag, self.idx2polarity = absa_data_reader.reverse_tag_map, absa_data_reader.reverse_polarity_map
        self.model = opt.model_class(embedding_matrix, opt, self.idx2tag, self.idx2polarity).to(opt.device)
        self.history_metrics = {
            'epoch': [], 'step' : [],
            'train_ap_precision' : [], 'train_ap_recall' : [], 'train_ap_f1' : [],
            'train_op_precision' : [], 'train_op_recall' : [], 'train_op_f1' : [],
            'train_triplet_precision' : [], 'train_triplet_recall' : [], 'train_triplet_f1' : [],
            'dev_ap_precision' : [], 'dev_ap_recall' : [], 'dev_ap_f1' : [],
            'dev_op_precision' : [], 'dev_op_recall' : [], 'dev_op_f1' : [],
            'dev_triplet_precision' : [], 'dev_triplet_recall' : [], 'dev_triplet_f1' : []
        }

        self.results = {
            'aspect_extraction': {'precision': [], 'recall': [], 'f1': []},
            'opinion_extraction': {'precision': [], 'recall': [], 'f1': []},
            'triplet_extraction': {'precision': [], 'recall': [], 'f1': []}
        }

        self._print_args()
        

        if torch.cuda.is_available():
            print('>>> cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('>>> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('>>> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                    
    def _save_history(self, epoch, step):
        train_ap_metrics, train_op_metrics, train_triplet_metrics = self._evaluate(self.train_data_loader)
        train_ap_precision, train_ap_recall, train_ap_f1 = train_ap_metrics
        train_op_precision, train_op_recall, train_op_f1 = train_op_metrics
        train_triplet_precision, train_triplet_recall, train_triplet_f1 = train_triplet_metrics
        
        dev_ap_metrics, dev_op_metrics, dev_triplet_metrics = self._evaluate(self.dev_data_loader)
        dev_ap_precision, dev_ap_recall, dev_ap_f1 = dev_ap_metrics
        dev_op_precision, dev_op_recall, dev_op_f1 = dev_op_metrics
        dev_triplet_precision, dev_triplet_recall, dev_triplet_f1 = dev_triplet_metrics

        self.history_metrics['epoch'].append(epoch)
        self.history_metrics['step'].append(step)

        self.history_metrics['train_ap_precision'].append(train_ap_precision)
        self.history_metrics['train_ap_recall'].append(train_ap_recall)
        self.history_metrics['train_ap_f1'].append(train_ap_f1)

        self.history_metrics['train_op_precision'].append(train_op_precision)
        self.history_metrics['train_op_recall'].append(train_op_recall)
        self.history_metrics['train_op_f1'].append(train_op_f1)
        
        self.history_metrics['train_triplet_precision'].append(train_triplet_precision)
        self.history_metrics['train_triplet_recall'].append(train_triplet_recall)
        self.history_metrics['train_triplet_f1'].append(train_triplet_f1)
        
        self.history_metrics['dev_ap_precision'].append(dev_ap_precision)
        self.history_metrics['dev_ap_recall'].append(dev_ap_recall)
        self.history_metrics['dev_ap_f1'].append(dev_ap_f1)
        
        self.history_metrics['dev_op_precision'].append(dev_op_precision)
        self.history_metrics['dev_op_recall'].append(dev_op_recall)
        self.history_metrics['dev_op_f1'].append(dev_op_f1)
        
        self.history_metrics['dev_triplet_precision'].append(dev_triplet_precision)
        self.history_metrics['dev_triplet_recall'].append(dev_triplet_recall)
        self.history_metrics['dev_triplet_f1'].append(dev_triplet_f1)
        
    def _train(self, optimizer):
        max_dev_f1 = 0
        best_state_dict_path = ''
        global_step = 0
        continue_not_increase = 0
        

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: {0}'.format(epoch+1))
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                targets = [sample_batched[col].to(self.opt.device) for col in self.opt.target_cols]
                outputs = self.model(inputs)
                loss = self.model.calc_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                #if epoch == self.opt.num_epoch - 1: #eliminar
                #    torch.save(inputs, "tensor/inputs.pt")
                #    torch.save(outputs, "tensor/outputs.pt")
                #    torch.save(targets, "tensor/targets.pt")
                
                if self.opt.save_history_metrics:
                    self._save_history(epoch, global_step)

                if global_step % self.opt.log_step == 0:
                    print(">>>>Dev Metrics<<<<")
                    dev_ap_metrics, dev_op_metrics, dev_triplet_metrics = self._evaluate(self.dev_data_loader)
                    dev_ap_precision, dev_ap_recall, dev_ap_f1 = dev_ap_metrics
                    dev_op_precision, dev_op_recall, dev_op_f1 = dev_op_metrics
                    dev_triplet_precision, dev_triplet_recall, dev_triplet_f1 = dev_triplet_metrics
                    print('dev_ap_precision: {:.4f}, dev_ap_recall: {:.4f}, dev_ap_f1: {:.4f}'.format(dev_ap_precision, dev_ap_recall, dev_ap_f1))
                    print('dev_op_precision: {:.4f}, dev_op_recall: {:.4f}, dev_op_f1: {:.4f}'.format(dev_op_precision, dev_op_recall, dev_op_f1))
                    print('loss: {:.4f}, dev_triplet_precision: {:.4f}, dev_triplet_recall: {:.4f}, dev_triplet_f1: {:.4f}'.format(loss.item(), dev_triplet_precision, dev_triplet_recall, dev_triplet_f1))
                    
                    if dev_triplet_f1 > max_dev_f1:
                        increase_flag = True
                        max_dev_f1 = dev_triplet_f1
                        best_state_dict_path = 'state_dict/'+self.opt.model+'_'+self.opt.dataset+'.pkl'
                        torch.save(self.model.state_dict(), best_state_dict_path)
                        print('>>> best model saved.')
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= self.opt.patience:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0    
        return best_state_dict_path

    def _evaluate(self, data_loader):
        # switch model to evaluation mode
        self.model.eval()
        t_ap_spans_all, t_op_spans_all, t_triplets_all  = None, None, None
        t_ap_spans_pred_all, t_op_spans_pred_all, t_triplets_pred_all  = None, None, None
        t_inputs_all = None #eliminar
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                t_ap_spans, t_op_spans, t_triplets = [t_sample_batched[col] for col in self.opt.eval_cols]
                t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)

                if t_ap_spans_all is None:
                    t_ap_spans_all = t_ap_spans
                    t_op_spans_all = t_op_spans
                    t_triplets_all = t_triplets
                    t_ap_spans_pred_all = t_ap_spans_pred
                    t_op_spans_pred_all = t_op_spans_pred
                    t_triplets_pred_all = t_triplets_pred
                    t_inputs_all = t_inputs #Eliminar
                else:
                    t_ap_spans_all = t_ap_spans_all + t_ap_spans
                    t_op_spans_all = t_op_spans_all + t_op_spans
                    t_triplets_all = t_triplets_all + t_triplets
                    t_ap_spans_pred_all = t_ap_spans_pred_all + t_ap_spans_pred
                    t_op_spans_pred_all = t_op_spans_pred_all + t_op_spans_pred
                    t_triplets_pred_all = t_triplets_pred_all + t_triplets_pred
                    t_inputs_all = t_inputs_all + t_inputs #Eliminar
        
        #eliminar
        inputs = t_inputs_all
        outputs = [t_ap_spans_pred_all, t_op_spans_pred_all, t_triplets_pred_all]
        targets = [t_ap_spans_all, t_op_spans_all, t_triplets_all] 
        torch.save(inputs, "tensor/test_inputs.pt")
        torch.save(outputs, "tensor/test_outputs.pt")
        torch.save(targets, "tensor/test_targets.pt")
        ########
                
        return self._metrics(t_ap_spans_all, t_ap_spans_pred_all), self._metrics(t_op_spans_all, t_op_spans_pred_all), self._metrics(t_triplets_all, t_triplets_pred_all)
        
    @staticmethod
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

    def run(self, repeats):
        if not os.path.exists('log/'):
            os.mkdir('log/')

        if not os.path.exists('state_dict/'):
            os.mkdir('state_dict/')

        f_out = open('log/'+self.opt.model+'_'+self.opt.dataset+'_test.json', 'w', encoding='utf-8')

        test_ap_precision_avg = 0
        test_ap_recall_avg = 0
        test_ap_f1_avg = 0
        test_op_precision_avg = 0
        test_op_recall_avg = 0
        test_op_f1_avg = 0
        test_triplet_precision_avg = 0
        test_triplet_recall_avg = 0
        test_triplet_f1_avg = 0
        for i in range(repeats):
            print('repeat: {0}'.format(i+1))
            
            #Keep weights values
            if opt.update_bert:
                for param in self.model.bert.base_model.parameters():
                    param.requires_grad = False
            
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

            # Allow bert update its weights during training
            if opt.update_bert:
                for param in self.model.bert.base_model.parameters():
                    param.requires_grad = True

            best_state_dict_path = self._train(optimizer)
            self.model.load_state_dict(torch.load(best_state_dict_path))
            test_ap_metrics, test_op_metrics, test_triplet_metrics = self._evaluate(self.test_data_loader)
            test_ap_precision, test_ap_recall, test_ap_f1 = test_ap_metrics
            test_op_precision, test_op_recall, test_op_f1 = test_op_metrics
            test_triplet_precision, test_triplet_recall, test_triplet_f1 = test_triplet_metrics

            print('test_ap_precision: {:.4f}, test_ap_recall: {:.4f}, test_ap_f1: {:.4f}'.format(test_ap_precision, test_ap_recall, test_ap_f1))
            print('test_op_precision: {:.4f}, test_op_recall: {:.4f}, test_op_f1: {:.4f}'.format(test_op_precision, test_op_recall, test_op_f1))
            print('test_triplet_precision: {:.4f}, test_triplet_recall: {:.4f}, test_triplet_f1: {:.4f}'.format(test_triplet_precision, test_triplet_recall, test_triplet_f1))
            
            # Save result metrics in dict
            self.results['aspect_extraction']['precision'].append(test_ap_precision)
            self.results['aspect_extraction']['recall'].append(test_ap_recall)
            self.results['aspect_extraction']['f1'].append(test_ap_f1)
            self.results['opinion_extraction']['precision'].append(test_op_precision)
            self.results['opinion_extraction']['recall'].append(test_op_recall)
            self.results['opinion_extraction']['f1'].append(test_op_f1)
            self.results['triplet_extraction']['precision'].append(test_triplet_precision)
            self.results['triplet_extraction']['recall'].append(test_triplet_recall)
            self.results['triplet_extraction']['f1'].append(test_triplet_f1)

            test_ap_precision_avg += test_ap_precision
            test_ap_recall_avg += test_ap_recall
            test_ap_f1_avg += test_ap_f1
            test_op_precision_avg += test_op_precision
            test_op_recall_avg += test_op_recall
            test_op_f1_avg += test_op_f1
            test_triplet_precision_avg += test_triplet_precision
            test_triplet_recall_avg += test_triplet_recall
            test_triplet_f1_avg += test_triplet_f1
            print('#' * 100)
        print("test_ap_precision_avg:", test_ap_precision_avg / repeats)
        print("test_ap_recall_avg:", test_ap_recall_avg / repeats)
        print("test_ap_f1_avg:", test_ap_f1_avg / repeats)
        print("test_op_precision_avg:", test_op_precision_avg / repeats)
        print("test_op_recall_avg:", test_op_recall_avg / repeats)
        print("test_op_f1_avg:", test_op_f1_avg / repeats)
        print("test_triplet_precision_avg:", test_triplet_precision_avg / repeats)
        print("test_triplet_recall_avg:", test_triplet_recall_avg / repeats)
        print("test_triplet_f1_avg:", test_triplet_f1_avg / repeats)

        json.dump(self.results, f_out)
        f_out.close()

        if self.opt.save_history_metrics:
            with open('log/history_metrics.json', 'w') as fp:
                json.dump(self.history_metrics, fp, indent=4)
            



if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ote', type=str)
    parser.add_argument('--case', type=str)
    parser.add_argument('--dataset', default='laptop14', type=str, help='laptop14, rest14, rest15, rest16')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=4, type=int)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--repeats', default=2, type=int)
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--bert_layer_index', default=10, type=int)
    parser.add_argument('--save_history_metrics', action='store_true')
    parser.add_argument('--update_bert', action='store_true')
    parser.add_argument('--lang', default='en', type=str)
    opt = parser.parse_args()

    model_classes = {
        'cmla': CMLA,
        'hast': HAST,
        'ote': OTE,
        'bote': BOTE
    }
    input_colses = {
        'cmla': ['text_indices', 'text_mask'],
        'hast': ['text_indices', 'text_mask'],
        'ote': ['text_indices', 'text_mask'],
        'bote': ['text_indices', 'text_mask', 'text_indices_bert', 'text_mask_bert', 'position_bert_in_naive', 'postag_indices', 'dependency_graph'],
    }
    target_colses = {
        'cmla': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
        'hast': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
        'ote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
        'bote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    #data_dirs = {
    #    'laptop14': 'datasets/lap14',
    #    'rest14': 'datasets/rest14',
    #    'rest15': 'datasets/rest15',
    #    'rest16': 'datasets/rest16',
    #    'reli': 'datasets/reli',
    #    'rehol': 'datasets/rehol'
    #}

    data_dirs = {
        'lap14_c_0': 'cross_validation/data/lap14/c_0',
        'lap14_c_1': 'cross_validation/data/lap14/c_1',
        'lap14_c_2': 'cross_validation/data/lap14/c_2',
        'lap14_c_3': 'cross_validation/data/lap14/c_3',
        'rest14_c_0': 'cross_validation/data/rest14/c_0',
        'rest14_c_1': 'cross_validation/data/rest14/c_1',
        'rest14_c_2': 'cross_validation/data/rest14/c_2',
        'rest14_c_3': 'cross_validation/data/rest14/c_3',
        'rest15_c_0': 'cross_validation/data/rest15/c_0',
        'rest15_c_1': 'cross_validation/data/rest15/c_1',
        'rest15_c_2': 'cross_validation/data/rest15/c_2',
        'rest15_c_3': 'cross_validation/data/rest15/c_3',
        'rest16_c_0': 'cross_validation/data/rest16/c_0',
        'rest16_c_1': 'cross_validation/data/rest16/c_1',
        'rest16_c_2': 'cross_validation/data/rest16/c_2',
        'rest16_c_3': 'cross_validation/data/rest16/c_3',
        'reli_c_0': 'cross_validation/data/reli/c_0',
        'reli_c_1': 'cross_validation/data/reli/c_1',
        'reli_c_2': 'cross_validation/data/reli/c_2',
        'reli_c_3': 'cross_validation/data/reli/c_3',
        'rehol_c_0': 'cross_validation/data/rehol/c_0',
        'rehol_c_1': 'cross_validation/data/rehol/c_1',
        'rehol_c_2': 'cross_validation/data/rehol/c_2',
        'rehol_c_3': 'cross_validation/data/rehol/c_3',
    }


    glove_files = {
        'en': 'glove.300d.txt',
        'pt': 'glove.300d_pt.txt',
        'es': 'glove.300d_es.txt',
    }

    spacy_languages = {
        'en': 'en_core_web_md',
        'pt': 'pt_core_news_sm',
        'es': 'es_core_news_md',
    }

    opt.model_class = model_classes[opt.model]
    opt.input_cols = input_colses[opt.model]
    opt.target_cols = target_colses[opt.model]
    opt.eval_cols = ['ap_spans', 'op_spans', 'triplets']
    opt.initializer = initializers[opt.initializer]
    opt.data_dir = data_dirs[opt.dataset]
    opt.glove_fname = glove_files[opt.lang]
    opt.spacy_lang = spacy_languages[opt.lang]
    
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run(opt.repeats)
