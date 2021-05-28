import numpy as np
import spacy
import pickle
import scipy.sparse as sp
from spacy.tokens import Doc
import argparse

# A white space tokenizer is created to replace spacy tokenizer
# to make compatible text_indices token length with postag_indices token length
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # Normalize by row
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def dependency_adj_matrix(text, nlp, undirected = 0, diag_one = 0, normalized = 0):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        if diag_one == 1: matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            if undirected == 1: matrix[child.i][token.i] = 1

    matrix = normalize(matrix) if normalized == 1 else matrix

    return matrix

def process(filename, nlp, opt):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines)):
        text, _, triplets = [s.lower().strip() for s in lines[i].partition("####")]
        adj_matrix = dependency_adj_matrix(text, nlp, opt.undirected, opt.diag_one, opt.normalized)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close() 

datasets = [
    ['rest14', 'en_core_web_md'],
    ['rest15', 'en_core_web_md'],
    ['rest16', 'en_core_web_md'],
    ['lap14', 'en_core_web_md'],
    ['reli', 'pt_core_news_sm'],
    ['rehol', 'pt_core_news_sm']
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--undirected', default=0, type=int)
    parser.add_argument('--diag_one', default=0, type=int)
    parser.add_argument('--normalized', default=0, type=int)
    opt = parser.parse_args()

    print("Generating adjacency matrices...")
    lang_above = ''
    for dataset in datasets:
        if dataset[1] != lang_above: # Cargar solamente cuando el idioma cambie
            print('Language '+ dataset[1] +' selected')
            nlp = spacy.load(dataset[1])
            nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        print("Generating matrix dataset "+dataset[0])
        process('./datasets/'+dataset[0]+'/total_triplets.txt', nlp, opt)
        lang_above = dataset[1]
    
    print("process finished")
