import numpy as np
import spacy
import pickle
import scipy.sparse as sp
from spacy.tokens import Doc

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

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # Normalize by row
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    matrix= normalize(matrix)
    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines)):
        text, _, triplets = [s.lower().strip() for s in lines[i].partition("####")]
        adj_matrix = dependency_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close() 

if __name__ == '__main__':
    process("./datasets/14rest/train_triplets.txt")
    process("./datasets/14rest/dev_triplets.txt")
    process("./datasets/14rest/test_triplets.txt")
    process("./datasets/15rest/train_triplets.txt")
    process("./datasets/15rest/dev_triplets.txt")
    process("./datasets/15rest/test_triplets.txt")
    process("./datasets/16rest/train_triplets.txt")
    process("./datasets/16rest/dev_triplets.txt")
    process("./datasets/16rest/test_triplets.txt")
    process("./datasets/14lap/train_triplets.txt")
    process("./datasets/14lap/dev_triplets.txt")
    process("./datasets/14lap/test_triplets.txt")
