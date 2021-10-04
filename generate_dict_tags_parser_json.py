import os
import spacy
import pandas as pd
import json
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)



if __name__ == '__main__':

    #folder = "/content/gdrive/MyDrive/dissertacao/asba/bote/"
    languages = [
        ['es', 'es_core_news_md', 
            [
                './datasets/restES/total_triplets.txt'
            ]
        ],
        ['en', 'en_core_web_md',
            [
                './datasets/rest14/total_triplets.txt',
                './datasets/rest15/total_triplets.txt',
                './datasets/rest16/total_triplets.txt',
                './datasets/lap14/total_triplets.txt'
            ]
        ],
        ['pt', 'pt_core_news_sm',
            [
                './datasets/rehol/total_triplets.txt',
                './datasets/reli/total_triplets.txt'
            ]
        ]
    ]

    for lang in languages:
        nlp = spacy.load(lang[1])
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        pairs_tag_dict = {'PAD': 0, 'UNK': 1}
        index = 2
        for path in lang[2]:
            data = pd.read_csv(path, sep="####", header=None, names = ["text", "triplet"], engine='python')
            
            for i, (s_ote, p_ote) in enumerate(zip(data["text"], data["triplet"])):
                doc = nlp(s_ote)
                for token in doc:
                    cat = token.tag_ + "__" + token.dep_

                    if cat not in pairs_tag_dict.keys():
                        pairs_tag_dict[cat] = index
                        index += 1

        with open('dict_tags_parser_tagger_'+lang[0]+'.json', 'w') as fp:
            json.dump(pairs_tag_dict, fp)