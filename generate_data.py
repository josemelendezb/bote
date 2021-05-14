import argparse
import pandas as pd
import numpy as np
import os
import pickle
import json

datasets = ['rest14', 'rest15', 'rest16', 'lap14', 'reli', 'rehol']
if __name__ == '__main__':
    
    for dataset in datasets:
        
        print('Loading data from '+dataset)
        data = pd.read_csv(os.getcwd()+'/datasets/'+dataset+'/total_triplets.txt', sep="|", header=None, names = ["content"], engine='python')
        path_matrix = open('./datasets/'+dataset+'/total_triplets.txt.graph','rb')
        matrix = pickle.load(path_matrix)
        path_matrix.close()

        data = data.sample(frac=1, random_state=31)

        # We extract the indexes of each review
        indexes = data.index.tolist()

        # We divide the indices into four groups. Two of the four groups will be used for the training data.
        # The other two are used for the validation and testing group. Later the groups are alternated
        split_limit = round(len(indexes)*0.25)
        fold_25a = indexes[0:split_limit]
        fold_25b = indexes[split_limit:2*split_limit]
        fold_25c = indexes[2*split_limit:3*split_limit]
        fold_25d = indexes[3*split_limit:]
        folds = [fold_25a, fold_25b, fold_25c, fold_25d]

        folds_indexes = [
            {'train': [0,1], 'dev': [2], 'test': [3]},
            {'train': [0,1], 'dev': [3], 'test': [2]},
            {'train': [2,3], 'dev': [0], 'test': [1]},
            {'train': [2,3], 'dev': [1], 'test': [0]}
        ]

        # We generate the clusters that contain the training, validation and test groups
        clustering_indexes = []
        for fold_index in folds_indexes:
            cluster = {}
            for tag, index in zip(fold_index.keys(), fold_index.values()):
                if tag == 'train':
                    cluster[tag] = folds[index[0]] + folds[index[1]]
                else:
                    cluster[tag] = folds[index[0]]

            clustering_indexes.append(cluster)

        print('Generating the dataset clusters from '+dataset)

        #with open('./cross_validation/data/index_reviews_clustering_'+ dataset +'.json', 'w') as fp:
        #    json.dump(clustering_indexes, fp)

        outfile = open('./cross_validation/index_reviews_clustering_'+ dataset +'.json', 'w')
        outfile.write(json.dumps(clustering_indexes, indent=4, sort_keys=False))
        outfile.close()

        for i, indexes in enumerate(clustering_indexes):
            train_data = data.loc[indexes['train']]
            dev_data = data.loc[indexes['dev']]
            test_data = data.loc[indexes['test']]

            train_matrix = list(map(matrix.get, indexes['train']))
            dev_matrix = list(map(matrix.get, indexes['dev']))
            test_matrix = list(map(matrix.get, indexes['test']))

            fout = open("./cross_validation/data/"+dataset+"/c_"+str(i)+"/train_triplets.txt.graph", 'wb')
            train_data.to_csv(os.getcwd()+"/cross_validation/data/"+dataset+"/c_"+str(i)+"/train_triplets.txt", sep = "|", header = False, index = False)
            train_data.to_csv(os.getcwd()+"/cross_validation/data/"+dataset+"/c_"+str(i)+"/train.txt", sep = "|", header = False, index = False)
            pickle.dump(train_matrix, fout)
            fout.close()
            
            fout = open("./cross_validation/data/"+dataset+"/c_"+str(i)+"/dev_triplets.txt.graph", 'wb')
            dev_data.to_csv(os.getcwd()+"/cross_validation/data/"+dataset+"/c_"+str(i)+"/dev_triplets.txt", sep = "|", header = False, index = False)
            dev_data.to_csv(os.getcwd()+"/cross_validation/data/"+dataset+"/c_"+str(i)+"/dev.txt", sep = "|", header = False, index = False)
            pickle.dump(dev_matrix, fout)
            fout.close()
            
            fout = open("./cross_validation/data/"+dataset+"/c_"+str(i)+"/test_triplets.txt.graph", 'wb')
            test_data.to_csv(os.getcwd()+"/cross_validation/data/"+dataset+"/c_"+str(i)+"/test_triplets.txt", sep = "|", header = False, index = False)
            test_data.to_csv(os.getcwd()+"/cross_validation/data/"+dataset+"/c_"+str(i)+"/test.txt", sep = "|", header = False, index = False)
            pickle.dump(test_matrix, fout)
            fout.close()
        
        print('Saved datasets from '+dataset+'\n')
    print('Saved datasets')
        
        
