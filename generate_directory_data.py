import os

datasets = ['rest14', 'rest15', 'rest16', 'lap14', 'reli', 'rehol']

if __name__ == '__main__':
    for dataset in datasets:
        for i in range(4): # They are four clusters to do cross validation per dataset
            path = 'cross_validation/data/'+dataset+'/c_'+str(i)+'/'
            try:
                os.makedirs(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s" % path)