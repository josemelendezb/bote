import argparse
import json


def diplay_results_dataset(dataset, models):
    aspect_extraction_all_models = {}
    opinion_extraction_all_models = {}
    triplet_extraction_all_models = {}

    for model in models:
        aspect_extraction = {'precision': [], 'recall': [], 'f1': []}
        opinion_extraction = {'precision': [], 'recall': [], 'f1': []}
        triplet_extraction = {'precision': [], 'recall': [], 'f1': []}
        for i in range(4):
            with open('./log/' + model+'_' + dataset + '_c_' + str(i) + '_test.json', 'r') as fp:
                results = json.load(fp)
            aspect_extraction['precision'].extend(results['aspect_extraction']['precision'])
            aspect_extraction['recall'].extend(results['aspect_extraction']['recall'])
            aspect_extraction['f1'].extend(results['aspect_extraction']['f1'])

            opinion_extraction['precision'].extend(results['opinion_extraction']['precision'])
            opinion_extraction['recall'].extend(results['opinion_extraction']['recall'])
            opinion_extraction['f1'].extend(results['opinion_extraction']['f1'])

            triplet_extraction['precision'].extend(results['triplet_extraction']['precision'])
            triplet_extraction['recall'].extend(results['triplet_extraction']['recall'])
            triplet_extraction['f1'].extend(results['triplet_extraction']['f1'])

        aspect_extraction_all_models[model] = {
            'precision': aspect_extraction['precision'],
            'recall': aspect_extraction['recall'],
            'f1': aspect_extraction['f1']
            }

        opinion_extraction_all_models[model] = {
            'precision': opinion_extraction['precision'],
            'recall': opinion_extraction['recall'],
            'f1': opinion_extraction['f1']
            }

        triplet_extraction_all_models[model] = {
            'precision': triplet_extraction['precision'],
            'recall': triplet_extraction['recall'],
            'f1': triplet_extraction['f1']
            }

    # Aspect Extraction
    print(dataset.upper(),'DATASET')
    print(20*'=')
    print('Aspect Extraction')
    print(20*'=')

    for metric in ['precision', 'recall', 'f1']:
        print('-'+metric)
        for model in models:
            metric_avg = sum(aspect_extraction_all_models[model][metric])/len(aspect_extraction_all_models[model][metric])
            print(' ',model, round(metric_avg, 4))

    print('\n')
    print(20*'=')
    print('Opinion Extraction')
    print(20*'=')
    for metric in ['precision', 'recall', 'f1']:
        print('-'+metric)
        for model in models:
            metric_avg = sum(opinion_extraction_all_models[model][metric])/len(opinion_extraction_all_models[model][metric])
            print(' ',model, round(metric_avg, 4))

    print('\n')
    print(20*'=')
    print('Triplet Extraction')
    print(20*'=')
    for metric in ['precision', 'recall', 'f1']:
        print('-'+metric)
        for model in models:
            metric_avg = sum(triplet_extraction_all_models[model][metric])/len(triplet_extraction_all_models[model][metric])
            print(' ',model, round(metric_avg, 4))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=['rest14', 'rest15', 'rest16', 'lap14', 'reli', 'rehol'])
    parser.add_argument("--models", nargs="+", default=['ote', 'cmla', 'bote'])
    opt = parser.parse_args()

    for dataset in opt.datasets:
        diplay_results_dataset(dataset, opt.models)
        print(50*'*')
        print(50*'*')

