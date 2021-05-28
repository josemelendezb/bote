#!/bin/bash
while getopts d:E: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        E) epochs=${OPTARG};;
    esac
done

if [ $dataset == 'rest14' ] || [ $dataset == 'rest15' ] || [ $dataset == 'rest16' ] || [ $dataset == 'lap14' ]
then
    bert_model='bert-base-uncased'
    lang='en'
    cased='uncased'
elif [ $dataset == 'reli' ] || [ $dataset == 'rehol' ]
then
    bert_model='neuralmind/bert-base-portuguese-cased'
    lang='pt'
    cased='cased'
else
    echo 'Select an appropriate dataset. Ex: rest14, rest15, rest16, lap14, reli or rehol.'
    sleep 6
    exit
fi

#datasets=( rest14 rest15 rest16 lap14 )
#for data in "${datasets[@]}"; do
for i in 0 1 2 3; do
    dataset_="${dataset}_c_${i}"
    #python train.py --model ote --case $cased --dataset $dataset_ --num_epoch $epochs --device cuda --patience 20 --repeats 5 --lang $lang --embed_dim 300 --hidden_dim 300
    #python train.py --model cmla --case $cased --dataset $dataset_ --num_epoch $epochs --device cuda --patience 20 --repeats 5 --lang $lang --embed_dim 300 --hidden_dim 300
    python train.py --model bote --case $cased --dataset $dataset_ --num_epoch $epochs --device cuda  --patience 20 --repeats 5 --lang $lang --bert_model $bert_model --bert_layer_index 10
done
#done
