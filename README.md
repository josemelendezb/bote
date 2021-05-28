# BOTE

**BOTE** - **B**ert for **O**pinion **T**riplet **E**xtraction
* Code and preprocessed dataset for paper titled A Deep Learning approach for Aspect Sentiment Triplet Extraction in Portuguese (preparing for submission)
* [Jose Melendez](https://github.com/josemelendezb) and [Glauber de Bona](https://scholar.google.com.br/citations?user=9OZKg7EAAAAJ&hl=en).

## Requirements

* Python 3.7.10
* PyTorch 1.8.1+cu101
* Numpy 1.19.5
* Transformers 4.6.1
* Spacy 2.2.4
* Pandas 1.1.5

## Get started
* Install Spacy Languages
```bash
python -m spacy download en_core_web_md
python -m spacy download pt_core_news_sm
python -m spacy download es_core_news_md
```

* Set enviroment
```bash
python generate_directory_data.py
python dependency_graph.py --undirected 1
python generate_data.py
```
## Usage

* Download pretrained GloVe embeddings (Englisg and Portuguese) with these links: [EN Glove](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and [PT Glove](http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip) . Extract files into `glove/`.

### Run standalone model
* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python train.py --model bote --case cased --dataset reli_c_0 --bert_model neuralmind/bert-base-portuguese-cased --lang pt
```
### Run experiments
* Run proposted model and baselines
```bash
chmod -R 777 run.sh
bash ./run.sh -d 'rehol' -E 60
```

### Display Results proposted model vs baselines
```bash
python cross_validation/display_results.py
```

## Task

An overview of the task aspect sentiment triplet extraction (ASTE) is given below

![model](/assets/subtasks.png)

Given a sentence $S=\{w_{1},w_{2},w_{3},...,w_{n}\}$ consisting of $n$ words, extracting all possible triplets $T=\{(a,o,p)_{m}\}_{m=1}^{|T|}$  from $S$, where $a$, $o$ and $p$ respectively denote an n-gram aspect term, an n-gram opinion term and a sentiment polarity; $a_{m}$ and $o_{m}$ can be represented as their start and end positions ($s_{m}$, $e_{m}$) in $S$ and $p_m \in \{Positive, Negative, Neutral\}$.