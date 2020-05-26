# Temporal Dependency Parser

## Overview

This repository contains code based on the Temporal Dependency Parser in https://github.com/yuchenz/tdp_ranking, as 
described in [Zhang & Xue, 2018](https://arxiv.org/pdf/1809.00370.pdf) and used in
[Zhang & Xue, 2019](https://www.aclweb.org/anthology/S19-1019). The temporal dependency tree structure used throughout
this repository is described in the earlier paper [Zhang & Xue, 2018](https://arxiv.org/pdf/1808.07599.pdf).

While the original model is written using [DyNet](https://dynet.readthedocs.io/en/latest/), this repository
rewrites their model in [TensorFlow](https://www.tensorflow.org/) 
and also adds a number of variations on the model using [BERT](https://github.com/google-research/bert/).

Also contained in this repository is the data from the authors' feasibility study of annotating temporal dependency
trees with AMT, as detailed in [Zhang & Xue, 2019](https://www.aclweb.org/anthology/S19-1019). A new, higher quality 
corpus is currently under development - please contact the authors of the paper for details. 

## Dependencies

This repository is designed to be run with an Anaconda environment. It is recommended to use two environments: one for
local development on CPU and another for training and running the model on GPU. This allows taking advantage of the
GPU-optimized `tensorflow-gpu` package as well as certain GPU-only Keras layers such as `tf.keras.layers.CuDNNLSTM`.

For convenience, the two environments are checked in as `environment.yml` and `environment_gpu.yml`. You can restore
a conda environment from these with

```bash
conda activate tdp_ranking
conda env update --file environment.yml
```

If you add any new packages, please update the `environment.yml` file with

```bash
conda env export > environment.yml
```

The primary packages that are required are:

* TensorFlow 1.13 (`tensorflow-mkl=1.13` for CPU, `tensorflow-gpu=1.13` for GPU) - note that precisely version 1.13 is 
required†
* NumPy (installed automatically when installing TensorFlow)
* `matplotlib` for plotting loss/accuracy graphs
* `pygraphviz` for plotting parse trees

† 1.12 doesn't support the `tf.compat.v1` API used here which allows forward compatibility with TF 1.14 and TF 2.0, 
while 1.14 has a memory leak when running the `bilstm` model. Hopefully 1.14 will be patched to fix this memory leak, 
and/or 2.0 will become more stable, and it will be possible to upgrade this repository to 1.14 or 2.0 with minimal code 
changes.

### Paths to set

In addition to downloading the dependencies, you should set the following paths:

* Paths to your Anaconda Python versions in [run_exp.sh](run_exp.sh) and [baseline.sh](baseline.sh)
* Optional: Path to the cache directory for TF Hub, which defaults to `/tmp` in [bert_layer.py](shared/bert_layer.py)

## Models

This repository contains four models: `bilstm`, the original architecture described in the paper,
`bert_bilstm` (`bilstm` but using frozen BERT embeddings), `bert_as_classifier` and `bert_as_classifier_alt` 
(which uses the same architecture as `bert_as_classifier` but different features).

### Framing the Problem as a Classification Task

The crux of all these models is that we can frame the tree parsing task as the much simpler classification task of 
choosing the correct parent and edge label for each child, i.e. time expression or event. So each model processes a 
list of (parent, child) tuples and returns, for each tuple, a score for each possible edge label, i.e. it classifies 
that (parent, child) tuple with an edge label. Precisely, the model returns a numerical score for each 
(parent, child, edge label) tuple. The fact that it is seeing the same child repeatedly with different possible parents 
does not concern it when it is predicting the edge label. Then, we take all the (parent, child) tuples with the same 
child and group their scores together to get a list of scores for each possible parent and edge label for that child. 
This is what we softmax (to get a probability distribution) and then choose the highest label from.

Choosing the parent has one caveat: it must satisfy an acyclicity constraint. That is, as we go through our children 
and add them and their parent to the tree, our newly chosen parent may not create a cycle. If it would create a cycle, 
we choose the next-highest-scoring parent which is not cyclic. This acyclicity constraint happens outside of the 
training loop of the model, in fact it outside of the models entirely, namely directly in 
[parse.py](parse.py), not [train.py](train.py) or any part of the model.
 For the BERT-as-Classifier models, even the grouping of tuples by their child happens outside of the model itself, 
 namely either when calculating the loss function of a batch during training or when returning the predictions for a 
 document during parsing. For the Bi-LSTM models, it happens outside of the feed-forward network which calculates the 
 tuple scores, but within the larger model as the final step.

This solves the problem of the number of predictions, namely possible parents + edges, potentially changing with every 
child. Even if we pad the number of parent candidates, the meaning of the parent at the i'th index would be changing. 
This however avoids that entirely by only making the model predict within a fixed set of edge labels 
(the 0'th edge label is always 'before', etc.)

This list of (parent, child) tuples is generated in [data_preparation.py](data_preparation.py) by listing the tree 
nodes, i.e. time expressions or events, in the order occurring in the text, and then for each node, taking the window 
of 10 nodes before and 3 nodes after, as well as the root and DCT nodes, as possible parents. This list is then padded 
to a consistent length in case there are fewer than 15 available nodes. The numbers of 10 nodes before and 3 nodes 
after are empirically determined according to the expert-annotated data.

This padding is not present in the original DyNet model and paper; this change is necessary to adapt the model to 
TensorFlow and allow it to run quickly. It may also improve the model accuracy by giving it fewer choices as the 
document length increases (the DyNet model allows it to choose from every single preceding node in the document, not 
capped at 10), though due to the general variability in experiment results we can't show this for certain.

### Baseline

The baseline classifier is improved upon the baseline in [Zhang & Xue, 2018](https://arxiv.org/pdf/1809.00370.pdf) 
and [Zhang & Xue, 2019](https://www.aclweb.org/anthology/S19-1019). This is done by adding two more rules to the 
original single rule.

The rules are as follows:

1. Attach all time expressions to the root with label 'depends on' (the only valid label for time expressions)
2. For events, if there is a time expression in the same sentence, attach to that with label 'overlap' 
(the most common label for news)
3. Otherwise, attach to the previous event or time expression in the text with label 'overlap'

### Bi-LSTM (Original Paper Model)

This model re-creates the original architecture described in [Zhang & Xue, 2018](https://arxiv.org/pdf/1809.00370.pdf) 
and implemented using [DyNet](https://dynet.readthedocs.io/en/latest/) in 
[https://github.com/yuchenz/tdp_ranking](https://github.com/yuchenz/tdp_ranking).

This model, referred to as `bilstm` in honor of its main distinguishing component, effectively takes two inputs: 
one document (news article), as a list of tokenized sentences, and a list of (parent, child) tuples as discussed above. 
For this model (but not for all models), the tuples are grouped by child, which facilitates grouping the scores by 
child at the end.

#### Bi-LSTM and Attention

The tokenized sentences are converted into a list of words and passed through a word embedding layer, then through a 
Bi-LSTM. This yields a sequence of vector representations corresponding to every word in the document.

Separately, the (parent, child) tuples are passed in. At this point, the parent and child tuple is simply an array 
containing the indices of parent and child's words in the document as well as their position in the document. 
For each tuple, we select the representations of the words corresponding to the parent and child from the Bi-LSTM 
output sequence. Since each node may have a varying number of words, we first pad these to a fixed length and then pass 
them through a simple attention mechanism which returns a weighted average of the word vectors according to which 
vectors it deems most important. The paper suggests this will learn a notion of headedness for phrases such as 
`take a picture` or `last Thursday`. This gives a fixed-length representation of the words in a node 
(namely the length of one word embedding). 

#### Features

The next step is to combine all the features, including the output of this attention mechanism, to yield the features 
for this (candidate parent, child) tuple. The full feature vector is the concatenation of the following 
(as declared in [features.py](features.py) and [numeric_features.py](numeric_features.py)):

* the Bi-LSTM output for the first and last words in the parent
* the Bi-LSTM output for the first and last words in the child
* the attended words for the parent
* the attended words for the child
* node distance features
    - whether the parent is the previous node in the document
    - whether the parent comes before the child and is in the same sentence (but is not the previous node)
    - whether the parent comes before the child and is more than one sentence away
    - whether the parent comes after the child
* whether the parent and child are in the same sentence or not
* time expression / event label features:
    - whether the child is a time expression and the parent is the root
    - whether the child and parent are both time expressions
    - whether the child is an event and the parent is DCT
* whether the parent is a padding node, not a real node in the document

These features are similar to the rules used in the baseline classifier described above and comparing the two is useful.
Notice that the numeric features (node distance, same sentence, label features, padding features) are in fact 
calculated outside of the model and passed in in advance as inputs. This allows the model to receive only numeric 
inputs rather than needing to receive the parent and child as `Node` objects which contain all the information 
required to construct these features. 

#### Feed-Forward Network

Finally, the features for each (parent, child) tuple are fed into the feed-forward network, which is a simple 
fully-connected network with one hidden layer, which predicts a score for each of the five edge labels for each tuple. 

Then, the scores are grouped by child so that the model returns a softmaxed probability distribution for each possible 
parent + edge label combination.

### Bi-LSTM with Frozen BERT Embeddings

This model, referred to as `bert_bilstm`, is identical to the Bi-LSTM model, except that instead of using randomly 
initialized word embeddings, frozen BERT embeddings are used. Each sentence in the sentence list is passed one at a 
time to BERT, which produces word embeddings contextualized to that sentence. That is, each word in the document now 
has a unique embedding, dependent on its sentence, rather than multiple occurrences of the same word in the document 
having the same embedding.

Since we are using a pre-trained, black box version of BERT from TensorFlow Hub for ease of integration, we simply 
pick the top layer of BERT as our embeddings, rather than using a lower hidden state or multiple states 
(which people have shown 
([[1]](https://github.com/hanxiao/bert-as-service/blob/master/docs/section/faq.rst#why-not-the-last-hidden-layer-why-second-to-last), 
[[2]](http://jalammar.github.io/images/bert-feature-extraction-contextualized-embeddings.png)) 
may be a better word embedding).

### BERT as a Classifier

This architecture is inspired by the fact that BERT does not merely produce contextual word embeddings, but can in fact
be used as a classifier in its own right, and is known to do extremely well on a number of classification tasks, 
either alone or with a dense layer after it.

Thus we feed in the (parent, child) tuples directly as two "sentences" into BERT, take the embedding of the [CLS] token
that BERT produces for that pair, and pass that into a single-layer feed-forward network (no hidden layer) to generate 
a score for each edge label for that pair. As above these scores are then combined at the last minute to aggregate them 
by child and softmax across all parents and edge labels for that child, before this is fed into the loss function or 
returned as a prediction, but again the classifier isn't really aware of this except through the feedback it gets from 
the loss function.

The make-up of the two "sentences" that BERT takes vary between the two variants of this model, as described below.

#### Initial Features

The two "sentences" are as explained in the diagram below. Note that because these sentences are fairly short, we only 
need a max sequence length of 128 tokens. This reduces the training time and number of trainable parameters compared 
to longer sequences.

#### Alternative Features

This model uses an entire paragraph / span of sentences as the second "sentence". This requires a max sequence length 
of 512 tokens for this corpus, resulting in a longer training time and more difficulty fitting the higher number of 
trainable parameters. However, this more closely resembles the inputs used by Google for the SQUAD question-answering 
task.


### Training and Loss Functions

All of these models are trained using the Adam optimizer and a simple cross-entropy loss function which compares the 
scores for all parents and edge labels for a child with the one-hot vector indicating which parent/edge label is 
correct. That is, the predicted vector is the concatenation of the scores for each label, in order, for the first 
parent, then the scores for each label for the second parent, and so forth. Notably this means that we cannot calculate 
the loss function for a single (parent, child) tuple; we must wait until we have the values for all the parents for a 
given child. The batch size is optimized for each model to ensure that this is achieved (since the loss function is 
calculated once per batch). 

#### Batch sizes for the Bi-LSTM models

The Bi-LSTM models (with and without BERT) take in one document per batch, that is, the batch size is 1. Each document 
is in fact a dictionary of various inputs including the sequence of words for the Bi-LSTM (or, the sentence list if 
using BERT) and the list of child/parent candidates in that document, grouped by child. This list is deconstructed into 
tuples and then grouped again by child at the end. 

We use one document per batch because the number of words in a document as well as the number of child/parent candidates 
varies from document to document. TensorFlow requires the input length for all samples in a batch to be the same, but 
the input length may differ between batches. Since the input length is different for each sample (document), we only 
include one document per batch.

##### Batch size for BERT

When using frozen BERT embeddings, we want to take advantage of the fact that processing multiple sentences as a batch 
is significantly faster. And in fact, our "batch" of one document contains multiple sentences. So for BERT, we flatten 
the input so that for each document, BERT sees a batch size equal to the number of sentences in the document. 
Afterwards, we then reshape the input back to having the additional batch dimension with batch size 1. 
This work is done in `BatchedBertLayer` in [bert_layer.py](shared/bert_layer.py)

#### Batch sizes for the BERT as Classifier model

We still want to take advantage of the fact that BERT is faster if it processes multiple inputs at once, but now BERT 
is processing child/parent candidate tuples (disguised as sentence pairs). The trick here is that no part of the model 
(BERT or dense layer) needs to know that the tuples are related by having the same child. However, we need to be able 
to group the scores it produces by child to calculate the loss. Knowing that the loss function is calculated once per 
batch, *each batch contains all the tuples for exactly one child*. That is, the batch size is equal to the window size 
for choosing the parents described above. The custom loss function reassembles the vector shape expected by the loss 
function, namely having the length of all parents + edge labels, before passing it to the standard Keras cross-entropy 
loss. It does this simply by flattening the scores returned by the batch. (The accuracy is handled similarly.)

Note that in doing this we actually lose the boundary between documents, since we have batches by child and there is no 
way of telling where the children of one document end and the next document begins, but this is actually irrelevant as 
the useful parts of the document are extracted at feature creation time and the document boundary does not matter 
thereafter.

It does create some slightly confusing reporting from Keras since we lose the document count when reporting progress 
along batches and epochs, instead seeing the count of all children across all documents, but this is a minor problem 
and only relevant if a human is actually watching the progress bar (it's not printed in the logs on GPU anyway). 

## Data

This repository contains four named "datasets", which combine the small quantity of expert-annotated data annotated by 
the authors, as well as the larger body annotated by Turkers, in various ways.

In [Zhang & Xue, 2019](https://www.aclweb.org/anthology/S19-1019), the authors chose to annotate the 
[TimeBank corpus](http://www.timeml.org/timebank/timebank.html), which already has events and time expressions labeled,
with temporal dependency trees. The [TimeBank-Dense](https://www.usna.edu/Users/cs/nchamber/caevo/#corpus) subset of the 
corpus is expert-annotated by the authors while the remaining, larger part of the TimeBank corpus is annotated 
using Amazon Mechanical Turk. Naturally and especially since the data was only collected for a feasbility study, the 
Turker-annotated data is larger in size but substantially lower in quality. (As mentioned, a better Turker-annotated
dataset is currently in progress.)

A number of combinations of these two parts of the corpus are useful for training the models:

* `tb-dense` uses only the expert-annotated data for training, validation and testing. This corresponds to the 
Timebank-Dense portion of the corpus, and uses the splits as defined in the original 
[TimeBank-Dense repository](https://github.com/nchambers/caevo/blob/master/src/main/java/caevo/Evaluate.java).
* `blended` treats the expert-annotated data as gold and the Turker data as silver, and implements blending as described 
in [Shnarch et al, 2018](https://aclweb.org/anthology/P18-2095). It uses the dev and test sets from the gold 
(Timebank-Dense) data.
* `mixed` also treats the expert-annotated data as gold, but uses a combined silver dataset which contains the expert 
annotations where they are available and just the Turker annotations for the remaining documents. (The Turker
annotations for the overlapping Timebank-Dense documents are not used.) The model is trained first on the combined
dataset, then refined just on the gold, but there is no blending period as in `blended` since the gold data is contained
in the combined data.
* `turker` is used for experimental purposes only and trains and evaluates solely on the Turker-annotated data, using
the same documents as in the Timebank-Dense dev and test splits for validation and testing.

The exact declaration of which filenames are used for which datasets can be found in 
[run_exp.sh](run_exp.sh).

### Details

`timebank-dense.yuchen-tdt.all` contains 36 documents manually annotated (expert-annotated) by Yuchen Zhang. 
These 36 documents are the Timebank-Dense corpus (a subset of the Timebank corpus) and the events and time expression 
annotations (including the event subclasses) are taken directly from Timebank.
The train/dev/test split splits this into 22 + 5 + 9 documents, as detailed in the 
[Timebank-Dense paper](https://github.com/nchambers/caevo/blob/master/src/main/java/caevo/Evaluate.java) 
(this is the reference to the splits provided by the website linked to in the original Timebank-Dense paper.)
These splits are recreated in `timebank-dense.yuchen-tdt.train` (22 documents), `timebank-dense.yuchen-tdt.dev` 
(5 documents) and `timebank-dense.yuchen-tdt.test` (9 documents).

`timebank.turker-tdt.all` contains 183 documents, namely all the documents from Timebank (of which Timebank-Dense is 
a subset), is annotated by AMT. Of these 183 documents, 9 have invalid trees. These 9 are not in the Timebank-Dense 
subset of the corpus.
Of the 36 documents out of these 183 which are in the Timebank-Dense subset, the same 5 and 9 as above have been split 
out as dev and test datasets, still using the Turker annotation for these documents. These can be found at 
`timebank.turker-tdt.dev` (5 documents) and `timebank.turker-tdt.test` (9 documents). The same dev/test split as above
is used to allow for comparison between the datasets, however to achieve true comparison, models trained on the Turker
 data should be evaluated on the expert-annotated dev and test sets.
The remaining documents are in `timebank.turker-tdt.train`, except that the 9 documents with invalid trees have been 
removed from the training set. This brings its size to 160 documents. 

The [latest paper from Zhang & Xue](https://www.aclweb.org/anthology/S19-1019) trains on a mixed dataset which 
combines the training set of the expert-annotated documents with the training set of the Turker-annotated documents. 
This set of documents, `timebank.turker-yuchen-tdt.train`, represents a similar dataset, except that when combining 
`timebank-dense.yuchen-tdt.train` and `timebank.turker-tdt.train`, we use only the expert annotations for the 22 
Timebank-Dense documents, and do not include the Turker annotations, since they are worse. 
Thus it contains 160 documents.
This training set is designed to be used with the expert-annotated dev and test sets `timebank-dense.yuchen-tdt.dev` 
and `timebank-dense.yuchen-tdt.test`.

## Training, Parsing and Evaluating

A full train/parse/evaluate run of each neural model can be done using the script 
[run_exp.sh](run_exp.sh) by passing in the following parameters:

1. Experiment ID (used when creating the filename to save the model and parsed files), can be any number or string
2. Number of iterations
3. Output directory for the model and parsed files
4. Model name; one of `bilstm`, `bert_bilstm`, `bert_as_classifier` or `bert_as_classifier_alt`
5. Dataset name; one of `mixed`, `blended`, `turker` or `tb-dense`, any other value will default to `tb-dense`
6. Flag identifying whether to print results on the dev and/or test files, and whether to evaluate on the unlabeled 
task or not; options are `--full` (labeled and unlabeled evaluations on the dev and test set), `--labeled-only` 
(labeled evaluation on test set only), or `--test-only` (labeled and unlabeled evaluations on just the test set), any 
other value will default to `--test-only`
7. CPU or GPU (`--cpu` or `--gpu`), defaults to `--cpu`
8. The `TE_label_set`, i.e. whether the model should use the subclasses of the time expressions and events in the 
labelled data or just the binary distinction Timex vs. Event, and, if subclasses are used, a name for the set of 
subclasses; options are `timex_event`, `time_ml`, `none` or `full`, though note that for the datasets available `full` 
is invalid and will be treated as `time_ml`.

```bash
./run_exp.sh 46591_bilstm 50 $output_dir bert_as_classifier mixed --labeled-only --gpu timex_event
```

This will call [train.py](train.py) to train the model for 
n iterations on the training set of the named dataset, evaluating each epoch on the dev
set and using early stopping on the dev set, then save the model. Then it will call 
[parse.py](parse.py) and use the saved model to parse the test 
(and optionally dev) set of that dataset, and save the parsed file. Finally, it will call
[eval.py](eval.py) to read in that parsed file, evaluate the 
accuracy / f1 score (calculated as attachment score; these are all the same) for the labeled (and optionally unlabeled) 
task, and print the results, in a format such as:

```
test doc 0: true_p = 5, false_p = 8, false_n = 8
test doc 1: true_p = 21, false_p = 11, false_n = 11
test doc 2: true_p = 8, false_p = 9, false_n = 9
test doc 3: true_p = 9, false_p = 1, false_n = 1
test doc 4: true_p = 5, false_p = 8, false_n = 8
test doc 5: true_p = 20, false_p = 12, false_n = 12
test doc 6: true_p = 10, false_p = 6, false_n = 6
test doc 7: true_p = 7, false_p = 8, false_n = 8
test doc 8: true_p = 3, false_p = 3, false_n = 3
macro average: f = 0.557; micro average: f = 0.571
TIMEX correct: 1.000
Children of DCT correct: 0.517
Children of other TIMEX correct: 0.686
```

In particular, the script takes care of handling all the identical parameters that need to be passed between `train.py`,
`parse.py` and `eval.py`, such as the location of the parsed file and parameters such as `TE_label_set`. It also
contains configuration for the various datasets, models and hyperparameters associated with the different combinations.
Finally, it handles environment details such as which Anaconda environment should be used and setting the log level for 
TensorFlow.

The model is saved in the specified output directory and can be loaded again for other tasks. Note that the `model_file`
parameter passed to the training script does not actually constitute a file, but is in fact the prefix for a number of 
files (two or three depending on the model); however passing this prefix to any of the train/parse/eval scripts will
be interpreted correctly by the model.

### Running the Baseline

The rule-based baseline model can be run with the much simpler script 
[baseline.sh](baseline.sh) as follows:

```bash
./baseline.sh overlap ./models
```

where the following parameters are passed:

* Default label to use for all events (`before`, `after`, `overlap`, `includes` or `Depend-on` - note that `Depend-on`, 
while a valid parameter for the script, is never a valid label choice for events)
* Output directory for the parsed file

This will use [parse.py](parse.py) and 
[eval.py](eval.py)
to print both the labeled and unlabeled evaluations on the TimeBank-Dense expert-annotated test set, which is the
test set used for all other models and datasets except `turker`.

The results for the labeled task are:

```
macro average: f = 0.552; micro average: f = 0.526
TIMEX correct: 1.000
Children of DCT correct: 0.103
Children of other TIMEX correct: 0.800
Children of events correct: 0.250

```

## Changing Hyperparameters

Hyperparameters which are common across all models and datasets are declared in 
[train.py](train.py) in the default arguments for the `arg_parser`.

These may be overridden on a per-model/per-dataset basis in 
[run_exp.sh](run_exp.sh).

If you wish to make changes to a hyperparameter, it is advisable to add a line in `run_exp.sh` unless you are confident 
of the effect it will have on all the different model/dataset combinations. 
(There is definitely some scope for refactoring this process to make it easier to see what the actual parameter is for 
each model/dataset, but it's non-trivial to ensure that the right defaults are applied in the right order and can still 
be overridden by the shell script calling `train.py` in case of hyperparameter tuning.

Currently `run_exp.sh` does not allow the hyperparameters for `train.py` to be passed to it from the command line
apart from the iterations and the `TE_label_set`, due to the number available
being quite high and cluttering the script when it most cases we are not interested in changing them. Formal 
hyperparameter tuning has not yet been conducted on this repository since other there have been other priorities and the
dataset is arguably too small at this point for it to be worthwhile. To do e.g. a grid search, it is recommended to
write a new script adapting `run_exp.sh` which holds parameters such as the dataset and output constant while instead
allowing the experiment script to vary the other hyperparameters which are passed to `train.py`.
