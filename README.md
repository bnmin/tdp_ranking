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

## Models

This repository contains four models: `bilstm`, the original architecture described in the paper,
`bert_bilstm` (`bilstm` but using frozen BERT embeddings), `bert_as_classifier` and `bert_as_classifier_alt` 
(which uses the same architecture as `bert_as_classifier` but different features).

Explanations and diagrams of the model architectures can be found on the 
[Models wiki page](http://e-gitlab.bbn.com/hross/tdp_ranking/wikis/models).

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

A more detailed overview of the datasets available in this repository can be found on the 
[Datasets wiki page](http://e-gitlab.bbn.com/hross/tdp_ranking/wikis/datasets). In particular, it is worth noting
that the Turker-annotated data contains a small number of invalid trees, which are not included in the training sets.

The exact declaration of which filenames are used for which datasets can be found in 
[run_exp.sh](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/run_exp.sh).

## Training, Parsing and Evaluating

A full train/parse/evaluate run of each neural model can be done using the script 
[run_exp.sh](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/run_exp.sh) by passing in the following parameters:

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

This will call [train.py](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/train.py) to train the model for 
n iterations on the training set of the named dataset, evaluating each epoch on the dev
set and using early stopping on the dev set, then save the model. Then it will call 
[parse.py](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/parse.py) and use the saved model to parse the test 
(and optionally dev) set of that dataset, and save the parsed file. Finally, it will call
[eval.py](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/eval.py) to read in that parsed file, evaluate the 
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
[baseline.sh](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/baseline.sh) as follows:

```bash
./baseline.sh overlap ./models
```

where the following parameters are passed:

* Default label to use for all events (`before`, `after`, `overlap`, `includes` or `Depend-on` - note that `Depend-on`, 
while a valid parameter for the script, is never a valid label choice for events)
* Output directory for the parsed file

This will use [parse.py](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/parse.py) and 
[eval.py](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/eval.py)
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
[train.py](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/train.py) in the default arguments for the `arg_parser`.

These may be overridden on a per-model/per-dataset basis in 
[run_exp.sh](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/run_exp.sh).

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

## Expected Results

A table of expected results for each model and dataset combination, using the hyperparameters in the code, can be found
on the [Results Summary wiki page](http://e-gitlab.bbn.com/hross/tdp_ranking/wikis/results-summary).

## Parsing SERIF XML

It is possible to use a previously trained model to parse SERIF XML into a temporal dependency tree.
A special shell script, [parse_serif.sh](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/parse_serif.sh), 
exists for this task since different preprocessing is needed. 

It takes the following arguments. In short, start by selecting a trained model, find its save file, and then pass the 
same parameters (model name, CPU/GPU) that were used when training the model. The model name cannot be
drawn from the save file as reading the save file depends on knowing which model it was created with.

1. Path to the SERIF XML file
2. Path to the trained model - note that as discussed above, this is not the path to an actual file, but rather the 
path/prefix for 2-3 files. The filename can be found by either looking at the experiment output directory and/or
observing how it is constructed in [run_exp.sh](http://e-gitlab.bbn.com/hross/tdp_ranking/blob/master/run_exp.sh) 
(parameter `model_file`)
3. The name of the trained model: `bilstm`, `bert_bilstm`, `bert_as_classifier` or `bert_as_classifier_alt`. Clearly, 
this **must match the actual model located at path above**.
4. Output directory for the parsed file and the parse tree image
5. CPU or GPU (`--cpu` or `--gpu`), defaults to `--cpu`. Again this **must match the environment the model was trained 
on** if you are using the Bi-LSTM or Bi-LSTM with BERT models. It's also highly recommended to run this on GPU for the 
BERT-as-Classifier models because even a forward pass is quite slow on CPU (expect up to 30min for a single document).

```bash
$model_name="timebank-dense.yuchen-tdt.train.bert_as_classifier-model.46572_bert_classifier-0"
./parse_serif.sh <path_to_serif>/AFP_ENG_19940512.0069.xml <path_to_model>/$model_name bert_as_classifier timex_event ./models --gpu
```

Note that parsing SERIF XML depends on the LearnIt repository: in `parse_serif.sh`, a checked out version of this
is added to the `PYTHONPATH`, but if you wish to use the newest version of LearnIt, you will need to
change that directory to your own, newer copy. Issue [#27](http://e-gitlab.bbn.com/text-group/learnit/issues/27) in the 
LearnIt repository is in progress to allow a PIP install of LearnIt instead of needing to have it present on the 
`PYTHONPATH`.