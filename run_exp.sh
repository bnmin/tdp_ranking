##########################
##  parameter settings  ##
##########################

# Get directory of this script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

exp_id=$1
iter=$2
model_dir=$3
classifier=$4
dataset=$5
eval_type=$6
env=$7
TE_label_set=$8
no_handcrafted=$9

# Set up data
blending_init=""
blending_epochs=""
blend_factor=""
early_stopping_warmup=""
early_stopping_threshold=""

if [[ ${dataset} == "turker" ]];
then
    train_file_stem=timebank.turker-tdt.train
    dev_file_stem=timebank.turker-tdt.dev
    test_file_stem=timebank.turker-tdt.test
elif [[ ${dataset} == "blended" ]];
then
    train_file_stem=timebank-dense.yuchen-tdt.train
    silver_train_file_stem=timebank.turker-tdt.train
    dev_file_stem=timebank-dense.yuchen-tdt.dev
    test_file_stem=timebank-dense.yuchen-tdt.test

    blend_factor="--blend_factor 0.75"
elif [[ ${dataset} == "mixed" ]];
then
    train_file_stem=timebank-dense.yuchen-tdt.train
    silver_train_file_stem=timebank.turker-yuchen-tdt.train
    dev_file_stem=timebank-dense.yuchen-tdt.dev
    test_file_stem=timebank-dense.yuchen-tdt.test

    if [[ ${classifier} == "bert_as_classifier" || ${classifier} == "bert_as_classifier_alt" ]];
    then
        early_stopping_warmup="--early_stopping_warmup 26"
        early_stopping_threshold="--early_stopping_threshold 3"
    fi
    blending_epochs="--blending_epochs 0"
    blend_factor="--blend_factor 0"
else
    train_file_stem=timebank-dense.yuchen-tdt.train
    dev_file_stem=timebank-dense.yuchen-tdt.dev
    test_file_stem=timebank-dense.yuchen-tdt.test

    if [[ ${classifier} == "bert_bilstm" ]];
    then
        early_stopping_warmup="--early_stopping_warmup 50"
    elif [[ ${classifier} == "bert_as_classifier" || ${classifier} == "bert_as_classifier_alt" ]];
    then
        early_stopping_warmup="--early_stopping_warmup 10"
        early_stopping_threshold="--early_stopping_threshold 3"
    fi
fi

data_dir=${script_dir}/data
train_file=${data_dir}/${train_file_stem}
dev_file=${data_dir}/${dev_file_stem}
test_file=${data_dir}/${test_file_stem}

if [[ ${silver_train_file_stem} ]];
then
    silver_train_file=${data_dir}/${silver_train_file_stem}
    silver_train_file_param="--silver_train_file ${silver_train_file}"
else
    silver_train_file=""
    silver_train_file_param=""
fi

# Set up Python environment
if [[ ${env} == "--gpu" ]];
then
    python=python # PUT YOUR ABSOLUTE PATH TO YOUR CONDA ENV WITH TENSORFLOW-GPU HERE
else
    python=python # PUT YOUR ABSOLUTE PATH TO YOUR CONDA ENV WITH TENSORFLOW (CPU) HERE
fi

# Set up parameters
if [[ ${TE_label_set} != "timex_event" && ${TE_label_set} != "none" ]]
then
    # For the datasets above, these three are the only valid TE_label_sets
    TE_label_set="time_ml"
fi

if [[ ${classifier} == "bert_as_classifier" || ${classifier} == "bert_as_classifier_alt" ]];
then
  learning_rate="--lr 0.0001"
else
  # This is Adam's default learning rate
  learning_rate="--lr 0.001"
fi

if [[ ${classifier} == "bert_as_classifier_alt" ]];
then
    max_seq_length="--max_sequence_length 512"
else
    max_seq_length="--max_sequence_length 128"
fi

if [[ ${no_handcrafted} == "--no_handcrafted_features" ]];
then
  handcrafted="--no_handcrafted_features"
else
  handcrafted=""
fi

echo exp_id ${exp_id}
echo iter ${iter}
echo train_file ${train_file}
echo silver_train_file ${silver_train_file}
echo dev_file ${dev_file}
echo test_file ${test_file}
echo labeled
echo TE_label_set ${TE_label_set}
echo handcrafted_features ${handcrafted}

model_file=${model_dir}/${train_file_stem}.${classifier}-model.${exp_id}

##################################
#  silence TensorFlow info logs  #
##################################

export KMP_WARNINGS=0

if [[ ${env} == "--gpu" ]];
then
    # Show only errors
    export TF_CPP_MIN_LOG_LEVEL=2
else
    # Show warnings and errors, but no info
    export TF_CPP_MIN_LOG_LEVEL=1
fi

################
##  training  ##
################

echo training ...
${python} -u ${script_dir}/train.py --train_file ${train_file} ${silver_train_file_param} --dev_file ${dev_file} --model_file ${model_file} --TE_label_set ${TE_label_set} --edge_label_set time_ml --classifier ${classifier} --labeled --iter ${iter} ${blending_init} ${blending_epochs} ${blend_factor} ${early_stopping_warmup} ${early_stopping_threshold} ${max_seq_length} ${learning_rate} ${handcrafted}

######################################
##  parse and evaluate on dev data  ##
######################################

if [[ ${eval_type} == "--full" ]];
then
    echo parsing dev data ...
    if [[ -f ${model_dir}/${dev_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} ]];
    then
        mv ${model_dir}/${dev_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} ~/.recycle
    fi

    ${python} ${script_dir}/parse.py --test_file ${dev_file} --model_file ${model_file} --parsed_file ${model_dir}/${dev_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} --TE_label_set ${TE_label_set} --classifier ${classifier} --labeled

    echo eval dev data ...
    ${python} ${script_dir}/eval.py --gold_file ${dev_file} --parsed_file ${model_dir}/${dev_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} --labeled
fi

#######################################
##  parse and evaluate on test data  ##
#######################################

if [[ ${eval_type} != "--no-eval" ]];
then
  echo parsing test data ...
  if [[ -f ${model_dir}/${test_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} ]];
  then
      mv ${model_dir}/${test_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} ~/.recycle
  fi

  ${python} ${script_dir}/parse.py --test_file ${test_file} --model_file ${model_file} --parsed_file ${model_dir}/${test_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} --TE_label_set ${TE_label_set} --classifier ${classifier} --labeled

  echo eval test data ...
  ${python} ${script_dir}/eval.py --gold_file ${test_file} --parsed_file ${model_dir}/${test_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id} --labeled
fi

#############################################
##  for models trained with labeled data,  ##
##    do unlabeled evaluations too         ##
#############################################

if [[ ${eval_type} != "--labeled-only" && ${eval_type} != "--no-eval" ]];
then
    if [[ ${eval_type} == "--full" ]];
    then
        echo unlabeled evaluations on dev data ...
        ${python} ${script_dir}/eval.py --gold_file ${dev_file} --parsed_file ${model_dir}/${dev_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id}
    fi

    echo unlabeled evaluations on test data ...
    ${python} ${script_dir}/eval.py --gold_file ${test_file} --parsed_file ${model_dir}/${test_file_stem}.stage2-${classifier}-parsed-labeled.${exp_id}
fi