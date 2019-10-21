# Get directory of this script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python=/nfs/raid88/u10/users/hross/software/anaconda3/envs/tdp_ranking/bin/python

test_file_stem=timebank-dense.yuchen-tdt.test
default_label=$1
model_dir=$2

data_dir=${script_dir}/data
test_file=${data_dir}/${test_file_stem}

if [[ -f ${model_dir}/${test_file_stem}.baseline-parsed-ul ]];
then
    mv ${model_dir}/${test_file_stem}.baseline-parsed-ul ~/.recycle
fi

echo unlabeled ...
${python} ${script_dir}/parse.py --test_file $test_file --classifier baseline --parsed_file ${model_dir}/${test_file_stem}.baseline-parsed-ul --default_label $default_label --TE_label_set time_ml
${python} ${script_dir}/eval.py --gold_file $test_file --parsed_file ${model_dir}/${test_file_stem}.baseline-parsed-ul

if [[ -f ${test_file}.baseline-parsed-l ]];
then
    mv ${test_file}.baseline-parsed-l ~/.recycle
fi

echo labeled ...
${python} ${script_dir}/parse.py --test_file $test_file --classifier baseline --parsed_file ${model_dir}/${test_file_stem}.baseline-parsed-l --default_label $default_label --TE_label_set time_ml --labeled
${python} ${script_dir}/eval.py --gold_file $test_file --parsed_file ${model_dir}/${test_file_stem}.baseline-parsed-l --labeled
