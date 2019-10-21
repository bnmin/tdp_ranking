##########################
##  parameter settings  ## 
##########################

# Get directory of this script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

serif_dir=$1
serif_filename=$2
model_file=$3
classifier=$4
output_dir=$5
env=$6

# Set up Python environment
if [[ ${env} == "--gpu" ]];
  then
      python=/nfs/raid88/u10/users/hross/software/anaconda3/envs/tdp_ranking_gpu_13/bin/python
  else
      python=/nfs/raid88/u10/users/hross/software/anaconda3/envs/tdp_ranking_13/bin/python
  fi

export PYTHONPATH=$PYTHONPATH:/nfs/raid88/u10/users/hross/projects/learnit

serif_file=${serif_dir}/${serif_filename}
parsed_file=${output_dir}/${serif_filename}.stage2-${classifier}-parsed-labeled

echo serif_file ${serif_file}
echo model_file ${model_file}


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

#######################
##  parse test data  ##
#######################

echo parsing data ...
${python} ${script_dir}/parse.py --test_file ${serif_file} --from_serif --model_file ${model_file} --classifier ${classifier} --parsed_file ${parsed_file} --TE_label_set timex_event --labeled

################
## plot trees ##
################

${python} ${script_dir}/plot_trees.py --data_file ${parsed_file} --output_dir ${output_dir}