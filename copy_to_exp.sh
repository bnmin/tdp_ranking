#!/usr/bin/env bash

# Run this from within the /bin directory of the experiment

# Copy scripts and Python files
cp /home/hross/raid88/projects/tdp_ranking/*.py .
cp /home/hross/raid88/projects/tdp_ranking/*.sh .
cp -r  /home/hross/raid88/projects/tdp_ranking/base_bilstm/ .
cp -r  /home/hross/raid88/projects/tdp_ranking/bert_as_classifier/ .
cp -r  /home/hross/raid88/projects/tdp_ranking/frozen_bert/ .
cp -r  /home/hross/raid88/projects/tdp_ranking/shared/ .

# Copy data
cp -r  /home/hross/raid88/projects/tdp_ranking/data/ .