#!/usr/bin/env bash

##############################################################################################################################################################
# This script is meant to run COMEBin binning after obtaining the embeddings.
# Author of pipeline: Ziye Wang.
# For questions, bugs, and suggestions, contact me at zwang17@fudan.edu.cn
##############################################################################################################################################################
VERSION="1.0.4"

help_message () {
  echo ""
  echo "COMEBin version: $VERSION"
  echo "Usage: bash run_comebin_bin.sh [options] -a contig_file -o output_dir -t num_threads"
  echo "Options:"
  echo ""
  echo "  -a STR          metagenomic assembly file"
  echo "  -o STR          output directory"
  echo "  -t INT          number of threads (default=5)"
  echo "";}

run_file_path=$(dirname $(which run_comebin_bin.sh))

if [[ $? -ne 0 ]]; then
  echo "cannot find run_comebin_bin.sh file - something went wrong with the installation!"
  exit 1
fi

num_threads=5

while getopts a:o:t: OPT; do
 case ${OPT} in
  a) contig_file=$(realpath ${OPTARG})
    ;;
  o) output_dir=$(realpath ${OPTARG})
    ;;
  t) num_threads=${OPTARG}
    ;;
  \?)
    exit 1
 esac
done

cd ${run_file_path}/COMEBin

if [ -z "${contig_file}" -o -z "${output_dir}" ]; then
  help_message
  exit 1
fi

emb_file=${output_dir}/comebin_res/embeddings.tsv
seed_file=${contig_file}.bacar_marker.2quarter_lencutoff_1001.seed

python main.py bin --contig_file ${contig_file} \
--emb_file ${emb_file} \
--output_path ${output_dir}/comebin_res \
--seed_file ${seed_file} --num_threads ${num_threads}

exit_code=$?

if [ $exit_code -eq 66 ]; then
    echo "Comebin couldn't find any bins. Exiting normally."
    exit 0
elif [ $exit_code -ne 0 ]; then
    echo "Something went wrong with running clustering. Exiting."
    exit 1
fi

python main.py get_result --contig_file ${contig_file} \
--output_path ${output_dir}/comebin_res \
--seed_file ${seed_file} --num_threads ${num_threads}

if [[ $? -ne 0 ]] ; then echo "Something went wrong with getting the results. Exiting."; exit 1; fi
