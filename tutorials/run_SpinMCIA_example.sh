#!/bin/bash
#SBATCH --job-name=test_lee
#SBATCH --error=../output_error/lee.err
#SBATCH --output=../output_error/lee.out
#SBATCH --time=24:00:00
#SBATCH --mem=100g
#SBATCH --cpus-per-task=8

SC_HIC=../sample_data/Preprocessed_Data/Lee/200kb/lee_sc_hic_200kb_resolution.h5ad # add scHi-C path
SECOND=../sample_data/Preprocessed_Data/Lee/200kb/lee_dna_methylation.h5ad # add second modality path 

OUTDIR=../results/Lee_200kb

mkdir -p "$OUTDIR"

python ../spinmcia/SpinMCIA.py \
  --sc_hic_h5ad_input_file "$SC_HIC" \
  --second_modality_h5ad_input_file "$SECOND" \
  --output_folder "$OUTDIR" \
  --col_preproc_method colprofile \
  --block_preproc_method unit_var \
  --num_PCs 70 \
  --tol 1e-9 \
  --max_iter 5000 \
  --deflationMethod block \
  --mode dense \
  --assume_same_order \
  ## --threshold 0.9 \ ## use only if the data needs to drop columns   

