#!/bin/bash
#SBATCH --job-name=test_lee
#SBATCH --error=../output_error/lee.err
#SBATCH --output=../output_error/lee.out
#SBATCH --time=24:00:00
#SBATCH --mem=100g
#SBATCH --cpus-per-task=8

SC_HIC=./sample_data/Preprocessed_Sample_Data/Lee_200kb_scHiC.h5ad # add scHi-C path
SECOND=./sample_data/Preprocessed_Sample_Data/Lee_dna_methylation.h5ad # add second modality path 

OUTDIR=../results/Lee_200kb

mkdir -p "$OUTDIR"

# Using the flexible N-modality argument list:
python ../spinmcia/spinmcia.py \
  --input_files "$SC_HIC" "$SECOND" \
  --modality_names "HiC" "DNA_Methylation" \
  --output_folder "$OUTDIR" \
  --col_preproc_method colprofile \
  --block_preproc_method unit_var \
  --num_PCs 70 \
  --tol 1e-9 \
  --max_iter 5000 \
  --deflationMethod block
  ## --thresholds 0.9 0.9 \ ## use only if the data needs to drop columns (pass one cutoff per modality)