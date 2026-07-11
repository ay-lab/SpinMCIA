#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import gc
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import anndata
import scanpy as sc
import scipy.sparse as sp
from numpy.linalg import norm
from scipy.sparse.linalg import svds

# -------------------- Utilities -------------------- #

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# -------------------- Data helpers -------------------- #

def adata_to_dense_df(adata: anndata.AnnData, name: str) -> pd.DataFrame:
    """Convert AnnData.X to a dense DataFrame with obs_names and var_names."""
    if sp.issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = adata.X
    df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    df.name = name
    # Ensure columns unique (defensive)
    if not df.columns.is_unique:
        cols = pd.Series(df.columns, dtype=str)
        seen: Dict[str, int] = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}-{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = pd.Index(new_cols)
    return df

# -------------------- Preprocessing -------------------- #

def col_preproc(df: pd.DataFrame, col_preproc_method: str) -> pd.DataFrame:
    temp_df = df.copy()
    method = col_preproc_method.lower()

    if method == "colprofile":
        # Non-negative
        min_val = temp_df.values.min()
        if min_val < 0:
            offset = np.floor(min_val)
            temp_df = temp_df + abs(offset)

        totsum = temp_df.values.sum()
        colsums = temp_df.sum(axis=0)
        row_contribs = temp_df.sum(axis=1) / (totsum if totsum != 0 else 1)

        nz_cols = colsums[colsums != 0].index
        temp_df.loc[:, nz_cols] = temp_df.loc[:, nz_cols].div(colsums[nz_cols], axis=1)

        temp_df = temp_df.sub(row_contribs, axis=0)

        denom = np.sqrt(colsums / (totsum if totsum != 0 else 1))
        denom.replace([np.inf, -np.inf], 0, inplace=True)
        denom = denom.fillna(0)
        temp_df = temp_df.mul(denom, axis=1)

    elif method == "standardized":
        temp_df = (temp_df - temp_df.mean()) / temp_df.std(ddof=0)
        temp_df = temp_df.replace([np.inf, -np.inf], 0).fillna(0)

    elif method == "centered_only":
        temp_df = temp_df - temp_df.mean()

    else:
        raise ValueError("Column preprocessing method not recognized.")
    return temp_df

def block_preproc(df: pd.DataFrame, block_preproc_method: str) -> pd.DataFrame:
    temp_df = df.copy()
    method = block_preproc_method.lower()

    if method == "unit_var":
        n = max(1, temp_df.shape[0] - 1)
        block_var = (np.linalg.norm(temp_df.values, ord='fro') ** 2) / n
        if block_var != 0:
            temp_df = temp_df * (1.0 / np.sqrt(block_var))
        else:
            warnings.warn("Data block has zero variance.")
        return temp_df

    elif method == "num_cols":
        temp_df = temp_df * (1.0 / max(1, temp_df.shape[1]))
        return temp_df

    elif method == "largest_sv":
        u, s, vt = svds(temp_df.values, k=1)
        eigval = s[0]
        if eigval != 0:
            temp_df = temp_df * (1.0 / eigval)
        else:
            warnings.warn("Data block has zero largest eigenvalue.")
        return temp_df

    elif method == "none":
        return temp_df

    else:
        raise ValueError("Block preprocessing method not recognized.")

def get_tv(ds: List[pd.DataFrame]) -> List[float]:
    def get_bv(X: pd.DataFrame) -> float:
        X_cent = X - X.mean()
        n_samp = X_cent.shape[0]
        return (np.linalg.norm(X_cent.values, ord='fro') ** 2) / max(n_samp - 1, 1)
    return [get_bv(df) for df in ds]

# -------------------- MCIA (NIPALS) -------------------- #

np.random.seed(42)

def nipals_iter(ds: List[pd.DataFrame], tol=1e-12, maxIter=1000) -> Dict[str, Any]:
    stop_crit = 2 * tol
    cov_squared_old = 0.0
    iteration = 0
    n_samples = ds[0].shape[0]

    gs = np.ones((n_samples, 1))

    while stop_crit > tol and iteration <= maxIter:
        bl_list = []
        bs_list = []

        for df in ds:
            bl_k = df.T @ gs
            bl_k_arr = bl_k.values.flatten()
            bl_norm = np.linalg.norm(bl_k_arr, ord=2)
            if bl_norm == 0 or np.isnan(bl_norm):
                warnings.warn("Zero/NaN norm in block loading. Stopping.")
                return {'global_scores': None, 'global_loadings': None, 'block_score_weights': None,
                        'block_scores': [None]*len(ds), 'block_loadings': [None]*len(ds), 'eigval': np.nan}
            bl_k_arr /= bl_norm
            bl_k = pd.DataFrame(bl_k_arr, index=df.columns)
            bl_list.append(bl_k)

        for df, bl_k in zip(ds, bl_list):
            bs_k = (df @ bl_k).values.flatten()
            bs_list.append(bs_k)

        bs_matrix = np.column_stack(bs_list)

        gw = bs_matrix.T @ gs
        gw_norm = np.linalg.norm(gw, ord=2)
        if gw_norm == 0 or np.isnan(gw_norm):
            warnings.warn("Zero/NaN norm in global weights. Stopping.")
            return {'global_scores': None, 'global_loadings': None, 'block_score_weights': None,
                    'block_scores': [None]*len(ds), 'block_loadings': [None]*len(ds), 'eigval': np.nan}
        gw /= gw_norm

        gs = bs_matrix @ gw

        gs_var = np.var(gs, ddof=1)
        if gs_var == 0 or np.isnan(gs_var):
            warnings.warn("Global score variance is zero/NaN. Early stop.")
            return {'global_scores': None, 'global_loadings': None, 'block_score_weights': None,
                    'block_scores': [None]*len(ds), 'block_loadings': [None]*len(ds), 'eigval': np.nan}
        gs_norm = gs.flatten() / np.sqrt(gs_var)

        cov_list = []
        for bs in bs_list:
            cov_bs_gs = np.cov(bs, gs_norm, ddof=1)[0, 1]
            if np.isnan(cov_bs_gs):
                warnings.warn("NaN covariance. Stopping.")
                return {'global_scores': None, 'global_loadings': None, 'block_score_weights': None,
                        'block_scores': [None]*len(ds), 'block_loadings': [None]*len(ds), 'eigval': np.nan}
            cov_list.append(cov_bs_gs**2)

        stop_crit = abs(sum(cov_list) - cov_squared_old)
        cov_squared_old = sum(cov_list)
        print_flush(f"[nipals_iter] Iteration {iteration}: stop_crit = {stop_crit}")
        iteration += 1

    if iteration > maxIter:
        warnings.warn("NIPALS iteration did not converge")
        return {'global_scores': None, 'global_loadings': None, 'block_score_weights': None,
                'block_scores': [None]*len(ds), 'block_loadings': [None]*len(ds), 'eigval': np.nan}

    eigval = float(np.var(gs, ddof=1))
    print_flush(f"[nipals_iter] Converged after {iteration} iterations with eigenvalue: {eigval}")

    gl = pd.Series(dtype=float)
    for bl_k, gw_k in zip(bl_list, gw):
        tmp = bl_k.iloc[:, 0] * gw_k[0]
        tmp = pd.Series(tmp.values, index=bl_k.index)
        gl = pd.concat([gl, tmp])

    gs_df = pd.DataFrame(gs, index=ds[0].index)

    block_scores = [pd.DataFrame(bs, index=ds[0].index) for bs in np.column_stack(bs_list).T]
    block_loadings = bl_list

    return {
        'global_scores': gs_df,
        'global_loadings': gl,
        'block_score_weights': gw,
        'block_scores': block_scores,
        'block_loadings': block_loadings,
        'eigval': eigval
    }

def deflate_block_bl(df: pd.DataFrame, bl: pd.DataFrame) -> pd.DataFrame:
    blv = bl.values.flatten()
    df_values = df.values - (df.values @ blv[:, None]) @ blv[None, :]
    return pd.DataFrame(df_values, index=df.index, columns=df.columns)

def deflate_block_gs(df: pd.DataFrame, gs: pd.DataFrame) -> pd.DataFrame:
    gsv = gs.values
    normed_gs = gsv / np.linalg.norm(gsv, ord=2)
    df_values = df.values - (normed_gs @ normed_gs.T) @ df.values
    return pd.DataFrame(df_values, index=df.index, columns=df.columns)

class NipalsResult:
    def __init__(self, global_scores, global_loadings, block_score_weights,
                 block_scores, block_loadings, eigvals, col_preproc_method,
                 block_preproc_method, block_variances, metadata):
        self.global_scores = global_scores
        self.global_loadings = global_loadings
        self.block_score_weights = block_score_weights
        self.block_scores = block_scores
        self.block_loadings = block_loadings
        self.eigvals = eigvals
        self.col_preproc_method = col_preproc_method
        self.block_preproc_method = block_preproc_method
        self.block_variances = block_variances
        self.metadata = metadata

def nipals_multiblock(data_blocks: List[pd.DataFrame],
                      col_preproc_method="colprofile",
                      block_preproc_method="unit_var",
                      num_PCs=10, tol=1e-9, max_iter=1000,
                      deflationMethod="block",
                      metadata: pd.DataFrame = None) -> NipalsResult:

    num_blocks = len(data_blocks)
    omics_names = [getattr(df, 'name', None) for df in data_blocks]
    if any(name is None for name in omics_names):
        omics_names = [f"omic{i+1}" for i in range(num_blocks)]
        for i, df in enumerate(data_blocks):
            df.name = omics_names[i]
    else:
        for i, df in enumerate(data_blocks):
            df.name = omics_names[i]

    for i, df in enumerate(data_blocks):
        if (df.columns is None) or (pd.isna(df.columns).any()):
            df.columns = [f"feature_{j+1}" for j in range(df.shape[1])]

    print_flush("Performing column-level pre-processing...")
    data_blocks = [col_preproc(df, col_preproc_method) for df in data_blocks]
    print_flush("Column pre-processing completed.")

    print_flush("Performing block-level pre-processing...")
    data_blocks = [block_preproc(df, block_preproc_method) for df in data_blocks]
    if block_preproc_method.lower() == "unit_var":
        block_vars = {name: 1 for name in omics_names}
    else:
        block_vars_list = get_tv(data_blocks)
        block_vars = dict(zip(omics_names, block_vars_list))
    print_flush("Block pre-processing completed.")
    gc.collect()

    print_flush("Computing order 1 scores...")
    nipals_result = nipals_iter(data_blocks, tol=tol, maxIter=max_iter)
    if nipals_result['global_scores'] is None:
        print_flush("NIPALS failed on the first component. Stopping.")
        return None

    global_scores = nipals_result['global_scores']
    global_loadings = nipals_result['global_loadings']
    block_score_weights = nipals_result['block_score_weights']

    block_scores = {}
    block_loadings = {}
    for i in range(num_blocks):
        bs = nipals_result['block_scores'][i]
        bl = nipals_result['block_loadings'][i]
        block_scores[omics_names[i]] = bs
        block_loadings[omics_names[i]] = bl

    eigvals = [nipals_result['eigval']]

    if num_PCs > 1:
        for pc in range(2, num_PCs + 1):
            print_flush(f"Computing order {pc} scores")
            if deflationMethod.lower() == "block":
                data_blocks = [deflate_block_bl(df, bl) for df, bl in zip(data_blocks, nipals_result['block_loadings'])]
            elif deflationMethod.lower() == "global":
                data_blocks = [deflate_block_gs(df, gs=nipals_result['global_scores']) for df in data_blocks]
            else:
                raise ValueError("Unknown deflationMethod (use 'block' or 'global').")

            nipals_result = nipals_iter(data_blocks, tol=tol, maxIter=max_iter)
            if nipals_result['global_scores'] is None:
                print_flush(f"NIPALS did not converge at PC {pc}. Stopping deflation loop.")
                break

            global_scores = pd.concat([global_scores, nipals_result['global_scores']], axis=1)
            global_loadings = pd.concat([global_loadings, nipals_result['global_loadings']], axis=1)
            block_score_weights = np.hstack((block_score_weights, nipals_result['block_score_weights']))
            eigvals.append(nipals_result['eigval'])

            for j in range(num_blocks):
                bs = nipals_result['block_scores'][j]
                bl = nipals_result['block_loadings'][j]
                block_scores[omics_names[j]] = pd.concat([block_scores[omics_names[j]], bs], axis=1)
                block_loadings[omics_names[j]] = pd.concat([block_loadings[omics_names[j]], bl], axis=1)

            gc.collect()

    pc_names = [f"PC{i+1}" for i in range(global_scores.shape[1])]
    global_scores.columns = pc_names

    if isinstance(nipals_result['global_loadings'], pd.Series):
        gl_df = pd.DataFrame(nipals_result['global_loadings'].values,
                             index=nipals_result['global_loadings'].index, columns=["PC1"])
    else:
        gl_df = nipals_result['global_loadings']

    for name in block_scores:
        if block_scores[name].shape[1] < len(pc_names):
            missing = len(pc_names) - block_scores[name].shape[1]
            fill = pd.DataFrame(np.nan, index=block_scores[name].index,
                                columns=[f"PC{block_scores[name].shape[1]+i+1}" for i in range(missing)])
            block_scores[name] = pd.concat([block_scores[name], fill], axis=1)
        block_scores[name].columns = pc_names

    for name in block_loadings:
        if block_loadings[name].shape[1] < len(pc_names):
            missing = len(pc_names) - block_loadings[name].shape[1]
            fill = pd.DataFrame(np.nan, index=block_loadings[name].index,
                                columns=[f"PC{block_loadings[name].shape[1]+i+1}" for i in range(missing)])
            block_loadings[name] = pd.concat([block_loadings[name], fill], axis=1)
        block_loadings[name].columns = pc_names

    metadata = metadata if metadata is not None else pd.DataFrame()

    return NipalsResult(
        global_scores=global_scores,
        global_loadings=gl_df,
        block_score_weights=block_score_weights,
        block_scores=block_scores,
        block_loadings=block_loadings,
        eigvals=np.array(eigvals),
        col_preproc_method=col_preproc_method.lower(),
        block_preproc_method=block_preproc_method.lower(),
        block_variances=block_vars,
        metadata=metadata
    )

# -------------------- Alignment helpers (N-Modality Generalized) -------------------- #

def relabel_obs_by_reference(adata: anndata.AnnData, ref_order: List[str]) -> anndata.AnnData:
    if adata.n_obs != len(ref_order):
        raise ValueError(f"Length mismatch when relabeling: adata has {adata.n_obs} rows, "
                         f"ref_order has {len(ref_order)}.")
    adata = adata.copy()
    adata.obs_names = pd.Index(ref_order)
    return adata

def align_n_modalities(adata_list: List[anndata.AnnData],
                       assume_same_order: bool = False,
                       cell_order_csv: str = None,
                       cell_order_col: str = "Cell_Name") -> List[anndata.AnnData]:
    """Align N AnnData objects by strict intersection or by row position if assume_same_order is set."""
    if assume_same_order:
        first_len = adata_list[0].n_obs
        for i, adata in enumerate(adata_list[1:], 1):
            if adata.n_obs != first_len:
                raise ValueError(f"--assume_same_order set, but n_obs differ: Modality 0={first_len}, Modality {i}={adata.n_obs}.")
        
        if cell_order_csv is not None:
            ref = pd.read_csv(cell_order_csv)
            if cell_order_col not in ref.columns:
                raise KeyError(f"Column '{cell_order_col}' not found in {cell_order_csv}. Available: {list(ref.columns)}")
            ref_order = ref[cell_order_col].astype(str).tolist()
            return [relabel_obs_by_reference(a, ref_order) for a in adata_list]
        else:
            # Check if all obs_names are already identical across all modalities
            all_identical = all(adata_list[0].obs_names.equals(a.obs_names) for a in adata_list[1:])
            if not all_identical:
                print_flush("[Align] --assume_same_order set with non-matching names. Overwriting all obs_names with shared cell_0, cell_1... indices.")
                shared = pd.Index([f"cell_{i}" for i in range(first_len)])
                new_list = []
                for a in adata_list:
                    a_copy = a.copy()
                    a_copy.obs_names = shared
                    new_list.append(a_copy)
                return new_list
            return adata_list

    # Default: align by intersection across all N modalities
    common = adata_list[0].obs_names
    for a in adata_list[1:]:
        common = common.intersection(a.obs_names)
    
    if len(common) == 0:
        raise ValueError("No shared cells across all modalities (obs_names intersection is empty). Use --assume_same_order if rows match by position.")
    
    common = common.sort_values()
    print_flush(f"[Align] Strict intersection aligned {len(common):,} cells across {len(adata_list)} modalities.")
    return [a[common].copy() for a in adata_list]

# -------------------- Thresholding helper -------------------- #

def apply_thresholding(adata: anndata.AnnData, threshold: float, mod_name: str, file_path: str, outdir: str, is_first_modality: bool = False) -> (anndata.AnnData, dict):
    """Apply sparsity filtering and generate provenance files."""
    n_cells = adata.n_obs
    if sp.issparse(adata.X):
        nonzeros_per_bin = np.array((adata.X != 0).sum(axis=0)).ravel()
        zeros_per_bin = n_cells - nonzeros_per_bin
    else:
        zeros_per_bin = (adata.X == 0).sum(axis=0)
    fraction_zeros = zeros_per_bin / n_cells
    keep_mask = fraction_zeros < threshold

    print_flush(f"[Threshold - {mod_name}] Columns before: {adata.n_vars:,}")
    print_flush(f"[Threshold - {mod_name}] Columns kept  : {int(keep_mask.sum()):,}")
    print_flush(f"[Threshold - {mod_name}] Columns drop  : {int((~keep_mask).sum()):,}")

    var_names = adata.var_names.to_list()
    kept_names = [var_names[i] for i, k in enumerate(keep_mask) if bool(k)]
    drop_names = [var_names[i] for i, k in enumerate(keep_mask) if not bool(k)]
    kept_idx   = np.where(keep_mask)[0]
    drop_idx   = np.where(~keep_mask)[0]

    # Use general filenames if multiple modalities are thresholded, or legacy names for the 1st modality
    prefix = "" if is_first_modality else f"{mod_name}_"
    np.save(os.path.join(outdir, f"{prefix}kept_mask.npy"), keep_mask.astype(bool))
    np.save(os.path.join(outdir, f"{prefix}dropped_mask.npy"), (~keep_mask).astype(bool))
    pd.Series(kept_idx).to_csv(os.path.join(outdir, f"{prefix}kept_indices.txt"), index=False, header=False)
    pd.Series(drop_idx).to_csv(os.path.join(outdir, f"{prefix}dropped_indices.txt"), index=False, header=False)
    pd.Series(kept_names).to_csv(os.path.join(outdir, f"{prefix}kept_bins.txt"), index=False, header=False)
    pd.Series(drop_names).to_csv(os.path.join(outdir, f"{prefix}dropped_bins.txt"), index=False, header=False)

    var_map = pd.DataFrame({"orig_index": np.arange(len(var_names), dtype=int),
                            "var_name": var_names, "keep": keep_mask.astype(bool)})
    var_map.to_csv(os.path.join(outdir, f"{prefix}var_map.tsv"), sep="\t", index=False)

    mask_summary = {
        "modality": mod_name,
        "n_vars_before": int(adata.n_vars),
        "n_vars_kept": int(keep_mask.sum()),
        "n_vars_dropped": int((~keep_mask).sum()),
        "threshold": float(threshold)
    }

    # Apply subsetting
    adata = adata[:, keep_mask].copy()

    # Save a filtered .h5ad next to input and copy into output
    if file_path and file_path.endswith('.h5ad'):
        base_noext = os.path.splitext(file_path)[0]
        filtered_path = f"{base_noext}_removed_bins_not_in_{int(threshold*100)}_percent_of_cells.h5ad"
        try:
            adata.write(filtered_path)
            print_flush(f"[Threshold - {mod_name}] Filtered AnnData saved to: {filtered_path}")
            filtered_copy = os.path.join(outdir, os.path.basename(filtered_path))
            if filtered_copy != filtered_path:
                adata.write(filtered_copy)
                print_flush(f"[Threshold - {mod_name}] (Copy) Filtered AnnData saved to: {filtered_copy}")
        except Exception as e:
            print_flush(f"[Threshold - {mod_name}] Could not save standalone filtered .h5ad: {e}")

    return adata, mask_summary

# -------------------- Main -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run NIPALS-MCIA for N Modalities with thresholding and flexible alignment")

    # N-Modality arguments
    parser.add_argument("--input_files", nargs='+', default=None, help="List of .h5ad or .csv files in order")
    parser.add_argument("--modality_names", nargs='+', default=None, help="List of names for each modality")

    # Backward compatibility arguments for legacy 2-modality scripts
    parser.add_argument("--sc_hic_h5ad_input_file", type=str, default=None)
    parser.add_argument("--second_modality_h5ad_input_file", type=str, default=None)
    parser.add_argument("--second_modality_name", type=str, default="SecondModality")
    parser.add_argument("--output_folder", type=str, required=True)

    # Metadata (optional)
    parser.add_argument("--metadata_file", type=str, default=None)

    # Preprocessing / MCIA
    parser.add_argument("--col_preproc_method", type=str, default="colprofile")
    parser.add_argument("--block_preproc_method", type=str, default="unit_var")
    parser.add_argument("--num_PCs", type=int, default=70)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--deflationMethod", type=str, default="block", choices=["block", "global"])
    parser.add_argument("--mode", type=str, default="dense")

    # Thresholding controls
    parser.add_argument("--threshold", type=float, default=None,
                        help="Single threshold applied to the FIRST modality (scHi-C).")
    parser.add_argument("--thresholds", nargs='+', type=float, default=None,
                        help="List of thresholds (one per modality). Pass >=1.0 or None to skip filtering for a specific modality.")

    # Alignment controls
    parser.add_argument("--assume_same_order", action="store_true",
                        help="Skip name intersection; assert same n_obs across all modalities; optionally relabel by reference CSV.")
    parser.add_argument("--cell_order_csv", type=str, default=None,
                        help="CSV defining desired cell order (e.g., updated_ref.csv).")
    parser.add_argument("--cell_order_col", type=str, default="Cell_Name",
                        help="Column in --cell_order_csv to use for obs_names.")

    args = parser.parse_args()

    # Resolve input files and names
    if args.input_files and args.modality_names:
        input_files = args.input_files
        modality_names = args.modality_names
    elif args.sc_hic_h5ad_input_file and args.second_modality_h5ad_input_file:
        input_files = [args.sc_hic_h5ad_input_file, args.second_modality_h5ad_input_file]
        modality_names = ["HiC", args.second_modality_name]
    else:
        print_flush("[ERROR] You must provide either (--input_files AND --modality_names) OR (--sc_hic_h5ad_input_file AND --second_modality_h5ad_input_file)")
        sys.exit(1)

    if len(input_files) != len(modality_names):
        print_flush("[ERROR] Number of --input_files must match --modality_names.")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)
    print_flush(f"[Init] Output directory: {args.output_folder}")

    # ---------- Step 1: Load all AnnData objects ---------- #
    adata_list = []
    for path, name in zip(input_files, modality_names):
        print_flush(f"[Load] Reading {name} from {path}...")
        if path.endswith('.h5ad'):
            adata = anndata.read_h5ad(path)
            adata.obs_names_make_unique(); adata.var_names_make_unique()
            adata_list.append(adata)
        elif path.endswith('.csv'):
            df_tmp = pd.read_csv(path, index_col=0)
            adata = anndata.AnnData(X=df_tmp.values, obs=pd.DataFrame(index=df_tmp.index), var=pd.DataFrame(index=df_tmp.columns))
            adata.obs_names_make_unique(); adata.var_names_make_unique()
            adata_list.append(adata)
        else:
            raise ValueError(f"Unsupported file format for {path}. Use .h5ad or .csv.")

    # ---------- Step 2: Thresholding (General N-Modality) ---------- #
    thresholds_list = [None] * len(adata_list)
    if args.thresholds:
        if len(args.thresholds) != len(adata_list):
            print_flush(f"[ERROR] Length of --thresholds ({len(args.thresholds)}) must match number of modalities ({len(adata_list)}).")
            sys.exit(1)
        thresholds_list = [t if (t is not None and t < 1.0) else None for t in args.thresholds]
    elif args.threshold is not None:
        thresholds_list[0] = args.threshold if args.threshold < 1.0 else None

    all_summaries = {}
    for i, (adata, t, mod_name, path) in enumerate(zip(adata_list, thresholds_list, modality_names, input_files)):
        if t is not None:
            is_first = (i == 0)
            adata_list[i], summary = apply_thresholding(adata, t, mod_name, path, args.output_folder, is_first_modality=is_first)
            all_summaries[mod_name] = summary
        else:
            print_flush(f"[Threshold] No filtering threshold applied to {mod_name}.")

    if all_summaries:
        with open(os.path.join(args.output_folder, "threshold_summary.json"), "w") as fh:
            json.dump(all_summaries, fh, indent=2)
        print_flush("[Save] Wrote threshold_summary.json")

    # ---------- Step 3: Align obs across all N modalities ---------- #
    adata_list = align_n_modalities(
        adata_list,
        assume_same_order=args.assume_same_order,
        cell_order_csv=args.cell_order_csv,
        cell_order_col=args.cell_order_col
    )
    print_flush(f"[Align] Cells used for NIPALS: {adata_list[0].n_obs:,}")

    # Save cells_used for provenance
    cells_used_path = os.path.join(args.output_folder, "cells_used.txt")
    pd.Series(adata_list[0].obs_names).to_csv(cells_used_path, index=False, header=False)
    print_flush(f"[Align] Wrote cells_used.txt -> {cells_used_path}")

    # ---------- Step 4: Build DataFrames FROM ALIGNED AnnData ---------- #
    print_flush("[Main] Converting aligned AnnData to dense DataFrames...")
    data_blocks = []
    for adata, name in zip(adata_list, modality_names):
        df = adata_to_dense_df(adata, name=name)
        print_flush(f"[Main] {name} DataFrame shape: {df.shape}")
        data_blocks.append(df)

    # Optional metadata
    metadata = None
    if args.metadata_file is not None:
        print_flush("[Main] Loading metadata CSV...")
        metadata = pd.read_csv(args.metadata_file)
        print_flush(f"[Main] Metadata shape: {metadata.shape}")

    # ---------- Step 5: Run SpinMCIA ---------- #
    if args.mode.lower() != "dense":
        warnings.warn("Only 'dense' mode is implemented; falling back to dense.")

    print_flush(f"[MCIA] Running SpinMCIA across {len(data_blocks)} modalities...")
    result = nipals_multiblock(
        data_blocks=data_blocks,
        col_preproc_method=args.col_preproc_method,
        block_preproc_method=args.block_preproc_method,
        num_PCs=args.num_PCs,
        tol=args.tol,
        max_iter=args.max_iter,
        deflationMethod=args.deflationMethod,
        metadata=metadata
    )

    if result is None:
        print_flush("[MCIA] Analysis did not converge. Exiting with code 1.")
        sys.exit(1)

    print_flush("[MCIA] Completed successfully.")

    # ---------- Step 6: Save outputs (INDEX PRESERVED + UNIVERSAL NPY) ---------- #
    outdir = args.output_folder
    os.makedirs(outdir, exist_ok=True)

    # Universal Feature Saving: Guaranteed .npy for every modality!
    for name, df in zip(modality_names, data_blocks):
        feature_npy_path = os.path.join(outdir, f"{name}_kept_features.npy")
        np.save(feature_npy_path, df.columns.to_numpy())
        print_flush(f"[Save] Feature list ({name}) -> {name}_kept_features.npy")

    result.global_scores.to_csv(os.path.join(outdir, "global_scores.csv"))
    print_flush("[Save] Global scores -> global_scores.csv")

    gl_df = result.global_loadings
    if isinstance(gl_df, pd.Series):
        gl_df = pd.DataFrame(gl_df.values, index=gl_df.index, columns=["PC1"])
    gl_df.to_csv(os.path.join(outdir, "global_loadings.csv"))
    print_flush("[Save] Global loadings -> global_loadings.csv")

    bsw_df = pd.DataFrame(
        result.block_score_weights,
        index=list(result.block_scores.keys()),
        columns=result.global_scores.columns
    )
    bsw_df.to_csv(os.path.join(outdir, "block_score_weights.csv"))
    print_flush("[Save] Block score weights -> block_score_weights.csv")

    for block_name, scores_df in result.block_scores.items():
        scores_df.columns = result.global_scores.columns
        scores_df.to_csv(os.path.join(outdir, f"block_scores_{block_name}.csv"))
        print_flush(f"[Save] Block scores ({block_name}) -> block_scores_{block_name}.csv")

    for block_name, load_df in result.block_loadings.items():
        load_df.columns = result.global_scores.columns
        load_df.to_csv(os.path.join(outdir, f"block_loadings_{block_name}.csv"))
        print_flush(f"[Save] Block loadings ({block_name}) -> block_loadings_{block_name}.csv")

    eigvals_df = pd.DataFrame(result.eigvals, index=result.global_scores.columns, columns=["Eigenvalue"])
    eigvals_df.to_csv(os.path.join(outdir, "eigvals.csv"), index=False)
    print_flush("[Save] Eigenvalues -> eigvals.csv")

    if result.block_variances is not None:
        pd.DataFrame.from_dict(result.block_variances, orient="index", columns=["Variance"]).to_csv(
            os.path.join(outdir, "block_variances.csv"))
        print_flush("[Save] Block variances -> block_variances.csv")

    if metadata is not None:
        metadata.to_csv(os.path.join(outdir, "metadata.csv"), index=False)
        print_flush("[Save] Metadata -> metadata.csv")

    print_flush("[Done] All results saved successfully.")

if __name__ == "__main__":
    main()