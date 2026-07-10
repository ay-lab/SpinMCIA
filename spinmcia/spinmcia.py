#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
import gc
import anndata
import scanpy as sc
import scipy.sparse as sp
from numpy.linalg import norm
from scipy.sparse.linalg import svds

###############################################################################
# Helper: Immediate printing (to flush to SLURM logs)
###############################################################################
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

###############################################################################
# Helper function to load data
###############################################################################
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print_flush(f"[load_data] Loading file: {file_path}")
    if ext == '.h5ad':
        adata = anndata.read_h5ad(file_path)
        if sp.issparse(adata.X):
            print_flush("[load_data] Data is sparse; converting to dense.")
            data = adata.X.toarray()
        else:
            print_flush("[load_data] Data is dense.")
            data = adata.X
        df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
        print_flush(f"[load_data] Data dimensions: {df.shape}")
        return df
    elif ext == '.csv':
        print_flush("[load_data] Loading CSV as dense DataFrame.")
        df = pd.read_csv(file_path, index_col=0)
        print_flush(f"[load_data] Data dimensions: {df.shape}")
        return df
    else:
        raise ValueError("Unsupported file extension. Please use .h5ad or .csv.")

###############################################################################
# Fast in-place NumPy version of 'colprofile'
###############################################################################
def colprofile_numpy(X):
    min_val = X.min()
    if min_val < 0:
        offset = np.floor(min_val)
        X += abs(offset)
    totsum = X.sum()
    colsums = X.sum(axis=0)
    row_contr = X.sum(axis=1) / totsum
    nz = colsums != 0
    X[:, nz] /= colsums[nz]
    X -= row_contr[:, None]
    weights = np.sqrt(colsums / totsum)
    X *= weights[None, :]
    return X

###############################################################################
# Column & Block Preprocessing
###############################################################################
def col_preproc(df, method):
    method = method.lower()
    if method == "colprofile":
        X = df.values.copy()
        colprofile_numpy(X)
        return pd.DataFrame(X, index=df.index, columns=df.columns)
    elif method == "standardized":
        return (df - df.mean()) / df.std()
    elif method == "centered_only":
        return df - df.mean()
    else:
        raise ValueError("Column preprocessing method not recognized.")

def block_preproc(df, block_preproc_method):
    temp_df = df.copy()
    method = block_preproc_method.lower()
    if method == "unit_var":
        block_var = np.linalg.norm(temp_df.values, ord='fro')**2 / max(1, temp_df.shape[0] - 1)
        if block_var != 0:
            temp_df *= (1 / np.sqrt(block_var))
        else:
            warnings.warn("Data block has zero variance.")
        return temp_df
    elif method == "num_cols":
        return temp_df * (1 / temp_df.shape[1])
    elif method == "largest_sv":
        u, s, vt = svds(temp_df.values, k=1)
        eigval = s[0]
        if eigval != 0:
            return temp_df * (1 / eigval)
        else:
            warnings.warn("Data block has zero largest eigenvalue.")
            return temp_df
    elif method == "none":
        return temp_df
    else:
        raise ValueError("Block preprocessing method not recognized.")

def get_tv(ds):
    def get_bv(df):
        Xc = df - df.mean()
        n = Xc.shape[0]
        return np.linalg.norm(Xc.values, ord='fro')**2 / max(n - 1, 1)
    return [get_bv(df) for df in ds]

###############################################################################
# NIPALS Engine
###############################################################################
def nipals_iter(ds, tol=1e-12, maxIter=1000):
    stop_crit = 2 * tol
    cov_squared_old = 0
    iteration = 0
    n_samples = ds[0].shape[0]
    gs = np.ones((n_samples, 1))
    while stop_crit > tol and iteration <= maxIter:
        bl_list, bs_list = [], []
        for df in ds:
            bl_k = df.T @ gs
            arr = bl_k.values.flatten()
            norm_arr = np.linalg.norm(arr, 2)
            if norm_arr == 0 or np.isnan(norm_arr):
                warnings.warn("Zero or NaN norm in block loading.")
                return {'global_scores': None}
            arr /= norm_arr
            bl_list.append(pd.DataFrame(arr, index=df.columns))
        for df, bl in zip(ds, bl_list):
            bs_list.append((df @ bl).values.flatten())
        bs_matrix = np.column_stack(bs_list)
        gw = bs_matrix.T @ gs
        norm_gw = np.linalg.norm(gw, 2)
        if norm_gw == 0 or np.isnan(norm_gw):
            warnings.warn("Zero or NaN norm in global weights.")
            return {'global_scores': None}
        gw /= norm_gw
        gs = bs_matrix @ gw
        var_gs = np.var(gs, ddof=1)
        if var_gs == 0 or np.isnan(var_gs):
            warnings.warn("Global score variance zero or NaN.")
            return {'global_scores': None}
        gs_normed = gs.flatten() / np.sqrt(var_gs)
        covs = [np.cov(bs, gs_normed, ddof=1)[0,1]**2 for bs in bs_list]
        stop_crit = abs(sum(covs) - cov_squared_old)
        cov_squared_old = sum(covs)
        print_flush(f"[nipals_iter] Iter {iteration}, stop_crit {stop_crit}")
        iteration += 1
    if iteration > maxIter:
        warnings.warn("NIPALS failed to converge")
        return {'global_scores': None}
    eigval = np.var(gs, ddof=1)
    print_flush(f"[nipals_iter] Converged at iter {iteration}, eigval {eigval}")
    gl = pd.Series(dtype=float)
    for bl, w in zip(bl_list, gw):
        gl = pd.concat([gl, bl.iloc[:,0]*w[0]])
    return {
        'global_scores': pd.DataFrame(gs, index=ds[0].index),
        'global_loadings': gl,
        'block_score_weights': gw,
        'block_scores': [pd.DataFrame(bs, index=ds[0].index) for bs in bs_list],
        'block_loadings': bl_list,
        'eigval': eigval
    }

def deflate_block_bl(df, bl):
    bl_arr = bl.values.flatten()
    X = df.values - (df.values @ bl_arr[:, None]) @ bl_arr[None, :]
    return pd.DataFrame(X, index=df.index, columns=df.columns)

def deflate_block_gs(df, gs):
    gs_norm = gs / np.linalg.norm(gs, 2)
    X = df.values - (gs_norm @ gs_norm.T) @ df.values
    return pd.DataFrame(X, index=df.index, columns=df.columns)

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

def nipals_multiblock(data_blocks, col_preproc_method="colprofile",
                      block_preproc_method="unit_var", num_PCs=10,
                      tol=1e-9, max_iter=1000, deflationMethod="block",
                      metadata=None):
    names = [getattr(df, 'name', f"omic{i+1}") for i, df in enumerate(data_blocks)]
    for df, name in zip(data_blocks, names):
        df.name = name
    print_flush("Starting column-level preproc...")
    data_blocks = [col_preproc(df, col_preproc_method) for df in data_blocks]
    print_flush("Starting block-level preproc...")
    data_blocks = [block_preproc(df, block_preproc_method) for df in data_blocks]
    block_vars = ({name:1 for name in names}
                  if block_preproc_method.lower()=="unit_var"
                  else dict(zip(names, get_tv(data_blocks))))
    res = nipals_iter(data_blocks, tol=tol, maxIter=max_iter)
    if res['global_scores'] is None:
        return None
    gs = res['global_scores']
    gl = res['global_loadings']
    bw = res['block_score_weights']
    bs = {names[i]: res['block_scores'][i] for i in range(len(names))}
    bl = {names[i]: res['block_loadings'][i] for i in range(len(names))}
    eigs = [res['eigval']]

    for pc in range(2, num_PCs+1):
        print_flush(f"Deflating for PC {pc}...")
        if deflationMethod.lower()=="block":
            data_blocks = [deflate_block_bl(df, res['block_loadings'][i]) for i, df in enumerate(data_blocks)]
        else:
            data_blocks = [deflate_block_gs(df, res['global_scores']) for df in data_blocks]

        res = nipals_iter(data_blocks, tol=tol, maxIter=max_iter)
        if res['global_scores'] is None:
            break
        gs = pd.concat([gs, res['global_scores']], axis=1)
        gl = pd.concat([gl, res['global_loadings']], axis=1)
        bw = np.hstack([bw, res['block_score_weights']])
        eigs.append(res['eigval'])
        for i,name in enumerate(names):
            bs[name] = pd.concat([bs[name], res['block_scores'][i]], axis=1)
            bl[name] = pd.concat([bl[name], res['block_loadings'][i]], axis=1)
        gc.collect()

    pcs = [f"PC{i}" for i in range(1, gs.shape[1]+1)]
    gs.columns = pcs
    gl.columns = pcs
    for name in names:
        bs[name].columns = pcs
        bl[name].columns = pcs

    if metadata is None:
        metadata = pd.DataFrame()
    return NipalsResult(gs, gl, bw, bs, bl, np.array(eigs),
                        col_preproc_method, block_preproc_method,
                        block_vars, metadata)

###############################################################################
# Main: Parse arbitrary N modalities (or legacy 2 modalities)
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Run N-modality Nipals MCIA pipeline.")
    
    # N-Modality arguments
    parser.add_argument("--input_files", nargs='+', default=None, help="List of .h5ad or .csv files in order")
    parser.add_argument("--modality_names", nargs='+', default=None, help="List of names for each modality")
    
    # Backward compatibility arguments
    parser.add_argument("--sc_hic_h5ad_input_file", default=None, help="Path to scHi-C .h5ad file")
    parser.add_argument("--second_modality_h5ad_input_file", default=None, help="Path to second modality .h5ad file")
    parser.add_argument("--second_modality_name", default="SecondModality", help="Name for the second modality")
    
    # General arguments
    parser.add_argument("--thresholds", nargs='+', type=float, default=None, help="Sparsity thresholds")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--metadata_file", default=None)
    parser.add_argument("--col_preproc_method", default="colprofile")
    parser.add_argument("--block_preproc_method", default="unit_var")
    parser.add_argument("--num_PCs", type=int, default=70)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--deflationMethod", default="block")
    parser.add_argument("--mode", default="dense", help="Legacy argument (ignored)")
    parser.add_argument("--assume_same_order", action="store_true", help="Legacy argument (ignored, alignment is automatic)")
    args = parser.parse_args()

    # Determine input files and names
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
        print_flush("[ERROR] Number of input files must match number of modality names.")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)
    data_blocks = []

    # 1. Load Data
    for i in range(len(input_files)):
        file_path = input_files[i]
        mod_name = modality_names[i]
        df = load_data(file_path)
        df.name = mod_name
        data_blocks.append(df)

    # 2. Strict Cell Alignment Engine
    print_flush("\n[Main] =========================================")
    print_flush("[Main] STARTING STRICT CELL ALIGNMENT")
    print_flush("[Main] =========================================")
    
    common_cells = data_blocks[0].index
    for i in range(1, len(data_blocks)):
        common_cells = common_cells.intersection(data_blocks[i].index)
    
    common_cells = common_cells.sort_values()
    print_flush(f"[Main] Found {len(common_cells)} perfectly matching cells across all {len(data_blocks)} modalities.")
    
    if len(common_cells) == 0:
        print_flush("[ERROR] Zero common cells found! Check your cell IDs (obs_names).")
        sys.exit(1)

    for i in range(len(data_blocks)):
        original_size = data_blocks[i].shape[0]
        mod_name = modality_names[i]
        data_blocks[i] = data_blocks[i].loc[common_cells]
        data_blocks[i].name = mod_name 
        print_flush(f"   -> {mod_name} resized from {original_size} to {data_blocks[i].shape[0]} cells.")
    print_flush("[Main] Alignment Complete.\n")

    metadata = pd.read_csv(args.metadata_file, index_col=0) if args.metadata_file else None

    # 3. Run Pipeline
    print_flush(f"[Main] Running pipeline on {len(data_blocks)} modalities...")
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
        print_flush("[Main] Convergence failed.")
        sys.exit(1)

    # 4. Save Outputs
    result.global_scores.to_csv(os.path.join(args.output_folder, "global_scores.csv"))
    result.global_loadings.to_csv(os.path.join(args.output_folder, "global_loadings.csv"))
    pd.DataFrame(result.block_score_weights, index=modality_names, columns=result.global_scores.columns).to_csv(os.path.join(args.output_folder, "block_score_weights.csv"))
    pd.DataFrame(result.eigvals, index=result.global_scores.columns, columns=["Eigenvalue"]).to_csv(os.path.join(args.output_folder, "eigvals.csv"))
    pd.DataFrame.from_dict(result.block_variances, orient="index", columns=["Variance"]).to_csv(os.path.join(args.output_folder, "block_variances.csv"))

    for name, df in result.block_scores.items():
        df.to_csv(os.path.join(args.output_folder, f"block_scores_{name}.csv"))
    for name, df in result.block_loadings.items():
        df.to_csv(os.path.join(args.output_folder, f"block_loadings_{name}.csv"))

    print_flush("[Main] All N-modality results saved successfully.")

if __name__ == "__main__":
    main()