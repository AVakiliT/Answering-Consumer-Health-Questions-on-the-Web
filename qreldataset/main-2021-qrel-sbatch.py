#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ipykernel/2022a/bin/ipython --ipython-dir=/tmp
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=rrg-smucker
#SBATCH --array=21-73
#SBATCH --time=0:20:0
# %%
import os

from utils.util import shell_cmd, unfixdocno

shell_cmd('module load StdEnv gcc cuda/11 faiss arrow/8 java')

from pathlib import Path

import pandas as pd

from tqdm import tqdm


# %%
files = sorted(list(
    Path('/project/6004803/smucker/group-data/c4-parquet').rglob('*.snappy.parquet')
))

qrels = pd.read_csv("/home/avakilit/resources21/qrels/qrels-35topics.txt",
                    header=None,
                    names="topic iter docno usefulness supportiveness credibility".split(),
                    sep=' ')

qrels = qrels.drop(columns="iter".split())
qrels.docno = qrels.docno.apply(unfixdocno)

n = int(os.environ['SLURM_ARRAY_TASK_ID'])
files_subset = files[n * 100:n * 100 + 100]


def func(file):
    df = pd.read_parquet(file)
    df = df.merge(qrels, on="docno", how="inner")
    return df


df = pd.concat(list(map(func, tqdm(files_subset))))
df = df.reset_index(drop=True)
df.to_parquet(f'qreldataset/2021-qrels-docs/{n * 100}-{n * 100 + 100}.snappy.parquet')
