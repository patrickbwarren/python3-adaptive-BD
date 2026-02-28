#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Describe raw Brownian dynamics simulation output
# Warren and Sear 2025/2026

import gzip
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='describe raw BD dataset')
parser.add_argument('dataset', help='raw input data file, eg *.dat.gz')
parser.add_argument('-s', '--show-dataset', action='store_true', help='also report dataset name')
args = parser.parse_args()

schema = {'k':float, 'Γ':float, 'Ds':float, 'Dp':float, 'R1':float,
          'α':float, 'Q':float, 'rc':float, 't_final':float, 'maxsteps':int,
          'ntrial':int, 'nsuccess':int, 't':float, 'Δt_final':float, 'Δr2':float,
          'traj':int, 'block':int, 'ntraj':int, 'nblock':int, 'code':str}

with gzip.open(args.dataset, 'rt') as fp:
    first_line = fp.readline()

if len(first_line.split('\t')) < len(schema): # wrangle dataset type, pipette or wall pore
    del schema['α'] # if wall pore then there is no α column

df = pd.read_csv(args.dataset, sep='\t', names=schema.keys(), dtype=schema)
df.sort_values(['Dp', 'Q', 'traj'], inplace=True)

def range_str(v, vals): # convert a list of values to a singleton, several values, or a range
    s = ', '.join([str(x) for x in np.sort(vals)]) if len(vals) < 10 else '--'.join([str(f(vals)) for f in [min, max]])
    return v, '  '+s, f'{len(vals):10}'

df2 = pd.DataFrame([range_str(col, df[col].unique()) for col in df.columns], columns=['column', 'range', 'count'])
if args.show_dataset:
    df2.insert(0, 'dataset', args.dataset)
header_row = pd.DataFrame(index=[-1], columns=df2.columns)
df2 = pd.concat([header_row, df2])
df2.loc[-1] = df2.columns

print('Dataset', args.dataset, 'contains', df.shape[0], 'records')
print('\n'.join(df2.to_string(justify='left', index=False).split('\n')[1:]))
