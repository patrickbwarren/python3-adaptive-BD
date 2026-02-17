#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analyse raw Brownian dynamics simulation output and compile to a
# spreadsheet.  For example:

# ./vardp_analyse.py data/vardp.dat.gz -o vardp_results.ods

import gzip
import argparse
import numpy as np
import pandas as pd

def range_str(v, vals): # convert a list of values to a singleton, several values, or a range
    s = ', '.join([str(x) for x in vals]) if len(vals) < 10 else '--'.join([str(f(vals)) for f in [min, max]])
    return v, '  '+s, f'{len(vals):10}'

parser = argparse.ArgumentParser(description='compile raw BD data to a spreadsheet')
parser.add_argument('dataset', help='raw input data file, eg *.dat.gz')
parser.add_argument('-c', '--column', default='Dp', help='select data column, default Dp')
parser.add_argument('-o', '--output', help='output compiled data to a spreadsheet, eg .ods, .xlsx')
args = parser.parse_args()

schema= {'k':float, 'Γ':float, 'Ds':float, 'Dp':float, 'R1':float, 
         'α':float, 'Q':float, 'rc':float, 't_final':float, 
         'ntrial':int, 'nsuccess':int, 't':float, 'Δt_final':float, 'Δr2':float, 
         'traj':int, 'block':int, 'ntraj':int, 'nblock':int, 'code':str}

with gzip.open(args.dataset, 'rt') as f:
    first_line = f.readline()

if len(first_line.split('\t')) < len(schema): # wrangle dataset type, pipette or wall pore
    del schema['α'] # if wall pore then there is no α column

col = args.column

df = pd.read_csv(args.dataset, sep='\t', names=schema.keys(), dtype=schema)
df.sort_values([col, 'Q', 'traj'], inplace=True)

df2 = df[['Q', col, 'block', 'Δr2']].groupby(['Q', col, 'block']).mean() # calculate mean square displacement per block

df2['RMSD'] = np.sqrt(df2.Δr2) # root mean square displacement (RMSD), per block
df2.reset_index(inplace=True) # pull multilevel index back to columns

ser1 = df2[['Q', col, 'RMSD']].groupby(['Q', col]).mean()['RMSD'] # the global mean RMSD
ser2 = df2[['Q', col, 'RMSD']].groupby(['Q', col]).sem()['RMSD'].rename('std_err') # std error in mean

df3 = pd.concat([ser1, ser2], axis=1).reset_index() # compile these into a new dataframe, pulling index (Q) back to a column

if args.output:
    with pd.ExcelWriter(args.output) as writer:
        for val in df3[col].unique():
            df3[df3[col] == val].drop(col, axis=1).to_excel(writer, sheet_name=f'{col}={val}', index=False)
    vals = ', '.join([str(val) for val in df[col].unique()])
    print(f'Data for {col} = {vals} written to', args.output)
else:
    print(df3)
