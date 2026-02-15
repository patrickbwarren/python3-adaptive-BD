#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analyse raw Brownian dynamics simulation output and compile to a
# spreadsheet.  For example:

# ./vardp_analyse.py data/vardp.dat.gz -o vardp_results.ods

import argparse
import numpy as np
import pandas as pd

def range_str(v, vals): # convert a list of values to a singleton, several values, or a range
    s = ', '.join([str(x) for x in vals]) if len(vals) < 10 else '--'.join([str(f(vals)) for f in [min, max]])
    return v, '  '+s, f'{len(vals):10}'

parser = argparse.ArgumentParser(description='compile raw BD data to a spreadsheet')
parser.add_argument('dataset', help='raw input data file, eg *.dat.gz')
parser.add_argument('-c', '--column', default='Dp', help='select data column, default Dp')
parser.add_argument('-d', '--describe', action='store_true', help='print a summary of the columns in the raw data')
parser.add_argument('-o', '--output', help='output compiled data to a spreadsheet, eg .ods, .xlsx')
args = parser.parse_args()

c = args.column

schema= {'k':float, 'Γ':float, 'Ds':float, 'Dp':float, 'R1':float, 
         'α':float, 'Q':float, 'rc':float, 't_final':float, 
         'ntrial':int, 'nsuccess':int, 't':float, 'Δt_final':float, 'Δr2':float, 
         'traj':int, 'block':int, 'ntraj':int, 'nblock':int, 'code':str}

df = pd.read_csv(args.dataset, sep='\t', names=schema.keys(), dtype=schema)
df.sort_values([c, 'Q', 'traj'], inplace=True)

if args.describe:
    dff = pd.DataFrame([range_str(c, df[c].unique()) for c in df.columns], columns=['column', 'range', 'count'])
    header_row = pd.DataFrame(index=[-1], columns=dff.columns)
    dff = pd.concat([header_row, dff])
    dff.loc[-1] = dff.columns
    print('Dataset', args.dataset, 'contains', df.shape[0], 'records')
    print('\n'.join(dff.to_string(justify='left', index=False).split('\n')[1:]))
    exit()

df2 = df[['Q', c, 'block', 'Δr2']].groupby(['Q', c, 'block']).mean() # calculate mean square displacement per block

df2['RMSD'] = np.sqrt(df2.Δr2) # root mean square displacement (RMSD), per block
df2.reset_index(inplace=True) # pull multilevel index back to columns

ser1 = df2[['Q', c, 'RMSD']].groupby(['Q', c]).mean()['RMSD'] # the global mean RMSD
ser2 = df2[['Q', c, 'RMSD']].groupby(['Q', c]).sem()['RMSD'].rename('std_err') # std error in mean

df3 = pd.concat([ser1, ser2], axis=1).reset_index() # compile these into a new dataframe, pulling index (Q) back to a column

if args.output:
    with pd.ExcelWriter(args.output) as writer:
        for x in df3[c].unique():
            df3[df3[c] == x].drop(c, axis=1).to_excel(writer, sheet_name=f'{c}={x}', index=False)
    vals = ', '.join([str(x) for x in df[c].unique()])
    print(f'Data for {c} = {vals} written to', args.output)
else:
    print(df3)
