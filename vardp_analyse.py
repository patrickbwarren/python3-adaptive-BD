#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analyse raw Brownian dynamics simulation output and compile to a
# spreadsheet.  For example:

# ./vardp_analyse.py data/vardp.dat.gz -o vardp_dat.ods


import argparse
import adaptive_bd
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='figure 1 in manuscript')
parser.add_argument('datafile', help='input data file, eg *.dat.gz')
parser.add_argument('-o', '--output', help='output compiled data to a spreadsheet, eg .ods, .xlsx')
args = parser.parse_args()

schema= {'k':float, 'Γ':float, 'Ds':float, 'Dp':float, 'R1':float, 
         'α':float, 'Q':float, 'rc':float, 't_final':float, 
         'ntrial':int, 'nsuccess':int, 't':float, 'Δt_final':float, 'Δr2':float, 
         'traj':int, 'block':int, 'ntraj':int, 'nblock':int, 'code':str}

df = pd.read_csv(args.datafile, sep='\t', names=schema.keys(), dtype=schema)
df.sort_values(['Dp', 'Q', 'traj'], inplace=True)

df2 = df[['Q', 'Dp', 'block', 'Δr2']].groupby(['Q', 'Dp', 'block']).mean() # calculate mean square displacement per block

df2['RMSD'] = np.sqrt(df2.Δr2) # root mean square displacement (RMSD), per block
df2.reset_index(inplace=True) # pull multilevel index back to columns

ser1 = df2[['Q', 'Dp', 'RMSD']].groupby(['Q', 'Dp']).mean()['RMSD'] # the global mean RMSD
ser2 = df2[['Q', 'Dp', 'RMSD']].groupby(['Q', 'Dp']).sem()['RMSD'].rename('std_err') # std error in mean
df4 = pd.concat([ser1, ser2], axis=1).reset_index() # compile these into a new dataframe, pulling index (Q) back to a column

Dpvals = df.Dp.unique()

if args.output:

    with pd.ExcelWriter(args.output) as writer:
        for Dp in Dpvals:
            df4[df4.Dp == Dp].to_excel(writer, sheet_name=f'Dp={Dp}', index=False)
    print('Data written to', args.output)

else:

    print(df4)

