#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

import adaptive_bd

parser = argparse.ArgumentParser(description='Ornstein-Uhlenbeck adaptive BD simulator')
parser.add_argument('code', nargs='?', default='', help='code name for run')
parser.add_argument('--seed', default=None, type=int, help='RNG seed')
parser.add_argument('--Dp', default=1.0, type=float, help='particle diffusion coeff, default 1.0 um^2/s')
parser.add_argument('--gamma', default=0.1, type=float, help='constant drift speed in -z direction, default 0.1 um/s')
parser.add_argument('--dt-init', default=0.05, type=float, help='initial time step, default 0.05 sec')
parser.add_argument('--eps', default='0.05,0.05', help='absolute and relative error, default 0.05 um and 0.05')
parser.add_argument('--q-lims', default='0.001,1.2', help='q limits, default as per article 0.001,1.2')
parser.add_argument('-t', '--tfinal', default='600', help='duration, default 600 sec')
parser.add_argument('-n', '--ntraj', default=5, type=int, help='number of trajectories, default 5')
parser.add_argument('-b', '--nblock', default=2, type=int, help='number of blocks, default 2')
parser.add_argument('-m', '--maxsteps', default='10000', help='max number of trial steps, default 10000')
parser.add_argument('-p', '--procid', default='0/1', help='process id, default 0 of 1')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

tf, max_steps = map(eval, [args.tfinal, args.maxsteps])
Dp, Δt_init, γ = args.Dp, args.dt_init, args.gamma

pid, njobs = map(int, args.procid.split('/')) # sort out process id and number of jobs
local_rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid] # select a local RNG stream

# Constant drift field: γ in negative z direction.
 
def constant_drift(r):
    u = np.array([0, 0, -γ])
    return u

# Instantiate an adaptive Brownian dynamics trajectory simulator with constant drift

adb = adaptive_bd.Simulator(rng=local_rng, drift=constant_drift)
adb.εabs, adb.εrel = eval(f'{args.eps}') # relative and absolute errors
adb.qmin, adb.qmax = eval(f'{args.q_lims}') # bounds for adaptation factor

# initial position at origin

r0 = np.zeros(3)

raw = [] # used to capture raw results
for block in range(args.nblock):
    for traj in range(args.ntraj):
        r, t, Δt, ntrial, nsuccess = adb.run(r0, Δt_init, max_steps, tf, Dp)
        raw.append((traj, block, ntrial, nsuccess, t, Δt, r[0], r[1], r[2])) # capture data

columns = ['traj', 'block', 'ntrial', 'nsuccess', 't', 'Δt_final', 'x', 'y', 'z']
results = pd.DataFrame(raw, columns=columns).set_index(['block', 'traj'])
results['Δt_mean'] = results.t / results.nsuccess
results['Δr2'] = np.sum(np.array([results.x, results.y, results.z])**2, axis=0)

selected_cols = ['ntrial', 'nsuccess', 't', 'Δt_mean', 'Δt_final', 'Δr2']

if args.verbose > 1:
    pd.set_option('display.max_rows', None)
    print(results[selected_cols])
    
if args.verbose:
    print(results[selected_cols].mean())
    rms = np.sqrt(results.Δr2.mean()) # root mean square (rms) displacement
    err = np.sqrt(results.Δr2.var() / args.ntraj) / (2*rms) # the error in the rms value
    print('rms Δr =', rms, '±', err)
    print('expected rms Δr =', np.sqrt(6*Dp*tf + c**2*tf**2))

if args.code:
    for block, traj in results.index:
        row = results.loc[block, traj]
        if not any(row.isna()):
            data = (row.ntrial, row.nsuccess, row.t, row.Δt_final, row.x, row.y, row.z,
                    traj, block, args.ntraj, args.nblock, args.code)
            forms = ('%i', '%i', '%g', '%e', '%e', '%e', '%e',
                     '%i', '%i', '%i', '%i', '%s')
            print('\t'.join(forms) % data)
