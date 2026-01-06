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
parser.add_argument('--z0', default=25.0, type=float, help='initial position in z, default 25.0 um')
parser.add_argument('--dt-init', default=0.05, type=float, help='initial time step, default 0.05 sec')
parser.add_argument('--eps', default='0.05,0.05', help='absolute and relative error, default 0.05 um and 0.05')
parser.add_argument('--q-lims', default='0.001,1.2', help='q limits, default as per article 0.001,1.2')
parser.add_argument('-t', '--tfinal', default='100', help='duration, default 100 sec')
parser.add_argument('-n', '--ntraj', default=5, type=int, help='number of trajectories, default 5')
parser.add_argument('-b', '--nblock', default=2, type=int, help='number of blocks, default 2')
parser.add_argument('-m', '--maxsteps', default='10000', help='max number of trial steps, default 10000')
parser.add_argument('-p', '--procid', default='0/1', help='process id, default 0 of 1')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

tf, max_steps = map(eval, [args.tfinal, args.maxsteps])
Dp, Δt_init = args.Dp, args.dt_init

pid, njobs = map(int, args.procid.split('/')) # sort out process id and number of jobs
local_rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid] # select a local RNG stream

# Create a derived class that overwrites the Heun-Euler step

def reflect(r, Δr): # reflect displacement in z = 0
    x, y, z = r[:]
    Δx, Δy, Δz = Δr[:]
    return np.array([Δx, Δy, np.abs(z + Δz) - z])

class ReflectingSimulator(adaptive_bd.Simulator):

    def heun_euler_trial_step(self, r, Δt, R):
        u = self.drift(r)
        Δrbar = reflect(r, u*Δt + self.sqrt2Dp*R) # eq 8
        ubar = self.drift(r + Δrbar)
        Δr = reflect(r, 0.5*(u + ubar)*Δt + self.sqrt2Dp*R) # eq 9
        return Δr, Δrbar

radb = ReflectingSimulator(rng=local_rng)
radb.εabs, radb.εrel = eval(f'{args.eps}') # relative and absolute errors
radb.qmin, radb.qmax = eval(f'{args.q_lims}') # bounds for adaptation factor

r0 = np.array([0, 0, args.z0]) # initial position

raw = [] # used to capture raw results
for block in range(args.nblock):
    for traj in range(args.ntraj):
        r, t, Δt, ntrial, nsuccess = radb.run(r0, Δt_init, max_steps, tf, Dp)
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
    print('expected rms Δr =', np.sqrt(6*Dp*tf))

if args.code:
    for block, traj in results.index:
        row = results.loc[block, traj]
        if not any(row.isna()):
            data = (row.ntrial, row.nsuccess, row.t, row.Δt_final, row.x, row.y, row.z,
                    traj, block, args.ntraj, args.nblock, args.code)
            forms = ('%i', '%i', '%g', '%e', '%e', '%e', '%e',
                     '%i', '%i', '%i', '%i', '%s')
            print('\t'.join(forms) % data)
