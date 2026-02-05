#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code is part of python3-adaptive-BD.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Copyright (c) 2025, 2026 Patrick B Warren <patrick.warren@stfc.ac.uk>

# Implement diffusiophoretic trapping in Landau-Squire jet.

import sys
import argparse
import adaptive_bd
import numpy as np
import pandas as pd
from numpy import pi as π
from models import Model

parser = argparse.ArgumentParser(description='pipette model adaptive BD simulator, units um and s')
parser.add_argument('code', nargs='?', default='', help='code name for run')
parser.add_argument('--seed', default=None, type=int, help='RNG seed')
parser.add_argument('-k', '--k', default=200, type=float, help='salt ratio, default 200')
parser.add_argument('-G', '--Gamma', default=150, type=float, help='value of Gamma, default 150 um^2/s')
parser.add_argument('-Q', '--Q', default=10, type=float, help='injection rate, units pL/s, default 10')
parser.add_argument('--log10Q', default=None, help='log10 injection rate, units pL/s, default unset')
parser.add_argument('--Ds', default=1610, type=float, help='salt diffusion coeff, default 1610 um^2/s')
parser.add_argument('--Dp', default=2.0, type=float, help='particle diffusion coeff, default 2.0 um^2/s')
parser.add_argument('--Rt', default=1.0, type=float, help='pipette radius, default 1.0 um')
parser.add_argument('--alpha', default=0.3, type=float, help='value of alpha, default 0.3')
parser.add_argument('--rc', default=1.0, type=float, help='cut off in um for regularisation, default 1.0')
parser.add_argument('--dt-init', default=0.05, type=float, help='initial time step, default 0.05 sec')
parser.add_argument('--eps', default='0.05,0.05', help='absolute and relative error, default 0.05 um and 0.05')
parser.add_argument('--q-lims', default='0.001,1.2', help='q limits, default as per article 0.001,1.2')
parser.add_argument('--perturb', default=1e-6, type=float, help='small perturbation to start, default 1e-6')
parser.add_argument('-t', '--tfinal', default='600', help='duration, default 600 sec')
parser.add_argument('-n', '--ntraj', default=5, type=int, help='number of trajectories, default 5')
parser.add_argument('-b', '--nblock', default=2, type=int, help='number of blocks, default 2')
parser.add_argument('-m', '--maxsteps', default='10000', help='max number of trial steps, default 10000')
parser.add_argument('-p', '--procid', default='0/1', help='process id, default 0 of 1')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
parser.add_argument('--info', action='store_true', help='provide info on computed quantities')
args = parser.parse_args()

pid, njobs = map(int, args.procid.split('/')) # sort out process id and number of jobs

local_rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid] # select a local RNG stream

tf = eval(args.tfinal) # final time

max_steps = eval(args.maxsteps) 

Dp, Δt_init = args.Dp, args.dt_init

Q = args.Q if args.log10Q is None else eval(f'10**({args.log10Q})') # extract from command line arguments

pip = Model('pipette').update(Q=Q, Γ=args.Gamma, k=args.k, Ds=args.Ds, R1=args.Rt, α=args.alpha, rc=args.rc)

if args.info:
    print(pip.info)
    exit()

# Instantiate an adaptive Brownian dynamics trajectory simulator

adb = adaptive_bd.Simulator(rng=local_rng, drift=pip.drift)

adb.εabs, adb.εrel = eval(f'{args.eps}') # relative and absolute errors
adb.qmin, adb.qmax = eval(f'{args.q_lims}') # bounds for adaptation factor

# initial position on-axis at stable fixed point or Rt, plus a small perturbation

z0 = pip.fixed_points[0] if pip.fixed_points is not None else args.Rt

r0 = np.array([0, 0, z0]) + local_rng.normal(0, args.perturb, 3)

raw = [] # used to capture raw results
for block in range(args.nblock):
    for traj in range(args.ntraj):
        r, t, Δt, ntrial, nsuccess = adb.run(r0, Δt_init, max_steps, tf, Dp)
        Δr2 = np.sum((r-r0)**2) # mean square displacement for the present trajectory
        raw.append((traj, block, ntrial, nsuccess, t, Δt, Δr2)) # capture data

columns = ['traj', 'block', 'ntrial', 'nsuccess', 't', 'Δt_final', 'Δr2']
results = pd.DataFrame(raw, columns=columns).set_index(['block', 'traj'])
results['Δt_mean'] = results.t / results.nsuccess
results['Δr'] = np.sqrt(results.Δr2)

selected_cols = ['ntrial', 'nsuccess', 't', 'Δt_mean', 'Δt_final', 'Δr2', 'Δr']

if args.verbose > 1:
    pd.set_option('display.max_rows', None)
    print(results[selected_cols])
    
if args.verbose:
    print(results[selected_cols].mean())
    print('Q =', 1e-3*pip.Q, 'pL/s')
    print('sqrt(α*Q*t) =', np.sqrt(pip.α*pip.Q*tf/(2*π**2*pip.R1))) # estimate assuming deterministic advection
    print('sqrt(6 Dp t) =', np.sqrt(6*Dp*tf)) # estimate assuming pure Brownian motion
    rms = np.sqrt(results.Δr2.mean()) # root mean square (rms) displacement
    err = np.sqrt(results.Δr2.var() / args.ntraj) / (2*rms) # the error in the rms value
    print('rms Δr =', rms, '±', err, '(rc =', pip.rc, ')')

if args.code:
    for block, traj in results.index:
        row = results.loc[block, traj]
        if not any(row.isna()):
            data = (pip.k, pip.Γ, pip.Ds, Dp, pip.R1, pip.α, 1e-3*pip.Q, pip.rc, tf,
                    row.ntrial, row.nsuccess, row.t, row.Δt_final, row.Δr2,
                    traj, block, args.ntraj, args.nblock, args.code)
            forms = ('%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g',
                     '%i', '%i', '%g', '%e', '%e',
                     '%i', '%i', '%i', '%i', '%s')
            print('\t'.join(forms) % data)
