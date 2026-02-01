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
import numpy as np
import pandas as pd
from numpy import pi as π
import adaptive_bd

parser = argparse.ArgumentParser(description='LS jet adaptive BD simulator, units um and s')
parser.add_argument('code', nargs='?', default='', help='code name for run')
parser.add_argument('--seed', default=None, type=int, help='RNG seed')
parser.add_argument('-k', '--k', default=200, type=float, help='salt ratio, default 200')
parser.add_argument('-G', '--Gamma', default=440, type=float, help='value of Gamma, default 440 um^2/s')
parser.add_argument('-Q', '--Q', default=10, type=float, help='injection rate, units pL/s, default 10')
parser.add_argument('--log10Q', default=None, help='log10 injection rate, units pL/s, default unset')
parser.add_argument('--Ds', default=1600, type=float, help='salt diffusion coeff, default 1600 um^2/s')
parser.add_argument('--Dp', default=2.0, type=float, help='particle diffusion coeff, default 2.0 um^2/s')
parser.add_argument('--Rt', default=0.5, type=float, help='pipette radius, default 0.5 um')
parser.add_argument('--alpha', default=0.3, type=float, help='value of alpha, default 0.3')
parser.add_argument('--rcut', default=None, help='cut off in um for regularisation, default none')
parser.add_argument('--dt-init', default=0.05, type=float, help='initial time step, default 0.05 sec')
parser.add_argument('--eps', default='0.05,0.05', help='absolute and relative error, default 0.05 um and 0.05')
parser.add_argument('--q-lims', default='0.001,1.2', help='q limits, default as per article 0.001,1.2')
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

k, Γ, Ds, Dp, Rt, α, Δt_init = args.k, args.Gamma, args.Ds, args.Dp, args.Rt, args.alpha, args.dt_init

Q = args.Q if args.log10Q is None else eval(f'10**({args.log10Q})') # extract from command line arguments

Q = 1e3 * Q # convert to um^3/s

rc = 0 if args.rcut is None else eval(args.rcut)

# derived quantities

rstar = π*Rt/α # where stokeslet and radial outflow match
vt = Q / (π*Rt**2) # flow speed (definition)
Pbyη = α*Rt*vt # from Secchi et al, should also = Q / r*
Pe = Pbyη / (4*π*Ds) # definition from Secchi et al
λ = Q / (4*π*Ds) # definition
λstar = λ / rstar # should be the same as Pe (salt)

Qcrit = 4*π*Ds*rstar*(np.sqrt(Γ/Ds)-np.sqrt(1/k))**2 # critical upper bound on Q

# the quadratic for the roots in units of r* is r^2 − (kΓbyD − kλ* − 1)r + kλ* = 0

b = k*Γ/Ds - k*λstar - 1 # the equation is r^2 − br + c = 0
Δ = b**2 - 4*k*λstar # discriminant of above

if Q > 0 and Δ > 0 and b > 0: # condition for roots to exist
    root = np.array((0.5*rstar*(b-np.sqrt(Δ)), 0.5*rstar*(b+np.sqrt(Δ))))
else:
    root = np.array((np.nan, np.nan))

if args.info:
    um, umpers, umsqpers, none = 'µm', 'µm/s', 'µm²/s', ''
    names = ['k', 'Γ', 'Ds', 'Γ / Ds', 'Dp', 'Q', 'Qcrit', 'Rt', 'α', 'vt', 'P/η',
             'r_c', 'r*', 'r1', 'r2', 'λ', 'λ*', 'Pe(salt)', 'Pe(part)', 'Δt(init)', 't_final',
             'max steps', '# traj', '# block', 'RNG seed']
    values = [k, Γ, Ds, Γ/Ds, Dp, 1e-3*Q, 1e-3*Qcrit, Rt, α, 1e-3*vt, Pbyη,
              rc, rstar, root[0], root[1], λ, λstar, Pe, Pe*Ds/Dp, Δt_init, tf,
              max_steps, args.ntraj, args.nblock, args.seed]
    units = [none, umsqpers, umsqpers, none, umsqpers, 'pL/s', 'pL/s', um, none, 'mm/s', umsqpers,
             um, um, um, um, um, none, none, none, 's', 's',
             none, none, none, none]
    table_d = {'name':pd.Series(names, dtype=str),
               'value':pd.Series(values, dtype=float),
               'unit':pd.Series(units, dtype=str)}
    table = pd.DataFrame(table_d).set_index('name')
    table.value = table.value.apply(lambda x: round(x, 3))
    pd.set_option('display.max_columns', None)
    print(sys.executable, ' '.join(sys.argv))
    print('\n'.join(table.to_string().split('\n')[2:])) # lose the first two lines
#    print(table)
#    print(table.transpose())
    exit()

# Drift field in LS jet problem (regularised)
 
def drift(rvec):
    x, y, z = rvec[:] # z is normal distance = r cosθ
    ρ = np.sqrt(x**2 + y**2) # in-plane distance = r sinθ
    r = np.sqrt(x**2 + y**2 + z**2) # radial distance from origin
    cosθ, sinθ = z/r, ρ/r # polar angle, as cos and sin
    sinθ_cosφ, sinθ_sinφ = x/r, y/r # avoids dividing by ρ which may be zero at the start
    ur = Q/(4*π*r**2) + Pbyη*cosθ/(4*π*r) - k*λ*Γ/(r*(r+k*λ)) # radial drift velocity
    ux = ur*sinθ_cosφ - Pbyη*sinθ_cosφ*cosθ/(8*π*r) # radial components
    uy = ur*sinθ_sinφ - Pbyη*sinθ_sinφ*cosθ/(8*π*r) # avoiding dividing by ρ
    uz = ur*cosθ + Pbyη*sinθ**2/(8*π*r) # normal z-component of velocity
    return np.zeros_like(rvec) if r < rc else np.array((ux, uy, uz))

# Instantiate an adaptive Brownian dynamics trajectory simulator

adb = adaptive_bd.Simulator(rng=local_rng, drift=drift)

adb.εabs, adb.εrel = eval(f'{args.eps}') # relative and absolute errors
adb.qmin, adb.qmax = eval(f'{args.q_lims}') # bounds for adaptation factor

# initial position on-axis at stable fixed point or Rt

z0 = Rt if np.isnan(root[0]) else root[0]
r0 = np.array([0, 0, z0])

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
    print('Q =', 1e-3*Q, 'pL/s')
    print('sqrt(α*Q*t) =', np.sqrt(α*Q*tf/(2*π**2*Rt))) # estimate assuming deterministic advection
    print('sqrt(6 Dp t) =', np.sqrt(6*Dp*tf)) # estimate assuming pure Brownian motion
    rms = np.sqrt(results.Δr2.mean()) # root mean square (rms) displacement
    err = np.sqrt(results.Δr2.var() / args.ntraj) / (2*rms) # the error in the rms value
    print('rms Δr =', rms, '±', err, '(rc =', rc, ')')

if args.code:
    for block, traj in results.index:
        row = results.loc[block, traj]
        if not any(row.isna()):
            data = (k, Γ, Ds, Dp, Rt, α, Q*1e-3, rc, tf,
                    row.ntrial, row.nsuccess, row.t, row.Δt_final, row.Δr2,
                    traj, block, args.ntraj, args.nblock, args.code)
            forms = ('%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g',
                     '%i', '%i', '%g', '%e', '%e',
                     '%i', '%i', '%i', '%i', '%s')
            print('\t'.join(forms) % data)
