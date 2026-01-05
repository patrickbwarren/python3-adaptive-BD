#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import pandas as pd
from numpy import pi as π
import adaptive_bd

parser = argparse.ArgumentParser(description='LS jet adaptive BD simulator, units um and s')
parser.add_argument('code', nargs='?', default='', help='code name for run')
parser.add_argument('--seed', default=None, type=int, help='RNG seed')
parser.add_argument('-w', '--width', default=500, type=float, help='width of plotting window, default 500')
parser.add_argument('-k', '--k', default=200, type=float, help='salt ratio, default 200')
parser.add_argument('-G', '--Gamma', default=440, type=float, help='value of Gamma, default 440 um^2/s')
parser.add_argument('-Q', '--Q', default=10, type=float, help='injection rate, units pL/s, default 10')
parser.add_argument('--log10Q', default=None, help='log10 injection rate, units pL/s, default unset')
parser.add_argument('--Ds', default=1600, type=float, help='salt diffusion coeff, default 1600 um^2/s')
parser.add_argument('--Dp', default=2.0, type=float, help='particle diffusion coeff, default 2.0 um^2/s')
parser.add_argument('--R1', default=0.5, type=float, help='pipette radius, default 0.5 um')
parser.add_argument('--alpha', default=0.3, type=float, help='value of alpha, default 0.3')
parser.add_argument('--rcut', default=None, help='cut off in um for regularisation, default none')
parser.add_argument('--dt-init', default=0.05, type=float, help='initial time step, default 0.05 sec')
parser.add_argument('--eps', default='0.05,0.05', help='absolute and relative error, default 0.05 um and 0.05')
parser.add_argument('--q-lims', default='0.001,1.2', help='q limits, default as per article 0.001,1.2')
parser.add_argument('-t', '--t-final', default='600', help='duration, default 600 sec')
parser.add_argument('-n', '--ntraj', default=5, type=int, help='number of trajectories, default 5')
parser.add_argument('-b', '--nblock', default=2, type=int, help='number of blocks, default 2')
parser.add_argument('-m', '--maxsteps', default=10000, type=int, help='max number of trial steps, default 10000')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
parser.add_argument('--cart', action='store_true', help='use cartesian drift field expressions')
parser.add_argument('--info', action='store_true', help='provide info on computed quantities')
args = parser.parse_args()

tf = eval(args.t_final) # final time

k, Γ, Ds, Dp, R1, α, Δt_init = args.k, args.Gamma, args.Ds, args.Dp, args.R1, args.alpha, args.dt_init

Q = args.Q if args.log10Q is None else eval(f'10**({args.log10Q})') # extract from command line arguments

Q = 1e3 * Q # convert to um^3/s

rc = 0 if args.rcut is None else eval(args.rcut)

# derived quantities

rstar = π*R1/α # where stokeslet and radial outflow match
v1 = Q / (π*R1**2) # flow speed (definition)
Pbyη = α*R1*v1 # from Secchi et al, should also = Q / r*
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
    names = ['k', 'Γ', 'Ds', 'Γ / Ds', 'Dp', 'Q', 'Qcrit', 'R1', 'α', 'v1', 'P/η',
             'r_c', 'r*', 'r1', 'r2', 'λ', 'λ*', 'Pe(salt)', 'Pe(part)', 'Δt(init)', 't_final',
             'max steps', '# traj', '# block', 'RNG seed']
    values = [k, Γ, Ds, Γ/Ds, Dp, 1e-3*Q, 1e-3*Qcrit, R1, α, 1e-3*v1, Pbyη,
              rc, rstar, root[0], root[1], λ, λstar, Pe, Pe*Ds/Dp, Δt_init, tf,
              args.maxsteps, args.ntraj, args.nblock, args.seed]
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

# Expressions for drift field in LS jet problem
 
def drift_spherical(r):
    x, y, z = r[:] # z is normal distance = r cosθ
    xy = np.sqrt(x**2 + y**2) # in-plane distance = r sinθ
    rr = np.sqrt(x**2 + y**2 + z**2) # radial distance from origin
    cosθ, sinθ = z/rr, xy/rr # polar angle, as cos and sin
    if rr < rc: # crude regularisation
        u = np.zeros_like(r)
    else:
        ur = Q/(4*π*rr**2) + Pbyη*cosθ/(4*π*rr) - k*λ*Γ/(rr*(rr+k*λ))
        uθ = - Pbyη*sinθ/(8*π*rr)
        uxy = ur*sinθ + uθ*cosθ # parallel xy-component of velocity
        uz = ur*cosθ - uθ*sinθ # normal z-component of velocity
        ux, uy = uxy*x/xy, uxy*y/xy # resolved x, y components
        u = np.array((ux, uy, uz))
    return u

def drift_cartesian(r):
    x, y, z = r[:] 
    rr = np.sqrt(x**2 + y**2 + z**2) # radial distance from origin
    if rr < rc: # crude regularisation
        u = np.zeros_like(r)
    else:
        A = Pbyη/(8*π*rr**3)
        B = Q/(4*π*rr**3) - k*λ*Γ/(rr**2*(rr+k*λ))
        u = np.array((A*x*z + B*x, A*y*z + B*y, A*(x**2 + y**2 + 2*z**2) + B*z))
    return u

# Instantiate an adaptive Brownian dynamics trajectory simulator

adb = adaptive_bd.Simulator(seed=args.seed)

adb.εabs, adb.εrel = eval(f'{args.eps}') # relative and absolute errors
adb.qmin, adb.qmax = eval(f'{args.q_lims}') # bounds for adaptation factor

adb.drift = drift_cartesian if args.cart else drift_spherical

# initial position on-axis (cartesian method) or off-axis (spherical polars)

z0 = R1 if np.isnan(root[0]) else root[0]
r0 = np.array([0, 0, z0]) if args.cart else np.array([1e-6, 2e-6, z0])

raw = [] # used to capture raw results
for block in range(args.nblock):
    for traj in range(args.ntraj):
        r, t, Δt, ntrial, nsuccess = adb.run(r0, Δt_init, max_steps, t_final, Dp):
        Δr2 = np.sum((r-r0)**2) # mean square displacement for the present trajectory
        raw.append((traj, block, ntrial, nsuccess, t, Δt, Δr2)) # capture data

columns = ['traj', 'block', 'ntrial', 'nsuccess', 't', 'Δt_final', 'Δr2']
data = pd.DataFrame(raw, columns=columns).set_index(['block', 'traj'])
data['Δt_mean'] = data.t / data.nsuccess
data['Δr'] = np.sqrt(data.Δr2)

selected_cols = ['ntrial', 'nsuccess', 't', 'Δt_mean', 'Δt_final', 'Δr2', 'Δr']

if args.verbose > 1:
    pd.set_option('display.max_rows', None)
    print(data[selected_cols])
    
if args.verbose:
    print(data[selected_cols].mean())
    print('Q =', 1e-3*Q, 'pL/s')
    print('sqrt(α*Q*t) =', np.sqrt(α*Q*tf/(2*π**2*R1))) # estimate assuming deterministic advection
    print('sqrt(6 Dp t) =', np.sqrt(6*Dp*tf)) # estimate assuming pure Brownian motion
    rms = np.sqrt(data.Δr2.mean()) # root mean square (rms) displacement
    err = np.sqrt(data.Δr2.var() / args.ntraj) / (2*rms) # the error in the rms value
    print('rms Δr =', rms, '±', err, '(rc =', rc, ')')

if args.code:
    for block, traj in data.index:
        row = data.loc[block, traj]
        if not any(row.isna()):
            data = (k, Γ, Ds, Dp, R1, α, Q*1e-3, rc, tf,
                    row.ntrial, row.nsuccess, row.t, row.Δt_final, row.Δr2,
                    traj, block, args.ntraj, args.nblock, args.code)
            forms = ('%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g',
                     '%i', '%i', '%g', '%e', '%e',
                     '%i', '%i', '%i', '%i', '%s')
            print('\t'.join(forms) % data)
