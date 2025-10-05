#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import pandas as pd
from numpy import pi as π

rng = np.random.default_rng()

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
parser.add_argument('--show', action='store_true', help='plot results')
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

drift = drift_cartesian if args.cart else drift_spherical

def l2norm(r): # standard vector norm
    return np.sqrt(np.sum(r**2))

# initial position on-axis (cartesian method) or off-axis (spherical polars)

z0 = R1 if np.isnan(root[0]) else root[0]
r0 = np.array([0, 0, z0]) if args.cart else np.array([1e-6, 2e-6, z0])

rng = np.random.default_rng(seed=args.seed) # initialise RNG

# Implement adaptive time step Brownian dynamics from
# Sammüller and Schmidt, J. Chem. Phys. 155, 134107 (2021)
# https://doi.org/10.1063/5.0062396
# ALGORITHM 2: embedded Heun–Euler trial step.
# Rejection Sampling with Memory (RSwM3) algorithm.
# Equation numbers below refer to this paper.

sqrt2Dp = np.sqrt(2*Dp) # for convenience

εabs, εrel = eval(f'{args.eps}') # relative and absolute errors
qmin, qmax = eval(f'{args.q_lims}') # bounds for adaptation factor

def heun_euler_trial_step(r, Δt, R):
    u = drift(r)
    Δrbar = u*Δt + sqrt2Dp*R # eq 8
    ubar = drift(r + Δrbar)
    Δr = 0.5*(u + ubar)*Δt + sqrt2Dp*R # eq 9
    return Δr, Δrbar

def adaptation_factor(Δr, Δrbar):
    E = l2norm(Δrbar - Δr) # eq 10
    τ = εabs + εrel * l2norm(Δr) # eq 11
    normE = E / τ # eq 12
    q = (1 / (2*normE))**2 if normE > 0 else qmax # eq 16 ; nominal adaptation factor
    return min(qmax, max(qmin, q)) # eq 17 ; bounded adaptation factor

def trial_step_details(traj, block, step, t, r, Δt, q, status): # returns a tuple capturing progress
    return (traj, block, step, t, r[0], r[1], r[2], l2norm(r), Δt, q, status)

progress = [] # used to accumulate progress data if requested
results = [] # used to capture final step number, elapsed time, mean square displacement

for block in range(args.nblock):
    for traj in range(args.ntraj):

        r, t, Δt = r0.copy(), 0.0, Δt_init # initial position, time, time step
        future_stack, using_stack = [], [] # use python lists as stacks
        R = rng.normal(0, np.sqrt(Δt), 3) # initial normally-distributed unscaled step size
        using_stack.append((Δt, R)) # initialise using stack per appendix C of paper
        reached_final_time = False
        ntrial, nsuccess = 0, 0 # keep track of number of attempted and successful trial steps

        if args.verbose > 2 or args.show:
            progress.append(trial_step_details(traj, block, 0, t, r, Δt, 1, 'initial'))

        for step in range(args.maxsteps):

            ntrial = ntrial + 1 # keep track of the number of attempted steps

            Δr, Δrbar = heun_euler_trial_step(r, Δt, R)
            q = adaptation_factor(Δr, Δrbar)

            if args.verbose > 3 or args.show:
                status = 'accept' if q > 1 else 'reject'
                progress.append(trial_step_details(traj, block, step, t, r, Δt, q, f'({status}) intermediate'))

            # The following is taken almost verbatim from the paper.
            # A stopping criterion has been added when a final time is reached.

            if q < 1: # reject the trial step
                reached_final_time = False # the step was rejected so won't reach final time this time
                Δts, Rs = 0.0, np.zeros(3)
                while using_stack: # the stack is not empty
                    Δtu, Ru = using_stack.pop()
                    Δts = Δts + Δtu ; Rs = Rs + Ru
                    if Δts < (1-q)*Δt:
                        future_stack.append((Δtu, Ru))
                    else:
                        ΔtM = Δts - (1-q)*Δt
                        qM = ΔtM / Δtu
                        Rbridge = qM*Ru + rng.normal(0, np.sqrt((1-qM)*qM*Δtu), 3)
                        future_stack.append(((1-qM)*Δtu, Ru-Rbridge))
                        using_stack.append((qM*Δtu, Rbridge))
                        break
                Δt = q*Δt ; R = R - Rs + Rbridge
            else: # q > 1, accept the trial step
                nsuccess = nsuccess + 1 # keep track of number of accepted trial steps
                t = t + Δt ; r = r + Δr ; Δt = q*Δt # update the time, position, time step
                if reached_final_time: # quit here if final time is now reached
                    break
                if t + Δt > tf: # next step would take us past the final time ..
                    Δt = tf - t # .. use this smaller time step instead ..
                    reached_final_time = True # .. and flag caught immediately above on the next cycle
                in_use_stack = []
                Δts, R = 0.0, np.zeros(3)
                while future_stack: # the stack is not empty
                    Δtf, Rf = future_stack.pop()
                    if Δts + Δtf < Δt:
                        Δts = Δts + Δtf ; R = R + Rf
                        using_stack.append((Δtf, Rf))
                    else:
                        qM = (Δt - Δts) / Δtf
                        Rbridge = qM*Rf + rng.normal(0, np.sqrt((1-qM)*qM*Δtf), 3)
                        future_stack.append(((1-qM)*Δtf, Rf-Rbridge))
                        using_stack.append((qM*Δtf, Rbridge))
                        Δts = Δts + qM*Δtf ; R = R + Rbridge
                        break
                Δtgap = Δt - Δts
                if Δtgap > 0:
                    Rgap = rng.normal(0, np.sqrt(Δtgap), 3)
                    R = R + Rgap
                    using_stack.append((Δtgap, Rgap))

        if args.verbose > 2 or args.show:
            progress.append(trial_step_details(traj, block, step, t, r, Δt, q, 'final'))

        Δr2 = np.sum((r-r0)**2) # mean square displacement for the present trajectory
        results.append((traj, block, ntrial, nsuccess, t, Δr2, Δt)) # capture results

columns = ['traj', 'block', 'ntrial', 'nsuccess', 't', 'Δr2', 'Δt_final']
results = pd.DataFrame(results, columns=columns).set_index(['block', 'traj'])
results['Δt_mean'] = results.t / results.nsuccess
results['Δr'] = np.sqrt(results.Δr2)

if progress:
    columns = ['traj', 'block', 'step', 't', 'x', 'y', 'z', 'r', 'Δt', 'q', 'status']
    progress = pd.DataFrame(progress, columns=columns).set_index(['traj', 'step'])

if args.verbose > 1:
    pd.set_option('display.max_rows', None)
    print(results[['ntrial', 'nsuccess', 't', 'Δt_mean', 'Δt_final', 'Δr2', 'Δr']])
    
if args.verbose > 2:
    print(progress)

if args.verbose:
    print(results[['ntrial', 'nsuccess', 't', 'Δt_mean', 'Δt_final', 'Δr2', 'Δr']].mean())
    print('Q =', 1e-3*Q, 'pL/s')
    print('sqrt(α*Q*t) =', np.sqrt(α*Q*tf/(2*π**2*R1))) # estimate assuming deterministic advection
    print('sqrt(6 Dp t) =', np.sqrt(6*Dp*tf)) # estimate assuming pure Brownian motion
    rms = np.sqrt(results.Δr2.mean()) # root mean square (rms) displacement
    err = np.sqrt(results.Δr2.var() / args.ntraj) / (2*rms) # the error in the rms value
    print('rms Δr =', rms, '±', err, '(rc =', rc, ')')

if args.code:
    for block, traj in results.index:
        row = results.loc[block, traj]
        if not any(row.isna()):
            data = (k, Γ, Ds, Dp, R1, α, Q*1e-3, rc, tf,
                    row.ntrial, row.nsuccess, row.t, row.Δt_final, row.Δr2,
                    traj, block, args.ntraj, args.nblock, args.code)
            forms = ('%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g', '%g',
                     '%i', '%i', '%g', '%e', '%e',
                     '%i', '%i', '%i', '%i', '%s')
            print('\t'.join(forms) % data)

if args.show:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    if Q > 0:
        w = args.width
        x, z = np.mgrid[-w:w:100j, -w:w:100j]
        r = np.sqrt(x**2 + z**2)
        cosθ, sinθ = z/r, x/r # note that θ is the angle to the polar (z) direction
        ur = Q/(4*π*r**2) + Pbyη*cosθ/(4*π*r) - k*λ*Γ/(r*(r+k*λ))
        uθ = - Pbyη*sinθ/(8*π*r)
        ux = ur*sinθ + uθ*cosθ
        uz = ur*cosθ - uθ*sinθ
        ax.streamplot(z, x, uz, ux)
        if not np.any(np.isnan(root)):
            ax.scatter(root, [0, 0], [100, 100], color='r', marker='x')
    for traj in progress.index.levels[0]:
        ax.plot(progress.loc[traj].z, progress.loc[traj].x, 'xr-')
    plt.show()
