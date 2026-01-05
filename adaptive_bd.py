#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Implement adaptive time step Brownian dynamics from
# Sammüller and Schmidt, J. Chem. Phys. 155, 134107 (2021)
# https://doi.org/10.1063/5.0062396
# ALGORITHM 2: embedded Heun–Euler trial step.
# Rejection Sampling with Memory (RSwM3) algorithm.
# Equation numbers below refer to this paper.

import numpy as np

def l2norm(r): # standard vector norm
    return np.sqrt(np.sum(r**2))

class Simulator:

    def __init__(self, seed=12345, eps_abs=0.05, eps_rel=0.05, q_min=0.001, q_max=1.2):
        self.εabs, self.εrel = eps_abs, eps_rel
        self.qmin, self.qmax = q_min, q_max
        self.zero = np.zeros(3) # for convenience
        self.rng = np.random.default_rng(seed=seed) # initialise RNG

    def drift(self): # default here is no drift field
        return self.zero

    def heun_euler_trial_step(self, r, Δt, R):
        u = self.drift(r)
        Δrbar = u*Δt + self.sqrt2Dp*R # eq 8
        ubar = drift(r + Δrbar)
        Δr = 0.5*(u + ubar)*Δt + sqrt2Dp*R # eq 9
        return Δr, Δrbar

    def adaptation_factor(self, Δr, Δrbar):
        E = l2norm(Δrbar - Δr) # eq 10
        τ = self.εabs + self.εrel * l2norm(Δr) # eq 11
        normE = E / τ # eq 12
        q = (1 / (2*normE))**2 if normE > 0 else qmax # eq 16 ; nominal adaptation factor
        return min(self.qmax, max(self.qmin, q)) # eq 17 ; bounded adaptation factor

    def run(self, r0, Δt_init=0.1, max_steps=100, t_final=10, Dp=1.0):
        self.sqrt2Dp = np.sqrt(2*Dp) # for convenience in Heun-Euler function
        r, t, Δt = r0.copy(), 0.0, Δt_init # initial position, time, time step
        future_stack, using_stack = [], [] # use python lists as stacks
        R = self.rng.normal(0, np.sqrt(Δt), 3) # initial normally-distributed unscaled step size
        using_stack.append((Δt, R)) # initialise using stack per appendix C of paper
        reached_final_time = False
        ntrial, nsuccess = 0, 0 # keep track of number of attempted and successful trial steps
        for step in range(max_steps):
            ntrial = ntrial + 1 # number of attempted steps
            Δr, Δrbar = heun_euler_trial_step(r, Δt, R, sqrt2Dp)
            q = adaptation_factor(Δr, Δrbar)
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
                        Rbridge = qM*Ru + self.rng.normal(0, np.sqrt((1-qM)*qM*Δtu), 3)
                        future_stack.append(((1-qM)*Δtu, Ru-Rbridge))
                        using_stack.append((qM*Δtu, Rbridge))
                        break
                Δt = q*Δt ; R = R - Rs + Rbridge
            else: # q > 1, accept the trial step
                nsuccess = nsuccess + 1 # number of accepted trial steps
                t = t + Δt ; r = r + Δr ; Δt = q*Δt # update the time, position, time step
                if reached_final_time: # quit here if final time is now reached
                    break
                if t + Δt > t_final: # next step would take us past the final time ..
                    Δt = t_final - t # .. use this smaller time step instead
                    reached_final_time = True # flag caught immediately above on the next cycle
                in_use_stack = []
                Δts, R = 0.0, np.zeros(3)
                while future_stack: # the stack is not empty
                    Δtf, Rf = future_stack.pop()
                    if Δts + Δtf < Δt:
                        Δts = Δts + Δtf ; R = R + Rf
                        using_stack.append((Δtf, Rf))
                    else:
                        qM = (Δt - Δts) / Δtf
                        Rbridge = qM*Rf + self.rng.normal(0, np.sqrt((1-qM)*qM*Δtf), 3)
                        future_stack.append(((1-qM)*Δtf, Rf-Rbridge))
                        using_stack.append((qM*Δtf, Rbridge))
                        Δts = Δts + qM*Δtf ; R = R + Rbridge
                        break
                Δtgap = Δt - Δts
                if Δtgap > 0:
                    Rgap = self.rng.normal(0, np.sqrt(Δtgap), 3)
                    R = R + Rgap
                    using_stack.append((Δtgap, Rgap))
        return r, t, Δt, ntrial, nsuccess # final position, time, time step, # trial steps, # successful steps

# That's it (or should be !)
