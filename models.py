#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python code to implement pipette and hole models
# Warren and Sear 2025/2026

import numpy as np
import pandas as pd
from numpy import pi as π

# shorthand for units used in preparing the table of parameters

none, um, umpersec, um2persec, um3persec, pLpersec = '', 'µm', 'µm/s', 'µm²/s', 'µm³/s', 'pL/s'

def tabulate(names, values, units, model_name='GENERIC'): # converts these lists into a nicely formatted table
    table_d = {'name':pd.Series(names, dtype=str),
               'value':pd.Series(values, dtype=float),
               'unit':pd.Series(units, dtype=str)}
    table = pd.DataFrame(table_d).set_index('name')
    table.loc['MODEL', 'unit'] = model_name # expected to be overwritten by the specific model name
    table.value = table.value.apply(lambda x: round(x, 3))
    return table

def table_to_string(table): # convert the above table to a printable string
    pd.set_option('display.max_columns', None)
    return '\n'.join(table.to_string().split('\n')[2:]) # lose the first two lines

class Model:

    def __init__(self, injector): # initialise base parameters, units are um and seconds
        pipette = injector == 'pipette' # true or false
        pore = injector == 'pore' # -- ditto --
        self.drift = self.pipette_drift if pipette else self.pore_drift if pore else None
        self.refresh = self.pipette_refresh if pipette else self.pore_refresh if pore else None
        self.update() # has the effect of setting all the defaults, per the next function

    def update(self, Q=1e-3, Γ=150, k=200, Ds=1610, R1=1.0, α=0.3, rc=1.0): # add more here as required
        self.Q = 1e3*Q # injection rate, convert to um^3/sec
        self.Γ = Γ # drift coefficient, here for DNA in LiCl
        self.k = k # in um^3/sec ; note conversion 1 pL/sec = 10^3 um^3/sec
        self.Ds = Ds # for NaCl
        self.R1 = R1 # pipette radius, or pore radius
        self.α = α # from Secchi at al
        self.rc = rc # default cut off
        self.refresh() # need to (re)run this to make sure all derived quantities are calculated
        return self

    def generic_refresh(self): # (re)calculate common derived parameters
        self.λ = self.Q / (4*π*self.Ds) # definition
        self.kλ = self.k * self.λ
        self.kλΓ = self.k * self.λ * self.Γ
        self.ΓkbyDs = self.Γ * self.k / self.Ds

    def pipette_refresh(self): # (re)calculate derived quantities for pipette
        self.generic_refresh()
        self.rstar = π*self.R1/self.α # where stokeslet and radial outflow match
        self.v1 = self.Q / (π*self.R1**2) # flow speed (definition)
        self.Pbyη = self.α*self.R1*self.v1 # from Secchi et al, should also = Q / r*
        self.Pe = self.Pbyη / (4*π*self.Ds) # definition from Secchi et al
        self.Qcrit = 4*π*self.Ds*self.rstar/self.k*(np.sqrt(self.ΓkbyDs)-1)**2 # critical upper bound on Q
        # the quadratic for the roots is z^2 − (kΓbyD − kλ* − 1)z + kλ* = 0 where z is in units of r*
        b = self.ΓkbyDs - self.kλ/self.rstar - 1 # the equation is r^2 − br + c = 0
        Δ = b**2 - 4*self.kλ/self.rstar # discriminant of above
        if self.Q > 0 and Δ > 0 and b > 0: # condition for roots to exist
            z1 = 0.5*self.rstar*(b - np.sqrt(Δ))
            z2 = 0.5*self.rstar*(b + np.sqrt(Δ))
            self.fixed_points = np.array([z1, z2])
        else:
            self.fixed_points = None
        self.parameters = self.pipette_parameters() # a pandas dataframe
        self.info = table_to_string(self.parameters) # as a string

    def pipette_drift(self, rvec): # pipette model, drift field
        x, y, z = rvec[:] # z is normal distance = r cosθ
        ρ = np.sqrt(x**2 + y**2) # in-plane distance = r sinθ
        r = np.sqrt(ρ**2 + z**2) # radial distance from origin
        cosθ, sinθ = z/r, ρ/r # polar angle, as cos and sin
        ur = self.Q/(4*π*r**2) + self.Pbyη*cosθ/(4*π*r) - self.kλΓ/(r*(r+self.kλ))
        uθ = - self.Pbyη*sinθ/(8*π*r)
        uρ = ur*sinθ + uθ*cosθ # parallel ρ-component of velocity
        uz = ur*cosθ - uθ*sinθ # normal z-component of velocity
        ux, uy = uρ*x/ρ, uρ*y/ρ # resolved x, y components
        return np.zeros_like(rvec) if r < self.rc else np.array((ux, uy, uz))

    def pore_refresh(self): # (re)calculate derived quantities for pore
        self.generic_refresh()
        self.Qcrit = 4*π*self.Ds*self.R1/(3*self.k)*np.sqrt(self.ΓkbyDs*(self.ΓkbyDs-3))
        # the quadratic for the roots is (1/2)(ΓkbyD-3)z^2 − 3kλz + (1/2)c^2 ΓkbyD = 0
        Δ = 9*self.kλ**2 - self.R1**2*self.ΓkbyDs*(self.ΓkbyDs - 3) # the discriminant
        if self.Q > 0 and Δ > 0: # condition for roots to exist
            z1 = (3*self.kλ - np.sqrt(Δ)) / (self.ΓkbyDs - 3)
            z2 = (3*self.kλ + np.sqrt(Δ)) / (self.ΓkbyDs - 3)
            self.fixed_points = np.array([z1, z2])
        else:
            self.fixed_points = None
        self.parameters = self.pore_parameters() # a pandas dataframe
        self.info = table_to_string(self.parameters) # as a string

    def pore_drift(self, rvec): # pore model, drift field using Sampson flow field
        x, y, z = rvec[:] # z is normal distance = r cosθ
        ρ = np.sqrt(x**2 + y**2) # in-plane distance = r sinθ
        r = np.sqrt(ρ**2 + z**2) # radial distance from origin
        cosθ, sinθ = z/r, ρ/r # polar angle, as cos and sin
        R1 = np.sqrt((ρ - self.R1)**2 + z**2) # used in the construction ..
        R2 = np.sqrt((ρ + self.R1)**2 + z**2) # .. of the oblate spheroidal coords ..
        λ = np.sqrt((R1 + R2)**2/(4*self.R1**2) - 1) # .. which are ..
        ζ = np.sqrt(1 - (R2 - R1)**2/(4*self.R1**2)) # .. these two quantities
        prefac = 3*self.Q/(2*π*self.R1**2) # prefac for Sampson flow field
        vρ = prefac * λ*ζ**2 / (λ**2 + ζ**2) * np.sqrt((1-ζ**2) / (1+λ**2))
        vz = prefac * ζ**3 / (λ**2 + ζ**2) # this and the above are the flow field components 
        fac = - 2*self.kλΓ/(r*(r+2*self.kλ)) if r > self.rc else 0 # note the extra factors of '2'
        uρ, uz = vρ + fac*sinθ, vz + fac*cosθ # resolved radial and axial components
        ux, uy = uρ*x/ρ, uρ*y/ρ # final pieces of cartesian components
        return np.array((ux, uy, uz))

    # Below here, to do with reporting parameters for information

    def common_parameters(self): # assemble lists of parameters common to both models
        names = ['MODEL', 'Q', 'Γ', 'k', 'Ds', 'R1', 'rc', 'λ', 'kλ', 'Γk/Ds']
        values = [np.inf, 1e-3*self.Q, self.Γ, self.k, self.Ds, self.R1, self.rc, self.λ, self.kλ, self.ΓkbyDs]
        units = ['GENERIC', pLpersec, um2persec, none, um2persec, um, um, um, um, none]
        return names, values, units

    def add_fixed_points(self, names, values, units):
        if self.fixed_points is not None:
            names.extend(['z1', 'z2'])
            values.extend(list(self.fixed_points))
            units.extend([um, um])
        return names, values, units

    def pipette_parameters(self): # add lists of parameters peculiar to pipette model
        names, values, units = self.common_parameters()
        v1, v1_units = (self.v1, umpersec) if self.v1 < 1e3 else (1e-3*self.v1, 'mm/s')
        names.extend(['Qcrit', 'α', 'r*', 'v1', 'P/η', 'Pe', 'λ*', 'kλ*'])
        values.extend([1e-3*self.Qcrit, self.α, self.rstar, v1, self.Pbyη, self.Pe,
                       self.λ/self.rstar, self.kλ/self.rstar])
        units.extend([pLpersec, none, um, v1_units, um2persec, none, none, none])
        names, values, units = self.add_fixed_points(names, values, units) 
        return tabulate(names, values, units, model_name='PIPETTE')

    def pore_parameters(self): # add lists of parameters peculiar to pore model
        names, values, units = self.common_parameters()
        names.extend(['Qcrit'])
        values.extend([1e-3*self.Qcrit])
        units.extend([pLpersec])
        names, values, units = self.add_fixed_points(names, values, units) 
        return tabulate(names, values, units, model_name='PORE')
