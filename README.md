## Adaptive time step algorithm for Brownian dynamics (BD)

### Summary

The code implements the rejection sampling with memory (RSwM3)
adaptive time step Brownian dynamics (BD) algorithm ('ALGORITHM 2') from 
[Sammüller and Schmidt, J. Chem. Phys. 155, 134107 (2021)](https://doi.org/10.1063/5.0062396), 
using embedded Heun–Euler trial steps.  The implementation is for
Brownian motion of a single particle in three dimensions, and is
intended to be partly pedagogical, as well as providing a code
repository for some simulations related to particle trapping by
diffusiophoresis (DP).

Equation numbers in the code refer to the above paper.

The algorithm is split off into `adaptive_bd.py` and drivers for the
following test cases are given:
* pure Brownian motion (free diffusion);
* Brownian motion in a linear drift field;
* Brownian motion in a harmonic trap (Ornstein-Uhlenbeck problem);
* bounded Brownian motion in a linear drift field (Chandrasekhar's sedimentation problem);
* Brownian motion with a non-potential drift field (DP trapping).

The code `orig_lsjet_adaptive.py` is retained for regression testing
from the initial commit.  Itcontains both the algorithm and the
application to DP trapping in the vicinity of slender pipette
injecting a salt solution.

Codes are designed to run as standalone python scripts and as batch
jobs within the condor high-throughput computing environment.  Some
supporting scripts are provided for this.

### Mathematical Results

#### Free diffusion

With diffusion coefficient $D$, starting from position $`x_0`$, and with
elapsed time $t$, the probability distribution function for the
trajectory end points is
```math
p(z, t)=\frac{1}{\sqrt{4\pi D t}}\exp\Bigl(-\frac{(x-x_0)^2}{4 D t}\Bigr)
```

#### Linear drift

In a linear drift field with drift speed $-\gamma$, the corresponding
expression is
```math
p(z, t)=\frac{1}{\sqrt{4\pi D t}}\exp\Bigl(-\frac{(x-x_0+\gamma t)^2}{4 D t}\Bigr)
```
#### Harmonic trap

This is the well-known [Ornstein-Uhlenbeck
process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process),
whose solution can be found online.  With drift field $`u_x=-\kappa
x`$, the distribution is
```math
p(z, t)=\frac{1}{\sqrt{4\pi D s}}\exp\Bigl(-\frac{(x-x_0 e^{-\gamma t})^2}{4 D s}\Bigr)
```
where the pseudo-time variable is 
```math
s=\frac{1-e^{-2\kappa t}}{2\kappa}
```




### Copying

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see
<http://www.gnu.org/licenses/>.

### Copyright

This program is copyright &copy; 2025 Patrick B Warren (STFC).

### Contact

Send email to patrick.warren{at}stfc.ac.uk.

