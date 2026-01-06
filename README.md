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

With diffusion coefficient $D$, starting from position $`z_0`$, and with
elapsed time $t$, the probability distribution function for the
trajectory end points is
```math
p(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(-\frac{(z-z_0)^2}{4 D t}\Bigr)\,.
```

#### Linear drift

In a linear drift field with drift speed $`u_z=-\gamma`$, the corresponding
expression is
```math
p(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(-\frac{(z-z_0+\gamma t)^2}{4 D t}\Bigr)\,.
```
#### Harmonic trap

This is the well-known 
[Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) 
process, whose solution can be found online.  With drift field $`u_z=-k z`$, 
the distribution is
```math
p(z, t)=\frac{1}{\sqrt{4\pi D s}}\>
\exp\Bigl(-\frac{(z-z_0 e^{-k t})^2}{4 D s}\Bigr)\,,
```
where the pseudo-time variable $`s=(1-e^{-2k t})/2k`$.

Note that the solution remains Gaussian at all times, and 'forgets'
the initial position with decay constant $k$ (which has units of
inverse time). The pseudo-time crosses over from $s=t$ at $k t\ll 1$
to the constant value $`s=(2k)^{-1}`$ for $k t\gg 1$.  

The drift field corresponds to motion in a harmonic trap potential
$U=\kappa z^2/2$.  The corresponding force $`f_z=-\partial U/\partial
z=-\kappa z`$ drives particles with a drift speed 
$`u_z=\mu f_z=-\mu\kappa z`$, where $\mu=\beta D$ is the mobility and
$\beta$ is inverse temperature in units of Boltzmann's constant.
Hence we identify $k=\beta D\kappa$, and the long-time limit of the
above expression $p\sim\exp(-k z^2/2D)=\exp(-\beta\kappa x^2/2)$; this
is Boltzmann-distributed, as expected.

#### Bounded linear drift

This is a model for the evolution of a sedimentation profile, where
gravity corresponds to the linear drift, and the base of the contained
provides a barrier wall.  The solution is somewhat more involved than
the preceding problems, and can be found in Chandrasekhar's famous
[Rev Mod Phys](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.15.1)
article (1943).  The result is also given in 
[lecture notes](https://www.ks.uiuc.edu/Services/Class/PHYS550/LectureNotes.html) 
(chapter 4) from Klaus Schulten.  The solution comprises three pieces, as
```math
p=p_1+p_2+p_3\,,
```
where the three pieces are
```math
p_1(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(-\frac{(z-z_0+\gamma t)^2}{4 D t}\Bigr)\,,
```
```math
p_2(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(\frac{\gamma z_0}{D}-\frac{(z+z_0+\gamma t)^2}{4 D t}\Bigr)\,,
```
```math
p_3(z, t)=\frac{\gamma e^{-\gamma z/D}}{2D}\>
\text{erfc}\Bigl(\frac{z+z_0-\gamma t}{\sqrt{4 D t}}\Bigr)\,.
```
One can show that this is normalised, $`\int_0^\infty \text{d}z\,p(z, t)=1`$ for all times.

At late times, one has $`p_1\to0`$ and $`p_2\to0`$, as the 'centres'
of the Gaussians drift further and further to $z<0$, whereas the final
contribution $`p_3\to (\gamma/D)e^{-\gamma z/D}`$.  

If we interpret the linear drift as corresponding to a gravitational
potential $U=mgz$, then as in the harmonic trap, the drift speed
$u_z=-\mu mg$, where $\mu=\beta D$ is the mobility as before. Hence we
identify $\gamma=\beta mgD$, and the late-stage steady state
$`e^{-\gamma z/D}=e^{-\beta mgz}`$ is the expected Boltzmann distribution.


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

