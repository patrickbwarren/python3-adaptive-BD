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

In the initial commit, the code `lsjet_adaptive.py` contains both the
algorithm and an application to DP trapping in the vicinity of slender
pipette injecting a salt solution.

The intent is to split off the actual algorithm, and demonstrate its
use with some test cases:
* pure Brownian motion;
* Brownian motion in a linear drift field;
* Brownian motion in a harmonic trap (Ornstein-Uhlenbeck problem);
* Brownian motion in a linear drift field with a wall (Chandrasekhar's sedimentation problem);
* Brownian motion with a non-potential drift field (DP trapping).

More to come...

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

