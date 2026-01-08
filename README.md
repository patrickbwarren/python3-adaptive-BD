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

Equation numbers in the code refer to the above paper.  A couple of
small adjustments have been made.  First, the logic around line 25 in
the listing ALGORITHM 2 in the paper has been reversed so the test
becomes
$`\Delta t_s + \Delta t_f > \Delta t`$
rather than
$`\Delta t_s + \Delta t_f < \Delta t`$.
The conditional blocks have also been swapped so there is
no effect on the algorithm, but this subtle change avoids runtime
warnings from spurious underflow errors when taking a square root in
generating the bridging random displacement
$`\mathbf{R}_\text{bridge}`$ at what was originally line 30.

Second, a feature was added to stop the simulation when a desired time
point is reached.  This is done by checking if the next trial time
step would take the simulation past the desired time point, and if so,
reducing the length of the time step so that the desired time point
would be exactly reached if the time step is accepted; if not
accepted, the simulation can proceed as usual, since a further reduction
cannot go past the desired time point.

The code is split into a module `adaptive_bd.py` which contains the
adaptive time step algorithm, and drivers for the following test cases:
* pure Brownian motion (free diffusion);
* Brownian motion in a linear drift field;
* Brownian motion in a harmonic trap (Ornstein-Uhlenbeck problem);
* bounded Brownian motion in a linear drift field (Chandrasekhar's sedimentation problem);
* Brownian motion in a diffusiophoretic trap (non-potential drift field).

The original code `orig_dp_trap.py` is retained for regression testing
from the initial commit.

The driver codes are designed to run as standalone python scripts, or
as batch jobs within the [HTCondor](https://htcondor.org/)
high-throughput computing environment.  Some supporting scripts are
provided for this.  For example
```bash
condor_submit condor.job args="-n 100 -b 100" \
seed=12345 exec=harmonic_trap.py name=htest njobs=50
```
followed by (when all the jobs have completed)
```bash
./cleanup.sh htest
```
submits 50 jobs, then consolidates the results into a single gzipped data file.

### Theory

#### Free diffusion and diffusion with linear drift

With diffusion coefficient $D$, starting from position $`z_0`$, and with
elapsed time $t$, the probability distribution function for the
trajectory end points is a Gaussian
```math
p(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(-\frac{(z-z_0)^2}{4 D t}\Bigr)\,.
```

In a linear drift field with drift speed $`u_z=-\gamma`$, the corresponding
expression is a displaced Gaussian
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
z=-\kappa z`$ drives particles with a drift speed $`u_z=\mu
f_z=-\mu\kappa z`$, where $\mu=\beta D$ is the mobility and $\beta$ is
inverse temperature in units of Boltzmann's constant.  Hence we
identify $k=\beta D\kappa$, and the long-time limit of the above
expression $p\sim\exp(-k z^2/2D)=\exp(-\beta\kappa x^2/2)$ is
Boltzmann-distributed, as expected.

#### Bounded linear drift

This is a model for the evolution of a sedimentation profile, where
gravity corresponds to the linear drift, and the base of the container
provides a barrier wall.  The solution is somewhat more involved than
the preceding problems, and can be found in Chandrasekhar's famous
[Rev Mod Phys](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.15.1)
article (1943).  The result is also given in 
[lecture notes](https://www.ks.uiuc.edu/Services/Class/PHYS550/LectureNotes.html) 
(chapter 4) from Klaus Schulten.  The solution comprises three pieces, as
$`p=p_1+p_2+p_3`$, where
```math
\begin{align}
&p_1(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(-\frac{(z-z_0+\gamma t)^2}{4 D t}\Bigr)\,,\\
&p_2(z, t)=\frac{1}{\sqrt{4\pi D t}}\>
\exp\Bigl(\frac{\gamma z_0}{D}-\frac{(z+z_0+\gamma t)^2}{4 D t}\Bigr)\,,\\
&p_3(z, t)=\frac{\gamma}{2D}\>\exp\Bigl(-\frac{\gamma z}{D}\Bigr)\>
\text{erfc}\Bigl(\frac{z+z_0-\gamma t}{\sqrt{4 D t}}\Bigr)\,.
\end{align}
```
One can show that the sum of these is normalised, 
$`\int_0^\infty\text{d}z\,p(z, t)=1`$ for all times.

At late times, $`p_1,\,p_2\to0`$ as the Gaussians drift further and
further into the physically inaccessible region $z<0$, and $`p_3\to
(\gamma/D)\, e^{-\gamma z/D}`$.  If we interpret the linear drift as
corresponding to a gravitational potential $U=mgz$, then the force
$`f_z=-\partial U/\partial z=-mg`$ and the drift speed $`u_z=-\mu mg`$
where $\mu=\beta D$ is the mobility as before. Hence we identify
$\gamma=\beta mgD$, and, as in the harmonic trap case, the late-stage
$`p_3\sim e^{-\gamma z/D}=e^{-\beta mgz}`$ is again
Boltzmann-distributed.

One can show that the mean position satisfies
```math
\begin{align}
&\langle z\rangle= \frac{1}{2}\Bigl[
\sqrt{\frac{4 Dt}{\pi}}\,
\exp\Bigl(-\frac{(z_0-\gamma t)^2}{4Dt}\Bigr)
+z_0+\frac{D}{\gamma}-\gamma t\\
&\qquad\qquad{}+\Bigl(z_0-\frac{D}{\gamma}-\gamma t\Bigr)
\text{erf}\Bigl(\frac{z_0-\gamma t}{\sqrt{4 Dt}}\Bigl)
-\frac{D}{\gamma}\,\exp\Bigl(\frac{\gamma z_0}{D}\Bigr)
\text{erfc}\Bigl(\frac{z_0+\gamma t}{\sqrt{4 Dt}}\Bigl)
\Bigr]\,.
\end{align}
```

#### Diffusiophoretic trap

This problem concerns the diffusiophoretic (DP) trapping of colloidal
particles and macromolecules.  
More to follow...

### Reflecting boundary

In the bounded linear drift problem, it seems quite difficult to
implement the effect of the wall, without introducing some bias in the
adaptive Brownian dynamics algorithm.  What works at least empirically
is to simulate in the full domain, with a 'reflected' drift speed,
$`u_z=-\gamma`$ for $z>0$ and $`u_z=+\gamma`$ for $z < 0$.  Then, at
the end one 'folds' the trajectories which end with $z<0$ back into
the $z>0$ half-space.

This works because the reflected solution is a solution of the
original problem from a reflected starting position.  Since the
Fokker-Planck equation is linear, the superposition of the original
and reflected solutions is also a solution, with starting positions at
$`z_0`$ and $`-z_0`$.  By symmetry, this superposition has zero flux
through the $z=0$ plane, and so keeping only that part with $z\ge 0$
and doubling it up for normalisation solves the original problem with
a reflecting wall at $z=0$.

This superposition trick can be embodied in the Brownian dynamics code
by reflecting the drift field so that $`u_z(z<0) = -u_z(z>0)`$,
keeping _all_ the trajectories, and reflecting those which end in
$z<0$.  The problem with this approach is that the drift field may be
discontinuous through $z=0$ ; this is so for the bounded linear drift
field problem for example, as indicated above.  In practice, perhaps
particularly with the adaptive time step methodology, this does not
seem to present much of a problem.

### Results

#### Free diffusion

The plots below show binned end points in the $x$, $y$, and $z$
directions, starting from the origin, for $2\times10^5$ trajectories,
with $D=1$ and after a time $t=600$.  The red line is the expected
distribution from the theory above.  The measured root mean square
displacement $`\sqrt{}\langle\Delta r^2\rangle = 59.94 \pm 0.06`$
should be compared to the exact result $`\sqrt{6 D t} = 60`$.  The
upper plot uses a linear scale whereas the lower plot shows the same
data with a log scale.  Results with linear drift are similar.

![Free diffusion end point distribution (linear scale)](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/ftest1.png)

![Free diffusion end point distribution (log scale)](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/ftest2.png)

#### Harmonic trap

The plots here show binned end points in the $z$ direction, for
$5\times10^5$ trajectories, with $D=1$, trapping parameter $`k=0.1`$,
starting position $`z_0=25`$, after a time $t=20$.  The red line is
the expected distribution from the theory above.  The measured mean
position $`\langle z\rangle = 3.381 \pm 0.005`$ should be compared to
the exact result $`z_0\,e^{-kt} \simeq 3.3834`$.  Likewise the
measured root mean square displacement from the mean position
$`\sqrt{}\langle\Delta z^2\rangle = 3.129 \pm 0.004`$ should be
compared to the exact result $`\sqrt{2 D s} \simeq 3.1332`$ where $`s
= (1-e^{-2k t})/2k \simeq 4.9084`$.  As before, the upper plot uses a
linear scale whereas the lower plot shows the same data with a log
scale.

![Harmonic trap end point distribution (linear scale)](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/htest1.png)

![Harmonic trap end point distribution (log scale)](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/htest2.png)

#### Bounded linear drift

This problem is implemented as indicated above, reflecting the linear
drift about $z=0$ and reflecting trajectories which end up with $z<0$.

The first plot shows binned end points in the $z$ direction, for
$5\times10^5$ trajectories, with $D = 1$, drift speed $\gamma = 0.2$,
starting position $`z_0=50`$, after a time $t = 120$.  The red line is
the expected distribution from the theory above.  The measured mean
position $`\langle z\rangle = 26.44 \pm 0.02`$ should be compared to
the result calculated by numerically integrating the distribution,
$`\langle z\rangle \simeq 26.4337`$.  Note that without the barrier,
the mean position would be $`z_0-\gamma t=26`$, so there is a subtle
slowing down as particles cannot get past the barrier at $z = 0$.

![Bounded linear drift end point distribution (t = 120)](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/btest1.png)

The second plot shows the situation at $t = 1200$, for $10^5$
trajectories, plotted on a log scale.  The expectation is for this to
be close to the steady state with $`p=(\gamma/D)\, e^{-\gamma z/D}`$
(red line).

![Bounded linear drift end point distribution (t = 1200)](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/btest2.png)

The final plot shows how the mean value $\langle z\rangle$ depends on
the time $t$.  The marked points are from the adaptive Brownian
dynamics simulations (each point is from $10^4$ trajectories, and the
errors are smaller than the symbols here).  The solid line is the
theoretical expression given above, the dashed line is the leading
behavior at $t\to 0$, namely $\langle z\rangle = z_0-\gamma t$, and
the dotted line is the asymptotic value at $t\to\infty$, namely
$\langle z\rangle\to D/\gamma$.

![Bounded linear drift mean position](https://github.com/patrickbwarren/python3-adaptive-BD/blob/main/figures/btest3.png)


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

This program is copyright &copy; 2025, 2026 Patrick B Warren (STFC).

### Contact

Send email to patrick.warren{at}stfc.ac.uk.

