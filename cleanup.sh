#!/usr/bin/bash

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

# Copyright (c) 2025, 2026 Patrick B Warren
# Email: patrickbwarren@gmail.com

# run with the name of the condor job

cat $1_*.out > $1.dat
gzip $1.dat

for ext in err log out
do
    rm -f $1_*.$ext
done

echo Created $1.dat.gz
