#!/usr/bin/env python3

import argparse
from datetime import timedelta

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('logfiles', nargs='+', help='the name of the output and job files')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

total, count = 0, 0

for log_file in args.logfiles:
    with open(log_file) as f:
        for line in f:
            if 'Total Remote' in line:
                h, m, s = [int(s) for s in line.split(',')[0].split()[2].split(':')]
                t = 3600*h + 60*m + s
                tmin = min(t, tmin) if count > 0 else t
                tmax = max(t, tmax) if count > 0 else t
                total = total + t
                count = count + 1
                if args.verbose:
                    print(log_file + ':', line.strip())

result = f'{count} jobs: total run time {timedelta(seconds=total)}, ' \
    f'range = {timedelta(seconds=tmin)}--{timedelta(seconds=tmax)}, ' \
    f'mean run time {timedelta(seconds=int(total/count))}'

print(result)
