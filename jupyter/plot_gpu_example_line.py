#!/usr/bin/env python

def plot_example_line (filepath):
    with open (filepath, 'rt') as f:
        lines = f.readlines ()

    if lines[-1] == '\n':
        lines.pop () # pop trailing blank line

    import re
    data = [[float (x) for x in re.split (r'[,\s]+', line.strip ())] for line in lines]

    from matplotlib import pyplot as plt
    import numpy as np

    array = np.r_[data]

    plt.plot ([np.hypot (a,b) for (a,b) in array])
    from os import path
    plt.title (path.basename (filepath).split ('.')[0])
    plt.show ()

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ('--file',
                         default='../install/bin/GpuExample1/line.txt',
                         help='Path to the line text file.'
                        )
    args = parser.parse_args ()

    plot_example_line (args.file)
