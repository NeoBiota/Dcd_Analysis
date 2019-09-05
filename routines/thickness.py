"""
routine to average and draw the membrane thickness
from the output of GridMAT
author: Paulina Mueller
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys


def main():
    # check for arguments
    if len(sys.argv) != 5:
        print("Argument missing")
        sys.exit()
    # get parameters
    try:
        fil = open(sys.argv[1])
    except IOError:
        print("Couldn't open input file")
        sys.exit()

    # save number of gridpoints
    grid_x = int(sys.argv[2])
    grid_y = int(sys.argv[3])
    # load data
    data = np.loadtxt(fil)
    fil.close()
    # check that file consists of full frames
    frames = len(data)/(grid_x*grid_y)

    # reshape
    try:
        data = data.reshape(frames, grid_x*grid_y)
        print(data)
    except ValueError:
        print("incomplete frames detected - exiting...")
        sys.exit(1)

    z_dat = np.mean(data, axis=0).reshape(grid_y, grid_x)
    x = range(1, grid_x+1)
    y = range(1, grid_y+1)

    plt.contourf(x, y, z_dat, cmap='RdGy')
    plt.colorbar()
    plt.title("Membrane thickness on grid")
    plt.show()


if __name__ == "__main__":
    main()
