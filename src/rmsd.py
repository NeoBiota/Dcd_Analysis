"""
methods to align coordinates and calculate the rmsd
after Prody - University of Pittsburgh
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # error does not affect behaviour


def get_align(ref, coord):
    """
    method to align coord with respect to ref
    """
    if not isinstance(ref, np.ndarray) and not isinstance(coord, np.ndarray):
        print("Value error, coordinate sets have to be arrays")
        return -1
    if ref.shape != coord.shape:
        print("Value error, inputs must be of the same shape")
        return -1
    if ref.shape[1] != 3:
        print("Value error, inputs must be coordinate arrays")
        return -1

    ref_com = ref.mean(0)
    coord_com = coord.mean(0)
    ref = ref - ref_com
    coord = coord - coord_com
    matrix = np.dot(coord.T, ref)

    U, s, Vh = np.linalg.svd(matrix)
    Id = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, np.sign(np.linalg.det(matrix))]])
    rotation = np.dot(Vh.T, np.dot(Id, U.T))
    translation = ref_com - np.dot(coord_com, rotation.T)

    return (rotation.T, translation)


def apply_align(coord, rotation, translation):
    return np.dot(coord, rotation) + translation


def calc_rmsd(ref, coords):
    """
    rmsd to ref per frame in coords
    """
    if not isinstance(ref, np.ndarray) and not isinstance(coords, np.ndarray):
        print("Value error, coordinate sets have to be arrays")
        return -1
    if ref.ndim != 2 or ref.shape[1] != 3:
        print('reference must have shape (n_atoms, 3)')
        return -1
    if ref.shape != coords.shape[-2:]:
        print('reference and target arrays must have the same '
              'number of atoms')
        return -1

    if coords.ndim == 2:
        return np.sqrt(((ref-coords) ** 2).sum() / ref.shape[0])
    else:
        rmsd = np.zeros(len(coords))
        for i, t in enumerate(coords):
            rmsd[i] = ((ref-t) ** 2).sum()
            return np.sqrt(rmsd / ref.shape[0])


def rmsd_per_frame(ref, coordsets, framenumbers=None):
    """
    aligns all coordsets to ref, then calculates rmsd
    """
    rmsd = np.zeros((len(coordsets), 2))
    if not framenumbers:
        for i, cor in enumerate(coordsets):
            aligned = apply_align(ref, *(get_align(ref, cor)))
            rmsd[i] = [calc_rmsd(ref, aligned), i]
    else:
        for i, cor in enumerate(coordsets):
            aligned = apply_align(ref, get_align(ref, cor))
            rmsd[i] = [calc_rmsd(ref, aligned), framenumbers[i]]

    return rmsd


def plot_rmsd(arr, title):
    """
    plotting the rmsd if arr is in format [rmsd, time]
    """
    plt.plot(arr[:, 1], arr[:, 0])
    plt.xlabel('time in ns')
    plt.ylabel('rmsd in Angstroem')
    plt.title(title)
    plt.show()


def rmsd_avg(arr):
    """
    calculating the root mean square deviation to the frame average in arr
    """
    if len(arr.shape) != 3 and arr.shape[2] == 3:
        print("input has wrong format! Must be array of shape ((frames, N, 3))")
        return 0
    ref = np.mean(arr.transpose(), 2).transpose()
    print(ref)

    return rmsd_per_frame(ref, arr)


def rmsf_per_atom(arr):
    """
    calculates the rmsf per atom in arr
    atoms are identiefied as cross section between coordinatesets
    """
    if len(arr.shape) != 3 and arr.shape[2] == 3:
        print("input has wrong format! Must be array of shape ((frames, N, 3))")
        return 0
    N = arr.shape[1]
    nframes = arr.shape[0]
    rmsf = np.zeros(N)
    total = np.zeros((N, 3))
    sqsum = np.zeros((N, 3))
    # msf: <x^2> - (<x>)^2
    for i in range(0, nframes):
        total += arr[i]
        sqsum += arr[i]**2
    rmsf = ((sqsum/nframes - (total/nframes)**2).sum(1))**0.5
    return rmsf
