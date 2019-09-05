"""
routine to calculate lipid order parameters from several dcd trajectory files
uses Robin M. Betz' vmd python modules
See yaml paramterfile for description of Input and Output options
Author: Paulina Mueller
August 2019
"""

import numpy as np
import yaml
import sys
from dcd_analysis import dcd_trajectory
from operator import itemgetter
from vmd import molecule, atomsel


def main():
    # check for parameter file
    if len(sys.argv) != 2:
        print("No parameter file given")
        sys.exit()
    # get parameters
    try:
        f = open(sys.argv[1])
    except IOError:
        print("Couldn't open parameter file")
        sys.exit()
    else:
        with f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    pdbfile = params['pdb']
    dcdfiles = params['dcd']
    lipid_dict = params['lipids']
    axis = params['axis']
    outname = params['outname']
    output = params['out']

    # read trajectories
    dcds = []
    for fil in dcdfiles:
        dcd = dcd_trajectory.DcdFile(fil)
        if not dcd._file:
            print('Problem with dcd file %s' % fil)
            continue
        dcd.parseHeader()
        dcds.append(dcd)

    # prepare the bonds that will be calculated
    lipid_bonds = make_bondlist(pdbfile, lipid_dict)

    # call calculation
    dcd_op_calc(dcds, lipid_bonds, lipid_dict, axis)

    # write output
    write_output(output, outname, lipid_dict)


def dcd_op_calc(dcds, lipid_bonds, lipid_dict, axis):
    """
    method to calculate the order parameters from bonds the dcd trajectory
    of a system specified by  the pdb for lipids specified

    averaging is done over frames
    """

    # get the total framenumber
    nframes = 0
    for dcd in dcds:
        nframes += dcd._n_csets
    # start looping
    for lipid, bonds, res_ids in lipid_bonds:
        # initalising the data container
        res_num = bonds.shape[1]
        # ops_per_frame: for every bond a number of frames with res_num entries
        # so every frame has to be read once per lipid
        ops_per_frame = np.zeros((bonds.shape[0],
                                  nframes, res_num))
        # initialise counter
        cnt = 0
        for dcd in dcds:
            for i in range(dcd._n_csets):
                coor = dcd.nextCoordset()
                for j, bond in enumerate(bonds):
                    # calc z vectors
                    vz = np.zeros((res_num, 3))
                    vz_norm2 = np.zeros(res_num)
                    # z-axis of molecular frame of reference is vector in direction of the bond
                    vz[:, 0] = itemgetter(bond[:, 0])(coor)[:, 0] - itemgetter(bond[:, 2])(coor)[:, 0]
                    vz[:, 1] = itemgetter(bond[:, 0])(coor)[:, 1] - itemgetter(bond[:, 2])(coor)[:, 1]
                    vz[:, 2] = itemgetter(bond[:, 0])(coor)[:, 2] - itemgetter(bond[:, 2])(coor)[:, 2]
                    # normalise
                    vz_norm2 = vz[:, 0]**2 + vz[:, 1]**2 + vz[:, 2]**2
                    vz[:, 0] = vz[:, 0]/np.sqrt(vz_norm2)
                    vz[:, 1] = vz[:, 1]/np.sqrt(vz_norm2)
                    vz[:, 2] = vz[:, 2]/np.sqrt(vz_norm2)

                    # calc x vectors
                    vtemp1 = np.zeros((res_num, 3))
                    vtemp2 = np.zeros((res_num, 3))
                    # first temporary vector
                    vtemp1[:, 0] = itemgetter(bond[:, 1])(coor)[:, 0] - itemgetter(bond[:, 2])(coor)[:, 0]
                    vtemp1[:, 1] = itemgetter(bond[:, 1])(coor)[:, 1] - itemgetter(bond[:, 2])(coor)[:, 1]
                    vtemp1[:, 2] = itemgetter(bond[:, 1])(coor)[:, 2] - itemgetter(bond[:, 2])(coor)[:, 2]
                    # second temporary vector
                    vtemp2[:, 0] = itemgetter(bond[:, 1])(coor)[:, 0] - itemgetter(bond[:, 0])(coor)[:, 0]
                    vtemp2[:, 1] = itemgetter(bond[:, 1])(coor)[:, 1] - itemgetter(bond[:, 0])(coor)[:, 1]
                    vtemp2[:, 2] = itemgetter(bond[:, 1])(coor)[:, 2] - itemgetter(bond[:, 0])(coor)[:, 2]
                    # product them
                    vx = np.cross(vtemp1, vtemp2)
                    # normalise
                    vx_norm2 = np.zeros(res_num)
                    vx_norm2 = vx[:, 0]**2 + vx[:, 1]**2 + vx[:, 2]**2
                    vx[:, 0] = vx[:, 0]/np.sqrt(vx_norm2)
                    vx[:, 1] = vx[:, 1]/np.sqrt(vx_norm2)
                    vx[:, 2] = vx[:, 2]/np.sqrt(vx_norm2)

                    # calc y
                    vy = np.cross(vz, vx)
                    # normalise
                    vy_norm2 = np.zeros(res_num)
                    vy_norm2 = vy[:, 0]**2 + vy[:, 1]**2 + vy[:, 2]**2
                    vy[:, 0] = vy[:, 0]/np.sqrt(vy_norm2)
                    vy[:, 1] = vy[:, 1]/np.sqrt(vy_norm2)
                    vy[:, 2] = vy[:, 2]/np.sqrt(vy_norm2)

                    # calculate the order parameter
                    ops_per_frame[j][i+cnt] = (-1)*(
                                            (0.6667)*(0.5*(3*(vx[:, axis]**2) - 1)) +
                                            (0.333)*(0.5*(3*(vy[:, axis])**2) - 1))

        # calculating the average over all the frames
        # ops_avg_per_frame: for every bond one number per residue (per avg_frames)
        ops_avg_per_frame = np.array([col for col in np.average(ops_per_frame, 1)])

        # increment counter
        cnt += dcd._n_csets

        # make lipid_dict holding the results of the calculation
        # dictionary is mutable
        lipid_dict[lipid] = (ops_avg_per_frame, res_ids)


def make_bondlist(pdbfile, lipid_dict):
    """
    returns bondlist pairs of atom serial (indices in dcd file) per res
    """
    try:
        id = molecule.load('pdb', pdbfile)
    except RuntimeError:
        print("Not able to read pdb file - aborting")
        sys.exit()
    # list of tuples of name and bond index array
    index_dict = []
    for lipid_name, lipid_param in lipid_dict.items():
        # lipid_param[0] is specified selectiontext for membrane components
        res_ids = set(atomsel(lipid_param[0], id).resid)
        # for each bond coordinates for 3 atoms, for every residue
        index_bond_list = np.zeros((len(lipid_param[1]), len(res_ids), 3), dtype=int)
        for i, bond in enumerate(lipid_param[1]):
            # sanity check for broken lipid residues
            atom1 = atomsel(lipid_param[0] +
                            " and name "+bond.split('-')[0]).serial
            atom2 = atomsel(lipid_param[0] +
                            " and name "+bond.split('-')[1]).serial
            atom3 = atomsel(lipid_param[0] +
                            " and name "+bond.split('-')[2]).serial
            if min(len(atom1), len(atom2), len(atom3)) == len(res_ids):
                index_bond_list[i] = zip(atom1, atom2, atom3)
            else:
                print("Warning - Incompletness detected")
                print("Have to skip bond %s, check bond or pdb" % bond)
                index_bond_list = np.delete(index_bond_list, i, 0)

        index_dict.append((lipid_name, index_bond_list, res_ids))
    return index_dict


def write_output(output, name, lipid_dict):
    """
    method to write calculations to files name+lipid
    """
    if "all" in output:
        # this tests if there are several values or just one average
        # output will be nrows=residues, ncolumns=bonds, block if avg_slive !=1
        if len(lipid_dict.values()[0][0].shape) == 2:
            for lipid, tup in lipid_dict.items():
                col1 = np.transpose(tup[0])
                # expand dims so [1, 2, 3] gets [[1], [2], [3]]
                col2 = np.expand_dims(np.array(list(tup[1]), dtype=int), 1)
                np.savetxt(name+lipid+'_all.dat', np.hstack([col2, col1]),
                           fmt='%i'+' %1.4f'*col1.shape[1])
        else:
            for lipid, tup in lipid_dict.items():
                col2 = np.expand_dims(np.array(list(tup[1]), dtype=int), 1)
                out = open(name+lipid+'_all.dat', 'wb')
                for i in range(0, tup[0].shape[0]):
                    col1 = np.transpose(tup[0][i])
                    np.savetxt(out, np.hstack([col2, col1]),
                               fmt='%i'+' %1.4f'*col1.shape[1])
                    out.write('\n \n \n')

    # averaging over residue
    # output will be nrows=bonds, ncolumns=according to avg_slice
    if "avg" in output:
        if len(lipid_dict.values()[0][0].shape) == 2:  # equals avg_slice == 1
            for lipid, tup in lipid_dict.items():
                np.savetxt(name+lipid+'_avg.dat', np.average(tup[0], 1), fmt='%1.4f')
        else:
            for lipid, tup in lipid_dict.items():
                np.savetxt(name+lipid+'_avg.dat', np.average(tup[0], 2), fmt='%1.4f')


if __name__ == "__main__":
    main()