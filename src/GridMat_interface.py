"""
Python interface to use GridMat-MD with dcd trajectories
Uses Robin M. Betz vmd python modules
Paulina Mueller, 07.2019
"""
from __future__ import division

import numpy as np
import os
import subprocess
import sys
import tempfile
import yaml

from dcd_trajectory import DcdFile
from vmd import atomsel, molecule

import logging
logging.basicConfig(filename='log.log', filemode='w', level=logging.INFO)


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

    # assign values
    try:
        logging.info("Input pdb file is %s" % params['pdb'])
        logging.info("Input dcd file is %s" % params['dcd'])
        logging.info("Input parameter file is %s" % params['perl_param'])
    except KeyError:
        print("Inputfile(s) missing. Check parameter file. Exiting...")
        sys.exit(0)

    # setting some defaults
    if 'resname' not in params:
        params['resname'] = 'DPPC'
    if 'atoms' not in params:
        params['atoms'] = 'P'
    if 'output' not in params:
        params['output'] = ["top_pbc", "bottom_pbc"]

    # initalise the trajectory
    properties, indices = initialise(params)

    # open outputs:
    outputs = open_outs(params['output'], params['dcd'].split('/')[-1])

    # loop over trajectories
    # initialise
    for fil in params['dcd']:
        dcd = DcdFile(fil, indices)
        if not dcd._file:
            logging.error("no dcd - aborting")
            sys.exit(1)
        dcd.parseHeader()

        # do the calculation
        grid_mat(dcd, outputs, params['perl_param'], properties)

    # clean up
    for key, outp in outputs.iteritems():
        outp.close()
        if os.path.exists("temp_%s.dat" % key):
            os.remove("temp_%s.dat" % key)


def grid_mat(dcd, outputs, param_file, properties):

    # begin looping over frames
    for i in range(dcd._n_csets):

        logging.info("Currently in frame "+str(i))
        # open output
        temp = tempfile.NamedTemporaryFile('w', delete=False)
        # get unitcell
        vectors = ",".join(([str(x/10) for x in dcd.nextUnitcell()[0:3]]))
        print(vectors)
        # get coordinates and write to tempfile
        coord = zip(properties, *[iter(np.ravel(dcd.nextCoordset()))]*3)
        for props, x, y, z in coord:
            temp.write("%s %s %s  " % props)
            temp.write("%s %s %s \n" % (x, y, z))

        temp.close()

        # call perl script
        pipe = subprocess.Popen(["perl", "GridMAT-dcd.pl",
                                 param_file, temp.name, vectors],
                                stdout=subprocess.PIPE)
        # catch output
        for line in pipe.stdout:
            logging.info(line)

        os.unlink(temp.name)

        # combine files
        for key, outp in outputs.iteritems():
            comb = open("temp_%s.dat" % key, 'r')
            outp.write(comb.read())
            comb.close()


def initialise(params):
    # open pdb
    try:
        id = molecule.load('pdb', params['pdb'])
    except RuntimeError:
        logging.error("Trouble with pdb... exiting")
        sys.exit(0)

    # get selections
    sels = []
    sels.append(atomsel("resname %s and name %s"
                        % (params['resname'], " ".join(params['atoms'])), id))
    if 'resname2' and 'atoms2' in params:
        sels.append(atomsel("resname2 %s and name %s"
                            % (params['resname'], " ".join(params['atoms2'])),
                            id))
    if 'protein' in params:
        sels.append(atomsel("protein", id))

    # get values
    indices = []
    atoms = []
    residues = []
    resids = []
    for sel in sels:
        indices.extend(sel.serial)
        atoms.extend(sel.name)
        residues.extend(sel.resname)
        resids.extend(sel.resid)

    # safe atom properties to write in temp file for perl script
    properties = zip(atoms, residues, resids)

    return (properties, indices)


def open_outs(output, name):
    file_arr = {}
    if "top_pbc" in output:
        file_arr["top_pbc"] = open("%s.top_thick.dat" % name, 'w')
    if "bottom_pbc" in output:
        file_arr["bottom_pbc"] = open("%s.bot_thick.dat" % name, 'w')
    if "average_pbc" in output:
        file_arr["average_pbc"] = open("%s.avg_thick.dat" % name, 'w')
    if "top_area" in output:
        file_arr["top_area"] = open("%s.top_area.dat" % name, 'w')
    if "bottom_area" in output:
        file_arr["bottom_area"] = open("%s.bot_area.dat" % name, 'w')
    return (file_arr)


if __name__ == "__main__":
    main()
