"""
pickle it - the class
"""

import cPickle as pickle
import numpy as np
import yaml
from vmd import molecule, atomsel


class Pickler:

    def __init__(self, struc_file):

        # open structure
        if struc_file.split('.')[-1] == 'pdb':
            try:
                self.id = molecule.load('pdb', struc_file)
            except RuntimeError:
                print("Trouble with structure file... exiting")
                self.struc_file = None
        elif struc_file.split('.')[-1] == 'psf':
            try:
                self.id = molecule.load('psf', struc_file)
            except RuntimeError:
                print("Trouble with structure file exiting")
                self.struc_file = None
        else:
            print("structure file unknown")
            self.struc_file = None

    def pickle_GridMat(self, yaml_file):
        """
        pickle atominformation to use vmd module free code in Don-Elias
        """

        # check if input is sufficient, else return
        try:
            parfile = open(yaml_file)
        except IOError:
            print("Couldn't open parameter file")
            return

        id = self.id
        # load params
        with parfile:
            params = yaml.load(parfile, Loader=yaml.FullLoader)

        # setting some defaults
        if 'resname' not in params:
            params['resname'] = 'DPPC'
        if 'atoms' not in params:
            params['atoms'] = 'P'
        if 'output' not in params:
            params['output'] = ["top_pbc", "bottom_pbc"]

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
            indices.extend(sel.index)
            atoms.extend(sel.name)
            residues.extend(sel.resname)
            resids.extend(sel.resid)

        # safe atom properties to write in temp file for perl script
        properties = zip(atoms, residues, resids)
        tup = (properties, indices)
        # pickle that
        pickle.dump(tup, open(params['pickle']+'.p', "wb"))

        return

    def pickle_Bondlist(self, yaml_file):
        """
        returns bondlist pairs of atom serial (indices in dcd file) per res
        """

        # check if input is sufficient, else return
        try:
            parfile = open(yaml_file)
        except IOError:
            print("Couldn't open parameter file")
            return

        # load params
        with parfile:
            params = yaml.load(parfile, Loader=yaml.FullLoader)
        index_dict = []

        # loop over lipids
        for lipid_name, lipid_param in params['lipids'].items():
            # lipid_param[0] is specified selectiontext for membrane components
            res_ids = set(atomsel(lipid_param[0], self.id).resid)
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

            pickle.dump(index_dict, open(params['pickle']+'.p', "wb"))

    def pickle_indices(self, selections, name, tup=None):
        """
        pickle indices of selections as name
        """
        indices = []
        if not tup:
            for text in selections:
                indices.extend(atomsel(text, self.id).index)
        else:
            for text in selections:
                indices.append(atomsel(text, self.id).index)
        pickle.dump(indices, open(name+'.p', "wb"))
