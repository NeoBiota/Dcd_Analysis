"""
Defines a class for reading and handeling dcd trajectory files
endianness is assumed to be small
"""
import numpy as np
from struct import unpack
from os.path import getsize


class DcdFile(object):

    """
    Instanciated with a dcd file (unformatted fortran binary)
    to read the coordinates and process it
    """

    def __init__(self, filename, indices=None):
        """
        Open dcd trajectory filename (string)
        to read the coordinate sets for atoms in array indices
        default reads all atoms
        """

        if not isinstance(filename, str):
            raise TypeError("filename argument must be a string")

        self._filename = filename
        self._file = None

        try:
            self._file = open(filename, 'rb')
        except IOError:
            print("Could not open file")
            return None

    # variables contained in the dcd file
        self._timestep = 1
        self._first_ts = 0
        self._framefreq = 1
        self._n_fixed = 0  # fixed atoms
        self._n_atoms = 0
        self._n_floats = 0  # number of floats in on coord set
        self._n_csets = 0  # number of coord sets, i.e. number of frames

    # variables calculated from parsing the dcd
        self._bytes_per_frame = None
        self._first_byte = None

    # variables that are specified by usage
        self._indices = indices  # indices of selected atoms

    # process variables
        self.nfi = -1  # number of frames

    def __del__(self):

        if DcdFile:
            if self._file:
                self._file.close()

    def parseHeader(self):
        """Read the header information from a dcd file.
        Returns 0 on success, negative error code on failure.
        natoms set to number of atoms per frame
        n_csets set to number of frames in dcd file
        first_ts set to starting timestep of dcd file
        framefreq set to timesteps between dcd saves
        delta set to value of trajectory timestep
        nfixed set to number of fixed atoms
        """

        dcd = self._file
        # perform check of native and file endianess
        check = unpack(b'ii', dcd.read(8))  # COUNT 8
        if check[0] == 84 and \
           check[1] == unpack(b'i', b'CORD')[0]:
            print("Dcd check successful")
        else:
            print("Problems with the fileformat")
            return -1

        # read header bytes
        bits = dcd.read(80)  # COUNT 88
        header = unpack(b'i'*9 + b'f' + b'i'*10, bits)

        # Store the number of sets of coordinates (NSET)
        self._n_csets = header[0]
        # Store ISTART, the starting timestep
        self._first_ts = header[1]
        # Store NSAVC, the number of timesteps between dcd saves
        self._framefreq = header[2]
        # Store NAMNF, the number of fixed atoms
        self._n_fixed = header[8]
        # Read in the timestep, DELTA
        self._timestep = header[9]*48.88821  # time scaling to fs

        # Read the end and start bits of the blocks to access data correctly
        # Should be: (endsize of header, size of next block, ntitle)
        # ntitle: number of blocks of size 80 with title information
        buffer = unpack(b'iii', dcd.read(12))  # COUNT 100
        if buffer != (84, 164, 2):
            print("Problems with the fileformat {0}".format(dcd.tell()))
            return -1

        # print title and remarks
        print("".join(unpack(b'c'*160, dcd.read(160))))  # COUNT 260
        # end and start bits with sanity check
        if unpack(b'ii', dcd.read(8)) != (164, 4):  # COUNT 268
            print("Problems with the fileformat at {0}".format(dcd.tell()))
            return -1

        # read in atom number
        self._n_atoms = unpack(b'i', dcd.read(4))[0]  # COUNT 272
        self._n_floats = (self._n_atoms + 2) * 3

        # end bits
        if unpack(b'i', dcd.read(4))[0] != 4:  # COUNT 276
            print("Problems with the fileformat {0}".format(dcd.tell()))
            return -1

        # number of floats per frame
        self._bytes_per_frame = 56 + self._n_floats * 4
        # start of the coordinates
        self._first_byte = self._file.tell()  # Should equal 276
        # check if calculated and given number of frames are consistent
        n_csets = (getsize(self._filename) - self._first_byte
                   ) // self._bytes_per_frame
        if n_csets != self._n_csets:
            print('DCD header claims {0} frames, file size '
                  'indicates there are actually {1} frames.'
                  .format(self._n_csets, n_csets))
            # self._n_csets = n_csets
        # set frame counter
        self.nfi = 0

        return self.nfi

    def nextCoordset(self):
        """Returns next coordinate set."""

        if self._file.closed:
            raise ValueError('I/O operation on closed file')
        if self._n_csets == 0:
            print('Error: Trajectory empty or header not parsed')
            return -1
        if self.nfi < self._n_csets:
            # Skip extended system coordinates (unit cell data)
            self._file.seek(56, 1)
            if self._indices is None:
                return self._nextCoordset()
            else:
                return self._nextCoordset()[self._indices]

    def nextCoordinates(self, indices):
        """
        get next coordinates for only now (not in object) specified indices
        """
        if self._file.closed:
            raise ValueError('I/O operation on closed file')
        if self._n_csets == 0:
            print('Error: Trajectory empty or header not parsed')
            return -1
        if self.nfi < self._n_csets:
            # Skip extended system coordinates (unit cell data)
            self._file.seek(56, 1)
        return self._nextCoordset()[indices]

    def _nextCoordset(self):

        n_floats = self._n_floats
        n_atoms = self._n_atoms
        xyz = np.fromstring(self._file.read(4 * n_floats),
                            np.float32)
        if len(xyz) != n_floats:
            return None
        xyz = xyz.reshape((3, n_atoms+2)).T[1:-1, :]
        xyz = xyz.reshape((n_atoms, 3))
        self.nfi += 1

        return xyz

    def nextUnitcell(self):
        """
        Returns unit cell [x, y, z, cos(alpha), cos(beta), cos(gamma)]
        sets file to beginning of full frame coordinate set
        """
        self._file.read(4)
        unitcell = np.fromstring(self._file.read(48), dtype=np.float64)
        unitcell = unitcell[[0, 2, 5, 1, 3, 4]]
        self._file.read(4)
        self._file.seek(-56, 1)

        return unitcell

    def find_frame(self, frame):
        """
        moves reader to beginning of frame
        """
        if self._n_csets == 0:
            print('Error: Trajectory empty or header not parsed')
            return -1
        if not isinstance(frame, (int, long)):
            print('ValueError: Frame number must be an integer')
            return -1
        if frame > self._n_csets:
            print('ValueError: Frame number greater than total frames')
            return -1
        dist = frame - self.nfi
        self._file.seek(dist*self._bytes_per_frame, 1)
        self.nfi = frame

    def traj_to_array(self, frames=None):
        """
        read coordsets in frames in array. Default is reading all frames
        """
        if not self._indices:
            n_atoms = self._n_atoms
        else:
            n_atoms = len(self._indices)
        if not frames:
            n_csets = self._n_csets
            coords = np.zeros((n_csets, n_atoms, 3))
            self.find_frame(0)
            self.nfi = 0
            for i in range(n_csets):
                coords[i] = self.nextCoordset()

        else:
            n_csets = len(frames)
            coords = np.zeros((n_csets, n_atoms, 3))
            for i, fr in enumerate(frames):
                self.find_frame(fr)
                coords[i] = self.nextCoordset()

        return coords

    def mk_time(self, arr):
        """
        scales the framenumber to the system time
        """
        fac = self._timestep*self._framefreq/np.power(10, 6)
        out = [(i*fac)+self._first_ts/np.power(10, 6) for i in arr]
        return out
