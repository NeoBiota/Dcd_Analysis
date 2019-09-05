"""
extract the lipidheads per area values from the GridMat logfile
"""

import numpy as np
import re
import sys


def main():

    try:
        log = open(sys.argv[1], 'r')
        out = open(sys.argv[3]+'.dat', 'w')
    except IndexError:
        print("Couldn't open an input or argument missing- Exiting...")
        sys.exit(1)
    except ValueError:
        print("Couldn't open an input or argument missing - Exiting...")
        sys.exit(1)

    if sys.argv[2] == '0':
        lpa_top = []
        lpa_bot = []
        for line in log:
            if "The average area per lipid in the top" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                lpa_top.append([num])
            if "The average area per lipid in the bot" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                lpa_bot.append([num])
        if len(lpa_bot) != len(lpa_top):
            print("incomplete logfile detected - aborting...")
            sys.exit(1)
        np.savetxt(out, np.hstack([lpa_bot, lpa_top]),
                   fmt='%1.4f'+' %1.4f')

    if sys.argv[2] == '1':
        lpa_top = []
        lpa_bot = []
        for line in log:
            if "The new area per lipid in the top" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                lpa_top.append([num])
            if "The new area per lipid in the bot" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                lpa_bot.append([num])
        if len(lpa_bot) != len(lpa_top):
            print("incomplete logfile detected - aborting...")
            sys.exit(1)
        np.savetxt(out, np.hstack([lpa_bot, lpa_top]),
                   fmt='%1.4f'+' %1.4f')

    if sys.argv[2] == '2':
        lpa_top = []
        lpa_bot = []
        nlpa_top = []
        nlpa_bot = []
        for line in log:
            if "The average area per lipid in the top" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                lpa_top.append([num])
            if "The average area per lipid in the bot" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                lpa_bot.append([num])
            if "The new area per lipid in the top" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                nlpa_top.append([num])
            if "The new area per lipid in the bot" in line:
                num = float(re.findall(r'\d+\.?\d*', line)[0])
                nlpa_bot.append([num])
        if len(lpa_bot) != len(lpa_top) != len(nlpa_bot) != len(nlpa_top):
            print("incomplete logfile detected - aborting...")
            sys.exit(1)
        np.savetxt(out, np.hstack([lpa_bot, nlpa_bot, lpa_top, nlpa_top]),
                   fmt='%1.4f'+' %1.4f'+' %1.4f'+' %1.4f')


if __name__ == "__main__":
    main()
