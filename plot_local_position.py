import os
import numpy
from pathlib import Path
import yaml
from matplotlib import pyplot as plt

dir = '/media/mf/AUTODRIVING-4TB/UGM Baru/autoriving-oskarnatan/datasetx/2026-02-09_route00/meta/'
files = os.listdir(dir)
files.sort()

plt.figure()

for file in files:
    with open(dir + file, 'r') as curr_metafile:
        current_meta = yaml.safe_load(curr_metafile)
        print(current_meta["local_position_xyz"][0])
        plt.plot(current_meta["local_position_xyz"][0], current_meta["local_position_xyz"][1], 'o-', color='red')

plt.show()