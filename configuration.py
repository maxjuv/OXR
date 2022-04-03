# !/Users/mpossovr/python_envs/py39/bin/python
import os
import sys
import getpass

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

print(os.system('python --version'))

# work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
work_dir = '/Volumes/VERBATIM_HD/Malo/OXR/'
print(work_dir)
# data_dir = work_dir + '/data/'
data_dir = f'/{work_dir}/data/'
scoring_dir = f'/{work_dir}/scoring/'
excel_dir = work_dir + '/excels/'
figure_dir = work_dir + '/pyFig/'
precompute_dir = work_dir + '/precompute/'

groups = ['Ox1r_dd','Ox2r_dd','Ox2r_d+','Ox1r_Ox2r_dd']
print('ONE GROUP HAS BEEN REMOVED')
# groups = ['Ox1r_dd','Ox1r_d+','Ox2r_dd','Ox2r_d+','Ox1r_Ox2r_dd']

if __name__ == '__main__':
    print(work_dir)
