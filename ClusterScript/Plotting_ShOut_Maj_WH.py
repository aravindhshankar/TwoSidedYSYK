import sys
import os
if not os.path.exists('./Sources'):
	print("Error - Path to sources directory not found")
sys.path.insert(1,'./Sources')
import h5py
import matplotlib.pyplot as plt


import numpy as np
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *

path_to_outfile = './Outputs'
outfile = 'MajWH_2147118.h5'
savepath = os.path.join(path_to_outfile, outfile)

if not os.path.exists(savepath):
	raise(Exception("Output file not found"))


data = h52dict(savepath, verbose = True)
print(data.keys())

fig, (ax1,ax2) = plt.subplots(2)

ax1.plot(data['omega'], data['rhoLL'])
ax1.set_xrange(-0.5,0.5)

ax2.plot(data['omega'], data['rhoLR'])
ax2.set_xrange(-0.5,0.5)



