with open ('install/bin/GpuExample1_line0.txt', 'rt') as f: lines = f.readlines ()

import re
data = [[float (x) for x in re.split (r'[,\s]+', line.strip ())] for line in lines]

from matplotlib import pyplot as plt
import numpy as np

array = np.r_[data]

plt.plot ([np.hypot (a,b) for (a,b) in array])
plt.show ()