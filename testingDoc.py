import numpy as np
from scipy.io import wavfile
import os
print(os.getcwd())

mywav = "sound01.wav"
print(os.path.dirname(mywav))
rate, data = wavfile.read(mywav)

print(data)
dataReduced = data[X[:, range(0, data.shape[0], 10)]]
print(dataReduced)
