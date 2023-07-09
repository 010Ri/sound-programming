import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio

fs, s = wavfile.read('mono.wav')
s = s.astype(np.float64)
length_of_s = len(s)
np.random.seed(0)
for n in range(length_of_s):
    s[n] = s[n] + 32768 + (np.random.rand() - 0.5)
    s[n] = s[n] / 65536 * 2 - 1

fs = 35280  # 44100 * 0.8

for n in range(length_of_s):
    s[n] = (s[n] + 1) / 2 * 65536
    s[n] = int(s[n] + 0.5)
    if s[n] > 65535:
        s[n] = 65535
    elif s[n] < 0:
        s[n] = 0

    s[n] -= 32768

wavfile.write('p8(output).wav', fs, s.astype(np.int16))

Audio('p8(output).wav')
