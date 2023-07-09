from IPython.display import display, Audio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

fs = 8000
duration = 1

length_of_s = int(fs * duration)
s = np.zeros(length_of_s)

for n in range(length_of_s):
    s[n] = 0.5 * np.sin(2 * np.pi * 1000 * n / fs)

for n in range(length_of_s):
    s[n] = (s[n] + 1) / 2 * 65536
    s[n] = int(s[n] + 0.5)
    if s[n] > 65535:
        s[n] = 65535
    elif s[n] < 0:
        s[n] = 0

    s[n] -= 32768

wavfile.write('p1(output).wav', fs, s.astype(np.int16))

Audio('p1(output).wav')
