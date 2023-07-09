import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio


def BPF(fs, fc, Q):
    fc /= fs
    fc = np.tan(np.pi * fc) / (2 * np.pi)
    a = np.zeros(3)
    b = np.zeros(3)
    a[0] = 1 + 2 * np.pi * fc / Q + 4 * np.pi * np.pi * fc * fc
    a[1] = (8 * np.pi * np.pi * fc * fc - 2) / a[0]
    a[2] = (1 - 2 * np.pi * fc / Q + 4 * np.pi * np.pi * fc * fc) / a[0]
    b[0] = 2 * np.pi * fc / Q / a[0]
    b[1] = 0
    b[2] = -2 * np.pi * fc / Q / a[0]
    a[0] = 1
    return a, b


fs = 8000
duration = 1

f0 = 120
T = fs / f0

length_of_s = int(fs * duration)
s0 = np.zeros(length_of_s)

x = 0
for n in range(length_of_s):
    s0[n] = - 2 * x + 1
    delta = f0 / fs
    if 0 <= x and x < delta:
        t = x / delta
        d = -t * t + 2 * t - 1
        s0[n] += d
    elif 1 - delta < x and x <= 1:
        t = (x - 1) / delta
        d = t * t + 2 * t + 1
        s0[n] += d

    x += delta
    if x >= 1:
        x -= 1

s1 = np.zeros(length_of_s)
fc = 800
Q = 800 / 100
a, b = BPF(fs, fc, Q)
for n in range(length_of_s):
    for m in range(0, 3):
        if n - m >= 0:
            s1[n] += b[m] * s0[n - m]

    for m in range(1, 3):
        if n - m >= 0:
            s1[n] += -a[m] * s1[n - m]

s2 = s1

s1 = np.zeros(length_of_s)
fc = 1200
Q = 1200 / 100
a, b = BPF(fs, fc, Q)
for n in range(length_of_s):
    for m in range(0, 3):
        if n - m >= 0:
            s1[n] += b[m] * s0[n - m]

    for m in range(1, 3):
        if n - m >= 0:
            s1[n] += -a[m] * s1[n - m]

s2 += s1

for n in range(int(fs * 0.01)):
    s2[n] *= n / (fs * 0.01)
    s2[length_of_s - n - 1] *= n / (fs * 0.01)

master_volume = 0.9
s2 /= np.max(np.abs(s2))
s2 *= master_volume

for n in range(length_of_s):
    s2[n] = (s2[n] + 1) / 2 * 65536
    s2[n] = int(s2[n] + 0.5)
    if s2[n] > 65535:
        s2[n] = 65535
    elif s2[n] < 0:
        s2[n] = 0

    s2[n] -= 32768

wavfile.write('p7(output).wav', fs, s2.astype(np.int16))

Audio('p7(output).wav')
