import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio


def Hanning_window(N):
    w = np.zeros(N)
    if N % 2 == 0:
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * n / N)
    else:
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / N)
    return w


fs, s = wavfile.read('p1(output).wav')
s = s.astype(np.float64)
length_of_s = len(s)
np.random.seed(0)
for n in range(length_of_s):
    s[n] = s[n] + 32768 + (np.random.rand() - 0.5)
    s[n] = s[n] / 65536 * 2 - 1

N = 512
shift_size = 64
number_of_frame = int((length_of_s - (N - shift_size)) / shift_size)

x = np.zeros(N)
w = Hanning_window(N)
S = np.zeros((int(N / 2 + 1), number_of_frame))

for frame in range(number_of_frame):
    offset = shift_size * frame
    for n in range(N):
        x[n] = s[offset + n] * w[n]

    X = np.fft.fft(x, N)
    X_abs = np.abs(X)

    for k in range(int(N / 2 + 1)):
        S[k, frame] = 20 * np.log10(X_abs[k])

plt.figure()
xmin = (N / 2) / fs
xmax = (shift_size * (number_of_frame - 1) + N / 2) / fs
ymin = 0
ymax = fs / 2
plt.imshow(S, aspect='auto', cmap='Greys', origin='lower',
           vmin=0, vmax=20, extent=[xmin, xmax, ymin, ymax])
plt.axis([0, length_of_s / fs, 0, fs / 2])
plt.xlabel('time [s]')
plt.ylabel('frequency [Hz]')
plt.savefig('spectrogram.png')
