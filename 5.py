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


fs, s = wavfile.read('vocal.wav')
s = s.astype(np.float64)
length_of_s = len(s)
np.random.seed(0)
for n in range(length_of_s):
    s[n] = s[n] + 32768 + (np.random.rand() - 0.5)
    s[n] = s[n] / 65536 * 2 - 1

N = 4096
shift_size = int(N / 2)
number_of_frame = int((length_of_s - (N - shift_size)) / shift_size)


x = np.zeros(N)
w = Hanning_window(N)
S_abs = np.zeros((int(N / 2 + 1), number_of_frame))
S_angle = np.zeros((int(N / 2 + 1), number_of_frame))


for frame in range(number_of_frame):
    offset = shift_size * frame
    for n in range(N):
        x[n] = s[offset + n] * w[n]

    X = np.fft.fft(x, N)
    X_abs = np.abs(X)
    X_angle = np.angle(X)

    for k in range(int(N / 2 + 1)):
        S_abs[k, frame] = X_abs[k]
        S_angle[k, frame] = X_angle[k]

plt.figure()
xmin = (N / 2) / fs
xmax = (shift_size * (number_of_frame - 1) + N / 2) / fs
ymin = 0
ymax = fs / 2
plt.imshow(S_abs, aspect='auto', cmap='Greys', origin='lower',
           vmin=0, vmax=1, extent=[xmin, xmax, ymin, ymax])
plt.axis([0, length_of_s / fs, 0, fs / 2])
plt.xlabel('time [s]')
plt.ylabel('frequency [Hz]')
plt.savefig('spectrogram.png')

plt.figure()
xmin = (N / 2) / fs
xmax = (shift_size * (number_of_frame - 1) + N / 2) / fs
ymin = 0
ymax = fs / 2
plt.imshow(S_angle, aspect='auto', cmap='Greys', origin='lower',
           vmin=-np.pi, vmax=np.pi, extent=[xmin, xmax, ymin, ymax])
plt.axis([0, length_of_s / fs, 0, fs / 2])
plt.xlabel('time [s]')
plt.ylabel('frequency [Hz]')
plt.savefig('spectrogram.png')

Y_abs = np.zeros(N)
Y_angle = np.zeros(N)
s1 = np.zeros(length_of_s)

for frame in range(number_of_frame):
    offset = shift_size * frame

    for k in range(int(N / 2)):
        Y_abs[k] = S_abs[k, frame]
        Y_angle[k] = S_angle[k, frame]

    for k in range(1, int(N / 2)):
        Y_abs[N - k] = Y_abs[k]
        Y_angle[N - k] = -Y_angle[k]

    Y = Y_abs * np.exp(1j * Y_angle)
    y = np.fft.ifft(Y, N)
    y = np.real(y)

    for n in range(N):
        s1[offset + n] += y[n]

for n in range(length_of_s):
    s1[n] = (s1[n] + 1) / 2 * 65536
    s1[n] = int(s1[n] + 0.5)
    if s1[n] > 65535:
        s1[n] = 65535
    elif s1[n] < 0:
        s1[n] = 0

    s1[n] -= 32768

wavfile.write('p6(output).wav', fs, s1.astype(np.int16))

Audio('p6(output).wav')
