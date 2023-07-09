import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio

fs = 8000
duration = 1

f0 = 600
T = fs / f0

length_of_s = int(fs * duration)
s = np.zeros(length_of_s)

x = 0
for n in range(length_of_s):
    s[n] = -2 * x + 1

    delta = f0 / fs

    x += delta
    if x >= 1:
        x -= 1

for n in range(length_of_s):
    s[n] = (s[n] + 1) / 2 * 65536
    s[n] = int(s[n] + 0.5)
    if s[n] > 65535:
        s[n] = 65535
    elif s[n] < 0:
        s[n] = 0

    s[n] -= 32768

# 周波数特性のグラフを作成
freq = np.fft.fftfreq(length_of_s, d=1/fs)
fft = np.fft.fft(s)

plt.plot(freq, np.abs(fft)/np.max(np.abs(fft)))  # 縦軸の最大値を1にする
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
# plt.title('Frequency Spectrum')
plt.xlim(0, fs/2)
plt.grid(True)
# plt.show()
plt.savefig('kadai_4.png')

wavfile.write('kadai_4.wav', fs, s.astype(np.int16))

Audio('kadai_4.wav')
