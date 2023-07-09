import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio

fs = 8000
duration = 0.004  # 4ミリ秒

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

# 時間軸を作成（ミリ秒単位）
t = np.arange(length_of_s) * 1000 / fs

# 波形のプロット
plt.plot(t, s/np.max(np.abs(s)))  # 縦軸の最大値を1にする
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
# plt.title('Sawtooth Wave')
plt.xlim(0, 4)  # 0から4ミリ秒まで表示
plt.grid(True)
plt.savefig('kadai_4_2.png')
