import numpy as np
import matplotlib.pyplot as plt

# 標本化周波数とサンプル数の設定
fs = 8000  # 標本化周波数8kHz
duration = 1.0  # サイン波の長さ1秒
samples = int(fs * duration)

# 時間軸を作成
t = np.arange(samples) / fs

# 4kHzのサイン波を生成
frequency = 4000
signal = np.sin(2 * np.pi * frequency * t)

# 周波数特性のグラフを作成
freq = np.fft.fftfreq(samples, d=1/fs)
fft = np.fft.fft(signal)

plt.plot(freq, np.abs(fft))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.xlim(0, fs/2)
plt.grid(True)
# plt.show()
plt.savefig('kadai_3.png')
