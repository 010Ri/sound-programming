import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


# 音声ファイル読み込み
wav_filename = "o_1.wav"
rate, data = scipy.io.wavfile.read(wav_filename)


# （振幅）の配列を作成
data = data / 32768

##### 周波数成分を表示する #####
# 縦軸：dataを高速フーリエ変換する（時間領域から周波数領域に変換する）
fft_data = np.abs(np.fft.fft(data))
# 横軸：周波数の取得　　#np.fft.fftfreq(データ点数, サンプリング周期)
freqList = np.fft.fftfreq(data.shape[0], d=1.0/rate)


##### BPFのfcとQの値を算出する #####
# プロットされたデータからピーク周波数を検出
peak_index = np.argmax(fft_data)
peak_freq = freqList[peak_index]

# ピーク周波数を中心周波数とする
fc = abs(peak_freq)

# 帯域幅を算出するためにsqrt(1/2)のピーク振幅を計算
half_power = fft_data[peak_index] * np.sqrt(1/2)

# ピーク振幅を下回る周波数範囲を求める
left_index = right_index = peak_index

while left_index > 0 and fft_data[left_index] > half_power:
    left_index -= 1

while right_index < len(fft_data) - 1 and fft_data[right_index] > half_power:
    right_index += 1

# 対応する周波数を取得
left_freq = freqList[left_index]
right_freq = freqList[right_index]

# 帯域幅を算出
bpf_width = right_freq - left_freq

# クオリティファクタを算出
Q = fc / bpf_width

print("中心周波数(fc):", fc)
print("クオリティファクタ(Q):", Q)
