import numpy as np
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


def calculate_correlation(original_audio, reconstructed_audio):
    correlation = np.corrcoef(
        original_audio[:len(reconstructed_audio)], reconstructed_audio)
    return correlation[0, 1]


def reconstruct_audio(spectrogram, fs, shift_size, num_components):
    # スペクトログラムの形状とパラメータを取得
    num_freq_bins, num_frames = spectrogram.shape
    frame_size = (num_freq_bins - 1) * 2

    # 再合成音声の初期化
    reconstructed_audio = np.zeros(num_frames * shift_size + frame_size)

    # サイン波の重ね合わせによる再合成
    for t in range(num_frames):
        frame_start = t * shift_size
        frame_end = frame_start + frame_size

        # 振幅スペクトルの大きい順にソートして上位 num_components を取得
        sorted_indices = np.argsort(spectrogram[:, t])[::-1][:num_components]

        for i, freq_bin in enumerate(sorted_indices):
            frequency = (freq_bin / (num_freq_bins - 1)) * (fs / 2)
            amplitude = spectrogram[freq_bin, t]

            time = np.arange(frame_start, frame_end)
            waveform = amplitude * \
                np.sin(2 * np.pi * frequency * time / fs)
            reconstructed_audio[frame_start:frame_end] += waveform

    return reconstructed_audio


# パラメータの設定
fs, s = wavfile.read('vocal.wav')
s = s.astype(np.float64)
length_of_s = len(s)

# ノイズの追加
np.random.seed(0)
for n in range(length_of_s):
    s[n] = s[n] + 32768 + (np.random.rand() - 0.5)
    s[n] = s[n] / 65536 * 2 - 1

# スペクトログラムの計算
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

# サイン波の再合成
num_components = 9  # 上位の振幅成分数を指定
reconstructed_audio = reconstruct_audio(S_abs, fs, shift_size, num_components)

# 元の音声と再構成音声の長さを一致させる
reconstructed_audio = reconstructed_audio[:len(s)]

# 相関係数を計算する
correlation = calculate_correlation(s, reconstructed_audio)
print("相関係数:", correlation)

# 音声の保存
wavfile.write('kadai_6.wav', fs,
              reconstructed_audio.astype(np.int16))

# 音声の再生
display(Audio('kadai_6.wav'))
