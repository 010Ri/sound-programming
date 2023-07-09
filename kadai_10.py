import soundfile as sf
import numpy as np
from scipy.fft import rfft, irfft
import scipy.signal as sg

# パラメータ
wav_name = "mono.wav"  # 読み込むWAVデータの名前
out_name = "kadai_10.wav"  # 出力するWAVデータの名前
window = "hann"     # 窓関数の種類
N = 1024            # FFT点数
r = 0.3            # スペクトル包絡の伸縮率

# WAVファイルを読み込む
x, fs = sf.read(wav_name)

# 短時間フーリエ変換(STFT)を行う X.shape=(n_bin, n_frame)
_, _, X = sg.stft(x, fs, window=window, nperseg=N)
X_phase = np.angle(X)   # 観測信号の位相
n_bin = X.shape[0]    # ビン数
n_frame = X.shape[1]    # フレーム数

# 各numpy配列を準備
ceps_l = np.zeros(N)       # 低次のケプストラム用の配列
ceps_h = np.zeros(N)       # 高次のケプストラム用の配列
H_tilde = np.zeros(n_bin)  # 伸縮後のスペクトル包絡用の配列
Y_abs = np.zeros(X.shape, dtype=np.float64)  # 出力信号の振幅用の配列
eps = np.finfo(np.float64).eps  # マシンイプシロン

# フレームごとにr培に伸縮したスペクトル包絡を求める
for i in range(n_frame):
    spec_log = np.log(np.abs(X[:, i])+eps)  # 対数変換
    ceps = irfft(spec_log)     # IFFTしてケプストラムを求める
    lifter = 72                # 低次のケプストラムを72点まで抽出
    ceps_l[0:lifter] = ceps[0:lifter]        # 低次の抽出（前半）
    ceps_l[N-lifter+1:] = ceps[N-lifter+1:]  # 低次の抽出（後半）
    ceps_h[lifter:N-lifter+1] = ceps[lifter:N-lifter+1]  # 高次の抽出
    H = np.real(rfft(ceps_l))  # FFTして実部だけ取り出す
    G = np.real(rfft(ceps_h))  # FFTして実部だけ取り出す
    # 対数スペクトル包絡をr倍に伸縮
    for k in range(n_bin):
        k2 = int(k/r)
        alpha = k/r - k2
        if k2 < n_bin-1:
            H_tilde[k] = (1-alpha)*H[k2] + alpha*H[k2+1]  # 線形補間
        else:  # k2がn_binを超えた場合
            H_tilde[k] = np.log(eps)  # -∞ に近いものを代入
    Y_abs[:, i] = np.exp(H_tilde+G)  # 振幅スペクトルを求める

# 位相と振幅でスペクトログラムを合成
Y = Y_abs * np.exp(X_phase)

# 逆短時間フーリエ変換(ISTFT)を行う
_, y = sg.istft(Y, fs=fs, window=window, nperseg=N)

# ファイルに書き込む
y = y/np.max(np.abs(y))  # ノーマライズ
sf.write(out_name, y, fs, subtype="PCM_16")
