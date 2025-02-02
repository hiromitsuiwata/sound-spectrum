import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# オーディオ設定
CHUNK = 1024  # フレームサイズ
FORMAT = pyaudio.paInt16  # 16bit PCM
CHANNELS = 1  # モノラル
RATE = 44100  # サンプリング周波数（44.1kHz）

# PyAudioの初期化
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# 周波数軸の計算（対数スケール用）
freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE)[:CHUNK // 2]  # 正の周波数成分のみ

# グラフの初期化
fig, ax = plt.subplots()
line, = ax.plot(freqs, np.zeros_like(freqs))  # 初期データ

ax.set_xscale("log")  # 横軸を対数スケール
ax.set_xlim(10, RATE // 2)  # 20Hz〜ナイキスト周波数まで
ax.set_ylim(-60, 150)  # dBスケールの範囲
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Real-time Audio Spectrum (Log Scale)")

# グラフ更新関数
def update(frame):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    fft_data = np.abs(np.fft.fft(data))[:CHUNK // 2]  # FFT計算（正の周波数成分のみ）
    
    # dBスケールに変換（log(0)対策で1e-10を加える）
    fft_db = 20 * np.log10(fft_data + 1e-10)
    
    line.set_ydata(fft_db)
    return line,

# アニメーションの開始
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.show()

# 終了処理
stream.stop_stream()
stream.close()
p.terminate()
