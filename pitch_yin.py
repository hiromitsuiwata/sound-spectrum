import numpy as np
import pyaudio
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pydub.effects import low_pass_filter, high_pass_filter
from scipy.fftpack import fft, ifft
from pydub import AudioSegment


# サンプリングレートとフレームサイズの設定
SAMPLE_RATE = 44100
CHUNK = 1024*4
FORMAT = pyaudio.paFloat32
CHANNELS = 1
signal = None
cumulative_mean_normalized_difference = None
diff = None
noise_profile = None
processed_audio = None

# YINアルゴリズムを用いたピッチ検出
def yin_pitch_detection(signal, threshold=0.1):
    global diff
    global cumulative_mean_normalized_difference

    tau_values = np.arange(1, len(signal))
    diff = np.zeros(len(signal))
    cumulative_mean_normalized_difference = np.zeros(len(signal))
    
    for tau in tau_values:
        diff[tau] = np.sum((signal[:-tau] - signal[tau:])**2)
    
    # W = len(signal)
    # max_tau = W // 2
    # for tau in range(max_tau + 1):
    #     diff[tau] = np.sum((signal[1:W-tau] - signal[1+tau:W])**2)
    
    cumulative_mean_normalized_difference[1:] = diff[1:] / np.cumsum(diff[1:])
    
    tau_candidates = np.where(cumulative_mean_normalized_difference[1:] < threshold)[0] + 1
    if len(tau_candidates) == 0:
        return None
    best_tau = tau_candidates[0]
    
    return SAMPLE_RATE / best_tau if best_tau > 0 else None

def apply_effects(audio):
    # ローパスフィルター
    audio = low_pass_filter(audio, 1000)
    # ハイパスフィルター
    audio = high_pass_filter(audio, 300)
    return audio

def noise_reduction(audio_data):
    global noise_profile
    
    # ノイズプロファイルの更新（最初の数フレーム）
    if noise_profile is None:
        noise_profile = np.abs(fft(audio_data))
        return audio_data
    
    # スペクトル減算
    spectrum = fft(audio_data)
    noise_reduction_factor = 0.9
    reduced_spectrum = spectrum - noise_reduction_factor * noise_profile
    reduced_spectrum = np.maximum(reduced_spectrum, 0)
    
    return np.real(ifft(reduced_spectrum))

# 音声入力のコールバック関数
def callback(in_data, frame_count, time_info, status):
    global signal
    global processed_audio
    signal = np.frombuffer(in_data, dtype=np.float32)

    # NumPy配列をpydubのAudioSegmentに変換
    audio = AudioSegment(
        signal.tobytes(), 
        frame_rate=SAMPLE_RATE,
        sample_width=signal.dtype.itemsize, 
        channels=CHANNELS
    )

    # エフェクトを適用
    processed_audio = apply_effects(audio)
    
    # ノイズリダクション
    processed_audio = noise_reduction(np.array(processed_audio.get_array_of_samples()).astype(np.float32))
    
    pitch = yin_pitch_detection(signal)
    print(f"Detected pitch: {pitch:.2f} Hz")
    
    return (in_data, pyaudio.paContinue)

# PyAudioの設定
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# 可視化の設定
fig, ax = plt.subplots()
x = np.arange(0, CHUNK)
ax.set_ylim(0, 10000000000000)
ax.set_yscale('log')
# ax.set_xlim(1000,1200)
line, = ax.plot(x, np.random.rand(CHUNK))


def update_plot(frame):
    line.set_ydata(signal)
    # line.set_ydata(diff)
    line.set_ydata(processed_audio)
    return line,

# アニメーションの設定（修正後）
ani = FuncAnimation(fig, update_plot, blit=True, cache_frame_data=False)


# ストリーム開始
stream.start_stream()
print("Listening... Press Ctrl+C to stop.")
try:
    # while True:
    #     pass
    plt.show()
except KeyboardInterrupt:
    print("Stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
