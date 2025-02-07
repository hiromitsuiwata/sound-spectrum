import numpy as np
import pyaudio
import collections

# サンプリングレートとフレームサイズの設定
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1

# バッファを作成
audio_buffer = collections.deque(maxlen=FRAME_SIZE)

# YINアルゴリズムを用いたピッチ検出
def yin_pitch_detection(signal, threshold=0.1):
    tau_values = np.arange(1, len(signal))
    diff = np.zeros(len(signal))
    cumulative_mean_normalized_difference = np.zeros(len(signal))
    
    for tau in tau_values:
        diff[tau] = np.sum((signal[:-tau] - signal[tau:])**2)
    
    cumulative_mean_normalized_difference[1:] = diff[1:] / np.cumsum(diff[1:])
    
    tau_candidates = np.where(cumulative_mean_normalized_difference[1:] < threshold)[0] + 1
    if len(tau_candidates) == 0:
        return None
    best_tau = tau_candidates[0]
    
    return SAMPLE_RATE / best_tau if best_tau > 0 else None

# 音声入力のコールバック関数
def callback(in_data, frame_count, time_info, status):
    signal = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    audio_buffer.extend(signal)
    
    if len(audio_buffer) >= FRAME_SIZE:
        print(audio_buffer)
        print(status)
        print(time_info)
        print(frame_count)
        exit()
        frame_signal = np.array(audio_buffer)
        pitch = yin_pitch_detection(frame_signal)
        if pitch and 50 < pitch < 5000:  # 可聴範囲の制限
            print(f"Detected pitch: {pitch:.2f} Hz")
    
    return (in_data, pyaudio.paContinue)

# PyAudioの設定
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=HOP_SIZE,
                stream_callback=callback)

# ストリーム開始
stream.start_stream()
print("Listening... Press Ctrl+C to stop.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
