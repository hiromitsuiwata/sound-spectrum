from matplotlib.lines import Line2D
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import threading
from matplotlib.animation import FuncAnimation
from numpy.fft import fft
from scipy.signal import windows

SAMPLE_RATE = 44100 # サンプリング周波数。1秒間に取得するフレーム数。チャンネル数が1の場合はデータ数と同じ。
CHUNK = pow(2,14) # シグナルを何フレーム単位で分割するか。FFTを行うので2のべき乗にする。
DT = 1.0 / SAMPLE_RATE # サンプリングの時間間隔
CHANNELS = 1 # チャンネル数
FORMAT = pyaudio.paFloat32

class Graph:
    def __init__(self):
        self.raw_data = np.zeros(CHUNK)
        self.windowed_data = np.zeros(CHUNK)
        self.complex = np.zeros(CHUNK)

        fig = plt.figure()
        ax1 = fig.add_subplot(321, title="Raw")
        ax2 = fig.add_subplot(323, title="Han")
        ax3 = fig.add_subplot(325, title="Windowed")
        ax4 = fig.add_subplot(322, title="Real part", ylim=(-2, 2))
        ax5 = fig.add_subplot(324, title="Imaginary part", ylim=(-2, 2))
        ax6 = fig.add_subplot(326, title="Spectrum", xscale="log", ylim=(0, 2))
        plt.tight_layout()

        x = np.arange(0, CHUNK)
        # 周波数軸の設定
        self.freqs = np.fft.fftfreq(CHUNK, DT)

        self.line1, = ax1.plot(x, np.zeros(CHUNK))
        self.line2, = ax2.plot(x, windows.hann(CHUNK))
        self.line3, = ax3.plot(x, np.zeros(CHUNK))
        self.line4, = ax4.plot(x, np.zeros(CHUNK))
        self.line5, = ax5.plot(x, np.zeros(CHUNK))
        self.line6, = ax6.plot(self.freqs[:len(self.freqs)//2], np.zeros(len(self.freqs)//2))
        
        self.ani = FuncAnimation(fig, self.__update_plot, cache_frame_data=False)

    def __update_plot(self, _):
        self.line1.set_ydata(self.raw_data)
        self.line3.set_ydata(self.windowed_data)
        self.line4.set_ydata(self.complex.real)
        self.line5.set_ydata(self.complex.imag)
        self.line6.set_ydata(np.abs(self.complex)[:len(self.freqs)//2])
    
    def set_raw_data(self, raw_data):
        self.raw_data = raw_data
    def set_windowed_data(self, windowed_data):
        self.windowed_data = windowed_data
    def set_complex_data(self, complex):
        self.complex = complex

class MyAudio:

    def callback(in_data, frame_count, time_info, status):
        raw_data = np.frombuffer(in_data, dtype=np.float32)
        graph.set_raw_data(raw_data)
        
        windowed_data = raw_data * windows.hann(CHUNK)
        graph.set_windowed_data(windowed_data)

        complex = fft(windowed_data)
        graph.set_complex_data(complex)
        
        return (in_data, pyaudio.paContinue)

    def audio_processing_thread():
        print("Start audio processing thread")
        # PyAudioの設定
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=MyAudio.callback)
        stream.start_stream()
        while stream.is_active():
            pass

    def start_audio():
        processing_thread = threading.Thread(target=MyAudio.audio_processing_thread, daemon=True)
        processing_thread.start()

if __name__ == "__main__":
    graph = Graph()
    MyAudio.start_audio()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted")
        exit(0)
