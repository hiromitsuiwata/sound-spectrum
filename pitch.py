from matplotlib.lines import Line2D
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from pydub.effects import low_pass_filter, high_pass_filter
import threading
from matplotlib.animation import FuncAnimation

SAMPLE_RATE = 44100
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32

class Graph:
    def __init__(self):
        self.raw_data = np.zeros(CHUNK)
        self.line = None
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = np.arange(0, CHUNK)
        ax1.set_title("Realtime audio wave")
        self.line, = ax1.plot(x, self.raw_data)
        self.ani = FuncAnimation(fig, self.__update_plot, cache_frame_data=False)

    def __update_plot(self, _):
        self.line.set_ydata(self.raw_data)
    
    def set_raw_data(self, raw_data):
        self.raw_data = raw_data

def callback(in_data, frame_count, time_info, status):
    raw_data = np.frombuffer(in_data, dtype=np.float32)
    graph.set_raw_data(raw_data)
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
                stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
        pass

if __name__ == "__main__":
    graph = Graph()

    processing_thread = threading.Thread(target=audio_processing_thread, daemon=True)
    processing_thread.start()

    try:
        # while True:
        #     pass
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted")
        exit(0)
