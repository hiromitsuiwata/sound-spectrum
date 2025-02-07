from matplotlib.lines import Line2D
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from pydub.effects import low_pass_filter, high_pass_filter
import threading
from matplotlib.animation import FuncAnimation

SAMPLE_RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
line: Line2D = None
raw_data = np.zeros(CHUNK)
ani = None

   
def update_plot(i):
    global raw_data
    line.set_ydata(raw_data)

def callback(in_data, frame_count, time_info, status):
    # print("callback")
    global raw_data
    raw_data = np.frombuffer(in_data, dtype=np.float32)
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

def init_graph():
    global line, ani
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, CHUNK)
    line, = ax.plot(x, np.zeros(CHUNK))
    ani = FuncAnimation(fig, update_plot, save_count=10, cache_frame_data=False)

if __name__ == "__main__":
    init_graph()

    processing_thread = threading.Thread(target=audio_processing_thread, daemon=True)
    processing_thread.start()

    try:
        # while True:
        #     pass
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted")
        exit(0)
