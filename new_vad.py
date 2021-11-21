import time
import traceback
import tkinter as tk
from threading import Thread
import queue
import torch
import torchaudio
import webrtcvad
import pyaudio
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)
from model import BCResNet

n_fft = 480
win_length = 480
hop_length = 160
n_mels = 40
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WINDOW_DURATION = 1.7
CHUNK = int((RATE * WINDOW_DURATION) // 10)

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
 
        
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, vad, frames):
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        change_label(is_speech)
        if is_speech:
            return True




def change_label(is_detected):
    global label
    if is_detected:
        label.configure(text='VAD: DETECTED', background="#67f0ae")
    else:
        label.configure(text='VAD: Not detected', background="#FF0000")



def main(anim_queue):
    p = pyaudio.PyAudio()

    spectrogrammer = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

    )

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    vad = webrtcvad.Vad(3)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BCResNet(2).to(device)
    #model = torch.nn.Sequential(torchvision.models.mobilenetv2.mobilenet_v2(), torch.nn.Linear(1000, 2)).to(device)
    model.eval()
    state_dict = torch.load("model_greeting100/model.pt",map_location=device)
    print(model.load_state_dict(state_dict))
    q = queue.Queue()
    window = np.zeros((int(RATE * WINDOW_DURATION),))

    @torch.no_grad()
    def detect_keyword(spec_queue,model):
        last_activation = 0
        active = False
        while True:
            try:
                spec = spec_queue.get(False)
            except queue.Empty:
                if time.time() - last_activation > 1:
                    kw_label.configure(text='Keyword: Not detected', background="#FF0000")
                continue
            spec_for_img = spec.clone().numpy()
            spec_for_img -= spec_for_img.min()
            spec_for_img /= spec_for_img.max()

            spec -= spec.max()
            # print(spec.shape)
            # print(spec.min())
            # print(spec.max())
            spec = spec.reshape(1,1,*tuple(spec.shape))
            # spec = convert_to_rgb(spec, device)
            res = torch.sigmoid(model(spec.to(device)))[0].item()
            #print(res)
            if res > 0.73:
                print("Здравствуйте!","res:",res)
                #last_activation = time.time()
                # if not active:
                #     active = True
                #     kw_label.configure(text='Keyword: DETECTED', background="#67f0ae")
            # if time.time() - last_activation > 1:
            #     kw_label.configure(text='Keyword: Not detected', background="#FF0000")
            #     active = False
            # print(time.time())
            # print(torch.sigmoid(res))
            # if res[1] > 0.99:
            #
            #


            anim_queue.put(spec_for_img)

    thread = Thread(target=detect_keyword,args=(q,model))
    thread.start()

    try:
        silent_frames = 0
        while True:
            #time.sleep(0.1)
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames = frame_generator(10, data, RATE)
            frames = list(frames)
            is_speech = vad_collector(RATE, vad, frames)
            # if is_speech:
                # if silent_frames > 0:
                #     window = np.zeros((int(RATE * WINDOW_DURATION),))
                # silent_frames = 0
            indata = np.frombuffer(data,np.int16)
            window = np.roll(window, -CHUNK, 0)
            window[-CHUNK:] = indata
            # else:
            #     silent_frames += 1
            #
            # if silent_frames < 2:
            torchdata = torch.from_numpy(window).float()
            q.put_nowait(torch.log(spectrogrammer(torchdata) + 1e-8))
            # else:
            #     window = np.zeros((int(RATE * WINDOW_DURATION),))


    except KeyboardInterrupt:
        print("exit")
    except Exception:
        traceback.print_exc()
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def anim_callback(frame):
    global anim_queue,spec_img
    while True:
        try:
            data = anim_queue.get_nowait()
        except queue.Empty:
            break
        spec_img.set_array(data)
        canvas.draw()
    return [spec_img]


root = tk.Tk()
root.geometry("400x400")
label = tk.Label(master=root, text='VAD: Not detected', width=30,height=3)
label.config(font=(None,14))

kw_label = tk.Label(master=root, text='Keyword: Not detected', width=30,height=3)
kw_label.configure(text='Keyword: Not detected', background="#FF0000",font=(None,14))
label.pack()
kw_label.pack()

window = np.zeros((41,171))
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_ylim(0,40)
spec_img = ax.imshow(window,vmin=0, vmax=1)

anim_queue = queue.Queue()
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

ani = FuncAnimation(fig, anim_callback, interval=100, blit=True)

thread = Thread(target=main, daemon=True,args=(anim_queue,))
thread.start()

tk.mainloop()
