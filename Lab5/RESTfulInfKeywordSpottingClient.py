import pyaudio
import argparse
import numpy as np
import pyaudio
from scipy import signal
from io import BytesIO
import wave
import base64
import datetime
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()
print(args)
model = args.model

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
      channels=1,
      rate= 48000, input=True,
      frames_per_buffer=2400)
stream.stop_stream()

frames = []
stream.start_stream()
for i in range(10):
  data = stream.read(2400) #, exception_on_overflow=False)
  frames.append(data)
stream.stop_stream()

audio = np.frombuffer(b''.join(frames), dtype=np.int16)
audio = signal.resample_poly(audio, 1 ,3)
audio = audio.astype(np.int16)
buf = BytesIO()
wavefile = wave.open(buf, 'wb')
wavefile.setnchannels(1)
wavefile.setsampwidth(2)
wavefile.setframerate(16000)
wavefile.writeframes(audio.tobytes())
wavefile.close()

#It is needed to send data over network since you can't send raw bytes
audio_b64_bytes = base64.b64encode(buf.read())
audio_string = audio_b64_bytes.decode()

timestamp = int(datetime.datetime.now().timestamp())

#Device IP address
ip = '2.44.137.33' + '/'
body = {
          'bn' : 'http://'+ip,
          'bi' : timestamp,
          'e' : [
              {'n':'audio', 'u':'/', 't':0, 'vb':audio_string}
            ]
        }

#Web service address
url = 'http://localhost:8080/' + model
#The json.dump() is done automatically
r = requests.put(url, json=body)

if(r.satus_code):
  rbody = r.json()
  label = rbody['label']
  prob = rbody['probability']

  print(f'{label} ({prob}%)')
else:
  raise KeyError(r.text)
