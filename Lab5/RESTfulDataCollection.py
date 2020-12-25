from board import D4
import adafruit_dht
import pyaudio
import wave
import base64
import datetime
import cherrypy

class DataCollector():
	exposed = True

	def __init__(self):
		self.dht_device = adafruit_dht.DHT11(D4)

		'''
		self.audio = pyaudio.PyAudio()
		self.stream = self.audio.open(format=pyaudio.paInt16,
			channels=1,
			rate= 48000, input=True,
			frames_per_buffer=4800)
		self.stream.wait()
		'''

	def GET(self,*path,**query):
		#Collect Temp and Hum
		temperature = self.dht_device.temperature
		humidity = self.dht_device.humidity
		'''
		#Record file audio
		frames = []
		self.stream.start_stream()
		for i in range(10):
			data = stream.read(4800) #, exception_on_overflow=False)
			frames.append(data)
		self.stream.stop_stream()
		#It is needed to send data over network since you can't send raw bytes
		audio_64_bytes = base64.b64encode(b''.join(frames))
		audio_string = audio_b64bytes.decode()
		'''

		ip = '2.44.137.33' + '/'
		timestamp = int(datetime.datetime.now().timestamp())
		body = {
					'bn' : 'http://'+ip,
					'bi' : timestamp,
					'e' : [
							{'n':'temperature', 'u':'Cel', 't':0, 'v':temperature},
							{'n':'humidity', 'u':'%RH', 't':0, 'v':humidity}
							#{'n':'audio', 'u':'/', 't':0, 'vb':audio_string}
						]
				}

		return json.dump(body)

if __name__ == '__main__':
	conf = { '/': {	'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
									'tools.sessions.on':True}
						}
	cherrypy.tree.mount(DataCollector(), '/', conf)

	cherrypy.config.update({'servet.socket_host':'192.168.1.8'})
	cherrypy.config.update({'servet.socket_port':8080})

	cherrypy.engine.start()
	cherrypy.engine.block()

