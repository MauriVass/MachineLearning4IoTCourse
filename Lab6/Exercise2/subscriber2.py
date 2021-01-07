#Add script from another folder
import sys
sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething
import time
import json

class Collector(DoSomething):
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here  
		r = msg.decode('utf-8')
		r = json.loads(r)
		print(f"Timestamp: {int(r['timestamp'])}")

if __name__ == "__main__":
	test = Collector("subscriber 2")
	test.run()
	test.myMqttClient.mySubscribe("/ASD123/timestamp/")

	a = 0
	while (a < 30):
		a += 1
		time.sleep(1)

	test.end()