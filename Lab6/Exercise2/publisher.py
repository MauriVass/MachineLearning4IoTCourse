import sys
sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething

import time
import datetime
import json

if __name__ == "__main__":
	test = DoSomething("publisher 1")
	test.run()

	a = 0
	#This is used to publish or not the timestamp (==2 -> publish)
	counter = 2
	while (a < 2000):
		date_time = datetime.datetime.now()
		timestamp = datetime.datetime.timestamp(date_time)

		date_str = str(date_time.date())
		time_str = str(date_time.time()).split('.')[0]
		message = {'date':date_str}
		message['time'] = time_str
		test.myMqttClient.myPublish ("/ASD123/datetime/", json.dumps(message))
		
		if(counter==2):
			message['timestamp'] = timestamp
			test.myMqttClient.myPublish ("/ASD123/timestamp/", json.dumps(message))
			counter=0

		counter += 1
		time.sleep(5)

	test.end()