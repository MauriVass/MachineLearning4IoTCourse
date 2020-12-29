#Add script from another folder
import sys
sys.path.insert(0, './../Exercise1')
from DoSomething2 import DoSomething
import time


if __name__ == "__main__":
	test = DoSomething("subscriber 2")
	test.run()
	test.myMqttClient.mySubscribe("/ASD123/date/time/timestamp/")

	a = 0
	while (a < 30):
		a += 1
		time.sleep(1)

	test.end()