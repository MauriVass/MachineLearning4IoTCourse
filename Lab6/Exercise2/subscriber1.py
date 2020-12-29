from DoSomething1 import DoSomething
import time

if __name__ == "__main__":
	test = DoSomething("subscriber 1")
	test.run()
	test.myMqttClient.mySubscribe("/ASD123/date/time/")

	a = 0
	while (a < 30):
		a += 1
		time.sleep(4)

	test.end()