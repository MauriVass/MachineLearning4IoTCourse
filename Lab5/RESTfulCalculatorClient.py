'''
Client application to invoke the RESTful calculator. Run RESTfulCalculatorServer.py before run this script
Input:
	- An operations to be executed: [add, sub, mul, div]
	- A list of values
	Example: python RESTfulCalculatorClient.py add 10 12.2 
Output:
	- A print with the summary of the operation
	Example: 10.0 + 12.2 = 23.2
'''

import requests
import sys
import json 

isGetReq = False

input_params = list(sys.argv)
command = input_params[1]
operands = {}
counter = 0
for i in input_params[2:]:
	operands[f'op{counter}'] = float(i)
	counter+=1

if(isGetReq):
	r = requests.get('http://localhost:8080/'+command, params=operands)
else:
	input_json = {}
	input_json['command'] = command
	input_json['operands'] = list(operands.values())
	input_json = json.dumps(input_json)
	r = requests.put('http://localhost:8080/', data=input_json)

if(r.status_code==200):
	#Print Result
	results = r.json()
	output = ''
	res_command = results['command']
	if(res_command=='add'):
		res_command = '+'
	elif(res_command=='sub'):
		res_command = '-'
	elif(res_command=='mul'):
		res_command = '+'
	elif(res_command=='div'):
		res_command = '/'

	counter = 0
	if(isGetReq):
		for k,v in results.items():
			if(k.find('op')!=-1):
				if(counter>0):
					output += f' {res_command} '
				counter+=1
				output += str(v)
			elif(k=='result'):
				output += f' = {str(v)}'
	else:
		operands = results['operands']
		for i in operands:
			if(counter>0):
				output += f' {res_command} '
			counter+=1
			output += str(i)
		output += f" = {results['result']}"
	print(output)
else:
	raise KeyError('Something went wrong!')
