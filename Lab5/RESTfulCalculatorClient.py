import requests
import sys

input_params = list(sys.argv)
command = input_params[1]
operands = {}
for i in input_params[2:]:
	val = int(i)
	operands[f'op{val-1}'] = val

r = requests.get('http://localhost:8080/'+command, params=operands)

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
	for k,v in results.items():
		if(k.find('op')!=-1):
			if(counter>0):
				output += f' {res_command} '
			counter+=1
			output += str(v)
		elif(k=='result'):
			output += f' = {str(v)}'
	print(output)
else:
	raise KeyError('Something went wrong!')
