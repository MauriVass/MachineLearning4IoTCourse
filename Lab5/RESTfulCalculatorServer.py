'''
RESTful web service that implements a simple calculator. It can accept GET or PUT requests
(GET)
Input :
	- An operations to be executed: [add, sub, mul, div]
	- A 'dictionary' of values 
	Example: http://localhost:8080/add?op1=10&op2=12.2
Output:
	- A json file
	Example: {
				"command": "add",
				"op1": 10.0
				"op2": 12.2
				"result": 22.2
			}

(PUT)
Input:
	- A json file in the body of the request
	Example: {
				"command": "add",
				"operands": [10, 9, 8, 7, 6, 5, 3, 2, 1]
			}
Output:
	- A json file
	Example: {"operands": [10, 9, 8, 7, 6, 5, 3, 2, 1], "command": "add", "result": 51.0}
'''
import random
import string
import cherrypy
import json 

class CalculatorWebService(object):
	#Required to be accessable online
	exposed=True

	def Calculation(self,commands,command,operands, operands_name=''):
		#Variable to generate a json file from
		output = {}

		if(operands_name==''):
			output['operands'] = operands

		output['command'] = command
		#Execute calculations
		res = 0
		for i,op in enumerate(operands):
			val = float(op)
			if(i==0):
				res = val
			else:
				if(command==commands[0]):
					res += val
				elif(command==commands[1]):
					res -= val
				elif(command==commands[2]):
					res *= val
				elif(command==commands[3]):
					if(val!=0):
						res /= val
					else:
						rep = f'Element: {operands_name[i]}' if operands_name!='' else f'Element position: {i}'
						raise cherrypy.HTTPError(404,f"Division by 0. {rep}")
			#Add name:value for each operand
			if(operands_name!=''):
				output[str(operands_name[i])] = val
		#Add the result
		output['result'] = res
		return output

	def GET(self,*path,**query):		
		#Possible commands
		commands = ['add', 'sub', 'mul', 'div']
		#Any or Multiple commands provided
		if(len(path)!=1):
			raise cherrypy.HTTPError(404,f"Use only 1 operation: ['add', 'sub', 'mul', 'div']. Used: {path}")
		command = str(path[0])
		#Command not expected
		if(command not in commands):
			raise cherrypy.HTTPError(404,f"Operation not recognized. Use: ['add', 'sub', 'mul', 'div']. Used: {command}")
		#Not enough operands
		if(len(query)<2):
			raise cherrypy.HTTPError(404,f"Use at least 2 operands. Used: {len(query)}")
		operands_name = list(query.keys())
		operands = list(query.values())
		output = self.Calculation(commands,command,operands,operands_name)
		return json.dumps(output)

	def POST(self,*path,**query):
		return

	def PUT(self,*path,**query):
		input = cherrypy.request.body.read()
		input = json.loads(input)

		#Any or Multiple commands provided
		if('command' not in input.keys()):
			raise cherrypy.HTTPError(404,f"Use only 1 operation: ['add', 'sub', 'mul', 'div']. Used: {input}")
		command = str(input['command'])
		#Possible commands
		commands = ['add', 'sub', 'mul', 'div']
		#Command not expected
		if(command not in commands):
			raise cherrypy.HTTPError(404,f"Operation not recognized. Use: ['add', 'sub', 'mul', 'div']. Used: {command}")

		if('operands' not in input.keys()):
			raise cherrypy.HTTPError(404,f"Operands not found. Input: {input}")
		operands = input['operands']
		#Not enough operands
		if(len(operands)<2):
			raise cherrypy.HTTPError(404,f"Use at least 2 operands. Used: {len(operands)}")

		operands = operands
		output = self.Calculation(commands,command,operands)

		return json.dumps(output)

	def DELETE(self,*path,**query):
		return

#'request.dispatch': cherrypy.dispatch.MethodDispatcher() => switch from default URL to HTTP compliant approch
conf = { '/': {	'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
								'tools.sessions.on':True} 
					}
cherrypy.tree.mount(CalculatorWebService(), '/', conf)

cherrypy.config.update({'servet.socket_host':'0.0.0.0'})
cherrypy.config.update({'servet.socket_port':'8080'})

cherrypy.engine.start()
cherrypy.engine.block()
