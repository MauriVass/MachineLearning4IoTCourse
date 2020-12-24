import random
import string
import cherrypy
import json 

class StringGeneratorWebService(object):
	#Required to be accessable online
	exposed=True

	def GET(self,*path,**query):
		#Variable to generate a json file from
		output = {}
		#Possible commands
		commands = ['add', 'sub', 'mul', 'div']
		#Any or Multiple commands provided
		if(len(path)!=1):
			raise cherrypy.HTTPError(404,f"Use only 1 operation: ['add', 'sub', 'mul', 'div']. Used: {path}")
		command = str(path[0])
		#Command not expected
		if(command not in commands):
			raise cherrypy.HTTPError(404,f"Operation not recognized. Use: ['add', 'sub', 'mul', 'div']. Used: {command}")
		output['command'] = command
		#Not enough operands
		if(len(query)<2):
			raise cherrypy.HTTPError(404,f"Use at least 2 operands. Used: {len(query)}")
		res = 0
		for i,items in enumerate(query.items()):
			val = float(items[1])
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
						raise cherrypy.HTTPError(404,f"Division by 0. Element: {items[0]}")
			output[str(items[0])] = val
		output['result'] = res
		return json.dumps(output)

	def POST(self,*path,**query):
		return
	def PUT(self,*path,**query):
		return
	def DELETE(self,*path,**query):
		return
#'request.dispatch': cherrypy.dispatch.MethodDispatcher() => switch from default URL to HTTP compliant approch
conf = { '/': {	'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
								'tools.sessions.on':True} 
					}
cherrypy.tree.mount(StringGeneratorWebService(), '/', conf)

cherrypy.config.update({'servet.socket_host':'0.0.0.0'})
cherrypy.config.update({'servet.socket_port':'8080'})

cherrypy.engine.start()
cherrypy.engine.block()
