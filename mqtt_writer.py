import os

class SummaryWriter:

	def __init__(self,host="10.0.0.1", comment=""):
		self.host = host
		self.mqtt_cmd = "mosquitto_pub -h " + host + " -t drl2/"
		self.comment = comment

	def reset(self):
		cmd = self.mqtt_cmd + "reset -m 0"
		self.run_command(cmd)
		cmd = self.mqtt_cmd + "score -m 0"
		self.run_command(cmd)
		cmd = self.mqtt_cmd + "episode -m 0"
		self.run_command(cmd)
		cmd = self.mqtt_cmd + "average -m 0"
		self.run_command(cmd)

	def log(self, value):
		cmd = self.mqtt_cmd + "log -m '" + value + "'"
		#print(cmd)
		self.run_command(cmd)	
		

	def add_scalar(self, name, value, step):
		cmd = self.mqtt_cmd + name + " -m {:.2f}".format(value)
		self.run_command(cmd)

	def run_command(self, cmd):
		#print(cmd)
		os.system(cmd)
		#print(cmd)

