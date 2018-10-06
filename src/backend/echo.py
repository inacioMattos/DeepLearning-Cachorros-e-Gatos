import zerorpc, sys


class EchoApi(object):
	def echo(string):
		print ("opa recebi texto")
		return string

def main():
	addr = "tcp://127.0.0.1:8080"
	s = zerorpc.Server(EchoApi)
	s.bind(addr)
	print('start running on {}'.format(addr))
	s.run()


main()