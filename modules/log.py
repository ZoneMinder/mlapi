class Log:
    def __init__(self):
        print('Initializing log')

    def debug(self, message):
        print('DEBUG: {}'.format(message))

    def error(self, message):
        print('ERROR: {}'.format(message)) 

    def info(self, message):
        print('INFO: {}'.format(message)) 
