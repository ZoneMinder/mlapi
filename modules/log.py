from datetime import datetime

class Log:
    def __init__(self):
        print('Initializing log')

    def debug(self, message):
        print('DEBUG: {}'.format(message))

    def error(self, message):
        print('ERROR: {}'.format(message)) 

    def info(self, message):
        print('INFO: {}'.format(message)) 

class ConsoleLog:
    ' console based logging function that is used if no logging handler is passed'
    def __init__(self):
        self.dtformat = "%b %d %Y %H:%M:%S.%f"
        self.level = 5

    def set_level(self,level):
        self.level = level

    def get_level(self):
        return self.level

    def Debug (self,level, message, caller=None):
        if level <= self.level:
            dt = datetime.now().strftime(self.dtformat)
            print ('{} [DBG {}] {}'.format(dt, level, message)) 

    def Info (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [INF] {}'.format( dt, message))

    def Warning (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{}  [WAR] {}'.format( dt, message))

    def Error (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [ERR] {}'.format(dt, message))

    def Fatal (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [FAT] {}'.format(dt, message))
        exit(-1)

    def Panic (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [PNC] {}'.format(dt, message))
        exit(-2)
