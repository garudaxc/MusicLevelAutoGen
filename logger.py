
class Logger:
    def __init__(self, filename = 'default.log', to_console = False):
        self.file = open(filename, 'w', encoding='utf-8')
        self.to_console = to_console

    def __make_string(self, objs):
        s = ''
        for a in objs: s = s + str(a) + ' '
        return s

    def info(self, *objs):
        str = self.__make_string(objs)
        
        self.file.write('info: '+str+'\n')
        if self.to_console:
            print(str)

    def error(self, *objs):
        str = self.__make_string(objs)
        
        self.file.write('error: '+str+'\n')
        if self.to_console:
            print(str)

def info(*objs):
    __log.info(*objs)

def error(*objs):
    __log.error(*objs)

__log = Logger(to_console=True)

def init(filename = 'default.log', to_console = False):
    globals()['__log'] = Logger(filename, to_console)

#if __name__ == '__main__':
    #__log = Logger(to_console=True)

