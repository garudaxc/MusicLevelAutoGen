import os.path


'''
记录生成关卡过程中的日志
普通信息保存的info.log，如果发成错误，会将错误码和信息输出到error.log
'''


class Logger:
    def __init__(self, pathname):
        self.path = pathname
        filename = os.path.join(pathname, 'info.log')
        self.file = open(filename, 'w', encoding='utf-8')  

    def __make_string(self, objs):
        s = ''
        for a in objs: s = s + str(a) + ' '
        return s

    def info(self, *objs):
        msg = self.__make_string(objs)
        
        self.file.write(msg+'\n')
        self.file.flush()        
        print(msg)

    def error(self, code, *objs):
        filename = os.path.join(self.path, 'error.log')

        with open(filename, 'w') as file:
            file.write(str(code)+'\n')
            msg = self.__make_string(objs)
            file.write(msg+'\n')
            print(msg)