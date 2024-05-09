import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.terminal.flush()  # 不启动缓冲,实时输出
            self.log.flush()
        except:
            pass

    def flush(self):
        pass
