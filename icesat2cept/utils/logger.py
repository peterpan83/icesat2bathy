import logging, os, sys
from datetime import datetime
import re
import logging.handlers


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Clogger():

    def __init__(self, stream_loglevel, file_loglevel, logname = None):
        self.stream_loglevel = int(stream_loglevel)
        self.file_loglevel = int(file_loglevel)
        logname = 'GAAC_GEN' if logname is None else logname
        self._logger = logging.getLogger(logname)
        self._logger.setLevel(10)

        formatter = logging.Formatter('%(asctime)s %(name)s %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S')
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(self.stream_loglevel)
        self._logger.addHandler(streamHandler)

        self.__current_fileHandler = None

    def set_logdir(self, log_dir):
        if self.__current_fileHandler is not None:
            self._logger.removeHandler(self.__current_fileHandler)

        now = datetime.now()
        filehandler = logging.FileHandler(filename=os.path.join(log_dir,
                                                                f'log_gaac_{now.year}{str.zfill(str(now.month), 2)}{str.zfill(str(now.day), 2)}-{str.zfill(str(now.hour), 2)}_'
                                                                f'{str.zfill(str(now.minute), 2)}_{str(now.second)}.txt'))
        formatter = logging.Formatter('%(asctime)s %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S')
        filehandler.setFormatter(formatter)
        filehandler.setLevel(self.file_loglevel)
        self.__current_fileHandler = filehandler
        self._logger.addHandler(filehandler)


    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def print(self, msg, level = 'INFO'):
        getattr(self, str.lower(level))(msg)

def get_root_logger():
    stream_loglevel = 10
    file_loglevel = 10
    _logger = Clogger(stream_loglevel=stream_loglevel, file_loglevel=file_loglevel, logname='ICECEPT')
    return _logger