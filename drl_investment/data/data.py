import abc


class DataBase(metaclass=abc.ABCMeta):
    def __init__(self):
        return
    
    @property
    @abc.abstractmethod
    def alpha101(self):
        return
    
    @property
    @abc.abstractmethod
    def alpha158(self):
        return
    
    @property
    @abc.abstractmethod
    def alpha360(self):
        return