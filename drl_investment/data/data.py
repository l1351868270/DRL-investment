import abc


def signed_power(x, a):
    return x**a
class DataBase(abc.ABC):
    def __init__(self):
        return
    
    @property
    def raw(self):
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
    

class Alpha101(object):
    def __init__(self, data=None):
        pass


class Alpha101Pytorch(object):
    def __init__(self, data=None):
        pass
