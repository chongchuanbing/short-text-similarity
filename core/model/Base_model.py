# -*- coding:utf-8 -*-

import abc

class Base_model(metaclass = abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def model_create(self):
        pass