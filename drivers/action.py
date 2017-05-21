import numpy as np

class Action():

    def __init__(self, delta, tr, tf):
        self.delta = delta
        self.tr = tr
        self.tf = tf

    @staticmethod
    def size():
        return 3

    @staticmethod
    def get_action(array):
        return Action(delta=array[0], tr=array[1], tf=array[2])

    def as_array(self):
        return np.asarray([self.delta, self.tr, self.tf])

    def __call__(self):
        return self.delta, self.tr, self.tf
