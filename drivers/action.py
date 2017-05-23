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
    def get_action(array, max_delta, max_t):
        return Action(delta=array[0]*max_delta*2 - max_delta, tr=array[1]*max_t*2 - max_t, tf=array[2]*max_t*2 - max_t)

    def as_array(self, max_delta, max_t):
        return np.asarray([self.delta/(max_delta*2) + .5, self.tr/(max_t*2)+.5, self.tf/(max_t*2)+.5])

    def __call__(self):
        return self.delta, self.tr, self.tf

if __name__ == "__main__":
    t_max = 150
    d_max = np.pi/2
    test = [
        (0, 0, 0),
        (np.pi/4, -100, 200)
    ]
    for delta, tr, tf in test:
        action = Action(delta, tr, tf)
        array = action.as_array(max_delta=d_max, max_t=t_max)
        results = Action.get_action(array, max_delta=d_max, max_t=t_max)
        for actual, result, name in zip(action(), results(), ["delta", "tr", "tf"]):
            assert abs((actual-result)/(actual+.001)) < 1e-3, "{0} was {1}, should have been {2}".format(name, result, actual)
