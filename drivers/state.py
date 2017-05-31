import numpy as np

from models.path import cospath_decay


class State():

    def __init__(self, Ux, Uy, r, wf, wr, path, wx, wy, wo, delta_psi, e, s, e_max=5, data=None):
        """
        :param Ux:
        :param Uy:
        :param r:
        :param x:
        :param y:
        :param o
        :param wf:
        :param wr:
        :param path:
        :param e_max:
        :param data:
        """
        self.Ux = Ux
        self.Uy = Uy
        self.r = r
        self.wx = wx
        self.wy = wy
        self.wo = wo
        self.wf = wf
        self.wr = wr
        self.path = path
        self.delta_psi = delta_psi
        self.e = e
        self.s = s
        self.e_max = e_max
        self.data = data if data is not None else {}

    @staticmethod
    def size():
        return 14  # 12 variables, with wf and wr size 2, and path remaining, excluding kappa

    def as_array(self, kappa_length, kappa_step_size=1):
        """
        returns an array representation of the state, including kappa
        :param kappa_length: the number of kappa values to include (in front of current location)
        :param kappa_step_size: the interval distance between kappa value given
        :return: 
        """
        return np.asarray(
            [
                getattr(self, attr) for attr in ['Ux', 'Uy', 'r', 'e', 'delta_psi', 'wx', 'wy', 'wo', 's']
            ]
            + [self.remainder()]
            + self.wf.tolist()
            + self.wr.tolist()
            + self.kappa(s=[self.s + i*kappa_step_size for i in range(kappa_length)]).tolist()
        )

    def kappa(self, s=None):
        if s is None:
            s = self.s
        return self.path.kappa(s)[0]

    def is_terminal(self):
        return self.reward() < 0 or self.remainder() <= 1e-2

    def reward(self, t_step=1):
        if abs(self.e) > self.e_max:
            return -100
        elif min(self.wr) < -1:
            return -10
        return self.Ux*t_step

    def remainder(self):
        return self.path.length() - self.s

    def x_front(self, model):
        return self.wx + np.cos(self.wo) * model.a

    def y_front(self, model):
        return self.wy + np.sin(self.wo) * model.a

    def x_back(self, model):
        return self.wx - np.cos(self.wo) * model.b

    def y_back(self, model):
        return self.wy - np.sin(self.wo) * model.b

    def __str__(self):
        return "Ux = {0}, Uy = {1}, wf = {2}, wr={3}, e = {4}, s = {5}".format(
            self.Ux,
            self.Uy,
            self.wf,
            self.wr,
            self.e,
            self.s
        )

if __name__ == "__main__":
    path = cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4)
    state = State(Ux=1, Uy=6, r=9, wf=np.asarray([1, 2]), wr=np.asarray([3, 4]), path=path, wx=4, wy=5, wo=7, delta_psi=9, e=1, s=2)
    state.as_array(kappa_length=5)
