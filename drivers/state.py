import numpy as np

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
        return 13  # 12 variables, with wf and wr size 2, excluding kappa

    def as_array(self):
        return np.asarray(
            [
                getattr(self, attr) for attr in ['Ux', 'Uy', 'r', 'e', 'delta_psi', 'wx', 'wy', 'wo', 's']
            ] + self.wf.asList() + self.wr.asList()
        )

    def kappa(self, s=None):
        if s is None:
            s = self.s
        return self.path.kappa(s)

    def is_terminal(self):
        return self.reward() < 0 or self.remainder() <= 0

    def reward(self):
        if self.path.length() < self.s:
            return self.Ux
        elif abs(self.e) > self.e_max:
            return -100
        elif min(self.wr) < -1:
            return -10
        return 0

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
