import numpy as np

from drivers.action import Action
from drivers.abstract_driver import AbstractDriver
from drivers.state import State
from models.car import CarModel
from models.gps import GPS
from models.path import cospath


class pidDriver(AbstractDriver):
    def __init__(self, V, kp, x_la, car, lookahead=0):
        super(AbstractDriver, self).__init__()
        self.V = V
        self.kp = kp
        self.x_la = x_la
        self.car = car
        self.lookahead = lookahead

    def get_action(self, state_batch):
        actions = []
        for state in state_batch:
            kappa = state.kappa(state.s)
            delta_fb = -self.kp * (state.e + self.x_la*state.delta_psi)
            K = self.car.m * self.car.mdf / self.car.cy - self.car.m*(1-self.car.mdf)/self.car.cy
            beta = (self.car.b - (self.car.a*self.car.m*state.Ux**2)/(self.car.cy*self.car.l))*kappa
            delta_ff = self.car.l*kappa + K*state.Ux**2*kappa - self.kp*self.x_la*beta
            delta = delta_fb + delta_ff
            delta = delta if abs(delta) < self.car.max_del else np.sign(delta)*self.car.max_del

            torque = 100*(self.V - state.Ux)
            torque = torque if abs(torque) < self.car.max_t else np.sign(torque)*self.car.max_t
            actions.append(Action(delta=delta, tr=torque, tf=torque))
        return actions

    def get_noisy_action(self, state_batch):
        actions = self.get_action(state_batch)
        actions = [
            Action.get_action(a.as_array(max_delta=1, max_t=1)*np.random.normal(loc=1, scale=0.05), max_delta=1, max_t=1)
            for a in actions
        ]
        return actions

    def train(self, R, states):
        # nothing to train
        pass

if __name__ == "__main__":
    t_step = .01
    max_i = 1000
    path = cospath(length=10, y_scale=10)
    model = CarModel()
    x = []
    y = []
    gps = GPS(car=model, o=np.pi/6)
    driver = pidDriver(V=10, kp=3*np.pi/180, x_la=15, car=model)

    class Path():
        def __init__(self, kappa):
            self.kapa = kappa

        def kappa(self, s):
            return [self.kapa]

    states = [
        State(
            Ux = 24.9966,
            Uy = -0.0451,
            r = 0.0608,
            e = 0.0447,
            delta_psi=0.0050,
            path = Path(0.0023),
            s = 0,
            wx = 0,
            wy = 0,
            wo = 0,
            wf = 0,
            wr = 0
        ),
        State(
            Ux = 24.9264,
            Uy = -0.2121,
            r = 0.1864,
            e= 0.2219,
            delta_psi=0.0105,
            path=Path(0.0074),
            s = 0,
            wx = 0,
            wy = 0,
            wo = 0,
            wf = 0,
            wr = 0
        )
    ]

    actions = driver.get_action(states)
    for action, delta in zip(actions, [0.0084, 0.0264]):
        assert abs(action.delta - delta)/action.delta < .03