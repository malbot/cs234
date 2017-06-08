import numpy as np

from models.path import cospath_decay

class RewardTypes(object):
    ERROR = "error"  # reward is inversely proportional to error from road
    NEGATIVE_ERROR = "neg_error"
    DISTANCE = "distance"  # reward is proportional to distance traveled
    NEGATIVE_DISTANCE = "neg_distance"  # negative reward is inversely proportional to distance traveled (ie, longer distance is less negative reward)
    SPEED = "speed"  # reward is proportional to speed

class State(object):

    reward_type = RewardTypes.NEGATIVE_ERROR
    negatively_reward_crash = True
    crash_cost = -1
    reward_increments = 5
    v = ['Ux', 'Uy', 'r', 'e', 'delta_psi', 's']

    def __init__(self, Ux, Uy, r, wf, wr, path, wx, wy, wo, delta_psi, e, s, e_max=10, data=None, t=0):
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
        self.time = t

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
                getattr(self, attr) for attr in State.v
            ]
            + [self.remainder()]
            + self.wf.tolist()
            + self.wr.tolist()
            + self.kappa(s=[self.s + i*kappa_step_size for i in range(kappa_length)]).tolist()
        )

    @staticmethod
    def array_value_mapping():
        dictionary = {
                v: i for i, v in zip(range(9), State.v)
            }
        dictionary["remainder"] = len(v)
        dictionary['wf'] = np.asarray([1,2] + len(State.v))
        dictionary['wr'] = np.asarray([3,4] + len(State.v))
        return dictionary

    def kappa(self, s=None):
        if s is None:
            s = self.s
        return self.path.kappa(s)[0]

    def is_terminal(self):
        return abs(self.e) > self.e_max or self.remainder() <= 1e-2 or min(self.wr) < -1

    def reward(self, t_step=1, previous_state=None, previous_action=None, max_path_length=1000):

        if (abs(self.e) > self.e_max or min(self.wr) < -1) and self.negatively_reward_crash:
            return self.crash_cost

        if self.reward_type == RewardTypes.ERROR:
            return  (self.e_max -1*abs(self.e))/self.e_max
        elif self.reward_type == RewardTypes.NEGATIVE_ERROR:
            return  -1*abs(self.e)/self.e_max
        elif self.reward_type == RewardTypes.DISTANCE:
            return self.s
        elif self.reward_type == RewardTypes.NEGATIVE_DISTANCE:
            return self.remainder()*-1
        elif self.reward_type == RewardTypes.SPEED:
            return (self.s - previous_state.s)/t_step

        raise ValueError("Reward type not recognized: {0}".format(self.reward_type))
        # if abs(self.e) > self.e_max or min(self.wr) < -1:
        #     return -100

        # if previous_action is not None and previous_state is not None and self.remainder() > 1:
        #     return int(self.s >= self.reward_increments > previous_state.s)
        # else:
        #     return int(self.remainder() <= 1)

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
        return "[r={r:.4g} ({x:.4g}, {y:.4g}, {o:.4g})]: Ux = {ux:.4g}, Uy = {uy:.4g}, wf = {wf}, wr={wr}, e = {e:.4g}, s = {s:.4g}".format(
            ux=self.Ux,
            uy=self.Uy,
            wf=self.wf,
            wr=self.wr,
            e=self.e,
            s=self.s,
            r=self.reward(),
            x=self.wx,
            y=self.wy,
            o=self.wo
        )

if __name__ == "__main__":
    path = cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4)
    state = State(Ux=1, Uy=6, r=9, wf=np.asarray([1, 2]), wr=np.asarray([3, 4]), path=path, wx=4, wy=5, wo=7, delta_psi=9, e=1, s=2)
    state.as_array(kappa_length=5)
