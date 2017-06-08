import numpy as np

from models.path import cospath_decay
from drivers.state import State, RewardTypes

class StateSimple(State):

    reward_type = RewardTypes.SPEED
    negatively_reward_crash = True
    crash_cost = -1
    reward_increments = 5
    v = ['Ux', 'Uy', 'r', 'e', 'delta_psi']

    @staticmethod
    def size():
        return len(StateSimple.v)+1  # 6 variables, with wf and wr size 2, and path remaining, excluding kappa

    def as_array(self, kappa_length, kappa_step_size=1):
        """
        returns an array representation of the state, including kappa
        :param kappa_length: the number of kappa values to include (in front of current location)
        :param kappa_step_size: the interval distance between kappa value given
        :return: 
        """
        return np.asarray(
            [
                getattr(self, attr) for attr in StateSimple.v
            ]
            + [self.remainder()]
            + self.kappa(s=[self.s + i*kappa_step_size for i in range(kappa_length)]).tolist()
        )

    @staticmethod
    def array_value_mapping():
        dictionary = {
                v: i for i, v in zip(range(9), StateSimple.v)
            }
        dictionary["remainder"] = len(StateSimple.v)
        return dictionary


    def crash(self):
        return abs(self.e) > self.e_max

    def __str__(self):
        return "[r={r:.4g} ({x:.4g}, {y:.4g}, {o:.4g})]: Ux = {ux:.4g}, Uy = {uy:.4g}, e = {e:.4g}, s = {s:.4g}".format(
            ux=self.Ux,
            uy=self.Uy,
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
