import numpy as np
import math
from driver import Action, State
from collections import namedtuple

PointSet = namedtuple('PointSet', ['x', 'y'])

class TireModel():

    def __init__(self, mu_s, mu_p, cx, cy):
        """
        Creates a tire model
        :param mu_s:
        :param mu_p:
        :param cx: N/rad in the x (longitudinal) direction
        :param cy: N/rad in the y (lateral) direction
        """
        self.mu_s = mu_s
        self.mu_p = mu_p
        self.cx = cx
        self.cy = cy

    def __call__(self, Fz, sigma, alpha, as_dict=False):
        """
        find the Fx and Fy forces by the wheel
        :param Fz: downward force
        :param sigma:
        :param alpha:
        :param as_dict: returns a dictionary instead of list
        :return: the Fx and Fy of the wheel
        if as_dict==True, returns {'Fx':Fx, 'Fy':Fy}
        """
        f = math.sqrt(self.cx ** 2 * (sigma / (1 + sigma)) ** 2 + self.cy**2 * (math.tan(alpha) / (1 + sigma)) ** 2)
        t = 3*self.mu_p*Fz
        if f < t:
            F = f - 1/t * (2-self.mu_s/self.mu_p)*(f**2) + 1/(t**2)*(1-(2*self.mu_s)/(3*self.mu_p))*(f**3)
        else:
            F = self.mu_s*Fz
        if f == 0:
            Fx = 0
            Fy = 0
        else:
            Fx = self.cx*(sigma/(1+sigma))*F/f
            Fy = -1*self.cy*(math.tan(alpha)/(1+sigma))*F/f

        if as_dict:
            return {
                'Fx': Fx,
                'Fy': Fy
            }
        else:
            return Fx, Fy


class CarModel():

    def __init__(self, mass=1650, tyre_model=TireModel, cx=200000, cy=100000, mdf=.57, length=2.5, Iz=2235, width=1.55, muf_p=1.1, muf_s=0.9, mur_p=1.2, mur_s=1.0, g=9.81, Re = 0.34, Jw=1.2, make_record = True):
        """
        Initializes the car model
        :param mass: mass of car
        :param tyre_model: tire model to use
        :param cx:
        :param cy:
        :param mdf: mass distribution on the front
        :param length: length of the car
        :param Iz: inertia moment of the car
        :param width: width of vehicle
        :param muf_p:
        :param muf_s:
        :param mur_p:
        :param mur_s:
        :param g: force of gravity
        :param Re:
        :param Jw:
        """
        self.tyre_model = {
            "front": tyre_model(mu_s = muf_s, mu_p=muf_p, cx=cx, cy=cy),
            "rear": tyre_model(mu_s=mur_s, mu_p=mur_p, cx=cx, cy=cy)
        }
        self.cx = cx
        self.cy = cy
        self.fz_front = mass*mdf*g
        self.fz_rear = mass*(1-mdf)*g
        self.mdf = mdf
        self.m = mass
        self.a = length*(1-mdf)
        self.b = length - self.a
        self.l = length
        self.Iz = Iz
        self.d = width
        self.Re = Re
        self.Jw = Jw
        self.record = [[[],[]],[[],[]],[[],[]]] if make_record else None

    def reset_record(self):
        self.record = []

    def get_records(self):
        return {
            "midpoint": PointSet(x=self.record[0][0], y=self.record[0][1]),
            "front": PointSet(x=self.record[1][0], y=self.record[1][1]),
            "rear": PointSet(x=self.record[2][0], y=self.record[2][1])
        }


    def make_test_state(self, Ux, Uy, r, path=None, x=0, y=0, o=0, delta_psi=0, e=0, s=0, wf=None, wr=None):
        """
        sets the state of the vehicle
        :param Ux: velocity in the x direction
        :param Uy: velocity the y direction (lateral)
        :param r: rate of rotation
        :return:
        """
        wf = np.ones(shape=2)*Ux/self.Re if wf is None else wf
        wr = np.ones(shape=2)*Ux/self.Re if wr is None else wr
        return State(Ux=Ux, Uy=Uy, r=r, wf=wf, wr=wr, path=path, wx=x, wy=y, wo=o, delta_psi=delta_psi, e=e, s=s)

    def start_state(self, Ux, Uy, r, path=None):
        """
        sets the state of the vehicle
        :param Ux: velocity in the x direction
        :param Uy: velocity the y direction (lateral)
        :param r: rate of rotation
        :return:
        """
        wf = np.ones(shape=2)*Ux/self.Re
        wr = np.ones(shape=2)*Ux/self.Re
        if path is not None:
            nx, ny = path.p(path.interp_offset)
            sx, sy = path.p(0)
            o = np.arctan2(ny-sy, nx-sx)
        else:
            o = 0
        return State(Ux=Ux, Uy=Uy, r=r, wf=wf, wr=wr, path=path, wx=path.x[0], wy=path.y[0], wo=o, delta_psi=0, e=0, s=0)

    def __call__(self, state, action, time, as_state=True):
        """
        updates the state of the vehicle
        :param delta: turn angle of the front wheels
        :param time: time step
        :param torque_f: torque on the front wheels
        :param torque_r: torque on the rear wheels
        :param as_state: returns output as dictionary
        :return: the new state
        """
        Uxr = np.array([
            state.Ux - self.d/2*state.r,
            state.Ux + self.d/2*state.r
        ])
        Uyr = np.array([
            state.Uy - self.b*state.r,
            state.Uy - self.b*state.r
        ])

        Uxf = np.array([
            state.Ux - self.d/2*state.r,
            state.Ux + self.d/2*state.r
        ])

        Uyf = np.array([
            state.Uy + self.a*state.r,
            state.Uy + self.a*state.r
        ])

        alphar = np.arctan2(Uyr, Uxr)
        alphaf = np.arctan2(Uyf, Uxf) - action.delta

        Vr = Uxr
        Vf = Uxf*np.cos(action.delta) + Uyf*np.sin(action.delta)
        sigmar = np.nan_to_num(np.divide(self.Re*state.wr - Vr, Vr))
        sigmaf = np.nan_to_num(np.divide(self.Re*state.wf - Vf, Vf))

        F_rl = self.tyre_model["rear"](Fz=self.fz_rear/2, alpha=alphar[0], sigma=sigmar[0], as_dict=True)
        F_rr = self.tyre_model["rear"](Fz=self.fz_rear/2, alpha=alphar[1], sigma=sigmar[1], as_dict=True)
        F_fl = self.tyre_model["front"](Fz=self.fz_front/2, alpha=alphaf[0], sigma=sigmaf[0], as_dict=True)
        F_fr = self.tyre_model["front"](Fz=self.fz_front/2, alpha=alphaf[1], sigma=sigmaf[1], as_dict=True)

        R_a = np.sqrt(self.a**2 + self.d**2/4)
        theta_a = np.arctan(self.d/(self.a*2))
        Ff = np.array([
            -F_fl['Fx']*np.sin(theta_a - action.delta) + F_fl['Fy']*np.cos(theta_a - action.delta),
            F_fr['Fx'] * np.sin(theta_a + action.delta) + F_fr['Fy']*np.cos(theta_a + action.delta)
        ])

        R_b = np.sqrt(self.b**2 + self.d**2/4)
        theta_b = np.arctan(self.d/(2*self.b))
        Fr = np.array([
            -F_rl['Fx']*np.sin(theta_b) - F_rl['Fy']*np.cos(theta_b),
            F_rr['Fx'] * np.sin(theta_b) - F_rr['Fy'] * np.cos(theta_b)
        ])

        drdt = (np.sum(Ff*R_a) + np.sum(Fr*R_b))/self.Iz
        dUydt = (
                    F_rl['Fy'] + F_rr['Fy']
                    + (F_fl['Fy'] + F_fr['Fy'])*np.cos(action.delta)
                    + (F_fl['Fx'] + F_fr['Fx'])*np.sin(action.delta)
                )/self.m - state.r*state.Ux
        dUxdt = (
                    F_rl['Fx'] + F_rr['Fx']
                    + (F_fl['Fx'] + F_fr['Fx'])*np.cos(action.delta)
                    - (F_fl['Fy'] + F_fr['Fy'])*np.sin(action.delta)
                )/self.m + state.r*state.Uy

        dwrdt = (action.tr - self.Re * np.array([F_rl['Fx'], F_rr['Fx']]))/self.Jw
        dwfdt = (action.tf - self.Re * np.array([F_fl['Fx'], F_fr['Fx']])) / self.Jw

        dx = (state.Ux * 2 + time * dUxdt) / 2 * time
        dy = (state.Uy * 2 + time * dUydt) / 2 * time
        do = (state.r * 2 + time * drdt) / 2 * time

        Ux = time*dUxdt + state.Ux
        Uy = time*dUydt + state.Uy
        r = time*drdt + state.r
        wr = time*dwrdt + state.wr
        wf = time*dwfdt + state.wf

        Wo = float(state.wo + do)
        if Wo > np.pi:
            Wo -= 2*np.pi
        elif Wo < -1*np.pi:
            Wo += 2*np.pi
        Wx = float(state.wx + (dx * np.cos(Wo) + dy * np.sin(Wo)))
        Wy = float(state.wy + (dx * np.sin(Wo) + dy * np.cos(Wo)))

        # dsdt = state.Ux * np.cos(state.delta_psi) - state.Uy * np.sin(state.delta_psi)
        # s = state.s + dsdt*time
        e, s, road_orientation = state.path.interpolate(Wx, Wy, old_s=None)
        delta_psi = Wo - road_orientation


        state = State(
                Ux=Ux,
                Uy=Uy,
                r=r,
                wf=wf,
                wr=wr,
                path=state.path,
                wo=Wo,
                wx=Wx,
                wy=Wy,
                e=e,
                s=s,
                delta_psi=delta_psi,
                e_max=state.e_max,
                road_orientation=road_orientation
        )

        if self.record is not None:
            for points, point in zip(self.record, [[state.wx, state.wy], [state.x_front(self), state.y_front(self)], [state.x_back(self), state.y_back(self)]]):
                points[0].append(point[0])
                points[1].append(point[1])

        if as_state:
            return state, dx, dy, do

        else:
            return dx, dy, do


def tyre_model_test():
    Cx = 200000
    Cy = 100000
    Mu_p = 1.1
    Mu_s = 0.9
    m = 1650 # mass of the gar
    mdf = .57 # mass distribution of the front
    g = 9.81 # gravity
    fz_front = m * mdf / 2 * g

    model = TireModel(mu_s=Mu_s, mu_p=Mu_p, cx=Cx, cy=Cy)
    inputs = [
        [0.1421e-15, 0],
        [-0.1421e-15, 0],
        [8.0036e-04, -0.0458],
        [6.2236e-04, -0.0464],
        [4.0686e-04, -0.0428]
    ]
    results = [
        [0.284e-10, 0],
        [-0.2842e-10, 0],
        [0.1097e3, 3.1376e3],
        [0.0848e3, 3.1649e3],
        [0.0572e3, 3.0125e3]
    ]
    for input, result in zip(inputs, results):
        output = model(fz_front, input[0], input[1])
        for o, r, type in zip(output, result, ['fx', 'fy']):
            type = "{type} = {actual}, should be {correct} is incorrect for sigma {sigma} and alpha {alpha}".format(
                type=type,
                sigma=input[0],
                alpha=input[1],
                actual=o,
                correct=r
            )
            if r == 0:
                assert o == 0, type
            else:
                assert abs(o - r)/r < .03, type


def car_model_test():
    t = 1.0000e-03
    model = CarModel()
    actions = [
        Action(delta=0, tr=0, tf=0),
        Action(delta=0.0408, tf=28.6115, tr=28.6115),
    ]
    results = [
        [0.0250, 0, 0],
        [0.0247, -4.2937e-04, 2.7073e-04]
    ]
    init = [
        [25, 0, 0, np.array([73.5294, 73.5294]), np.array([73.5294, 73.5294])],
        [24.7139, -0.4294, 0.2707, np.array([72.1132, 73.3479]), np.array([72.0391, 73.2730])]
    ]

    class TestPath():
        def kappa(self, s=0):
            return [s]

    for action, result, state, run in zip(actions, results, init, range(len(actions))):

        state = model.make_test_state(Ux=state[0], Uy=state[1], r=state[2], path=TestPath(), wf=state[3], wr=state[4])
        output = model(state=state, action=action, time=t, as_state=False)
        for o, r, type in zip(output, result, ['dx', 'dy', 'do']):
            type = "run {t}: {type} = {actual}, should be {correct} is incorrect".format(
                type=type,
                actual=o,
                correct=r,
                t=run + 1
            )
            if r == 0:
                assert o == 0, type
            else:
                assert abs(o - r)/r < .01, type

if __name__ == "__main__":
    tyre_model_test()
    car_model_test()

