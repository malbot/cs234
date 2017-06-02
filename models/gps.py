from collections import namedtuple

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

from drivers.action import Action
from models import car

PointSet = namedtuple('PointSet', ['x', 'y'])

class GPS:

    def __init__(self, car, x=0, y=0, o=0):
        self.x = x
        self.y = y
        self.o = o
        self.s = 0
        self.e = 0
        self.delta_psi = 0
        self.car = car
        self.record = [[[],[]],[[],[]],[[],[]]]
        self.make_record = False

    def start_record(self):
        self.record = [[[],[]],[[],[]],[[],[]]]
        self.make_record = True

    def end_record(self):
        self.make_record = False

    def __call__(self, dx=0, dy=0, do=0, Ux=0, Uy=0, r=0, time=0, kappa=0):
        self.o += do
        # if self.o > np.pi:
        #     self.o = np.pi - self.o
        # elif self.o < -np.pi:
        #     self.o = -np.pi - self.o
        self.x += dx*np.cos(self.o) + dy*np.sin(self.o)
        self.y += dy*np.cos(self.o) + dx*np.sin(self.o)
        dsdt = Ux*np.cos(self.delta_psi) - Uy*np.sin(self.delta_psi)
        dedt = Uy*np.cos(self.delta_psi) + Ux*np.sin(self.delta_psi)
        dDpdt = r - dsdt*kappa
        self.s += dsdt*time
        self.e += dedt*time
        self.delta_psi += dDpdt*time

        if self.make_record:
            for points, point in zip(self.record, [[self.x, self.y], [self.x_front(), self.y_front()], [self.x_back(), self.y_back()]]):
                points[0].append(point[0])
                points[1].append(point[1])

        return self.x, self.y, self.o*180/np.pi, self.s, self.e

    def get_records(self):
        return {
            "midpoint": PointSet(x=self.record[0][0], y=self.record[0][1]),
            "front": PointSet(x=self.record[1][0], y=self.record[1][1]),
            "rear": PointSet(x=self.record[2][0], y=self.record[2][1])
        }

    def x_front(self):
        return self.x + np.cos(self.o) * self.car.a

    def y_front(self):
        return self.y + np.sin(self.o) * self.car.a

    def x_back(self):
        return self.x - np.cos(self.o) * self.car.b

    def y_back(self):
        return self.y - np.sin(self.o) * self.car.b


if __name__ == "__main__":
    car = car.CarModel()
    gps = GPS(car=car)
    gps.start_record()
    state = gps()
    tstep = .001
    state = car.make_test_state(Ux=0, Uy=0, r=0)
    for i in range(1000):
        action = Action(-.001*i, 100, 100)
        state = car(state=state, action=action, time=tstep)
        print(list(gps()) + list(car.state().items()))
    # print([np.sqrt((xf - xb)**2 + (yf - yb)**2) for xf, yf, xb, yb in zip(*gps.get_records()['front'], *gps.get_records()['back'])])
    handles = []
    for name, points in gps.get_records().items():
        handles.append(plt.plot(points.x, points.y, label=name)[0])
    plt.legend(handles=handles)
    plt.show()