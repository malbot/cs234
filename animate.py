import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation
from time import time
from path import Path
from gps import PointSet
from bar import Progbar

class CarAnimation():

    def __init__(self, animate_car = False):
        self.animate_car = animate_car

    def __call__(self, frame):

        if self.states is not None:
            state = self.states[frame]
            # self.bar.update(frame, exact=self.states[frame].items())
            self.data_text.set_text("\n".join(["{0} = {1}".format(attr, state[attr]) for attr in ["e", "kappa", 's', "delta_psi", "road_orientation", "wo"]]))
        else:
            self.bar.update(frame)

        if self.animate_car:
            self.car_plot.set_data(
                [self.front_wheels.x[frame], self.rear_wheels.x[frame]],
                [self.front_wheels.y[frame], self.rear_wheels.y[frame]]
            )
            return self.car_plot
        else:
            self.front_wheels_plot.set_data(self.front_wheels.x[:frame], self.front_wheels.y[:frame])
            self.rear_wheels_plot.set_data(self.rear_wheels.x[:frame], self.rear_wheels.y[:frame])
            return self.front_wheels, self.rear_wheels,

    def animate(self, front_wheels, rear_wheels, path, interval= 200, states = None):
        """
        animates the front/rear wheels along the path
        :param front_wheels: the PointSet that the front wheels follow
        :param rear_wheels: the PointSet the rear wheels follow
        :param path: the Path the car is trying to follow
        :param interval: the interval in miliseconds that each frame is played
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.grid()

        self.path_plot,  = ax.plot(path.x, path.y, '-', lw=2)
        self.front_wheels_plot,  = ax.plot([], [], lw=2)
        self.rear_wheels_plot,  = ax.plot([], [], lw=2)
        self.car_plot,  = ax.plot([], [], lw=3)
        self.rear_wheels = rear_wheels
        self.front_wheels = front_wheels
        self.path = path
        self.states = states
        self.bar = Progbar(target=len(front_wheels.x))

        # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        # energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        self.data_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ani = FuncAnimation(fig=fig, func=self, frames=range(0, len(front_wheels.x),4), interval=interval)

        plt.show()

if __name__ == "__main__":
    interval = .1
    time = np.array(range(0, int(np.ceil(np.pi*4/interval))))*interval
    x = np.array([cos(t) for t in time])
    path = Path(time, x)
    rear_wheels = PointSet(time + .1, x + .1)
    front_wheels = PointSet(time - .1, x - .1)
    animation = CarAnimation(animate_car=True)
    animation.animate(front_wheels=front_wheels, rear_wheels=rear_wheels, path=path)