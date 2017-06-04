import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy import cos

from models.path import Path
from models.gps import PointSet


class CarAnimation():

    def __init__(self, animate_car = False):
        self.animate_car = animate_car
        self.path_plot = None
        self.front_wheels_plot = None
        self.rear_wheels_plot = None
        self.car_plot = None
        self.rear_wheels = None
        self.front_wheels = None
        self.path = None
        self.states = None
        self.data_text = None

    def __call__(self, frame):

        if self.states is not None:
            state = self.states[frame]
            self.data_text.set_text("\n".join(["{0} = {1}".format(key, state[key]) for key in sorted(state.keys())]))

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

    def animate(self, front_wheels, rear_wheels, path, interval=200, states=None, save_to=None, t_step=1):
        """
        animates the front/rear wheels along the path
        :param front_wheels: the PointSet that the front wheels follow
        :param rear_wheels: the PointSet the rear wheels follow
        :param path: the Path the car is trying to follow
        :param interval: the interval in miliseconds that each frame is played
        :param states:
        :param save_to: if none, will display animiate, if string, will save to that file (adding extension)
        :param t_step: time step between states
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

        # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        # energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        self.data_text = ax.text(1.01, 0.0, '', transform=ax.transAxes)
        ani = FuncAnimation(fig=fig, func=self, frames=iter(range(0, len(front_wheels.x), 20)), interval=interval)

        if save_to is not None:
            mpl.verbose.set_level("helpful")
            ani.save(filename="{file}.mp4".format(file=save_to), writer="avconv", codec="libx264", fps=int(1/(t_step*20)))
        else:
            plt.show()
        plt.close()

if __name__ == "__main__":
    time_step = .1
    time = np.array(range(0, int(np.ceil(np.pi * 4 / time_step)))) * time_step
    x = np.array([cos(t) for t in time])
    p = Path(time, x)
    r_wheels = PointSet(time + .1, x + .1)
    f_wheels = PointSet(time - .1, x - .1)
    animation = CarAnimation(animate_car=True)
    animation.animate(front_wheels=f_wheels, rear_wheels=r_wheels, path=p)