import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from shapely.geometry import LineString, Point

class Path():
    interp_offset = 0.1
    new_s_max_distance = 1
    interp_interval = 0.1

    def __init__(self, x, y):
        assert len(x) == len(y), 'length of x and y are not equal'

        self.x = x
        self.y = y

        d = np.cumsum(
            [0] +
            [
                np.sqrt((x_o - x_n)**2 + (y_o - y_n)**2) for x_o, x_n, y_o, y_n in zip(
            self.x[:-1],
            self.x[1:],
            self.y[:-1],
            self.y[1:]
        )])
        self.l = np.sum([np.sqrt((x1-x2)**2 + (y1-y2)**2) for x1, x2, y1, y2 in zip(self.x[:-1], self.x[1:], self.y[:-1], self.y[1:])])
        self.d = d.tolist()
        dx = np.gradient(x)
        ddx = np.gradient(dx)
        dy = np.gradient(y)
        ddy = np.gradient(dy)
        self._kappa = (dx * ddy - dy * ddx) / np.power(np.power(dx, 2) + np.power(dy, 2), 3 / 2)
        self.kappa_interp = interp1d(self.d, self._kappa, fill_value=0, bounds_error=False)
        self.x_interp = interp1d(self.d, self.x, fill_value=self.x[-1], bounds_error=False)
        self.y_interp = interp1d(self.d, self.y, fill_value=self.y[-1], bounds_error=False)
        self.l = np.sum([np.sqrt((x1-x2)**2 + (y1-y2)**2) for x1, x2, y1, y2 in zip(self.x[:-1], self.x[1:], self.y[:-1], self.y[1:])])
        self.line = LineString(zip(self.x, self.y))

    def kappa(self, s):
        if s is not list:
            s = [s]
        return [self.kappa_interp(i) for i in s]

    def length(self):
        return self.l

    def s(self, x, y, old_s=None):
        if old_s is not None:
            line = LineString([self.line.interpolate(s) for s in np.arange(start=old_s, stop=old_s + self.new_s_max_distance, step=self.interp_interval)])
            return old_s + line.project(Point(x, y))
        else:
            return self.line.project(Point(x, y))

    def p(self, s):
        point = self.line.interpolate(s)
        return point.x, point.y

    def interpolate(self, x, y, s=None, old_s=None):
        if s is None:
            s = self.s(x, y, old_s)
        x_line = float(self.x_interp(s))
        y_line = float(self.y_interp(s))
        orientation = np.arctan2(
            self.y_interp(s+self.interp_offset) - self.y_interp(s-self.interp_offset),
            self.x_interp(s+self.interp_offset) - self.x_interp(s-self.interp_offset)
        )
        x_line_front = float(self.x_interp(s + self.interp_offset))
        y_line_front = float(self.y_interp(s + self.interp_offset))
        e = np.sqrt((x_line - x) ** 2 + (y_line - y) ** 2)*np.sign(np.cross([x_line_front - x_line, y_line_front - y_line], [x - x_line, y - y_line]))
        return e, s, orientation


def cospath(length, interval=0.1, y_scale=1, frequency=1):
    return cospath_decay(length=length, interval=interval, y_scale=y_scale, frequency=1, decay_amplitude=0, decay_frequency=0)


def cospath_decay(length, interval=0.1, y_scale=1, frequency=1, decay_amplitude=0, decay_frequency=0):
    x = np.array(range(0, int(np.ceil(length / interval)))) * interval
    y = [np.sin(t / x[-1] * frequency * 2 * np.pi * np.exp(decay_frequency*t)) * np.exp(decay_amplitude * t) * y_scale for t in x]
    return Path(x, y)


def circle_path(radius, interval=0.1, revolutions=1, decay = 0, reverse_direction=False):
    """
    creates a circular path
    :param radius: the radius of the circle
    :param interval: the distance between points in the path
    :param revolutions: float the number of times to go around the circle
    :param decay:
    :param reverse_direction:
    :return:
    """
    x = []
    y = []
    angle_interval = np.arccos((2*(radius**2) - interval **2) / (2*radius*radius))
    for angle in np.arange(start=0, step=angle_interval, stop=2*np.pi*revolutions - angle_interval):
        r = radius - radius * decay * angle/(2*np.pi)
        x_value = radius - np.cos(angle)*r
        x.append(x_value if not reverse_direction else -1*x_value)
        y.append(np.sin(angle)*r)

    return Path(x=y, y=x)


def strait_path(length, interval=0.1):
    x = np.arange(start=0, stop=length, step=interval)
    return Path(x=x, y=np.random.normal(0, .01, size=np.size(x)))

if __name__ == "__main__":
    path = circle_path(radius=10, interval=1, revolutions=.95, decay=0)
    plt.plot(path.x, path.y)
    print(path._kappa)

    points = [
        (7.5, 9, 1),
        (7.5, 11, 1),
        (11, 9, -1),
        (11, 11, -1),
        (-7.5, 9, 1),
        (-7.5, 11, 1),
        (-11, 9, -1),
        (-11, 11, -1),
        (1, 18, 1),
        (-1, 18, 1),
        (1, 21, -1),
        (-1, 21, -1)
    ]
    pass_test = True
    for x, y, sign in points:
        e, _, _ = path.interpolate(x, y)
        if e*sign < 0:
            pass_test = False
            print("({x}, {y}) should be {sign}, but is {notsign}".format(
                x=x,
                y=y,
                sign=sign,
                notsign=-1 * sign
            ))
            plt.scatter([x], [y], marker="+" if e > 0 else "_")
    if pass_test:
        print("pass all tests")
    plt.show()