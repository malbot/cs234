import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

from bar import Progbar
from drivers.pidDriver import pidDriver
from models.animate import CarAnimation
from models.car2 import CarModel
from models.path import circle_path, cospath_decay, strait_path

t_step = .001
# path = cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4)
# path = circle_path(radius=40, interval=.1, revolutions=1.5, decay=.6)
# path = circle_path(radius=200, interval=.1, revolutions=.8, decay=0)
path = strait_path(200)
# path = cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4)
# path = circle_path(radius=100, interval=.1, revolutions=.8, decay=0)
model = CarModel()
state = model.start_state(Ux=9, Uy=0, r=0, path=path)
driver = pidDriver(V=15, kp=3 * np.pi / 180, x_la=15, car=model)
bar = Progbar(target=int(path.length())+1)
data = []
t = 0
max_t = int(path.length()*1.5/driver.V)
print("Max t = {0}".format(max_t))
while not state.is_terminal(): # and t < max_t:
    t += t_step
    data.append({
        **{
            attr: getattr(state, attr) for attr in ['Ux', 'Uy', 'r', 'e', 'delta_psi', 'wx','wy', 'wo', 's']
        },
        "kappa": state.kappa()
    })
    if state.s > path.length():
        print(state)
    action = driver.get_action([state])[0]
    state, dx, dy, do = model(state=state, action=action, time=t_step)
    bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("t", t)])
bar.update(int(path.length())+1)
print(state)

records = model.get_records()
animator = CarAnimation(animate_car=True)
animator.animate(front_wheels=records['front'], rear_wheels=records['rear'], path=path, interval=1, states=data, save_to="test")

handles = []
for name, points in records.items():
    print(len(points.x))
    handles.append(plt.plot(points.x, points.y, label=name)[0])
handles.append(plt.plot(path.x, path.y, label='path')[0])
plt.legend(handles=handles)
plt.show()
