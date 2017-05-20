import numpy as np
from matplotlib import pyplot as plt
from models.path import circle_path
from models.pidDriver import pidDriver

from bar import Progbar
from models.animate import CarAnimation
from models.car2 import CarModel

t_step = .001
# path = cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4)
path = circle_path(radius=40, interval=.1, revolutions=1.5, decay=.6)
# path = strait_path(10000)
model = CarModel()
state = model.start_state(Ux=9, Uy=0, r=0, path=path)
driver = pidDriver(V=15, kp=3 * np.pi / 180, x_la=15, car=model, lookahead=5)
bar = Progbar(target=int(path.length())+1)
data = []
while (not state.is_terminal()):
    data.append({
        **{
            attr: getattr(state, attr) for attr in ['Ux', 'Uy', 'r', 'e', 'delta_psi', 'wx','wy', 'wo', 's']
        },
        "kappa": state.kappa()
    })
    action = driver.get_policy([state])[0]
    state, dx, dy, do = model(state=state, action=action, time=t_step)
    bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux)])
bar.update(int(path.length())+1)
print(state)

records = model.get_records()
animator = CarAnimation(animate_car=True)
animator.animate(front_wheels=records['front'], rear_wheels=records['rear'], path=path, interval=1, states=data)

handles = []
for name, points in records.items():
    print(len(points.x))
    handles.append(plt.plot(points.x, points.y, label=name)[0])
handles.append(plt.plot(path.x, path.y, label='path')[0])
plt.legend(handles=handles)
plt.show()