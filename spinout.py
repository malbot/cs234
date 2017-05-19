import numpy as np
from driver import Driver, State, Action
from path import Path
from car2 import CarModel
from matplotlib import pyplot as plt
from pidDriver import pidDriver
from animate import CarAnimation
from bar import Progbar

with open("spinout_road.csv", 'r') as f:
    x, y = tuple(f)
    path = Path(x = [float(v) for v in x.split(",")], y = [float(v) for v in y.split(",")])

t_step = .001
model = CarModel(muf_p=1.2, muf_s=1.0, mur_p=1.1, mur_s=0.9)
state = model.start_state(Ux=30, Uy=0, r=0, path=path)
state.e_max = 10
driver = pidDriver(V=30, kp=3 * np.pi / 180, x_la=15, car=model)
bar = Progbar(target=int(path.length())+1)
data = []
while (not state.is_terminal()):
    data.append({
        **{
            attr: getattr(state, attr) for attr in ['Ux', 'Uy', 'r', 'e', 'delta_psi', 'wx','wy', 'wo', 's', "road_orientation"]
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