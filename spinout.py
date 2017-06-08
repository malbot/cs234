import numpy as np
from matplotlib import pyplot as plt

from bar import Progbar
from drivers.pidDriver import pidDriver
from models.animate import CarAnimation
from models.car2 import SimpleCar
from models.path import Path

with open("spinout_road.csv", 'r') as f:
    x, y = tuple(f)
    path = Path(x=[float(v) for v in x.split(",")], y=[-1*float(v) for v in y.split(",")])

with open("data.csv", "r") as f:
    keys = ["{}_z".format(key) for key in f.readline().split(",")]
    comparison = []
    for line in f:
        values = [float(item) for item in line.split(",")]
        comparison.append({
            k: v for k, v in zip(keys, values)
        })

# path = circle_path(radius=40, interval=.1, revolutions=1.5, decay=.6)
t_step = .001
v = 30
model = SimpleCar(muf_p=1.2, muf_s=1.0, mur_p=1.1, mur_s=0.9)
state = model.start_state(Ux=v, Uy=0, r=0, path=path)
state.e_max = 10
driver = pidDriver(V=v, kp=3 * np.pi / 180, x_la=15, car=model)
bar = Progbar(target=int(path.length())+1)
data = []
t = 0
while (not state.is_terminal()):
    data.append({
        **{
            attr: getattr(state, attr) for attr in ['Ux', 'Uy', 'r', 'e', 'delta_psi', 'wx', 'wy', 'wo', 's']
        },
        "kappa": state.kappa(),
        "time": t_step*t,
        **state.data,
        **comparison[min(t, len(comparison) - 1)],
        "Tv": np.sqrt(state.Ux**2 + state.Uy**2),
        "Tv_w": 0 if t == 0 else np.sqrt(((old_state.wx - state.wx)/t_step)**2 + ((old_state.wy - state.wy)/t_step)**2),
        "r_w": 0 if t == 0 else abs(old_state.wo - state.wo)/t_step
    })
    t += 1
    action = driver.get_action([state])[0]
    old_state = state
    state, dx, dy, do = model(state=state, action=action, time=t_step)
    bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux)])
bar.target = int(state.s)
bar.update(int(state.s))
print(state)

records = model.get_records()
animator = CarAnimation(animate_car=True)
animator.animate(front_wheels=records['front'], rear_wheels=records['rear'], path=path, interval=1)

handles = []
for name, points in records.items():
    print(len(points.x))
    handles.append(plt.plot(points.x, points.y, label=name)[0])
handles.append(plt.plot(path.x, path.y, label='path')[0])
plt.legend(handles=handles)
plt.show()
