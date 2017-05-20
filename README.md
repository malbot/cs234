# cs234
final project for cs234

Teach a model to drive a car using Reinforcement learning.

2 different vehicle dynamic models are included, the first (car.py) calculates the e (distance from car to path), s (distance traveled along path), and \(\delta_\psi\) (angle between car and path) from the dynamics of the vehicle.

The second model (car2) calculates the values by interpolating the vehicles current position with the path. This is more accurate, however, the interpolation is somewhat computationally intensive.

Currently, only a simple feed-forward/feed-back controller exists, working on how to build a Actor-critic reinforcement learning vehicle controller.
