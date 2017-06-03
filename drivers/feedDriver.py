import tensorflow as tf
from tensorflow.contrib import layers

from drivers.simple_driver import SimpleDriver
from drivers.state import State

class FeedDriver(SimpleDriver):

    def __init__(self, car_model, save_dir=None):
        self.car = car_model
        super(FeedDriver, self).__init__(save_dir=save_dir)

    def actor(self, scope, state, keep_prob, reuse=False):
        path_kappa = state[:,State.size():]
        state = state[:,:State.size()]
        mapping = State.array_value_mapping()
        with tf.variable_scope(scope):
            V = tf.get_variable(name="V", shape=[1], initializer=layers.xavier_initializer())
            t_factor = tf.get_variable('t_factor', shape=[1], initializer=layers.xavier_initializer())
            kp = tf.get_variable(name="kp", shape=[1], initializer=layers.xavier_initializer())
            x_la = tf.get_variable(name="x_la", shape=[1], initializer=layers.xavier_initializer())

            kappa = layers.fully_connected(
                inputs=path_kappa,
                num_outputs=1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer()

            )
            e = tf.expand_dims(input=state[:, mapping['e']], axis=1)
            Ux = tf.expand_dims(input=state[:, mapping['Ux']], axis=1)
            delta_psi = tf.expand_dims(input=state[:, mapping['delta_psi']], axis=1)

            delta_fb = -kp * (e + x_la*delta_psi)
            K = self.car.m * self.car.mdf / self.car.cy - self.car.m*(1-self.car.mdf)/self.car.cy
            beta = (self.car.b - (self.car.a * self.car.m * Ux ** 2) / (self.car.cy * self.car.l)) * kappa
            delta_ff = self.car.l * kappa + K * tf.pow(Ux, 2) * kappa - kp * x_la * beta
            delta = tf.minimum(tf.maximum(delta_fb + delta_ff, -1), 1)

            torque = tf.minimum(tf.maximum(t_factor * (V - Ux), -1), 1)
        return tf.concat([delta, torque, torque], axis=1)

if __name__ == "__main__":
    with tf.Session() as Session:
        model = FeedDriver(FeedDriver.get_car_model())
        model.test(sess=Session)