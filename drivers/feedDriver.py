import tensorflow as tf
from tensorflow.contrib import layers, rnn
import numpy as np

from drivers.simple_driver import SimpleDriver
from drivers.state import State

class FeedDriver(SimpleDriver):
    kappa_length = 100
    rnn_state_size = 50
    kappa_step_size = .25

    def __init__(self, car_model, save_dir=None):
        self.car = car_model
        super(FeedDriver, self).__init__(save_dir=save_dir)

    def actor(self, scope, state, keep_prob, reuse=False):
        path_kappa = state[:,State.size():]
        path = tf.concat([
            tf.expand_dims(path_kappa, axis=2),
            tf.expand_dims(
                tf.tile(
                    input=tf.expand_dims(
                        tf.constant(
                            np.arange(start=0, stop=self.kappa_step_size * self.kappa_length,
                                      step=self.kappa_step_size),
                            dtype=tf.float32
                        ),
                        axis=0),
                    multiples=[tf.shape(state)[0], 1]
                ),
                axis=2)
        ],
            axis=2
        )
        state = state[:,:State.size()]
        mapping = State.array_value_mapping()
        with tf.variable_scope(scope):
            conv1 = layers.conv2d(inputs=path, num_outputs=32, kernel_size=[16], stride=2, padding="SAME",
                                  scope="conv1")
            conv2 = layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=[8], stride=2, padding="SAME",
                                  scope="conv2")
            fw_init_state = (
                layers.fully_connected(
                    inputs=state,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="fw_init_state_c"
                ),
                layers.fully_connected(
                    inputs=state,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="fw_init_state_h"
                ))
            bw_init_state = (
                layers.fully_connected(
                    inputs=state,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="bw_init_state_c"
                ),
                layers.fully_connected(
                    inputs=state,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="bw_init_state_h"
                ))
            fw_cell = rnn.BasicLSTMCell(self.rnn_state_size, reuse=reuse)
            bw_cell = rnn.BasicLSTMCell(self.rnn_state_size, reuse=reuse)
            inputs = [
                tf.reduce_sum(t, axis=1)
                for t
                in tf.split(value=conv2, axis=1, num_or_size_splits=conv2.get_shape().dims[1].value)
            ]
            outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                initial_state_bw=bw_init_state,
                initial_state_fw=fw_init_state,
                scope="biLSTM"
            )
            lstm_states = tf.concat([state_fw[0], state_fw[1], state_bw[0], state_bw[1]], axis=1)

            val = layers.fully_connected(
                inputs=lstm_states,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer(),
                num_outputs=3,
                activation_fn=tf.tanh
            )
            kappa = tf.expand_dims(
                input=val[:,0],
                axis=1
            )
            kp = tf.expand_dims(
                input=tf.get_variable(
                    name="kp",
                    shape=[1],
                    initializer=tf.constant_initializer(3 * np.pi / 180)
                ) + val[:,1],
                axis=1
            )
            x_la = tf.expand_dims(
                input=tf.get_variable(
                    name="x_la",
                    shape=[1],
                    initializer=tf.constant_initializer(15)
                ) + val[:,2],
                axis=1
            )
            e = tf.expand_dims(input=state[:, mapping['e']], axis=1)
            Ux = tf.expand_dims(input=state[:, mapping['Ux']], axis=1)
            delta_psi = tf.expand_dims(input=state[:, mapping['delta_psi']], axis=1)

            delta_fb = -kp * (e + x_la * delta_psi)
            K = self.car.m * self.car.mdf / self.car.cy - self.car.m * (1 - self.car.mdf) / self.car.cy
            beta = (self.car.b - (self.car.a * self.car.m * tf.pow(Ux, 2)) / (self.car.cy * self.car.l)) * kappa
            delta_ff = self.car.l * kappa + K * tf.pow(Ux, 2) * kappa - kp * x_la * beta
            delta = (delta_fb + delta_ff) / self.car.max_del


            torque = layers.fully_connected(
                inputs=lstm_states,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer(),
                num_outputs=1,
                activation_fn=tf.tanh
            )

            # torque = (V - Ux)*t_factor
        return tf.concat([delta, torque, torque], axis=1)

    def critic(self, scope, state, action, keep_prob, reuse=False):
        """

        :param scope: 
        :param state: 
        :param action
        :param keep_prob: 
        :param reuse: 
        :return: 
        """
        path = tf.concat([
            tf.expand_dims(state[:, State.size():], axis=2),
            tf.expand_dims(
                tf.tile(
                    input=tf.expand_dims(
                        tf.constant(
                            np.arange(start=0, stop=self.kappa_step_size * self.kappa_length,
                                      step=self.kappa_step_size),
                            dtype=tf.float32
                        ),
                        axis=0),
                    multiples=[tf.shape(state)[0], 1]
                ),
                axis=2)
        ],
            axis=2
        )
        state_action = tf.concat([state[:, :State.size()], action], axis=1)
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = layers.conv2d(inputs=path, num_outputs=32, kernel_size=[16], stride=2, padding="SAME",
                                  scope="conv1")
            conv2 = layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=[8], stride=2, padding="SAME",
                                  scope="conv2")
            fw_init_state = (
                layers.fully_connected(
                    inputs=state_action,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="fw_init_state_c"
                ),
                layers.fully_connected(
                    inputs=state_action,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="fw_init_state_h"
                ))
            bw_init_state = (
                layers.fully_connected(
                    inputs=state_action,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="bw_init_state_c"
                ),
                layers.fully_connected(
                    inputs=state_action,
                    biases_initializer=layers.xavier_initializer(),
                    num_outputs=self.rnn_state_size,
                    reuse=reuse,
                    weights_initializer=layers.xavier_initializer(),
                    scope="bw_init_state_h"
                ))
            fw_cell = rnn.BasicLSTMCell(self.rnn_state_size, reuse=reuse)
            bw_cell = rnn.BasicLSTMCell(self.rnn_state_size, reuse=reuse)
            inputs = [
                tf.reduce_sum(t, axis=1)
                for t
                in tf.split(value=conv2, axis=1, num_or_size_splits=conv2.get_shape().dims[1].value)
            ]
            outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                initial_state_bw=bw_init_state,
                initial_state_fw=fw_init_state,
                scope="biLSTM"
            )
            lstm_states = tf.concat([state_fw[0], state_fw[1], state_bw[0], state_bw[1]], axis=1)
            lstm_states_expand = tf.expand_dims(lstm_states, axis=2)
            w1 = tf.tile(
                tf.get_variable(
                    shape=[1, state_action.shape.dims[1].value, 4 * self.rnn_state_size],
                    initializer=layers.xavier_initializer(),
                    name="w1"
                ),
                multiples=[tf.shape(state_action)[0], 1, 1]
            )
            state_expand = tf.tile(tf.expand_dims(input=state_action, axis=1), multiples=[1, self.critic_hidden_length, 1])
            b1 = tf.get_variable(shape=[self.critic_hidden_length], initializer=layers.xavier_initializer(), name="b1")
            v1 = tf.reduce_mean(tf.matmul(tf.matmul(state_expand, w1), lstm_states_expand), axis=2) + b1
            linear1 = layers.fully_connected(
                inputs=tf.concat([v1, lstm_states, state_action], axis=1),
                biases_initializer=layers.xavier_initializer(),
                num_outputs=self.critic_hidden_length,
                reuse=reuse,
                weights_initializer=layers.xavier_initializer(),
                activation_fn=tf.tanh,
                scope="connected_1"
            )
            linear2 = layers.fully_connected(
                inputs=linear1,
                biases_initializer=layers.xavier_initializer(),
                num_outputs=1,
                reuse=reuse,
                weights_initializer=layers.xavier_initializer(),
                scope="connected_2"
            )
            return linear2


if __name__ == "__main__":
    with tf.Session() as Session:
        model = FeedDriver(FeedDriver.get_car_model())
        model.test(sess=Session)