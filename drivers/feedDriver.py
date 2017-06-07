import tensorflow as tf
from tensorflow.contrib import layers, rnn
import numpy as np

from drivers.simple_driver import SimpleDriver
from drivers.state import State

class FeedDriver(SimpleDriver):
    kappa_length = 3
    rnn_state_size = 50
    kappa_step_size = 1
    torque_nn_layers = 3
    pretrain_iterations = 100
    train_episodes = 50
    train_epochs = 20

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
            fw_cell = rnn.BasicLSTMCell(self.rnn_state_size)
            bw_cell = rnn.BasicLSTMCell(self.rnn_state_size)
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
                inputs=path_kappa,
                weights_initializer=tf.constant_initializer(1.0/self.kappa_length),
                biases_initializer=tf.constant_initializer(0),
                num_outputs=3
            )
            kappa = tf.expand_dims(
                input=val[:,0],
                axis=1
            )
            # kp = tf.expand_dims(
            #     input=tf.get_variable(
            #         name="kp",
            #         shape=[1],
            #         initializer=tf.constant_initializer(3 * np.pi / 180)
            #     ) + val[:,1],
            #     axis=1
            # )
            # x_la = tf.expand_dims(
            #     input=tf.get_variable(
            #         name="x_la",
            #         shape=[1],
            #         initializer=tf.constant_initializer(15)
            #     ) + val[:,2],
            #     axis=1
            # )
            kp = tf.get_variable(name="kp", shape=[1], initializer=tf.constant_initializer(3 *np.pi/180))
            x_la = tf.get_variable(name="x_la", shape=[1], initializer=tf.constant_initializer(15))
            e = tf.expand_dims(input=state[:, mapping['e']], axis=1)
            Ux = tf.expand_dims(input=state[:, mapping['Ux']], axis=1)
            delta_psi = tf.expand_dims(input=state[:, mapping['delta_psi']], axis=1)

            delta_fb = -kp * (e + x_la * delta_psi)
            K = self.car.m * self.car.mdf / self.car.cy - self.car.m * (1 - self.car.mdf) / self.car.cy
            beta = (self.car.b - (self.car.a * self.car.m * tf.pow(Ux, 2)) / (self.car.cy * self.car.l)) * kappa
            delta_ff = self.car.l * kappa + K * tf.pow(Ux, 2) * kappa - kp * x_la * beta
            delta = tf.tanh((delta_fb + delta_ff) / self.car.max_del)


            torque = layers.fully_connected(
                inputs=lstm_states,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer(),
                num_outputs=1,
                activation_fn=tf.sigmoid
            )

            # torque = (V - Ux)*t_factor
        return tf.concat([delta, torque, torque], axis=1)

    def critic(self, scope, state, action, keep_prob, reuse=False):
        """
    torque_nn_layers
        :param scope:
        :param state:
        :param action
        :param keep_prob:
        :param reuse:
        :return:
        """
        mapping = State.array_value_mapping()
        with tf.variable_scope(scope, reuse=reuse):
            values = tf.stack([
                state[:, mapping['s']],
                state[:, mapping['e']],
                state[:, mapping['Ux']],
                state[:, mapping['Uy']],
                state[:, mapping['r']],
                state[:, mapping['delta_psi']],
                action[:, 0],
                state[:, State.size()]
            ], axis=1)

            r = layers.fully_connected(
                inputs=values,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer(),
                num_outputs=1,
            )
            return r
    #
    # def critic(self, scope, state, action, keep_prob, reuse=False):
    #     """
    #
    #     :param scope:
    #     :param state:
    #     :param action
    #     :param keep_prob:
    #     :param reuse:
    #     :return:
    #     """
    #     path = tf.concat([
    #         tf.expand_dims(state[:, State.size():], axis=2),
    #         tf.expand_dims(
    #             tf.tile(
    #                 input=tf.expand_dims(
    #                     tf.constant(
    #                         np.arange(start=0, stop=self.kappa_step_size * self.kappa_length,
    #                                   step=self.kappa_step_size),
    #                         dtype=tf.float32
    #                     ),
    #                     axis=0),
    #                 multiples=[tf.shape(state)[0], 1]
    #             ),
    #             axis=2)
    #     ],
    #         axis=2
    #     )
    #     state_steering = tf.concat([
    #         state[:, :State.size()],
    #         tf.expand_dims(action[:,1], axis=1)
    #     ], axis=1)
    #     state_torque = tf.concat([
    #         tf.expand_dims(state[:, State.array_value_mapping()['Ux']], axis=1),
    #         tf.expand_dims(state[:, State.array_value_mapping()['delta_psi']], axis=1),
    #         tf.expand_dims(state[:, State.array_value_mapping()['e']], axis=1),
    #         state[:, State.size():],
    #         action[:,1:]
    #     ], axis=1)
    #     with tf.variable_scope(scope, reuse=reuse):
    #         conv1 = layers.conv2d(inputs=path, num_outputs=32, kernel_size=[16], stride=2, padding="SAME",
    #                               scope="conv1")
    #         conv2 = layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=[8], stride=2, padding="SAME",
    #                               scope="conv2")
    #         path_convolution = layers.flatten(conv2)
    #         linear1 = layers.fully_connected(
    #             inputs=tf.concat([path_convolution, state_steering], axis=1),
    #             biases_initializer=layers.xavier_initializer(),
    #             num_outputs=self.critic_hidden_length,
    #             reuse=reuse,
    #             weights_initializer=layers.xavier_initializer(),
    #             activation_fn=tf.tanh,
    #             scope="connected_1"
    #         )
    #         steering_quality = layers.fully_connected(
    #             inputs=linear1,
    #             biases_initializer=layers.xavier_initializer(),
    #             num_outputs=1,
    #             reuse=reuse,
    #             weights_initializer=layers.xavier_initializer(),
    #             scope="connected_2"
    #         )
    #
    #         v = tf.concat([path_convolution, state_torque], axis=1)
    #         for i in range(self.torque_nn_layers):
    #             v = layers.fully_connected(
    #                 inputs=v,
    #                 biases_initializer=layers.xavier_initializer(),
    #                 num_outputs=self.rnn_state_size if i < self.torque_nn_layers - 1 else 1,
    #                 reuse=reuse,
    #                 weights_initializer=layers.xavier_initializer(),
    #                 scope="fully_connected_layer_{0}".format(i)
    #             )
    #
    #         q = layers.fully_connected(
    #             inputs=tf.concat(steering_quality, v, axis=1),
    #             biases_initializer=layers.xavier_initializer(),
    #             num_outputs=1,
    #             reuse=reuse,
    #             weights_initializer=layers.xavier_initializer(),
    #             scope="final_quality"
    #         )
    #
    #         return q


if __name__ == "__main__":
    with tf.Session() as Session:
        model = FeedDriver(FeedDriver.get_car_model())
        model.test(sess=Session)
