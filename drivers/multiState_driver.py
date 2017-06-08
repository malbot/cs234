import tensorflow as tf
from tensorflow.contrib import layers, rnn
import numpy as np
import os
import random
import time
import sys

from drivers.state2 import StateSimple as State
from drivers.action import Action
from models.car_simple import CarSimple
from models.path import circle_path, cospath, cospath_decay, strait_path
from drivers.pidDriver import pidDriver
from bar import Progbar
from models.animate import CarAnimation

class MultiStateDriver(object):
    kappa_length = 40
    rnn_state_size = 50
    kappa_step_size = .25
    critic_hidden_length = 100
    actor_hidden_length = 100
    gamma = .9  # reward discount
    lr = 1e-3
    clip_norm = 10
    c_scope = "q"
    c_target_scope = "q_target"
    a_scope = "pi"
    a_target_scope = "pi_target"
    t_step = 0.01
    batch_size = 130
    random_action_stddev = 0.005  # stddev of percentage of noise to add to actions
    initial_velocity = 10
    alpha = 0 #0.01
    tau = 0.01 # learning rate of the target networks
    keep_prob = .9
    pretrain_iterations = 1000
    train_epochs = 10
    train_episodes = 100
    train_buffer = 100
    history = 5

    def __init__(self, car_model = None, save_dir=None):
        """
        adds the placeholders that will be used by both the actor and critic
        actor_state: placeholder for states to feed to actor
        critic_state: placeholder for states to feed to critic
        action: action taken while training actor
        g: reward G, the total sum of rewards for a single episode from the given state onward (excludes previous rewards)
        r: reward R, the reward for a single (s,a,s') tuple
        :return: 
        """

        if save_dir is None:
            save_dir = self.default_save_dir()

        print("Initializing {class_name} model".format(class_name=type(self).__name__))

        self.save_path = "{dir}/{name}".format(name=type(self).__name__, dir=save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.car = car_model if car_model is not None else self.get_car_model()

        # set placeholders
        self.current_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.history, State.size() + self.kappa_length])
        self.next_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.history, State.size() + self.kappa_length])
        self.current_action_placeholder = tf.placeholder(tf.float32, shape=[None, Action.size()])
        self.reward_placeholder = tf.placeholder(tf.float32, shape=[None])
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope("simple_driver"):
            self.pi = self.actor(scope=self.a_scope, state=self.current_state_placeholder, keep_prob=self.keep_prob_placeholder)
            self.pi_target = self.actor(scope=self.a_target_scope, state=self.next_state_placeholder, keep_prob=1)
            self.pi_noisy = self.noisy_actor(pi=self.pi)
            self.q = self.critic(state=self.current_state_placeholder, action=self.current_action_placeholder, scope=self.c_scope, keep_prob=self.keep_prob_placeholder)
            self.q_target = self.critic(state=self.next_state_placeholder, action=self.pi_target, scope=self.c_target_scope, keep_prob=1)

            self.c_loss = self.critic_loss(q=self.q, q_target=self.q_target, r=self.reward_placeholder)
            self.a_loss = self.actor_loss(pi=self.pi, q_scope=self.c_scope)
            self.a_lost_pretrain = self.pretrain_actor_loss(self.pi)

            self.c_train, self.c_norm = self.get_gradient(f=self.c_loss, grad_scope=self.c_scope, scope="c_train")
            self.a_train, self.a_norm = self.get_gradient(f=self.a_loss, grad_scope=self.a_scope, scope="a_train")
            self.a_pretrain, self.a_norm_pretrain = self.get_gradient(f=self.a_lost_pretrain, grad_scope=self.a_scope, scope="a_pretrain")

            self.update_actor_target = self.update_target_network(learning_network_scope=self.a_scope,
                                                                  target_network_scope=self.a_target_scope)
            self.update_critic_target = self.update_target_network(learning_network_scope=self.c_scope,
                                                                   target_network_scope=self.c_target_scope)

    @staticmethod
    def default_save_dir():
        return "logs"

    def actor(self, scope, state, keep_prob, reuse=False):
        path_kappa = state[:,self.history-1,State.size():]
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
        state = state[:,0,:State.size()]
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

        :param scope: 
        :param state: 
        :param action
        :param keep_prob: 
        :param reuse: 
        :return: 
        """
        path_kappa = state[:,self.history-1,State.size():]
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
        with tf.variable_scope(scope, reuse=reuse):
            if sys.version_info[0] == 2:
                history_cell = rnn.BasicLSTMCell(self.rnn_state_size)
                prediction_cell = rnn.BasicLSTMCell(self.rnn_state_size*2)
            elif sys.version_info[0] == 3:
                history_cell = rnn.BasicLSTMCell(self.rnn_state_size, reuse=reuse)
                prediction_cell = rnn.BasicLSTMCell(self.rnn_state_size*2, reuse=reuse)
            else:
                raise EnvironmentError("WTF python version is this?")

            inputs = [state[:, i, :] for i in range(self.history)]
            _, history_lstm_states = rnn.static_rnn(
                cell=history_cell,
                inputs=inputs,
                scope="historyLSTM",
                dtype=tf.float32
            )
            history = layers.fully_connected(
                inputs=tf.concat([history_lstm_states[0], history_lstm_states[1], action], axis=1),
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer(),
                activation_fn=tf.tanh,
                reuse=reuse,
                scope="action_embedding",
                num_outputs=self.critic_hidden_length*2
            )
            history2 = layers.fully_connected(
                inputs=history,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=layers.xavier_initializer(),
                activation_fn=tf.tanh,
                reuse=reuse,
                scope="action_embedding2",
                num_outputs=self.critic_hidden_length * 2
            )
            _, state = rnn.static_rnn(
                cell=prediction_cell,
                inputs=[path[:, i, :] for i in range(self.kappa_length)],
                initial_state=tf.tuple([
                    history2[:, self.critic_hidden_length:],
                    history2[:, :self.critic_hidden_length]
                ])
            )
            linear1 = layers.fully_connected(
                inputs=tf.concat([state[0], state[1]], axis=1),
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

    def noisy_actor(self, pi):
        """
        returns actions with small randomness added
        :param pi: the non-noisy actor
        :return: 
        """
        random_action = pi + tf.random_normal(shape=tf.shape(pi), mean=0, stddev=0.05)
        return random_action

    def critic_loss(self, q, q_target, r):
        """
        loss function for the critic
        :param q: critic NN
        :param q_target:
        :param r: rewards
        :return: 0d tensor of average loss over all batches
        """
        return tf.reduce_mean(tf.square(r + self.gamma * q_target - q))

    def actor_loss(self, pi, q_scope):
        """
        returns the loss tensor for actor network pi, using critic network in scope q_scope as for finding the quality
        of actions taken by the actor network
        :param pi: actor network to evaluate
        :param q_scope: scope of critic network, not target critic network
        :return: 
        """
        q = self.critic(state=self.current_state_placeholder, action=pi, scope=q_scope, reuse=True, keep_prob=1)
        grad = tf.reduce_mean(-1 * q)
        return grad

    def update_target_network(self, learning_network_scope, target_network_scope):
        """

        :param learning_network_scope: 
        :param target_network_scope: 
        :return: 
        """
        learning = {
            "/".join([k for k in t.name.split("/") if k != learning_network_scope]): t for t in
            tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            if "/{0}/".format(learning_network_scope) in t.name
        }
        target = {
            "/".join([k for k in t.name.split("/") if k != target_network_scope]): t for t in
            tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            if "/{0}/".format(target_network_scope) in t.name
        }
        assignment = tf.tuple([
            tf.assign(value=learning[n] * self.tau + (1 - self.tau) * target[n], ref=target[n])
            for n in learning.keys()
        ])
        return assignment

    def pretrain_actor_loss(self, pi):
        """
        used for pre-training the actor against a simple driver
        :param pi: actor NN
        :return: 0d tensor of average loss over all batches
        """
        return tf.reduce_mean(tf.square(pi - self.current_action_placeholder))

    def get_gradient(self, f, grad_scope, scope):
        """
        returns the training function for a loss function
        :param f: the loss function to take gradient of
        :param grad_scope: scope of variables to include in output 
        :param scope: new scope of gradient functions
        :return: tensor that applies gradients to variables of loss tensor that are of scope grad_scope
        """
        opt = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = opt.compute_gradients(f)
        grads_and_vars = [
            (tf.clip_by_norm(t=grad, clip_norm=self.clip_norm), var)
            for grad, var in grads_and_vars
            if "/{0}/".format(grad_scope) in var.name
        ]
        with tf.variable_scope(scope):
            opt = opt.apply_gradients(grads_and_vars)
            norm = tf.global_norm([grad for grad, var in grads_and_vars])
        return opt, norm

    def init(self, session):
        """
        initializes the graph network/session
        :return: 
        """
        init = tf.global_variables_initializer()
        session.run(init)

    def train_critic(self, session, state, action, reward, next_state):
        """
        applies the gradient of a single batch to the critic, given (s,a,r,s') tuple
        :param session: tensorflow session
        :param state: is the current state s in the tuple
        :param action: is the action a from the tuple
        :param reward: is the reward for taking the (s,a) in the tuple
        :param next_state: the next state after state s and action a
        :return: 
        """
        input_feed_dict = {
            self.reward_placeholder: reward,
            self.current_state_placeholder: state,
            self.next_state_placeholder: next_state,
            self.current_action_placeholder: action,
            self.keep_prob_placeholder: self.keep_prob
        }

        output_feed = [self.c_train, self.c_loss, self.c_norm]

        _, loss, norm = session.run(output_feed, input_feed_dict)

        return loss, norm

    def train_actor(self, session, state):
        """
        applies the gradient of a single batch to the actor
        :param session: tensorflow session
        :param state: states to take gradient of actor at
        :return: 
        """
        input_feed_dict = {
            self.current_state_placeholder: state,
            self.keep_prob_placeholder: self.keep_prob
        }

        output_feed = [self.a_train, self.a_loss, self.pi, self.a_norm]

        _, loss, a, norm = session.run(output_feed, input_feed_dict)
        input_feed_dict[self.current_action_placeholder] = a
        # q = session.run([self.q], input_feed_dict)
        return loss, norm

    def pretrain_actor(self, session, state, action):
        """

        :param session: 
        :param state: 
        :param action: 
        :return: 
        """
        input_feed_dict = {
            self.current_state_placeholder: state,
            self.current_action_placeholder: action,
            self.keep_prob_placeholder: self.keep_prob
        }

        output_feed = [self.a_pretrain, self.a_lost_pretrain, self.pi]

        _, loss, a = session.run(output_feed, input_feed_dict)
        return loss, a

    def update_targets(self, session):
        """
        updates the target networks using the learning networks
        :param session: 
        :return: 
        """
        input_feed_dict = {}
        output_feed = [self.update_actor_target, self.update_critic_target]
        session.run(output_feed, input_feed_dict)

    def get_noisy_action(self, session, state):
        """
        returns the actions the Actor thinks should be take for each state
        :param session: tensorflow session
        :param state: states to get actions of
        :return: 
        """
        input_feed_dict = {
            self.current_state_placeholder: state,
            self.keep_prob_placeholder: 1
        }

        output_feed = [self.pi_noisy]

        results = session.run(output_feed, input_feed_dict)
        return results[0]

    def get_action(self, session, state):
        """
        returns the actions the Actor thinks should be take for each state
        :param session: tensorflow session
        :param state: states to get actions of
        :return: 
        """
        input_feed_dict = {
            self.current_state_placeholder: state,
            self.keep_prob_placeholder: 1
        }

        output_feed = [self.pi]

        results = session.run(output_feed, input_feed_dict)
        return results[0]

    def get_q(self, session, state, action):
        """

        :param session: 
        :param state: 
        :param action: 
        :return: 
        """
        input_feed_dict = {
            self.current_action_placeholder: action,
            self.current_state_placeholder: state,
            self.keep_prob_placeholder: 1
        }
        output_feed = [self.q]
        results = session.run(output_feed, input_feed_dict)
        return results[0]

    def run_training(self, session, car, paths, episodes, replay_buffer_size, replay):
        """

        :param session: 
        :param car: 
        :param paths: 
        :param episodes: 
        :param replay_buffer_size: 
        :param replay:
        :return: 
        """
        experience = []
        bar = Progbar(target=np.sum(paths[i % len(paths)].length() for i in range(episodes)))
        bar_offset = 0
        for epi in range(episodes):
            state = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=paths[epi % len(paths)])
            history = [
                state.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size)
                for _ in range(self.history)
            ]
            c = 0
            while not state.is_terminal():
                a = self.get_noisy_action(session, np.expand_dims(history, axis=0))[0]
                action = Action.get_action(a, max_delta=car.max_del, max_t=car.max_t)
                sp, _, _, _ = car(
                    state=state,
                    action=action,
                    time=self.t_step
                )
                history.append(sp.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size))
                state = sp
                c += 1
                if c > self.history:
                    r = sp.reward(t_step=self.t_step, previous_state=state, previous_action=action)
                    experience.append((
                        np.asarray(history[:-1]),
                        a,
                        r,
                        np.asarray(history[1:])
                    ))
                    if len(experience) > replay_buffer_size:
                        del experience[0]
                    for _ in range(replay):
                        batch = experience if len(experience) < self.batch_size else random.sample(experience, k=self.batch_size)
                        states = np.array([s for s, a, r, sp in batch])
                        actions = np.array([a for s, a, r, sp in batch])
                        rewards = np.array([r for s, a, r, sp in batch])
                        state_primes = np.array([sp for s, a, r, sp in batch])

                        critic_loss, critic_norm = self.train_critic(
                            session=session,
                            state=states,
                            action=actions,
                            reward=rewards,
                            next_state=state_primes
                        )
                        actor_loss, actor_norm = self.train_actor(session=session, state=states)
                        self.update_targets(session=session)

                        bar.update(current=bar_offset + state.s, exp_avg=[
                            ("critic loss", critic_loss),
                            ("critic norm", critic_norm),
                            ("actor loss", actor_loss),
                            ("actor norm", actor_norm)
                        ])

                        if len(experience) < replay_buffer_size:
                            break
                del history[0]
            bar_offset = state.path.length() + bar_offset
        bar.update(
            current=bar.target,
            exp_avg=[
                ("critic loss", critic_loss),
                ("critic norm", critic_norm),
                ("actor loss", actor_loss),
                ("actor norm", actor_norm)
            ]
        )

    def run_pretrain(self, session, car, other_driver, paths, num_episodes, reiterate, train_only_critic=False):
        """

        :param session: 
        :param car: 
        :param other_driver: 
        :param paths: 
        :param num_episodes: 
        :param reiterate: 
        :param train_only_critic:
        :return: 
        """
        training_tuples = []

        print("collecting s,a pairs")
        for t in range(num_episodes):
            print("learning episode {0}".format(t))
            path = paths[t % len(paths)]
            bar = Progbar(target=int(path.length()) + 1)
            s = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=path)
            state_history = []
            while not s.is_terminal():
                a = other_driver.get_action([s])[0]
                sp, _, _, _ = car(state=s, action=a, time=self.t_step)
                state_history.append(sp.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size))
                if len(state_history) > self.history:
                    training_tuples.append((
                        np.asarray(state_history[:-1]),
                        a.as_array(max_delta=car.max_del, max_t=car.max_t),
                        sp.reward(t_step=self.t_step, previous_action=a, previous_state=s),
                        np.asarray(state_history[1:])
                    ))

                    a = other_driver.get_noisy_action([s])[0]
                    sp, _, _, _ = car(state=s, action=a, time=self.t_step)
                    bar.update(int(s.s), exact=[("e", sp.e), ("Ux", sp.Ux), ("s", sp.s)])
                    del state_history[0]
                s = sp
            bar.target = int(s.s)
            bar.update(int(s.s))

        print("Training with {0} examples".format(len(training_tuples)))
        bar = Progbar(target=reiterate * len(training_tuples))
        loss = [0.0]
        critic_loss = 0
        for t in range(reiterate):
            random.shuffle(training_tuples)
            for i in range(0, len(training_tuples) - 1, self.batch_size):
                batch = training_tuples[i:i + self.batch_size]
                states = np.array([s for s, a, r, sp in batch])
                actions = np.array([a for s, a, r, sp in batch])
                rewards = np.array([r for s, a, r, sp in batch])
                state_primes = np.array([sp for s, a, r, sp in batch])
                if not train_only_critic:
                    actor_loss, a = self.pretrain_actor(session=session, state=states, action=actions)
                    loss.append(actor_loss * .01 + .99 * loss[-1])
                c_loss, mean = self.train_critic(
                    session=session,
                    state=states,
                    action=actions,
                    reward=rewards,
                    next_state=state_primes
                )
                critic_loss = c_loss * .01 + .99 * critic_loss
                self.update_targets(session=session)
                bar.update(
                    t * len(training_tuples) + i + len(batch),
                    exact=[("actor loss", loss[-1]), ("critic loss", critic_loss), ("critic norm", mean), ("i", i)]
                )
        return loss, critic_loss

    def test_model(self, session, path, car):
        """

        :param session: 
        :param path: 
        :param car: 
        :return: 
        """
        car.reset_record()
        state = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=path)
        bar = Progbar(target=int(path.length()) + 1)
        t = 0.0
        ux = []
        actions = []
        states = []
        q_values = []
        rewards = []
        history = [
            state.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size)
            for _ in range(self.history)
        ]
        while not state.is_terminal() and t < 1.5 * path.length() * self.initial_velocity:
            t += self.t_step
            action = self.get_action(session, np.expand_dims(history, axis=0))
            q_values.append(self.get_q(session=session, state=np.expand_dims(history, axis=0), action=action))
            a = np.reshape(action, newshape=[np.size(action, 1)])
            action = Action.get_action(a, max_delta=car.max_del, max_t=car.max_t)

            states.append(state)
            actions.append(action)

            state_p, _, _, _ = car(state=state, action=action, time=self.t_step)
            rewards.append(state_p.reward(t_step=self.t_step, previous_state=state, previous_action=action))

            state = state_p
            history.append(state_p.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size))
            del history[0]

            ux.append(state.Ux)
            bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])
        states.append(state)
        records = car.get_records()
        return np.sum(rewards), np.mean(ux), state.s, records, states, actions, q_values, rewards

    def save_model(self, session, paths, car):
        summary = ""
        results = []
        reward = []
        for path, i in zip(paths, range(1, len(paths) + 1)):
            r, ux, s, record, states, actions, q_values, rewards = self.test_model(session=session, path=path, car=car)
            summary += "Path {i}, r = {r}, mean speed = {ux}, total distance = {s}".format(
                i=i,
                ux=ux,
                r=r,
                s=s
            )
            results.append(({
                                "ux": ux,
                                "r": r,
                                "i": i,
                                "s": s,
                                "path": list(zip(path.x, path.y)),
                            },
                            record,
                            states,
                            actions,
                            rewards,
                            path,
                            q_values,
                            i
            ))
            reward.append(r)
        save_path = "{path}/{timestamp}_{reward:.3g}".format(
            reward=np.mean(reward),
            timestamp=time.time(),
            path=self.save_path
        )
        os.mkdir(save_path)
        saver = tf.train.Saver()
        saver.save(session, "{path}/model.ckpt".format(path=save_path))
        animator = CarAnimation(animate_car=True)
        # animator.animate(front_wheels=records['front'], rear_wheels=records['rear'], path=path, interval=1)
        print("Model saved to in directory {path}".format(path=save_path))
        for result, record, states, actions, rewards, path, q_values, i in results:
            animator.animate(
                front_wheels=record['front'],
                rear_wheels=record['rear'],
                path=path,
                save_to="{path}/animation_{i}".format(path=save_path, i=i),
                t_step=self.t_step
            )
            # with open("{path}/path_{i}.txt".format(path=save_path, i=i), "w") as file:
            #     json.dump(result, file)
            with open("{path}/states_{i}.txt".format(path=save_path, i=i), "w") as file:
                for t, s in zip(range(len(states)), states):
                    file.write("{t:.3g} || {s}\n".format(s=s, t=t))
            with open("{path}/action_{i}.txt".format(path=save_path, i=i), "w") as file:
                for t, a, q, r in zip(range(len(actions)), actions, q_values, rewards):
                    file.write("{t:.3g} || {a} => {q}, {r}\n".format(a=a, q=q, r=r, t=t))

        return save_path, np.mean(reward)

    def load_model(self, session):
        try:
            folders = [
                "{path}/{f}".format(path=self.save_path, f=f)
                for f
                in os.listdir(self.save_path)
                if os.path.isdir("{path}/{f}".format(path=self.save_path, f=f))
            ]
        except FileNotFoundError:
            folders = []
        if len(folders) > 0:
            list.sort(folders)
            best_model_path = folders[-1]
            saver = tf.train.Saver()
            try:
                saver.restore(session, "{path}/model.ckpt".format(path=best_model_path))
                print("Loaded model from {path}".format(path=best_model_path))
                return True
            except:
                print("Unable to load model from {path}".format(path=best_model_path))
                return False
        else:
            print("No model could be found")
            return False

    @staticmethod
    def get_car_model():
        return CarSimple()

    @staticmethod
    def get_baseline_driver_model(car_model):
        return pidDriver(V=15, kp=3 * np.pi / 180, x_la=15, car=car_model)

    def test(self, sess):
        training_paths = [
            # cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4),
            circle_path(radius=100, interval=.1, revolutions=.5, decay=0),
            # cospath_decay(length=200, y_scale=-10, frequency=2, decay_amplitude=0, decay_frequency=0),
            circle_path(radius=200, interval=.1, revolutions=.5, decay=0, reverse_direction=True),
            strait_path(length=100)
        ]
        test_paths = training_paths + [
            # cospath_decay(length=100, y_scale=-10, frequency=.5, decay_amplitude=0, decay_frequency=1.0e-4),
            circle_path(radius=50, interval=.1, revolutions=.5, decay=0),
            # cospath_decay(length=200, y_scale=-10, frequency=3, decay_amplitude=0, decay_frequency=0),
            circle_path(radius=150, interval=.1, revolutions=.5, decay=0)
        ]

        model = self.get_car_model()
        good_driver = self.get_baseline_driver_model(model)

        learning_driver = self
        learning_driver.init(sess)
        loaded = self.load_model(sess)
        if not loaded:
            print("pretrain")
            learning_driver.run_pretrain(session=sess, car=model, other_driver=good_driver, paths=training_paths,
                                         num_episodes=len(training_paths), reiterate=self.pretrain_iterations)
            save_path, r = self.save_model(
                session=sess,
                paths=test_paths,
                car=model
            )
            print("Post pre-training had average reward of {r}".format(r=r))
        else:
            print("Loaded older model, not pretraining")
        print("training")
        for i in range(self.train_epochs):
            learning_driver.run_training(
                session=sess,
                car=model,
                paths=training_paths,
                episodes=self.train_episodes,
                replay_buffer_size=self.train_buffer,
                replay=1
            )
            save_path, r = self.save_model(
                session=sess,
                paths=test_paths,
                car=model
            )
            print("Epoc {i} had average reward of {r}".format(i=i, r=r))
        # plt.semilogy(loss)
        # plt.show()
        print("testing")
        rewards = []
        for path in test_paths:
            reward, Ux, s, records, states, actions, q_values, _ = learning_driver.test_model(session=sess, path=path,
                                                                                              car=model)

            model.reset_record()
            state = model.start_state(Ux=learning_driver.initial_velocity, Uy=0, r=0, path=path)
            bar = Progbar(target=int(path.length()) + 1)
            t = 0.0
            r = 0.0
            ux = []
            while not state.is_terminal() and t < 1.5 * path.length() * learning_driver.initial_velocity:
                t += learning_driver.t_step
                action = good_driver.get_action([state])[0]
                state_p, _, _, _ = model(state=state, action=action, time=learning_driver.t_step)
                r += state_p.reward(t_step=learning_driver.t_step, previous_action=action, previous_state=state)
                state = state_p
                ux.append(state.Ux)
                bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])
            rewards.append((reward, r, Ux, np.mean(ux), s, state.s))
        print("\n")
        for nn, pid, nn_ux, pid_ux, nn_s, pid_s in rewards:
            print(
                "Reward diff- pid: {pid} ({pid_ux} [m/s] / {pid_s} [m]), nn: {nn} ({nn_ux} [m/s] / {nn_s} [m]), diff {diff}".format(
                    pid=pid,
                    nn=nn,
                    diff=(nn - pid) / pid,
                    pid_ux=pid_ux,
                    nn_ux=nn_ux,
                    pid_s=pid_s,
                    nn_s=nn_s
                ))

    def test_critic(self, sess):
        training_paths = [
            # cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4),
            circle_path(radius=100, interval=.1, revolutions=.5, decay=0),
            # cospath_decay(length=200, y_scale=-10, frequency=2, decay_amplitude=0, decay_frequency=0),
            circle_path(radius=200, interval=.1, revolutions=.5, decay=0, reverse_direction=True),
            strait_path(length=100)
        ]
        test_paths = training_paths + [
            # cospath_decay(length=100, y_scale=-10, frequency=.5, decay_amplitude=0, decay_frequency=1.0e-4),
            circle_path(radius=50, interval=.1, revolutions=.5, decay=0),
            # cospath_decay(length=200, y_scale=-10, frequency=3, decay_amplitude=0, decay_frequency=0),
            circle_path(radius=150, interval=.1, revolutions=.5, decay=0)
        ]

        model = self.get_car_model()
        good_driver = self.get_baseline_driver_model(model)

        learning_driver = self
        learning_driver.init(sess)
        loss = []
        error_rate = []
        for i in range(100):
            _, c_loss = learning_driver.run_pretrain(
                session=sess,
                car=model,
                other_driver=good_driver,
                paths=training_paths,
                num_episodes=len(training_paths),
                reiterate=100,
                train_only_critic=True
            )
            loss.append(c_loss)

            for path in test_paths:
                car = self.get_car_model()
                s = car.start_state(Ux=10, Uy=0, r=0, path=path)
                rewards = []
                states = [s]
                actions = []
                quality = []
                while not s.is_terminal():
                    a = good_driver.get_action(state_batch=[s])[0]
                    sp, _, _, _ = car(state=s, action=a, time=self.t_step)
                    states.append(sp)
                    actions.append(a)
                    rewards.append(sp.reward(t_step=self.t_step, previous_action=a, previous_state=s))
                    quality.append(self.get_q(
                        session=sess,
                        state=np.expand_dims(
                            s.as_array(kappa_step_size=self.kappa_step_size, kappa_length=self.kappa_length), axis=0),
                        action=np.expand_dims(a.as_array(max_delta=car.max_del, max_t=car.max_t), axis=0)
                    ))
                    s = sp
                discounted_rewards = [0.0]
                for r in rewards[::-1]:
                    discounted_rewards.append(discounted_rewards[-1] * self.gamma + r)
                discounted_rewards = discounted_rewards[::-1][:-1]
                error_rate.append(np.mean([abs(r - q) / max(abs(r), abs(q)) for r, q in zip(rewards, quality)]))
            print("Loss = {l}".format(l=loss[-10::]))
            print("error = {e}".format(e=error_rate[-10::]))


if __name__ == "__main__":
    with tf.Session() as Session:
        driver_model = MultiStateDriver()
        driver_model.test(sess=Session)