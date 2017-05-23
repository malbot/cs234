import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
import random
from matplotlib import pyplot as plt

from drivers.abstract_driver import AbstractDriver
from drivers.state import State
from drivers.action import Action
from models.path import circle_path, cospath, cospath_decay
from models.car2 import CarModel
from drivers.pidDriver import pidDriver
from bar import Progbar
from models.animate import CarAnimation


class SimpleDriver(AbstractDriver):
    kappa_length = 20  # number of kappa values of the path to include
    critic_hidden_length = 100
    actor_hidden_length = 100
    gamma = .9  # reward discount
    lr = 0.001
    clip_norm = 10
    c_scope = "v"
    a_scope = "pi"
    t_step = 0.01
    batch_size = 100
    random_action_stddev = 0.05  # stddev of percentage of noise to add to actions
    initial_velocity = 10
    alpha = 0 #0.01

    def add_placeholders(self):
        """
        adds the placeholders that will be used by both the actor and critic
        actor_state: placeholder for states to feed to actor
        critic_state: placeholder for states to feed to critic
        action: action taken while training actor
        g: reward G, the total sum of rewards for a single episode from the given state onward (excludes previous rewards)
        r: reward R, the reward for a single (s,a,s') tuple
        :return: 
        """
        self.current_state = tf.placeholder(tf.float32, shape=[None, State.size() + self.kappa_length])
        self.next_state = tf.placeholder(tf.float32, shape=[None, State.size() + self.kappa_length])
        self.action = tf.placeholder(tf.float32, shape=[None, Action.size()])
        self.g = tf.placeholder(tf.float32, shape=[None])
        self.r = tf.placeholder(tf.float32, shape=[None])

    def critic(self, state, scope, reuse=False):
        """
        adding a 2-layer NN for the critic (state value prediction)
        :return: 
        """
        with tf.variable_scope(scope, reuse=reuse):
            # layer 1
            w1 = tf.get_variable(
                shape=[state.shape[1], self.critic_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w1"
            )
            b1 = tf.get_variable(
                shape=[self.critic_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b1"
            )
            v1 = tf.nn.softmax(tf.matmul(state, w1) + b1)

            # layer 2
            w2 = tf.get_variable(
                shape=[self.critic_hidden_length, 1],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w2"
            )
            b2 = tf.get_variable(
                shape=[1],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b2"
            )
            return tf.nn.softmax(tf.matmul(v1, w2) + b2)

    def actor(self, scope, reuse=False):
        """
        adding a 3-layer NN for the actor (action generation from current state)
        :return: 
        """
        with tf.variable_scope(scope, reuse=reuse):
            # layer 1
            w1 = tf.get_variable(
                shape=[self.current_state.shape[1], self.actor_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w1"
            )
            b1 = tf.get_variable(
                shape=[self.actor_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b1"
            )
            v1 = tf.matmul(self.current_state, w1) + b1

            # # layer 2
            # w2 = tf.get_variable(
            #     shape=[self.critic_hidden_length, self.critic_hidden_length],
            #     dtype=tf.float32,
            #     initializer=contrib.layers.xavier_initializer(),
            #     name="w2"
            # )
            # b2 = tf.get_variable(
            #     shape=[self.critic_hidden_length],
            #     dtype=tf.float32,
            #     initializer=contrib.layers.xavier_initializer(),
            #     name="b2"
            # )
            # v2 = tf.matmul(v1, w2) + b2

            # layer 3
            w3 = tf.get_variable(
                shape=[self.critic_hidden_length, Action.size()],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w3"
            )
            b3 = tf.get_variable(
                shape=[Action.size()],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b3"
            )
            return tf.matmul(v1, w3) + b3

    def noisy_actor(self, scope):
        """
        returns actions with small randomness added
        :param scope: must be same scope as actor scope
        :return: 
        """
        action = self.actor(scope=scope, reuse=True)
        random_action = action + tf.random_normal(shape=tf.shape(action), mean=0, stddev=0.05)*action
        return random_action

    def critic_loss(self, v):
        """
        loss function for the critic
        :param v: critic NN
        :return: 0d tensor of average loss over all batches
        """
        return tf.reduce_mean(tf.square(self.g - v)) + self.alpha*tf.norm(self.pi)

    def actor_loss(self, critic_scope, pi, v):
        """
        loss function for actor
        :param critic_scope: scope of the critic
        :param pi: actor NN
        :param v: critic NN
        :return: 0d tensor of average loss over all batches
        """
        next_v = self.critic(scope=critic_scope, state=self.next_state, reuse=True)
        return tf.reduce_mean(tf.square(pi - self.action)*(v - self.r - self.gamma*next_v))

    def pretrain_actor_loss(self, pi):
        """
        used for pre-training the actor against a simple driver
        :param pi: actor NN
        :return: 0d tensor of average loss over all batches
        """
        return tf.reduce_mean(tf.square(pi - self.action)) + self.alpha*tf.norm(self.pi)

    def get_training(self, loss, grad_scope):
        """
        returns the training function for a loss function
        :param loss: the loss function to take gradient of
        :param grad_scope: scope of variables to include in output 
        :return: tensor that applies gradients to variables of loss tensor that are of scope grad_scope
        """
        opt = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = opt.compute_gradients(loss)
        grads_and_vars = [
            (tf.clip_by_norm(t=grad, clip_norm=self.clip_norm), var)
            for grad, var in grads_and_vars
            if "/{0}/".format(grad_scope) in var.name
        ]
        return opt.apply_gradients(grads_and_vars)

    def init(self, session):
        """
        builds the necessary framework
        :return: 
        """
        self.add_placeholders()
        with tf.variable_scope("simple_driver"):
            self.v = self.critic(state=self.current_state, scope=self.c_scope)
            self.pi = self.actor(scope=self.a_scope)
            self.pi_noisy = self.noisy_actor(scope=self.a_scope)
            self.c_loss = self.critic_loss(v=self.v)
            self.a_loss = self.actor_loss(critic_scope=self.c_scope, pi=self.pi, v=self.v)
            self.a_lost_pretrain = self.pretrain_actor_loss(self.pi)
            self.c_train = self.get_training(loss=self.c_loss, grad_scope=self.c_scope)
            self.a_train = self.get_training(loss=self.a_loss, grad_scope=self.a_scope)
            self.a_pretrain = self.get_training(loss=self.a_lost_pretrain, grad_scope=self.a_scope)

        init = tf.global_variables_initializer()
        session.run(init)

    def train_critic(self, session, batch_g, batch_states):
        """
        applies the gradient of a single batch to the critic
        :param session: tensorflow session
        :param batch_g: batch of G values (rewards from current state to end of episode)
        :param batch_states: states corresponding to the G values
        :return: 
        """
        input_feed_dict = {}
        input_feed_dict[self.g] = batch_g
        input_feed_dict[self.current_state] = batch_states

        output_feed = [self.c_train]

        session.run(output_feed, input_feed_dict)

    def train_actor(self, session, batch_states, batch_actions, batch_next_states):
        """
        applies the gradient of a single batch to the actor
        :param session: tensorflow session
        :param batch_states: states to take gradient of actor at
        :param batch_actions: actions take from batch_states
        :param batch_next_states: the states the model ended at after taking batch_actions at batch_states
        :return: 
        """
        input_feed_dict = {}
        input_feed_dict[self.action] = batch_actions
        input_feed_dict[self.current_state] = batch_states
        input_feed_dict[self.next_state] = batch_next_states

        output_feed = [self.a_train]

        session.run(output_feed, input_feed_dict)

    def pretrain_actor(self, session, batch_states, batch_actions):
        input_feed_dict = {}
        input_feed_dict[self.action] = batch_actions
        input_feed_dict[self.current_state] = batch_states

        output_feed = [self.a_pretrain, self.a_lost_pretrain, self.pi]

        _, loss, a = session.run(output_feed, input_feed_dict)
        return loss, a

    def get_noisy_action(self, session, batch_states):
        """
        returns the actions the Actor thinks should be take for each state
        :param session: tensorflow session
        :param batch_states: states to get actions of
        :return: 
        """
        input_feed_dict = {}
        input_feed_dict[self.current_state] = batch_states

        output_feed = [self.pi_noisy]

        results = session.run(output_feed, input_feed_dict)
        return results[0]

    def get_action(self, session, batch_states):
        """
        returns the actions the Actor thinks should be take for each state
        :param session: tensorflow session
        :param batch_states: states to get actions of
        :return: 
        """
        input_feed_dict = {}
        input_feed_dict[self.current_state] = batch_states

        output_feed = [self.pi]

        results = session.run(output_feed, input_feed_dict)
        return results[0]

    def get_random_action(self, session, batch_states):
        """
        returns the actions the Actor thinks should be take for each state, with small noise added to it
        :param session: tensorflow session
        :param batch_states: states to get actions of
        :return: 
        """
        input_feed_dict = {}
        input_feed_dict[self.current_state] = batch_states

        output_feed = [self.pi]

        results = session.run(output_feed, input_feed_dict)
        return results[0]

    def discounted_rewards(self, r_list):
        if len(r_list) == 1:
            return r_list
        g_tp1 = self.discounted_rewards(r_list[1:])
        return [r_list[0] + self.gamma*g_tp1[0]] + g_tp1

    def run_training(self, session, car, paths, critic_iterations, actor_iterations, iterations):

        for _ in range(iterations):

            # collect s,g values for the current policy
            critic_tuples = []
            for _ in range(critic_iterations):
                state = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=random.choice(paths))
                states = []
                rewards = []
                while not state.is_terminal():
                    action = self.get_action(session, [state.as_array()])
                    state_prime, _, _, _ = car(state=state, action=Action.get_action(action), time=self.t_step)
                    r = state_prime.reward()
                    states.append([state.as_array()])
                    rewards.append(r)
                    state = state_prime
                g = self.discounted_rewards(rewards)
                critic_tuples.append(zip(states, g))

            # train the critic with the collected s,g values, order randomized to improve stability/convergence
            random.shuffle(critic_tuples)
            for i in range(0, int(np.floor((len(critic_tuples)-1)/self.batch_size)), self.batch_size):
                batch = critic_tuples[i:i+self.batch_size]
                state = np.array([s for s, r in batch])
                rewards = np.array([r for s, r in batch])
                self.train_critic(session=session, batch_states=state, batch_g=rewards)

            for _ in range(actor_iterations):
                pass

    def run_pretrain(self, session, car, other_driver, paths, num_episodes, reiterate):

        training_tuples = []

        print("collecting s,a pairs")
        for t in range(num_episodes):
            print("learning episode {0}".format(t))
            path = random.choice(paths)
            bar = Progbar(target=int(path.length())+1)
            state = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=path)
            while not state.is_terminal():
                action = other_driver.get_action([state])[0]
                training_tuples.append((state, action))
                state, _, _, _ = car(state=state, action=action, time=self.t_step)
                bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])
            bar.target = int(state.s)
            bar.update(int(state.s))

        print("Training with {0} examples".format(len(training_tuples)))
        bar = Progbar(target=reiterate*len(training_tuples))
        loss = [0.0]
        for t in range(reiterate):
            random.shuffle(training_tuples)
            for i in range(0, len(training_tuples) - 1, self.batch_size):
                batch = training_tuples[i:i + self.batch_size]
                states = np.array([
                    s.as_array(kappa_length=self.kappa_length)
                    for s, a in batch
                ])
                actions = np.array([
                    a.as_array(max_delta=car.max_del, max_t=car.max_t)
                    for s, a in batch
                ])
                l, a = self.pretrain_actor(session=session, batch_states=states, batch_actions=actions)
                loss.append(l*.01 + .99*loss[-1])
                bar.update(
                    t*len(training_tuples) + i + len(batch),
                    exact=[("loss", loss[-1]), ("i", i)]
                )
        return loss

    def test_model(self, session, path, car):
        car.reset_record()
        state = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=path)
        bar = Progbar(target=int(path.length())+1)
        t = 0.0
        r = 0.0
        ux = []
        while not state.is_terminal() and t < 1.5*path.length()*self.initial_velocity:
            t += self.t_step
            s = np.array(state.as_array(kappa_length=self.kappa_length))
            s = np.reshape(s, newshape=[1, np.alen(s)])
            action = self.get_action(session, s)
            a = np.reshape(action, newshape=[np.size(action, 1)])
            action = Action.get_action(a, max_delta=car.max_del, max_t=car.max_t)
            state, _, _, _ = car(state=state, action=action, time=self.t_step)
            r = state.reward() + self.gamma*r
            ux.append(state.Ux)
            bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])

        # records = model.get_records()
        # animator = CarAnimation(animate_car=True)
        # animator.animate(front_wheels=records['front'], rear_wheels=records['rear'], path=path, interval=1)
        return r, np.mean(ux), state.s

if __name__ == "__main__":
    training_paths = [
        cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4),
        circle_path(radius=100, interval=.1, revolutions=.8, decay=0),
        cospath_decay(length=200, y_scale=-10, frequency=2, decay_amplitude=0, decay_frequency=0),
        circle_path(radius=200, interval=.1, revolutions=.8, decay=0)
    ]
    test_paths = training_paths + [
        cospath_decay(length=100, y_scale=-10, frequency=.5, decay_amplitude=0, decay_frequency=1.0e-4),
        circle_path(radius=50, interval=.1, revolutions=.8, decay=0),
        cospath_decay(length=200, y_scale=-10, frequency=3, decay_amplitude=0, decay_frequency=0),
        circle_path(radius=150, interval=.1, revolutions=.8, decay=0)
    ]

    model = CarModel()
    good_driver = pidDriver(V=15, kp=3 * np.pi / 180, x_la=15, car=model, lookahead=5)

    with tf.Session() as sess:
        learning_driver = SimpleDriver()
        learning_driver.init(sess)
        loss = learning_driver.run_pretrain(session=sess, car=model, other_driver=good_driver, paths=training_paths, num_episodes=10, reiterate=100)
        # plt.semilogy(loss)
        # plt.show()

        rewards = []
        for path in test_paths:
            reward, Ux, s = learning_driver.test_model(session=sess, path=path, car=model)

            model.reset_record()
            state = model.start_state(Ux=learning_driver.initial_velocity, Uy=0, r=0, path=path)
            bar = Progbar(target=int(path.length()) + 1)
            t = 0.0
            r = 0.0
            ux = []
            while not state.is_terminal() and t < 1.5 * path.length() * learning_driver.initial_velocity:
                t += learning_driver.t_step
                action = good_driver.get_action([state])[0]
                state, _, _, _ = model(state=state, action=action, time=learning_driver.t_step)
                r = state.reward() + learning_driver.gamma*r
                ux.append(state.Ux)
                bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])
            rewards.append((reward, r, Ux, np.mean(ux), s, state.s))
        print("\n")
        for nn, pid, nn_ux, pid_ux, nn_s, pid_s in rewards:
            print("Reward diff- pid: {pid} ({pid_ux} / {pid_s}), nn: {nn} ({nn_ux} / {nn_s}), diff {diff}".format(
                pid=pid,
                nn=nn,
                diff=(nn-pid)/pid,
                pid_ux=pid_ux,
                nn_ux=nn_ux,
                pid_s=pid_s,
                nn_s=nn_s
            ))
