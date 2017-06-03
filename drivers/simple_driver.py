# noinspection PyPep8
import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
import random
import time
import json
import os

# from drivers.abstract_driver import AbstractDriver
from drivers.state import State
from drivers.action import Action
from models.path import circle_path, cospath, cospath_decay, strait_path
from models.car2 import CarModel
from drivers.pidDriver import pidDriver
from bar import Progbar
from models.animate import CarAnimation


class SimpleDriver(object):
    kappa_length = 20  # number of kappa values of the path to include
    kappa_step_size = .5
    critic_hidden_length = 100
    actor_hidden_length = 100
    gamma = .999  # reward discount
    lr = 1e-5
    clip_norm = 10
    c_scope = "v"
    c_target_scope = "v_target"
    a_scope = "pi"
    a_target_scope = "pi_target"
    t_step = 0.01
    batch_size = 100
    random_action_stddev = 0.05  # stddev of percentage of noise to add to actions
    initial_velocity = 10
    alpha = 0 #0.01
    tau = 0.01 # learning rate of the target networks
    keep_prob = .9

    def __init__(self, save_dir=None):
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

        # set placeholders
        self.current_state_placeholder = tf.placeholder(tf.float32, shape=[None, State.size() + self.kappa_length])
        self.next_state_placeholder = tf.placeholder(tf.float32, shape=[None, State.size() + self.kappa_length])
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

    def critic(self, state, action, scope, keep_prob, reuse=False):
        """
        adding a 2-layer NN for the critic (state value prediction)
        :param state: 
        :param action: 
        :param scope: 
        :param keep_prob:
        :param reuse: 
        :return: 
        """
        with tf.variable_scope(scope, reuse=reuse):
            # layer 1
            w1 = tf.get_variable(
                shape=[state.shape[1] + action.shape[1], self.critic_hidden_length],
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
            v1 = tf.matmul(tf.nn.dropout(tf.concat([state, action], axis=1), keep_prob=keep_prob), w1) + b1

            # layer 2
            w2 = tf.get_variable(
                shape=[self.critic_hidden_length, self.critic_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w2"
            )
            b2 = tf.get_variable(
                shape=[self.critic_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b2"
            )
            v2 = tf.matmul(tf.nn.dropout(v1, keep_prob=keep_prob), w2) + b2

            # layer 3
            w3 = tf.get_variable(
                shape=[self.critic_hidden_length, 1],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w3"
            )
            b3 = tf.get_variable(
                shape=[1],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b3"
            )
            return tf.matmul(v2, w3) + b3

    def actor(self, scope, state, keep_prob, reuse=False):
        """
        adding a 3-layer NN for the actor (action generation from current state)
        :param scope: 
        :param state:
        :param keep_prob:
        :param reuse: 
        :return: 
        """
        with tf.variable_scope(scope, reuse=reuse):
            # layer 1
            w1 = tf.get_variable(
                shape=[state.shape[1], self.actor_hidden_length],
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
            v1 = tf.nn.tanh(tf.matmul(tf.nn.dropout(state, keep_prob=keep_prob), w1) + b1)

            # layer 2
            w2 = tf.get_variable(
                shape=[self.actor_hidden_length, self.actor_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="w2"
            )
            b2 = tf.get_variable(
                shape=[self.actor_hidden_length],
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer(),
                name="b2"
            )
            v2 = tf.nn.tanh(tf.matmul(tf.nn.dropout(v1, keep_prob=keep_prob), w2) + b2)

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
            return tf.nn.tanh(tf.matmul(v2, w3) + b3)

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
        return tf.reduce_mean(tf.square(r + self.gamma*q_target - q))

    def actor_loss(self, pi, q_scope):
        """
        returns the loss tensor for actor network pi, using critic network in scope q_scope as for finding the quality
        of actions taken by the actor network
        :param pi: actor network to evaluate
        :param q_scope: scope of critic network, not target critic network
        :return: 
        """
        q = self.critic(state=self.current_state_placeholder, action=pi, scope=q_scope, reuse=True, keep_prob=1)
        grad = tf.reduce_mean(-1*q)
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
            tf.assign(value=learning[n]*self.tau + (1-self.tau)*target[n], ref=target[n])
            for n in learning.keys()
        ])
        return assignment

    def pretrain_actor_loss(self, pi):
        """
        used for pre-training the actor against a simple driver
        :param pi: actor NN
        :return: 0d tensor of average loss over all batches
        """
        return tf.reduce_mean(tf.square(pi - self.current_action_placeholder)) + self.alpha * tf.norm(self.pi)

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
        R = []
        bar = Progbar(target=np.sum(paths[i % len(paths)].length() for i in range(episodes)))
        critic_loss = 0
        critic_norm = 0
        actor_loss = 0
        actor_norm = 0
        bar_offset = 0
        for epi in range(episodes):
            s = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=paths[epi % len(paths)])
            while not s.is_terminal():
                bar.update(current=bar_offset + s.s, exp_avg=[
                    ("critic loss", critic_loss),
                    ("critic norm", critic_norm),
                    ("actor loss", actor_loss),
                    ("actor norm", actor_norm)
                ])
                a = self.get_noisy_action(session, [s.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size)])[0]
                sp, _, _, _ = car(
                    state=s,
                    action=Action.get_action(a, max_delta=car.max_del, max_t=car.max_t),
                    time=self.t_step
                )
                r = sp.reward(t_step=self.t_step)
                R.append((
                    s.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size),
                    a,
                    r,
                    sp.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size)
                ))
                if len(R) > replay_buffer_size:
                    del R[0]
                state_p = sp
                for _ in range(replay):
                    batch = R if len(R) < self.batch_size else random.sample(R, k=self.batch_size)
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
                    if len(R) < replay_buffer_size:
                        break

                s = state_p
            bar_offset = s.path.length() + bar_offset
        bar.update(
            current=bar.target,
            exp_avg=[
                ("critic loss", critic_loss),
                ("critic norm", critic_norm),
                ("actor loss", actor_loss),
                ("actor norm", actor_norm)
            ]
        )


    def run_pretrain(self, session, car, other_driver, paths, num_episodes, reiterate):
        """
        
        :param session: 
        :param car: 
        :param other_driver: 
        :param paths: 
        :param num_episodes: 
        :param reiterate: 
        :return: 
        """
        training_tuples = []

        print("collecting s,a pairs")
        for t in range(num_episodes):
            print("learning episode {0}".format(t))
            path = paths[t % len(paths)]
            bar = Progbar(target=int(path.length())+1)
            s = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=path)
            while not s.is_terminal():
                a = other_driver.get_action([s])[0]
                sp, _, _, _ = car(state=s, action=a, time=self.t_step)
                training_tuples.append((
                    s.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size),
                    a.as_array(max_delta=car.max_del, max_t=car.max_t),
                    sp.reward(t_step=self.t_step),
                    sp.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size)
                ))

                a = other_driver.get_noisy_action([s])[0]
                sp, _, _, _ = car(state=s, action=a, time=self.t_step)
                bar.update(int(s.s), exact=[("e", sp.e), ("Ux", sp.Ux), ("s", sp.s)])
                s = sp
            bar.target = int(s.s)
            bar.update(int(s.s))

        print("Training with {0} examples".format(len(training_tuples)))
        bar = Progbar(target=reiterate*len(training_tuples))
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
                actor_loss, a = self.pretrain_actor(session=session, state=states, action=actions)
                c_loss, mean = self.train_critic(
                    session=session,
                    state=states,
                    action=actions,
                    reward=rewards,
                    next_state=state_primes
                )
                critic_loss = c_loss*.01 + .99*critic_loss
                self.update_targets(session=session)
                loss.append(actor_loss*.01 + .99*loss[-1])
                bar.update(
                    t*len(training_tuples) + i + len(batch),
                    exact=[("actor loss", loss[-1]), ("critic loss", critic_loss), ("critic norm", mean), ("i", i)]
                )
        return loss

    def test_model(self, session, path, car):
        """
        
        :param session: 
        :param path: 
        :param car: 
        :return: 
        """
        car.reset_record()
        state = car.start_state(Ux=self.initial_velocity, Uy=0, r=0, path=path)
        bar = Progbar(target=int(path.length())+1)
        t = 0.0
        r = 0.0
        ux = []
        actions = []
        states = []
        while not state.is_terminal() and t < 1.5*path.length()*self.initial_velocity:
            t += self.t_step
            s = np.array(state.as_array(kappa_length=self.kappa_length, kappa_step_size=self.kappa_step_size))
            s = np.reshape(s, newshape=[1, np.alen(s)])
            action = self.get_action(session, s)
            a = np.reshape(action, newshape=[np.size(action, 1)])
            action = Action.get_action(a, max_delta=car.max_del, max_t=car.max_t)

            states.append(state)
            actions.append(action)

            state, _, _, _ = car(state=state, action=action, time=self.t_step)
            r += state.reward(t_step=self.t_step)
            ux.append(state.Ux)
            bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])

        records = car.get_records()
        return r, np.mean(ux), state.s, records, states, actions

    def save_model(self, session, paths, car):
        summary = ""
        results = []
        reward = []
        for path, i in zip(paths, range(1, len(paths)+1)):
            r, ux, s, record, states, actions = self.test_model(session=session, path=path, car=car)
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
                "path": list(zip(path.x, path.y))
            },
                record,
                states,
                actions,
                path,
                i
            ))
            reward.append(r)
        save_path = "{path}/{reward:.3g}_{timestamp}".format(
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
        for result, record, states, actions, path, i in results:
            animator.animate(
                front_wheels=record['front'],
                rear_wheels=record['rear'],
                path=path,
                save_to="{path}/animation_{i}".format(path=save_path, i=i)
            )
            with open("{path}/path_{i}.txt".format(path=save_path, i=i), "w") as file:
                json.dump(result, file)
            with open("{path}/states_{i}.txt".format(path=save_path, i=i), "w") as file:
                for s in states:
                    file.write("{s}\n".format(s=s))
            with open("{path}/action_{i}.txt".format(path=save_path, i=i), "w") as file:
                for a in actions:
                    file.write("{a}\n".format(a=a))

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
            saver.restore(session, "{path}/model.ckpt".format(path=best_model_path))
            print("Loaded model from {path}".format(path=best_model_path))
        else:
            print("No model could be found")

    @staticmethod
    def get_car_model():
        return CarModel()

    def test(self, sess):
        training_paths = [
            cospath_decay(length=100, y_scale=-10, frequency=1, decay_amplitude=0, decay_frequency=1.0e-4),
            circle_path(radius=100, interval=.1, revolutions=.8, decay=0),
            cospath_decay(length=200, y_scale=-10, frequency=2, decay_amplitude=0, decay_frequency=0),
            circle_path(radius=200, interval=.1, revolutions=.8, decay=0),
            strait_path(length=200)
        ]
        test_paths = training_paths + [
            cospath_decay(length=100, y_scale=-10, frequency=.5, decay_amplitude=0, decay_frequency=1.0e-4),
            circle_path(radius=50, interval=.1, revolutions=.8, decay=0),
            cospath_decay(length=200, y_scale=-10, frequency=3, decay_amplitude=0, decay_frequency=0),
            circle_path(radius=150, interval=.1, revolutions=.8, decay=0)
        ]

        model = self.get_car_model()
        good_driver = pidDriver(V=15, kp=3 * np.pi / 180, x_la=15, car=model)

        learning_driver = self
        learning_driver.init(sess)
        self.load_model(sess)
        print("pretrain")
        loss = learning_driver.run_pretrain(session=sess, car=model, other_driver=good_driver, paths=training_paths,
                                            num_episodes=len(training_paths), reiterate=300)
        save_path, r = self.save_model(
            session=sess,
            paths=training_paths,
            car=model
        )
        print("Post pre-training had average reward of {r}".format(r=r))
        print("training")
        for i in range(10):
            learning_driver.run_training(
                session=sess,
                car=model,
                paths=training_paths,
                episodes=100,
                replay_buffer_size=100,
                replay=1
            )
            save_path, r = self.save_model(
                session=sess,
                paths=training_paths,
                car=model
            )
            print("Epoc {i} had average reward of {r}".format(i=i, r=r))
        # plt.semilogy(loss)
        # plt.show()
        print("testing")
        rewards = []
        for path in test_paths:
            reward, Ux, s, records, states, actions = learning_driver.test_model(session=sess, path=path, car=model)

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
                r += state.reward(t_step=learning_driver.t_step)
                ux.append(state.Ux)
                bar.update(int(state.s), exact=[("e", state.e), ("Ux", state.Ux), ("s", state.s)])
            rewards.append((reward, r, Ux, np.mean(ux), s, state.s))
        print("\n")
        for nn, pid, nn_ux, pid_ux, nn_s, pid_s in rewards:
            print("Reward diff- pid: {pid} ({pid_ux} [m/s] / {pid_s} [m]), nn: {nn} ({nn_ux} [m/s] / {nn_s} [m]), diff {diff}".format(
                pid=pid,
                nn=nn,
                diff=(nn - pid) / pid,
                pid_ux=pid_ux,
                nn_ux=nn_ux,
                pid_s=pid_s,
                nn_s=nn_s
            ))


if __name__ == "__main__":
    with tf.Session() as Session:
        driver_model = SimpleDriver()
        driver_model.test(sess=Session)
