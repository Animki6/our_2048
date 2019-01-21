# -*- coding: utf-8 -*-
import csv
import gym
import numpy as np
import cPickle as pickle
import gym_2048

do_render = False
resume = False


class Learning2048:
    def __init__(self):
        self.gamma = 0.99  # rewards in backpropagation calculating coefficient
        self.size = 4
        self.x_size = self.size**2
        self.hlayer_neurons_number = 200
        self.actions_number = 4
        self.update_interval = 10  # co ile gier robić update'a
        self.decay_rate = 0.8  # decay factor for rmsprop lieaky sum of grad^2
        self.learning_rate = 0.01
        self.epx, self.eph, self.epdlogp, self.epr = [], [], [], []
        self.illegal_actions = []
        self.illegal_moves_count = 0

        self.running_reward = None
        self.reward_sum = 0
        self.game_moves = 0
        self.game_number = 0

        if resume:
            self.model = pickle.load(open('save.p', 'rb'))
        else:
            self.model = {}
            # initialising Weights
            self.model['W1'] = np.random.randn(self.hlayer_neurons_number, self.x_size) / np.sqrt(2.0 / self.x_size)
            self.model['W2'] = np.random.randn(self.actions_number, self.hlayer_neurons_number) / np.sqrt(2.0 / self.hlayer_neurons_number)

        # two empty model-shaped dicts:
        self.grad_buffer = {k: np.zeros_like(v) for k, v in
                       self.model.iteritems()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  # rmsprop memory


    def write_out_csv(self):
        row = ','.join([str(self.game_number), str(self.game_moves), str(self.illegal_moves_count), str(max(self.current_state)), str(self.reward_sum)])
        with open('wyniki.csv', 'a') as fd:
            fd.write(row)
            fd.write('\n')


    def propagate_rewards(self, r):
      """ take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      running_add = 0
      for t in reversed(xrange(0, r.size)):
        running_add = running_add * self.gamma + r[t]
        discounted_r[t] = running_add
      return discounted_r


    def run_network_forward(self, x):
        # calculate output
        #return h_layer, output
        h_layer = np.dot(self.model['W1'], x)
        h_layer[h_layer<0] = 0  # ReLu
        out_raw = np.dot(self.model['W2'], h_layer)
        e_x = np.exp(out_raw - np.max(out_raw))
        output = e_x / e_x.sum()  # softmax

        return h_layer, output

    def update_weights(self, h, grad):
        # calculate weights
        # return weights
        # dW2 = np.dot(h.T, grad).ravel()
        dW2 = np.dot(h.T, grad).T
        dh = np.dot(grad, self.model['W2'])
        dh[dh<=0] = 0  # relu
        dW1 = np.dot(dh.T, self.epx)
        return {'W1': dW1, 'W2': dW2}

    def determine_action(self, net_output):
        y = np.zeros_like(net_output)
        # max_index = net_output.index(max(net_output))
        max_index = net_output.argmax()
        if max_index.size != 1:
            max_index = max_index[0]
        available_moves = [0, 1, 2, 3]
        action = max_index if np.random.uniform() < net_output[max_index] else np.random.choice(available_moves)
        # is this cheating? :
        while action in self.illegal_actions:
            available_moves.remove(action)
            if available_moves == []:
                break
            action = np.random.choice(available_moves)
        y[action] = 1
        return action, y


    def start_learning(self):
        env = gym.make('2048-v0')
        self.game_number = 0

        self.current_state = env.reset()

        self.illegal_actions = []
        xs, hs, dlogps, drs = [], [], [], []



        while True:
            if do_render:
                env.render()

            # policy_forward - czyli obliczenie na podstawie aktualniego wejscia i stanu sieci
            # unlike w projekcie z pongiem - nie liczymy różnicy miedzy stanem obecnym a poprzednim
            h_layer, output = self.run_network_forward(self.current_state)

            action, y = self.determine_action(output)
            while action in self.illegal_actions:
                # thanks God we have some randomness!
                action, y = self.determine_action(output)


            # robimy step i sprawdzamy czy nowa akcja też nie jest illegal
            self.current_state, reward, done, info = env.step(action)

            if reward == -1.0:
                # illegal move reward - prevent the same move!
                self.illegal_actions.append(action)
                self.illegal_moves_count += 1
            else:
                self.illegal_actions = []  # clear illegal actions if any
                # record some values for later use
                xs.append(self.current_state)  # observation
                hs.append(h_layer)  # hidden state
                dlogps.append(y - output)
                drs.append(reward)
                self.reward_sum += reward
                self.game_moves += 1


            if done:
                self.game_number += 1

                # TODO refactor x, h, (y-approx), rewards
                self.epx = np.vstack(xs)
                self.eph = np.vstack(hs)
                self.epdlogp = np.vstack(dlogps)
                self.epr = np.vstack(drs)
                xs, hs, dlogps, drs = [], [], [], []  # reset array memory

                fuzzy_rewards = self.propagate_rewards(self.epr)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                fuzzy_rewards -= np.mean(fuzzy_rewards)
                fuzzy_rewards /= np.std(fuzzy_rewards)

                # (y - approx) = (y - approx) * fuzzy_rewards
                self.epdlogp *= fuzzy_rewards
                d_weights = self.update_weights(self.eph, self.epdlogp)

                for k in self.model:
                    self.grad_buffer[k] += d_weights[k]  # accumulate grad over batch

                # perform RMSprop
                if self.game_number % self.update_interval == 0:
                    for key, value in self.model.iteritems():
                        g = self.grad_buffer[key]
                        self.rmsprop_cache[key] = self.decay_rate * self.rmsprop_cache[key] + (1 - self.decay_rate) * g**2
                        self.model[key] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[key]) + 1e-5)
                        self.grad_buffer[key] = np.zeros_like(value) # reset grad buffer

                # bookkeepineg????
                self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                if self.game_number % 100 == 0: pickle.dump(self.model, open('save.p', 'wb'))

                print('***************GAME_OVER***************\ngame %d:  finished, moves: %d, illegal: %d, reward: %f, reward/moves: %f, max: %d' % (self.game_number, self.game_moves, self.illegal_moves_count, self.reward_sum, self.reward_sum/self.game_moves, max(self.current_state)))
                self.write_out_csv()
                # reset game
                reward_sum = 0
                self.current_state = env.reset()
                self.game_moves = 0
                self.illegal_moves_count = 0
                self.illegal_actions = []



if __name__ == '__main__':

    instance = Learning2048()
    instance.start_learning()