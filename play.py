# -*- coding: utf-8 -*-
import gym
import numpy as np
import cPickle as pickle
import gym_2048

do_render = False

gamma = 0.99  # rewards in backpropagation calclating coefficient
size = 4
resume = False
x_size = size*size
hlayer_neurons_number = 200
actions_number = 4
update_interval = 10 # co ile gier robić update'a
decay_rate = 0.99 # decay factor for rmsprop lieaky sum of grad^2
learning_rate = 1e-4
epx, eph, epdlogp, epr = [], [], [], []

if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  # initialising Weights
  model['W1'] = np.random.randn(hlayer_neurons_number,x_size) / np.sqrt(2.0/x_size)
  model['W2'] = np.random.randn(actions_number, hlayer_neurons_number) / np.sqrt(2.0/hlayer_neurons_number)

# two empty model-shaped dicts:
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def propagate_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def run_network_forward(x):
    # calculate output
    #return h_layer, output
    h_layer = np.dot(model['W1'], x)
    h_layer[h_layer<0] = 0  # ReLu
    out_raw = np.dot(model['W2'], h_layer)
    e_x = np.exp(out_raw - np.max(out_raw))
    output = e_x / e_x.sum()  # softmax

    return h_layer, output

def update_weights(h, grad, epx):
    # calculate weights
    # return weights
    # dW2 = np.dot(h.T, grad).ravel()
    dW2 = np.dot(h.T, grad).T
    dh = np.dot(grad, model['W2'])
    dh[dh<=0] = 0  # relu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2':dW2}

def determine_action(net_output, illegal):
    # very naive for now
    # TODO add some random factor
    y = np.zeros_like(net_output)
    # max_index = net_output.index(max(net_output))
    max_index = net_output.argmax()
    if max_index.size != 1:
        max_index = max_index[0]
    available_moves = [0, 1, 2, 3]
    # thats cheating:
    # while max_index in illegal:
    #     available_moves.remove(max_index)
    #     if available_moves == []:
    #         break
    #     max_index = np.random.choice(available_moves)
    y[max_index] = 1
    return max_index, y


def main():
    env = gym.make('2048-v0')
    game_number = 0

    current_state = env.reset()

    prev_x = None
    illegal_actions = []
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    game_moves = 0


    while True:
        if do_render:
            env.render()

        # policy_froward - czyli obliczenie na podstawie aktualniego wejscia i stanu sieci
        # unlike w projekcie z pongiem - nie liczymy różnicy miedzy stanem obecnym a poprzednim
        h_layer, output = run_network_forward(current_state)

        # record some values for later use
        xs.append(current_state)  # observation
        hs.append(h_layer)  # hidden state

        action, y = determine_action(output, illegal_actions)
        # print('Action: %d' % action)
        dlogps.append(y - output)

        # robimy step
        current_state, reward, done, info = env.step(action)
        game_moves += 1
        reward_sum += reward

        drs.append(reward)
        if reward == 0.0:
            # illegal move reward - prevent the same move!
            illegal_actions.append(action)
        else:
            illegal_actions = []  # clear illegal actions

        if done:
            game_number += 1

            # TODO refactor x, h, (y-approx), rewards
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            fuzzy_rewards = propagate_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            fuzzy_rewards -= np.mean(fuzzy_rewards)
            fuzzy_rewards /= np.std(fuzzy_rewards)

            # (y - approx) = (y - approx) * fuzzy_rewards
            epdlogp *= fuzzy_rewards
            d_weights = update_weights(eph, epdlogp, epx)

            for k in model:
                grad_buffer[k] += d_weights[k]  # accumulate grad over batch

            # perform RMSprop
            if game_number % update_interval == 0:
                for key, value in model.iteritems():
                    g = grad_buffer[key]
                    rmsprop_cache[key] = decay_rate * rmsprop_cache[key] + (1 - decay_rate) * g**2
                    model[key] += learning_rate * g / (np.sqrt(rmsprop_cache[key]) + 1e-5)
                    grad_buffer[key] = np.zeros_like(value) # reset grad buffer

            # bookkeepineg????
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            if game_number % 10 == 0: pickle.dump(model, open('save.p', 'wb'))

            print('***************GAME_OVER***************\ngame %d:  finished, moves: %d, reward: %f, reward/moves: %f, max: %d' % (game_number, game_moves, reward_sum, reward_sum/game_moves, max(current_state)))

            # reset game
            reward_sum = 0
            current_state = env.reset()
            prev_x = None
            game_moves = 0



if __name__ == '__main__':
    main()