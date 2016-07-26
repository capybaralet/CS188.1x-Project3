from pylab import *
import scipy
import scipy.stats
import numpy
np = numpy
from collections import OrderedDict
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query_cost', default=.01, type=float)
parser.add_argument('--fixed_mdp', default=1, type=int)
locals().update(parser.parse_args().__dict__)

fixed_battery=1


"""
The number of queries should be based on the total cost, NOT fixed. 

We should save more information (what??)

"""

# NTS: a lot of bugs caused by using "state" instead of "current_state"... maybe should renameNTS: a lot of bugs caused by using "state" instead of "current_state"... maybe should rename

# hyper-parameters:
grid_width = 8
epsilon = 0.1
prob_random_move = 0.1
prob_random_reset = 0.001
gamma = .9 # discount factor
prob_zero_reward = .9
learning_rate = .1

num_experiments = 200
#num_experiments = 1
num_steps = 100000

# states (lexical order)
states = range(grid_width**2)
# actions: stay, N, E, S, W
actions = range(5)


def row_and_column(state):
    return state / grid_width, state % grid_width

# basic environment dynamics
def next_state(state, action):
    row, column = row_and_column(state)
    if action == 1 and row > 0:
        return state - grid_width
    if action == 2 and column < grid_width - 1:
        return state + 1
    if action == 3 and row < grid_width - 1:
        return state + grid_width
    if action == 4 and column > 0:
        return state - 1
    else:
        return state

#################################################################
# generally useful functions

def updated_running_average(avg, new_value, update_n):
    return (avg * (update_n - 1) + new_value) / update_n

def multinomial_entropy(probs): # TODO: stability?
    return -np.sum(probs * np.log(probs))

def binomial_entropy(p):
    return multinomial_entropy([p, 1-p])
 
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#################################################################
# Bayesian Stuff
# TODO: precompute entropies?

# learn a distribution over states (dirichlet-multinomial) 
def state_entropy():
    return scipy.stats.dirichlet([sum(nvisit) + .5 for nvisit in nvisits]).entropy()
def state_probability(state):
    (sum(nvisits[state]) + .5) / (step + .5*len(states))


# learn a distribution over rewards (beta-bernoulli)
# we use a lookup table to speed up computations
# the list version was too large.  I'm trying a dictionary instead...
#beta_entropy_lookup = np.inf * np.ones((num_steps, num_steps))
beta_entropy_lookup = {}

def reward_entropy():
    # empirical counts
    alpha_ = total_r_observed[current_state][action]
    beta_ = nqueries[current_state][action] - alpha_
    if (alpha_, beta_) in beta_entropy_lookup:
        entry = beta_entropy_lookup[(alpha_, beta_)]
    else:
        entry = scipy.stats.beta(alpha_ + .5, beta_ + 1).entropy()
        beta_entropy_lookup[(alpha_, beta_)] = entry
    return entry

def expected_reward(state, action): 
    alpha = total_r_observed[state][action] + .5
    beta = ( nqueries[state][action] + 1) - alpha
    return alpha / (alpha + beta)#(total_r_observed[state][action] + .5 ) / ( nqueries[state][action] + 1)


# TODO:
# learn a distribution over Q_values (Gaussian, even though it's probably the wrong thing...)
# initialize means based on prior over rewards (E[r] = .5) and discount rate
# initialize std = mean
mean_Q_values = [[.5 * (1 / 1 - gamma),] * len(actions),] *len(states)
std_Q_values = [[.5 * (1 / 1 - gamma),] * len(actions),] *len(states)

#
def softmax_entropy():
    return multinomial_entropy(softmax(Q_values[current_state]))

#################################################################
# Query functions
query_fns = OrderedDict()

# simple baselines
max_num_queries = 10000.
#query_fns['first N steps'] = lambda : step < max_num_queries
query_fns['first N steps (based on query cost)'] = lambda : step < max_num_queries * (query_cost / .1)
query_fns['first N state visits'] = lambda : sum(nvisits[current_state]) < (max_num_queries * (query_cost / .1) / len(states))
query_fns['first N (state,action) visits'] = lambda : nvisits[current_state][action] < (max_num_queries * (query_cost / .1) / (len(states) * len(actions)))
#query_probability_decay = 1 - 1. / max_num_queries 
#query_fns['decaying probability'] = lambda : np.random.binomial(1, query_probability_decay**step)
#query_fns['every time'] = lambda : True

# query based on entropy of softmax policy:
# FIXME: querying too much... also, we should be looking MORE at those with lower entropy, I think...
#query_fns['softmax entropy threshold'] = lambda : softmax_entropy() > 1
#query_fns['stochastic softmax entropy'] = lambda : np.random.binomial(1, softmax_entropy() / multinomial_entropy([.2,.2,.2,.2,.2]))

# query based on entropy of P(r|s,a):
query_fns['r entropy threshold'] = lambda : reward_entropy() > -1
#query_fns['stochastic r entropy'] = lambda : np.random.binomial(1, np.exp(reward_entropy()))

# query based on number of state visits:
#query_fns['prob = proportion of state visits'] = lambda : np.random.binomial(1, (sum(nvisits[current_state]) + 50) / (sum(nvisits) + 50. * len(states)))
#query_fns['prob = proportion of state-action visits'] = lambda : np.random.binomial(1, (sum(nvisits[current_state][action]) + 10) / (sum(nvisits) + 10. * len(states) * len(actions)))

# query based on expected reward
#query_fns['geq than average expected reward'] = lambda : expected_reward(current_state, action) >= mean([expected_reward(s,a) for s in states for a in actions])
query_fns['proportion of expected reward'] = lambda : np.random.binomial(1, expected_reward(current_state, action) / sum(expected_reward(s,a) for s in states for a in actions))

# query based on expected reward
#query_fns['geq than average Q'] = lambda : Q_values[current_state][action] >= mean(Q_values)
query_fns['proportion of Q'] = lambda : np.random.binomial(1, (Q_values[current_state][action] + 1) / (sum(Q_values) + 1))

# query based on entropy of P(s):
# TODO: we want this to actually depend on the state we're in, otherwise, it seems to mostly just scale with sqrt(nvisits)
#query_fns['s entropy threshold'] = lambda : state_entropy() > -240
#query_fns['stochastic s entropy'] = lambda : np.random.binomial(1, np.exp(state_entropy())) # TODO: scaling...



#################################################################
# LEARNING

def update_q(state0, action, state1, reward, query): 
    if query: 
        reward = expected_reward(state0, action)
    old = Q_values[state0][action] 
    new = reward + gamma*np.max(Q_values[state1])
    Q_values[state0][action] = (1-learning_rate)*old + learning_rate*new




#query_fn = query_fns['every time']

t1 = time.time()
all_results = {}

for name,query_fn in query_fns.items():
    print "\n query_cost=", query_cost, '\n'
    print "name=", name
    all_results[query_cost] = {}

    performances = []
    for nex in range(num_experiments):
        # reward probabilities (rewards are stochastic bernoulli)
        if fixed_mdp:
            reward_probabilities = np.load('/u/kruegerd/CS188.1x-Project3/fixed_mdp0.npy')
        elif fixed_battery:
            reward_probabilities = np.load('/u/kruegerd/CS188.1x-Project3/200mdps.npy')[nex*len(states): (nex+1)*len(states)]
        else:
            reward_probabilities = np.random.binomial(1, 1 - prob_zero_reward, len(states)) * np.random.uniform(0, 1, len(states))
        for n in range(grid_width):
            print reward_probabilities[n*grid_width: (n+1)*grid_width]


        # reset for a new experiment
        Q_values = [[0,0,0,0,0] for state in states]
        total_r_observed = [[0,0,0,0,0] for state in states] 
        nvisits = [[0,0,0,0,0] for state in states]
        nqueries = [[0,0,0,0,0] for state in states]
        current_state = 0
        total_reward = 0

        for step in range(num_steps):
            #state_counts[state] += 1
            action = np.argmax(Q_values[current_state])
            if np.random.binomial(1, epsilon): # take a random action
                action = np.argmax(np.random.multinomial(1, [.2, .2, .2, .2, .2]))
            nvisits[current_state][action] += 1
            reward = np.random.binomial(1, reward_probabilities[current_state])
            total_reward += reward 
            query = query_fn()
            if query:
                nqueries[current_state][action] += 1
                # TODO: naming these two
                total_r_observed[current_state][action] += reward

            old_state = current_state
            if np.random.binomial(1, prob_random_move): # action has random effect
                current_state = next_state(current_state, np.argmax(np.random.multinomial(1, [.2,.2,.2,.2,.2])))
            else:
                current_state = next_state(current_state, action)
            if np.random.binomial(1, prob_random_reset): # reset to initial state
                current_state = 0

            #simple q-learner 
            update_q(old_state, action, current_state, reward, query)

        total_observed_reward = sum([ sum(r_observed) for r_observed in total_r_observed])
        total_nqueries = sum([ sum(nqueries_s) for nqueries_s in nqueries])
        total_query_cost = query_cost * total_nqueries
        performance = total_reward - total_query_cost

        if 0:#printing:
            print "total_reward =", total_reward
            print "total_observed_reward =", total_observed_reward
            print "total_nqueries =", total_nqueries
            print "performance =", performance
            print ''
        print "total_nqueries =", total_nqueries
        print "performance =", performance
        performances.append(performance)
        print time.time() - t1

        save_str = '/u/kruegerd/CS188.1x-Project3/results/'
        save_str += 'total_nqueries_' + name + '__query_cost=' + str(query_cost)
        if fixed_mdp:
            save_str += '__fixed_mdp'
        np.save(save_str, total_nqueries)

        save_str = '/u/kruegerd/CS188.1x-Project3/results/'
        save_str += 'performances_' + name + '__query_cost=' + str(query_cost)
        if fixed_mdp:
            save_str += '__fixed_mdp'
        np.save(save_str, performances)

    #hist(performances, 50)

