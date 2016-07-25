from pylab import *
import numpy
np = numpy
from collections import OrderedDict
import time

# hyper-parameters:
grid_width = 8
epsilon = 0.1
prob_random_move = 0.1
prob_random_reset = 0.001
query_cost = .1 # overridden in for loop!
gamma = .9 # discount factor
prob_zero_reward = .9
learning_rate = .1

num_experiments = 300

# states (lexical order)
states = range(grid_width**2)
# actions: stay, N, E, S, W
actions = range(5)
# reward probabilities (rewards are stochastic bernoulli)
reward_probabilities = np.random.binomial(1, 1 - prob_zero_reward, len(states)) * np.random.uniform(0, 1, len(states))


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
    # requires scipy version>0.15
    return scipy.stats.dirichlet([nvisit + .5 for nvisit in nvisits]).entropy()
def state_probability(state):
    (sum(nvisits[state]) + .5) / (step + .5*len(states))

# learn a distribution over rewards (beta-bernoulli)
def reward_entropy():
    alpha = total_r_observed[state][action] + .5
    beta = ( nqueries[state][action] + 1) - alpha
    return scipy.stats.beta(alpha, beta).entropy()
def expected_reward(state, action): 
    alpha = total_r_observed[state][action] + .5
    beta = ( nqueries[state][action] + 1) - alpha
    return alpha / (alpha + beta)#(total_r_observed[state][action] + .5 ) / ( nqueries[state][action] + 1)

# learn a distribution over Q_values (Gaussian, even though it's probably the wrong thing...)
# initialize means based on prior over rewards (E[r] = .5) and discount rate
# initialize std = mean
mean_Q_values = [[.5 * (1 / 1 - gamma),] * len(actions),] *len(states)
std_Q_values = [[.5 * (1 / 1 - gamma),] * len(actions),] *len(states)

# TODO: replace w/Gaussian
def state_entropy():
    return scipy.stats.dirichlet([nvisit + .5 for nvisit in nvisits]).entropy()
def state_probability(state):
    (sum(nvisits[state]) + .5) / (step + .5*len(states))

#
def softmax_entropy():
    return multinomial_entropy(softmax(Q_values[state]))

#################################################################
# Query functions
query_fns = OrderedDict()

# simple baselines
max_num_queries = 10000
query_fns['first N steps'] = lambda : step < max_num_queries
query_fns['first N state visits'] = lambda : nvisits[current_state] < (max_num_queries / len(states))
query_fns['first N (state,action) visits'] = lambda : nvisits[current_state][action] < (max_num_queries / len(states) / len(actions))
query_probability_decay = 1 - 1. / max_num_queries 
query_fns['decaying probability'] = lambda : np.random.binomial(1, query_probability_decay**step)
query_fns['every time'] = lambda : True


# query based on expectation of P(r|s,a):
query_fns['r entropy threshold'] = lambda : reward_entropy() > -1
query_fns['stochastic r entropy'] = lambda : np.binomial(1, np.exp(reward_entropy()))

# query based on entropy of P(r|s,a):
query_fns['r entropy threshold'] = lambda : reward_entropy() > -1
query_fns['stochastic r entropy'] = lambda : np.binomial(1, np.exp(reward_entropy()))

# query based on entropy of P(s):
query_fns['s entropy threshold'] = lambda : state_entropy() > -1
query_fns['stochastic s entropy'] = lambda : np.binomial(1, np.exp(state_entropy()))

# query based on entropy of softmax policy:
query_fns['softmax entropy threshold'] = lambda : softmax_entropy() > 1
query_fns['stochastic softmax entropy'] = lambda : np.binomial(1, softmax_entropy() / multinomial_entropy([0,0,0,0,0]))

# TODO:
# query based on entropy of "bayesian" policy:
# (this policy takes actions according to their true probability of being optimal, given that we have distributions over Q_values)
#query_fns['softmax entropy threshold'] = lambda : softmax_entropy() > 1
#query_fns['stochastic softmax entropy'] = lambda : np.binomial(1, softmax_entropy() / multinomial_entropy([0,0,0,0,0]))


query_fn = query_fns['every time']

#################################################################
# LEARNING

def update_q(state0, action, state1, reward, query): 
    if query: 
        reward = expected_reward(state0, action)
    old = Q_values[state0][action] 
    new = reward + gamma*np.max(Q_values[state1])
    Q_values[state0][action] = (1-learning_rate)*old + learning_rate*new

nsteps = 50000


for n in range(grid_width):
    print reward_probabilities[n*grid_width: (n+1)*grid_width]

t1 = time.time()

all_results = {}
for query_cost in [.01]:#, .03, .01]:
    print "\n query_cost=", query_cost, '\n'
    all_results[query_cost] = {}

    performances = []
    for nex in range(num_experiments):
        # reset for a new experiment
        Q_values = [[0,0,0,0,0] for state in states]
        total_r_observed = [[0,0,0,0,0] for state in states] 
        nvisits = [[0,0,0,0,0] for state in states]
        nqueries = [[0,0,0,0,0] for state in states]
        current_state = 0
        total_reward = 0

        for step in range(nsteps):
            #state_counts[state] += 1
            action = np.argmax(Q_values[current_state])
            if np.random.binomial(1, epsilon): # take a random action
                action = np.argmax(np.random.multinomial(1, [.2, .2, .2, .2, .2]))
            reward = np.random.binomial(1, reward_probabilities[current_state])
            total_reward += reward 
            query = query_fn()#current_state, action, step)
            if query:
                nqueries[state][action] += 1
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
        print "performance =", performance
        performances.append(performance)

    hist(performances, 50)

print time.time() - t1
