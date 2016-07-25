import numpy
np = numpy

"""
TODO:

	implement Q-learning

	track uncertainty of Q (or A (advantage function))

	learn P(r | s, a)
	learn P(s)
	^ use these to make cooler query functions

	------------------------------------------------
	implement PSRL

"""


# hyper-parameters:
grid_width = 10
prob_random_action = 0.
prob_random_reset = 0.
query_cost = .01
gamma = .999 # discount factor
prob_zero_reward = .9

# states (lexical order)
states = range(grid_width**2)

# reward probabilities (rewards are stochastic bernoulli)
reward_probabilities = np.random.binomial(1, 1 - prob_zero_reward, len(states)) * np.random.uniform(0, 1, len(states))



##################################
def row_and_column(state):
	return state / grid_width, state % grid_width

actions = range(5) # stay, N, E, S, W

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
# GENERAL functions

def multinomial_entropy(probs): # TODO: stability?
	return -np.sum(probs * np.log(probs))
 
#################################################################
# Query functions

# query each time
def query_fn(s, a, step):
	return True

# query first N times
max_num_queries = 10000
def query_fn(s, a, step):
	return step < max_num_queries

# query first N times in state s
max_num_queries = 10000 / len(states)
def query_fn(s, a, step):
	return sum(nqueries[s]) < max_num_queries

# query first N times if state s, taking action a
max_num_queries = 10000 / len(states) / len(actions)
def query_fn(s, a, step):
	return nqueries[s][a] < max_num_queries

# query with time-dependent probability
query_probability_decay = .999 # TODO: make this expectation = max_num_queries
def query_fn(s, a, step):
	return query_probability_decay**step 

# query based on entropy of action probability
entropy_threshold = 1.5  # TODO: what's a good value here?
def query_fn(s, a, step):
	return multinomial_entropy(policy[s]) > entropy_threshold

"""
# TODO: query when expected returns under softmax policy is greater than something... 
def query_fn(s, a, step):
	return nqueries[s][a] < max_num_queries
"""

#################################################################
# learn a probability distribution over states:
state_probs = []


# learn a dirichlet-multinomial probability distribution over states probabilities:
state_posteriors = []


#################################################################

"""
	# greedy policy:
	action = np.argmax(policy[current_state])
	# uniform random policy:
	action = np.argmax(np.random.multinomial(1, [.2,.2,.2,.2,.2]))
"""


# stochastic policy, given in terms of action probabilities
policy = [[.2, .2, .2, .2, .2] for state in states]

# policy should probably be a function (taking Q_values)

Q_values = [[0,0,0,0,0] for state in states]
nqueries = [[0,0,0,0,0] for state in states]

nsteps = 50000
current_state = 0
total_reward = 0
total_observed_reward = 0

# TODO: discounting
# TODO: more policies
for step in range(nsteps):
	# this is how the policy would work:
	action = np.argmax(np.random.multinomial(1, policy[current_state]))

	# greedy Q-learning:
	action = np.argmax(Q_values[current_state])

	if np.random.binomial(1, prob_random_action, 1)[0]: # take a random action
		action = np.argmax(np.random.multinomial(1, [.2, .2, .2, .2, .2], 1))

	query = query_fn(current_state, action, step)
	if query:
		nqueries[state][action] += 1
	# TODO: +1 thing?? (is this the right reward?)
	reward = gamma**step * np.random.binomial(1, reward_probabilities[current_state], 1)
	total_reward += reward
	observed_reward = query * reward
	total_observed_reward += observed_reward 
	current_state = next_state(current_state, action)
	if np.random.binomial(1, prob_random_reset, 1)[0]: # reset to initial state
		current_state = 0

	# TODO: learning



total_nqueries = sum([ sum(nqueries_s) for nqueries_s in nqueries])
total_query_cost = query_cost * total_nqueries
# TODO: more complicated stuff, e.g.:
#total_query_cost = query_cost(query_history)

performance = total_reward - total_query_cost
print "total_reward =", total_reward
print "total_observed_reward =", total_observed_reward
print "total_nqueries =", total_nqueries
print "performance =", performance


