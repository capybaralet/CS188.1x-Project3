import numpy
np = numpy
import gridworld


# hyper-parameters:
grid_width = 3
prob_random_action = 0.1
prob_random_reset = 0.001
query_cost = .01
gamma = .99 # discount factor
prob_zero_reward = .9

learning_rate = .1

# states (lexical order)
states = range(grid_width**2)

# reward probabilities
reward_probabilities = np.random.binomial(1, 1 - prob_zero_reward, len(states)) * np.random.uniform(0, 1, len(states))



##################################
def row_and_column(state):
	return state / grid_width, state % grid_width

"""
def state_from_row_and_column(row, column):
	state = row * grid_width + column
	assert 0 <= state and state <= len(states) 
	return state
"""

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




#################################################################

# TODO: policy should just be action probabilities...
# policy should probably be a function (taking Q_values)
policy = [[.2, .2, .2, .2, .2] for state in states]

Q_values = [[0,0,0,0,0] for state in states]
total_r_observed = [[0,0,0,0,0] for state in states] 

nqueries = [[0,0,0,0,0] for state in states]
#policy = {state: [.2, .2, .2, .2, .2] for state in states}
#nqueries = {state: [0,0,0,0,0] for state in states}

nsteps = 100000
current_state = 0
total_reward = 0
total_observed_reward = 0

# TODO: discounting
# TODO: more policies

def expected_reward(state, action): 
    return (total_r_observed[state][action] + .5 ) / ( nqueries[state][action] + 10)



action_names = ['north', 'south', 'east', 'west']
class QDummy(object): 
    def getQValue(self, state, action):
        s = state[0] + state[1]*grid_width

        a = action_names.index(action)+1


        return Q_values[s][a]

dummy_agent = QDummy()
dummy_grid = gridworld.Gridworld([[' '] *grid_width ] * grid_width)

def reverse(t): 
    return (t[1], t[0])

import graphicsGridworldDisplay
display = graphicsGridworldDisplay.GraphicsGridworldDisplay(dummy_grid, 75, 10.0)
display.start()

def update_q(state0, action, state1, reward, query): 
    if not query: 
        reward = expected_reward(state0, action)

    old = Q_values[state0][action] 
    new = reward + gamma*np.max(Q_values[state1])

    Q_values[state0][action] = (1-learning_rate)*old + learning_rate*new

for step in range(nsteps):
	# uniform random policy:
	action = np.argmax(np.random.multinomial(1, [.2,.2,.2,.2,.2]))
	# "thompson-sampling" greedy (proportional agent):
	action = np.argmax(np.random.multinomial(1, policy[current_state]))
	# greedy policy:
	action = np.argmax(Q_values[current_state])
	if np.random.binomial(1, prob_random_action, 1)[0]: # take a random action
		action = np.argmax(np.random.multinomial(1, [.2, .2, .2, .2, .2], 1))

	query = query_fn(current_state, action, step)
	# +1 thing??
	reward = np.random.binomial(1, reward_probabilities[current_state], 1)
	total_reward += reward 

	if query:
                nqueries[state][action] += 1
                total_r_observed[current_state][action] += reward
                total_observed_reward += reward 

        old_state = current_state
	current_state = next_state(current_state, action)

        if np.random.uniform(0,1) < prob_random_reset: #np.random.binomial(1, prob_random_reset, 1)[0]: # reset to initial state
		current_state = 0

	# TODO: learning

        #simple q-learner 
        update_q(old_state, action, current_state, reward, query)

for i in range(8):
    print reward_probabilities[i*7:(i+1)*7]
display.displayQValues(dummy_agent, reverse(row_and_column(current_state)), "q vals")
print total_r_observed
display.pause()

total_nqueries = sum([ sum(nqueries_s) for nqueries_s in nqueries])
total_query_cost = query_cost * total_nqueries
# TODO: more complicated stuff, e.g.:
#total_query_cost = query_cost(query_history)

performance = total_reward - total_query_cost
print "total_reward =", total_reward
print "total_nqueries =", total_nqueries
print "performance =", performance


