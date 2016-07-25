import numpy
np = numpy
from collections import OrderedDict

"""
IDEAS for query functions:
    possible inputs:
        P(s)
        P(r | s, a)
        pi(s)

    want to query more if state might be:
        high value and reachable
    want to query more if we're uncertain...

    want to avoid querying inevitable states
        (e.g. if there were a gate-state right in from of the start-state.)
        need an environment to test this...)

    we also just care about finding the best action from a given state



THINGS TO MONITOR:
    learning of everything that is learned:
        Q-function
        P(r), P(s)
    what is queried
    how often it visits the good states vs. not.


QUESTIONS / musings...:

    how much would observing r reduce the entropy of what is P(a is best | s, theta) P(theta)??
    simpler: we can measure entropy of Q(s,a) and sum over a as a measure of uncertainty.

    Should we use P(r) to give rewards to the Q-learner?  Or should we use the reward we actually observe?

    Think about bandit case!
        has this been done here?

    When we are confident that we shouldn't explore a state, we also wouldn't want to query
    Intuitively, we should query when we're exploring

    value of querying is based on figuring out how likely the information gain is to change your action

    should we be using lambda > 0?


TODO:
    track uncertainty of Q (or A (advantage function))
    use P(s), P(r | s, a) to make cooler query functions

    ------------------------------------------------
    model-based
        PSRL with extra cost for less visited s,a
        Dave has PSRL
        Rmax - query cost (?)



"""


# hyper-parameters:
grid_width = 4
epsilon = 0.1
prob_random_move = 0.0
prob_random_reset = 0.001
query_cost = .1 # overridden in for loop!
gamma = .999 # discount factor
prob_zero_reward = .9

learning_rate = .1

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

def updated_running_average(avg, new_value, update_n):
    return (avg * (update_n - 1) + new_value) / update_n

def multinomial_entropy(probs): # TODO: stability?
    return -np.sum(probs * np.log(probs))
 
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#################################################################
# Query functions
query_fns = OrderedDict()

# query each time
def query_fn(s, a, step):
    return True
query_fns['always query'] = query_fn

# query with time-dependent probability
query_probability_decay = .9999 
def query_fn(s, a, step):
    return np.random.binomial(1, query_probability_decay**step)
query_fns['decaying query probability'] = query_fn

# query first N times
max_num_queries = 10000
def query_fn(s, a, step):
    return step < max_num_queries
query_fns['query first N times'] = query_fn



"""
# FIXME: almost always queries!
# query first N times in state s
#max_num_queries = 10000 / len(states)
def query_fn(s, a, step):
    return sum(nqueries[s]) < (max_num_queries / len(states))
query_fns['query first N state visits'] = query_fn

# FIXME: almost always queries!
# query first N times if state s, taking action a
#max_num_queries = 10000 / len(states) / len(actions)
def query_fn(s, a, step):
    return nqueries[s][a] < (max_num_queries / len(states) / len(actions))
query_fns['query first N state,action visits'] = query_fn

# query based on entropy of action probability
entropy_threshold = 1.5  # TODO: what's a good value here?
def query_fn(s, a, step):
    return multinomial_entropy(policy[s]) > entropy_threshold
query_fns['policy entropy threshold'] = query_fn
# TODO: query when expected returns under softmax policy is greater than something... 
def query_fn(s, a, step):
    return nqueries[s][a] < max_num_queries
"""

#################################################################
# learn a dirichlet-multinomial probability distribution over states probabilities
state_counts = [.5,]*grid_width**2



#################################################################

# TODO: more policies
"""
class ValueIteration(object):
    def __init__(self, iterations=10000):
        for i in range(0, self.iterations):
            newValues = np.zeros(len(states))
            for state in states:
                maxActionValue = -1*float('inf')
                maxAction = None
                possibleActions = actions
                for action in possibleActions:
                    actionSumSPrime = self.getQValue(state, action)

                    #Find the maximum action
                    if maxActionValue < actionSumSPrime:
                        maxAction = action
                        maxActionValue = actionSumSPrime

                v_kPlus1 = maxActionValue
                newValues[state] = v_kPlus1
            self.values = newValues

# greedy policy:
action = np.argmax(policy[current_state])
# uniform random policy:
action = np.argmax(np.random.multinomial(1, [.2,.2,.2,.2,.2]))

# stochastic policy, given in terms of action probabilities
#policy = [[.2, .2, .2, .2, .2] for state in states]
"""

def expected_reward(state, action): 
    return (total_r_observed[state][action] + .5 ) / ( nqueries[state][action] + 1)

def update_q(state0, action, state1, reward, query): 
    if query: 
        reward = expected_reward(state0, action)
    old = Q_values[state0][action] 
    new = reward + gamma*np.max(Q_values[state1])
    Q_values[state0][action] = (1-learning_rate)*old + learning_rate*new

all_results = {}
for query_cost in [.1, .03, .01]:
    print "\n query_cost=", query_cost, '\n'
    all_results[query_cost] = {}
    for query_fn_name in query_fns:
        all_results[query_cost][query_fn_name] = {}
        all_results[query_cost][query_fn_name]['total_reward'] = 0
        all_results[query_cost][query_fn_name]['total_observed_reward'] = 0
        all_results[query_cost][query_fn_name]['total_nqueries'] = 0
        all_results[query_cost][query_fn_name]['performance'] = 0
        for nex in range(4):
            query_fn = query_fns[query_fn_name]
            print query_fn_name

            Q_values = [[0,0,0,0,0] for state in states]
            total_r_observed = [[0,0,0,0,0] for state in states] 
            nqueries = [[0,0,0,0,0] for state in states]

            nsteps = 200000
            current_state = 0
            total_reward = 0
            total_observed_reward = 0

            for step in range(nsteps):
                state_counts[state] += 1

                # this is how the policy would work:
                #action = np.argmax(np.random.multinomial(1, policy[current_state]))

                # greedy Q-learning:
                action = np.argmax(Q_values[current_state])
                if np.random.binomial(1, epsilon, 1)[0]: # take a random action
                    action = np.argmax(np.random.multinomial(1, [.2, .2, .2, .2, .2]))

                # TODO: softmax Q-learning:
                #action = softmax(Q_values[current_state])


                # +1 thing??
                reward = np.random.binomial(1, reward_probabilities[current_state], 1)
                total_reward += reward 
                
                query = query_fn(current_state, action, step)
                if query:
                    nqueries[state][action] += 1
                    # TODO: naming these two
                    total_r_observed[current_state][action] += reward
                    total_observed_reward += reward 

                old_state = current_state
                if np.random.binomial(1, prob_random_move, 1)[0]: # action has random effect
                    current_state = next_state(current_state, np.argmax(np.random.multinomial(1, [.2,.2,.2,.2,.2])))
                else:
                    current_state = next_state(current_state, action)
                if np.random.binomial(1, prob_random_reset, 1)[0]: # reset to initial state
                    current_state = 0

                # TODO: more learning
                #simple q-learner 
                update_q(old_state, action, current_state, reward, query)

            total_nqueries = sum([ sum(nqueries_s) for nqueries_s in nqueries])
            total_query_cost = query_cost * total_nqueries
            # TODO: more complicated stuff, e.g.:
            #total_query_cost = query_cost(query_history)

            performance = total_reward - total_query_cost
            all_results[query_cost][query_fn_name]['total_reward'] = updated_running_average(all_results[query_cost][query_fn_name]['total_reward'], total_reward, nex+1)
            all_results[query_cost][query_fn_name]['total_observed_reward'] = updated_running_average(all_results[query_cost][query_fn_name]['total_observed_reward'], total_observed_reward, nex+1)
            all_results[query_cost][query_fn_name]['total_nqueries'] = updated_running_average(all_results[query_cost][query_fn_name]['total_nqueries'], total_nqueries, nex+1)
            all_results[query_cost][query_fn_name]['performance'] = updated_running_average(all_results[query_cost][query_fn_name]['performance'], performance, nex+1)

            if 0:#printing:
                print "total_reward =", total_reward
                print "total_observed_reward =", total_observed_reward
                print "total_nqueries =", total_nqueries
                print "performance =", performance
                print ''

        if 1:#printing:
            print "total_reward =", all_results[query_cost][query_fn_name]['total_reward']
            print "total_observed_reward =", all_results[query_cost][query_fn_name]['total_observed_reward']
            print "total_nqueries =",  all_results[query_cost][query_fn_name]['total_nqueries']
            print "performance =", all_results[query_cost][query_fn_name]['performance']
            print ''


