from pylab import *
import scipy
#import scipy.stats
import numpy
np = numpy
from collections import OrderedDict
import time


"""
N.B. for a proper analysis, we would want to look at how number of queries affects performance (e.g. if it pretty much determines performance, that would be somewhat unfortunate)
"""

#################################################################
# Query functions
query_fns = OrderedDict()

# simple baselines
query_fns['every time'] = lambda : True
query_fns['decaying probability'] = lambda : np.random.binomial(1, query_probability_decay**step)
max_num_queries = 10000.
query_fns['first N steps'] = lambda : step < max_num_queries
query_fns['first N state visits'] = lambda : sum(nvisits[current_state]) < (max_num_queries / len(states))
query_fns['first N (state,action) visits'] = lambda : nvisits[current_state][action] < (max_num_queries / (len(states) * len(actions)))
query_probability_decay = 1 - 1. / max_num_queries 

# query based on entropy of softmax policy:
# FIXME: querying too much... also, we should be looking MORE at those with lower entropy, I think...
#query_fns['softmax entropy threshold'] = lambda : softmax_entropy() > 1
#query_fns['stochastic softmax entropy'] = lambda : np.random.binomial(1, softmax_entropy() / multinomial_entropy([.2,.2,.2,.2,.2]))

# query based on entropy of P(r|s,a):
query_fns['r entropy threshold'] = lambda : reward_entropy() > -1
query_fns['stochastic r entropy'] = lambda : np.random.binomial(1, np.exp(reward_entropy()))

# query based on number of state visits:
query_fns['prob = proportion of state visits'] = lambda : np.random.binomial(1, (sum(nvisits[current_state]) + 50) / (sum(nvisits) + 50. * len(states)))
query_fns['prob = proportion of state-action visits'] = lambda : np.random.binomial(1, (sum(nvisits[current_state][action]) + 10) / (sum(nvisits) + 10. * len(states) * len(actions)))

# query based on expected reward
query_fns['geq than average expected reward'] = lambda : expected_reward(current_state, action) >= mean([expected_reward(s,a) for s in states for a in actions])
query_fns['proportion of expected reward'] = lambda : np.random.binomial(1, expected_reward(current_state, action) / sum(expected_reward(s,a) for s in states for a in actions))

# query based on expected Q
query_fns['geq than average Q'] = lambda : Q_values[current_state][action] >= mean(Q_values)
query_fns['proportion of Q'] = lambda : np.random.binomial(1, (Q_values[current_state][action] + 1) / (sum(Q_values) + 1))

# query based on entropy of P(s):
# TODO: we want this to actually depend on the state we're in, otherwise, it seems to mostly just scale with sqrt(nvisits)
#query_fns['s entropy threshold'] = lambda : state_entropy() > -240
#query_fns['stochastic s entropy'] = lambda : np.random.binomial(1, np.exp(state_entropy())) # TODO: scaling...


# TODO: we want to know who wins, each time, and do some sort of ranking thing...

for name in query_fns:
    print name 
all_mean_performances = []
#for query_cost in [1.,.3,.1,.03,.01,.003,.001]:
for query_cost in [2.,3.,4.,5.,10.,20.,50.,100.]:
    #for fixed_mdp in [0,1]: 
    for fixed_mdp in [ 1]: 
        figure()
        title = str(query_cost)
        if fixed_mdp:
            title = "fixed_mdp___" + title
        suptitle(title)
        n = 0
        mean_performances = []
        for name,query_fn in query_fns.items():
            n += 1
            #if (3 < n < 6) or n == 7 or n == 9 or n == 11 or n == 13:
            if (3 < n < 6) or n == 7 or n == 11 or n == 13:
                save_dir = '/Users/david/CS188.1x-Project3/results/'
                save_str = ''
                if fixed_mdp:
                    save_str += 'fixed_mdp__'
                save_str += 'performances_' + name + '__query_cost=' + str(query_cost)
                performances = np.load(save_dir + save_str + '.npy')
                #print save_str, mean(performances)
                #mean_performances.append(np.mean(performances))
                mean_performances.append(np.median(performances))
                all_mean_performances.append(mean_performances)
                #subplot(4,4,n)
                #hist(performances, 50)

        #figure()
        bar(range(len(mean_performances)), mean_performances)
        ylim(58000, 75000)



