from pylab import *
import scipy
#import scipy.stats
import numpy
np = numpy
from collections import OrderedDict
import time


"""
N.B. for a proper analysis, we would want to look at how number of queries affects performance (e.g. if it pretty much determines performance, that would be somewhat unfortunate)
# TODO: we want to know who wins, each time, and do some sort of ranking thing...
"""

#################################################################
# Query functions
query_fns = OrderedDict()
query_fns['first N steps (based on query cost)'] = lambda : step < max_num_queries * (query_cost / .1)
query_fns['first N state visits'] = lambda : sum(nvisits[current_state]) < (max_num_queries * (query_cost / .1) / len(states))
query_fns['first N (state,action) visits'] = lambda : nvisits[current_state][action] < (max_num_queries * (query_cost / .1) / (len(states) * len(actions)))
query_fns['r entropy threshold'] = lambda : reward_entropy() > -1
query_fns['proportion of expected reward'] = lambda : np.random.binomial(1, expected_reward(current_state, action) / sum(expected_reward(s,a) for s in states for a in actions))
query_fns['proportion of Q'] = lambda : np.random.binomial(1, (Q_values[current_state][action] + 1) / (sum(Q_values) + 1))


for name in query_fns:
    print name 
all_mean_performances = []
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
            if 1:#(3 < n < 6) or n == 7 or n == 11 or n == 13:
                save_dir = '/Users/david/CS188.1x-Project3/results/'
                save_str = ''
                save_str += 'total_nqueries_' + name + '__query_cost=' + str(query_cost)
                if fixed_mdp:
                    save_str += '__fixed_mdp'
                performances = np.load(save_dir + save_str + '.npy')
                #print save_str, mean(performances)
                mean_performances.append(np.mean(performances))
                #mean_performances.append(np.median(performances))
                all_mean_performances.append(mean_performances)
                #subplot(4,4,n)
                #hist(performances, 50)

        #figure()
        bar(range(len(mean_performances)), mean_performances)
        #ylim(58000, 75000)



