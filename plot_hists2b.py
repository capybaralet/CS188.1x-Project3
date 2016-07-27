from pylab import *
import scipy
#import scipy.stats
import numpy
np = numpy
from collections import OrderedDict
import time
import os


"""
N.B. for a proper analysis, we would want to look at how number of queries affects performance (e.g. if it pretty much determines performance, that would be somewhat unfortunate)
# TODO: we want to know who wins, each time, and do some sort of ranking thing...
"""

#################################################################
# Query functions
query_fns = OrderedDict()
query_fns['first N (state,action) visits'] = lambda : nvisits[current_state][action] < (max_num_queries * (query_cost / .1) / (len(states) * len(actions)))
query_fns['proportion of expected reward'] = lambda : np.random.binomial(1, expected_reward(current_state, action) / sum(expected_reward(s,a) for s in states for a in actions))


for name in query_fns:
    print name 
save_paths = []

#query_cost = [.1,1.,2.,5.,10.,20.,50.]
query_costs = [1.,2.,3.,4.,5.,6.,10.]
fixed_mdp = 1

for query_cost in query_costs:
    mean_performances = []
    for budget in ['high', 'med', 'low', 'very_low']:
        try:
            for name,query_fn in query_fns.items():
                save_str = 'performances_' + name + '__query_cost=' + str(query_cost) + '__budget=' + str(budget) + '__fixed_mdp'
                save_path = os.environ['HOME'] + '/CS188.1x-Project3/results/'+ save_str
                save_paths.append(save_path)
                performances = np.load(save_path + '.npy')
                mean_performances.append(np.mean(performances))
        except:
            pass

    figure()
    title = str(query_cost)
    suptitle(title)
    #subplot(122)
    bar(range(len(mean_performances)), mean_performances)
    #ylim(20000,70000)
    #subplot(121)
    #hist(performances, 25)

