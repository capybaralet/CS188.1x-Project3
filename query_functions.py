
"""
Experiment flow:
    1. make sure Q-learning can reliably solve the task
        harder than it seems... will just look at the distributions, first, and see what's up there...
    2. for each query_function, query_cost combination, run 20 experiments and plot histogram...


MATH TODO:
    1. figure out bayesian policy
    2. figure out information gain
    3. thompson sampling of Q function?

IDEAS for query functions:
    query based on uncertainty of P(s)
    query based on uncertainty of P(r)
    query based on uncertainty of P(r | s, a)
    query based on uncertainty of softmax pi(s)
    query based on uncertainty of "proper bayesian" pi(s)
        how do we compute that?
        query based on reduction of uncertainty in same...
    query based on expected value of state (higher = query more)

HOW TO CHOSE when to query:
    probabilistic
    threshold (+ budget)
    information gain (TODO)


NOTES on query functions:
    want to query more if state might be:
        high value and reachable
    want to query more if we're uncertain...

    ** want to avoid querying inevitable states
        (e.g. if there were a gate-state right in from of the start-state.)
        need an environment to test this...)

    we also just care about finding the best action from a given state


highly speculative:
    looking at P(r|s,a), or pi(a | s) as point estimates seems to lead to wireheading, whereas looking at the conjugate priors doesn't???


"""





