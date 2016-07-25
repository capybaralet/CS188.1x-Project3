

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



    LIST:
        query based on uncertainty of P(s)
        query based on uncertainty of P(r)
        query based on uncertainty of P(r | s, a)
        query based on uncertainty of softmax pi(s)
        query based on uncertainty of "proper bayesian" pi(s)


"""
