import random
import sys
import mdp
import environment
import util
import optparse  
import gridworld

class Opts(object):
    def __init__(self, display=True): 
        self.speed = 1.0
        self.gridSize=150

        self.discount=.9
        self.learningRate=.1
        self.epsilon=.1

        self.iters =10
        self.episodes=10
        self.manual=False
        
        self.quiet=True
        self.pause=False
        self.display=display
    
def runAgent(a, mdp, opts):

    env = gridworld.GridworldEnvironment(mdp)
    
    import textGridworldDisplay

    import graphicsGridworldDisplay
    display = textGridworldDisplay.TextGridworldDisplay(mdp)

    if opts.display:
        display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)

    display.start()

    
    ###########################
    # RUN EPISODES
    ###########################

    # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
    displayCallback = lambda state: None
    
    if opts.display:
        displayCallback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")

    messageCallback = lambda x: gridworld.printString(x)
    if opts.quiet:
        messageCallback = lambda x: None

    # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
    pauseCallback = lambda : None
    if opts.pause:
        pauseCallback = lambda : display.pause()

    # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
    if opts.manual:
        decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
    else:
        decisionCallback = a.getAction  


    returns = 0
    
    for episode in range(1, opts.episodes+1):
        returns += gridworld.runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
    
    if opts.episodes > 0:
        print
        print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)
        print
        print
