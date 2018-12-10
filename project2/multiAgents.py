# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent


class ReflexAgent(Agent):
    """
        A reflex agent chooses an action at each choice point by examining
        its alternatives via a state evaluation function.
        The code below is provided as a guide.    You are welcome to change
        it in any way you see fit, so long as you don't touch our method
        headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        def absoluteDistance(x, y):
            z = abs(x[0] - y[0]) + abs(x[1] - y[1])
            return z

        score = successorGameState.getScore()
        foodDis = [1000]
        ghostDis = [1000]

        for food in newFood.asList():
            foodDis.append(absoluteDistance(newPos, food))
        minFoodDis = min(foodDis)

        for ghost in newGhostStates:
            ghostDis.append(absoluteDistance(newPos, ghost.getPosition()))
        minGhostDis = min(ghostDis)

        if minGhostDis > 5:
            minGhostDis = 100
        """ Tried some values but can not make it, so take a look at some values others used from the internet """

        return score + 10 / minFoodDis + minGhostDis


def scoreEvaluationFunction(currentGameState):
    """
        This default evaluation function just returns the score of the state.
        The score is the same one displayed in the Pacman GUI.
        This evaluation function is meant for use with adversarial search agents
        (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
        This class provides some common elements to all of your
        multi-agent searchers.    Any methods defined here will be available
        to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
        You *do not* need to make any changes here, but you can if you want to
        add functionality to all your adversarial search agents.    Please do not
        remove anything, however.
        Note: this is an abstract class: one that should not be instantiated.    It's
        only partially specified, and designed to be extended.    Agent (game.py)
        is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
        Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
        Directions.STOP:
            The stop direction, which is always legal
        gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        "*** YOUR CODE HERE ***"

        pacman_index = 0
        ghost_index = range(1, gameState.getNumAgents())
        largenum = sys.maxint

        def TERMINAL(state, depth):
            return state.isWin() or state.isLose() or depth == 0

        def MINIMAX_DECISION(state, depth):
            results=[]
            for action in state.getLegalActions(pacman_index):
                successor=MIN_VALUE(state.generateSuccessor(pacman_index, action), depth)
                results.append((action,successor))
            action, value = max(results, key=lambda t: t[1])

            return action, value

        def MIN_VALUE(state, depth):
            if TERMINAL(state, depth):
                return self.evaluationFunction(state)

            v = largenum
            for ghost in ghost_index:
                for action in state.getLegalActions(ghost):
                    successor = state.generateSuccessor(ghost, action)
                    v = min(v, MAX_VALUE(successor, depth - 1))

            return v

        def MAX_VALUE(state, depth):
            if TERMINAL(state, depth):
                return self.evaluationFunction(state)

            v = -largenum
            for action in state.getLegalActions(pacman_index):
                if action==Directions.STOP:
                    continue
                successor = state.generateSuccessor(pacman_index, action)
                v = max(v, MIN_VALUE(successor, depth - 1))

            return v

        action, value = MINIMAX_DECISION(gameState, self.depth)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        pacman_index = 0
        ghost_index = range(1, gameState.getNumAgents())
        largenum = sys.maxint

        def TERMINAL(state, depth):
            return state.isWin() or state.isLose() or depth < 0

        def A_B_SEARCH(state, depth):

            results = []
            for action in state.getLegalActions(pacman_index):
                successor = MIN_VALUE(state.generateSuccessor(pacman_index, action), depth, -largenum, largenum)
                results.append((action, successor))
            action, value = max(results, key=lambda t: t[1])

            return action, value

        def MIN_VALUE(state, depth, a, b):
            if TERMINAL(state, depth):
                return self.evaluationFunction(state)

            v = largenum
            for ghost in ghost_index:
                for action in state.getLegalActions(ghost):
                    successor = state.generateSuccessor(ghost, action)
                    v = min(v, MAX_VALUE(successor, depth - 1, a, b))
                    if v <= a: return v
                    b = min(b, v)
            return v

        def MAX_VALUE(state, depth, a, b):
            if TERMINAL(state, depth):
                return self.evaluationFunction(state)

            v = -largenum
            for action in state.getLegalActions(pacman_index):
                if action==Directions.STOP:
                    continue
                successor = state.generateSuccessor(pacman_index, action)
                v = max(v, MIN_VALUE(successor, depth - 1, a, b))
                if v >= b: return v
                a = max(a, v)
            return v

        action, value = A_B_SEARCH(gameState, self.depth)

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
        Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
            Returns the expectimax action using self.depth and self.evaluationFunction
            All ghosts should be modeled as choosing uniformly at random from their
            legal moves.
        """
        "*** YOUR CODE HERE ***"

        pacman_index = 0
        ghost_index = range(1, gameState.getNumAgents())
        largenum = sys.maxint

        def TERMINAL(state, depth):
            return state.isWin() or state.isLose() or depth < 0

        def MINIMAX_DECISION(state, depth):
            results = []
            for action in state.getLegalActions(pacman_index):
                successor = MIN_VALUE(state.generateSuccessor(pacman_index, action), depth)
                results.append((action, successor))
            action, value = max(results, key=lambda t: t[1])

            return action, value

        def MIN_VALUE(state, depth):
            if TERMINAL(state, depth):
                return self.evaluationFunction(state)

            total = 0
            count = 0
            for ghost in ghost_index:
                for action in state.getLegalActions(ghost):
                    successor = state.generateSuccessor(ghost, action)
                    total += MAX_VALUE(successor, depth - 1)
                    count += 1

            return float(total) / float(count)

        def MAX_VALUE(state, depth):
            if TERMINAL(state, depth):
                return self.evaluationFunction(state)

            v = -largenum
            for action in state.getLegalActions(pacman_index):
                if action==Directions.STOP:
                    continue
                successor = state.generateSuccessor(pacman_index, action)
                v = max(v, MIN_VALUE(successor, depth - 1))

            return v

        action, value = MINIMAX_DECISION(gameState, self.depth)

        return action


def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: I originally planned on starting with my evaluation function
        from part 1 and improving it, but then I realized that it performed
        really well for this part, so I just reused it.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    initialScore=currentGameState.getScore()
    free=True

    def absoluteDistance(x, y):
        z = abs(x[0] - y[0]) + abs(x[1] - y[1])
        return z

    score = currentGameState.getScore()
    foodDis = [1000]
    ghostDis = [1000]

    for food in newFood.asList():
        foodDis.append(absoluteDistance(newPos, food))
    minFoodDis = min(foodDis)

    for ghost in newGhostStates:
        if ghost.scaredTimer !=0:
            continue
        ghostDis.append(absoluteDistance(newPos, ghost.getPosition()))
        free=False
    minGhostDis = min(ghostDis)

    if minGhostDis > 5:
        minGhostDis = 100
    """ Tried some values but can not make it, so take a look at some values others used from the internet """

    return score + 10 / minFoodDis + minGhostDis


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
        Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
            Returns an action.    You can use any method you want and search to any depth you want.
            Just remember that the mini-contest is timed, so you have to trade off speed and computation.
            Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
            just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
