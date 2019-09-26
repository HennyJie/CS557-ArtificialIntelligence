'''
@Description: In User Settings Edit
@Author: your name
@Date: 2014-02-13 23:34:14
@LastEditTime: 2019-09-25 22:17:23
@LastEditors: Please set LastEditors
'''
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print "successorGameState: ", successorGameState

        newPos = successorGameState.getPacmanPosition()
        # print "newPos: ", newPos

        currentFood = currentGameState.getFood()  # food available from current state
        # print "currentFood: ", currentFood

        # food available from successor state (excludes food@successor)
        newFood = successorGameState.getFood()
        # print "newFood: ", newFood

        # power pellets/capsules available from current state
        currentCapsules = currentGameState.getCapsules()
        # print "currentCapsules: ", currentCapsules

        # capsules available from successor (excludes capsules@successor)
        newCapsules = successorGameState.getCapsules()
        # print "newCapsules: ", newCapsules

        newGhostStates = successorGameState.getGhostStates()
        # print "newGhostStates: ", newGhostStates

        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        # print "newScaredTimes: ", newScaredTimes

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def maxValue(gameState, depth, totalGhostAgentsNumber):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = -float("inf")
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                value = max(value, minValue(
                    nextState, 1, depth, totalGhostAgentsNumber))
            return value

        def minValue(gameState, agentIndex, depth, totalGhostAgentsNumber):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == totalGhostAgentsNumber:
                    value = min(value, maxValue(
                        nextState, depth+1, totalGhostAgentsNumber))
                else:
                    value = min(value, minValue(
                        nextState, agentIndex+1, depth, totalGhostAgentsNumber))
            return value

        bestAction = None
        totalGhostAgentsNumber = gameState.getNumAgents() - 1
        value = -float("inf")
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = minValue(nextState, 1, 0, totalGhostAgentsNumber)
            if nextValue > value:
                value = nextValue
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, alpha, beta, depth, totalGhostAgentsNumber):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = -float("inf")
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                value = max(value, minValue(nextState, alpha, beta,
                                            1, depth, totalGhostAgentsNumber))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def minValue(gameState, alpha, beta, agentIndex, depth, totalGhostAgentsNumber):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == totalGhostAgentsNumber:
                    value = min(value, maxValue(nextState, alpha, beta,
                                                depth+1, totalGhostAgentsNumber))
                else:
                    value = min(value, minValue(nextState, alpha, beta,
                                                agentIndex+1, depth, totalGhostAgentsNumber))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        bestAction = None
        totalGhostAgentsNumber = gameState.getNumAgents() - 1
        value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = minValue(nextState, alpha, beta,
                                 1, 0, totalGhostAgentsNumber)
            if nextValue > value:
                value = nextValue
                bestAction = action
            alpha = max(alpha, value)

        return bestAction


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
        def maxValue(gameState, depth, totalGhostAgentsNumber):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = -float("inf")
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                value = max(value, expectValue(nextState, 1, depth,
                                               totalGhostAgentsNumber))
            return value

        def expectValue(gameState, agentIndex, depth, totalGhostAgentsNumber):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = 0
            allPossibilitesNumber = len(gameState.getLegalActions(agentIndex))
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == totalGhostAgentsNumber:
                    value += maxValue(nextState, depth+1,
                                      totalGhostAgentsNumber) / allPossibilitesNumber
                else:
                    value += expectValue(nextState, agentIndex+1, depth,
                                         totalGhostAgentsNumber) / allPossibilitesNumber
            return value

        totalGhostAgentsNumber = gameState.getNumAgents() - 1
        value = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = expectValue(nextState, 1, 0, totalGhostAgentsNumber)
            if nextValue > value:
                value = nextValue
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currenPacmanPosition = currentGameState.getPacmanPosition()
    currentFoodsPosition = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentGhostsPosition = [ghostState.getPosition()
                             for ghostState in currentGhostStates if ghostState.scaredTimer == 0]
    currentCapsulePosition = currentGameState.getCapsules()

    currenctScore = currentGameState.getScore()
    # if currentGameState.isWin():
    #     newScore = oldScore + 1000
    # if currentGameState.isLose():
    #     newScore = oldScore - 1000

    neareastFoodDistance = min(util.manhattanDistance(
        food, currenPacmanPosition) for food in currentFoodsPosition)
    foodScore = neareastFoodDistance

    neareastGhostDistance = min(util.manhattanDistance(
        ghost, currenPacmanPosition) for ghost in currentGhostsPosition)
    ghostScore = 2 * neareastGhostDistance

    nearestCapsulesDistance = min(util.manhattanDistance(
        currenPacmanPosition, capsule) for capsule in currentCapsulePosition)
    capsuleScore = nearestCapsulesDistance

    newScore = currenctScore - foodScore + 2*ghostScore + capsuleScore

    return newScore


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
