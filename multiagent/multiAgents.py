'''
@Description: In User Settings Edit
@Author: your name
@Date: 2014-02-13 23:34:14
@LastEditTime: 2019-09-26 11:37:04
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
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()  # food available from current state
        # food available from successor state (excludes food@successor)
        newFood = successorGameState.getFood()
        # power pellets/capsules available from current state
        currentCapsules = currentGameState.getCapsules()
        # capsules available from successor (excludes capsules@successor)
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currenctScore = successorGameState.getScore()

        foodScore = 0
        newFoodPositions = newFood.asList()
        if len(newFoodPositions):
            neareastFoodDistance = min(util.manhattanDistance(
                food, newPos) for food in newFoodPositions)
            if neareastFoodDistance == 0:
                foodScore -= 10
            foodScore = -neareastFoodDistance

        ghostScore = 0
        newGhostPositions = [ghostState.getPosition()
                             for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        if len(newGhostPositions):
            neareastGhostDistance = min(util.manhattanDistance(
                ghost, newPos) for ghost in newGhostPositions)
            if neareastGhostDistance == 0:
                return float("-inf")
            ghostScore = 2 * neareastGhostDistance

        capsuleScore = 0
        if len(newCapsules):
            nearestCapsulesDistance = min(util.manhattanDistance(
                newPos, capsule) for capsule in newCapsules)
            capsuleScore = - nearestCapsulesDistance

        scaredTimeScore = sum(newScaredTimes)

        newScore = currenctScore + foodScore + \
            ghostScore + capsuleScore + scaredTimeScore

        return newScore


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

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        totalGhostAgentsNumber = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions(0)

        bestAction = max(legalActions, key=lambda action: minValue(
            gameState.generateSuccessor(0, action), 1, 0, totalGhostAgentsNumber))

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

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

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
            legalActions = gameState.getLegalActions(0)
            if not legalActions or depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = -float("inf")
            for action in legalActions:
                nextState = gameState.generateSuccessor(0, action)
                value = max(value, expectValue(nextState, 1, depth,
                                               totalGhostAgentsNumber))
            return value

        def expectValue(gameState, agentIndex, depth, totalGhostAgentsNumber):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = 0
            probability = 1.0 / len(legalActions)

            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == totalGhostAgentsNumber:
                    value += maxValue(nextState, depth+1,
                                      totalGhostAgentsNumber) * probability
                else:
                    value += expectValue(nextState, agentIndex+1, depth,
                                         totalGhostAgentsNumber) * probability
            return value

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        totalGhostAgentsNumber = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions(0)

        # bestAction = max(expectValue(gameState.generateSuccessor(
        #     0, action), 1, 0, totalGhostAgentsNumber) for action in legalActions)

        bestAction = max(legalActions, key=lambda action: expectValue(
            gameState.generateSuccessor(0, action), 1, 0, totalGhostAgentsNumber))

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
      The newScore I desinged is based on currenctScore, but adjusted with the consideration of neareastFoodDistance, 
      neareastGhostDistance, nearestCapsulesDistance and currentScaredTime.

      1. foodScore: The distance from the current position of pacman to its nearest food. Pacman's goal is to eat all 
        the food as soon as possible, so the position where pacman has the minimun distance to the nearest food should 
        has the higher score. That means the weight of foodScore should be negative. In particular, if the nearestFoodDistance
        is 0, I will add 10 to give it a "bounus". The weight of foodScore is -1.

      2. ghostScore: The distance from the current position of pacman to its nearest ghost. Pacman need to avoid all ghosts, 
        so the longer the distance from pacman to the closest ghost is, the higher the score is. That means the weight of 
        foodScore should be positive. In particular, if the nearestGhostDistance is 0, I returned "-infinity" cause we should 
        absolutely avoid this position. The weight of ghostScore is 2.

      3. capsuleScore: The distance from the current position of pacman to its nearest capsule. Capsules can make the ghost 
        "sleep" for a moment, which is a good chance for pacman to move around. That means the weight of foodScore should be 
        negative. Since there are few capsules, I give weight of capsuleScore -2.

      4. scaredTimeScore: The sum of scaredTimer for all ghosts at currentGameState. A higher currentScaredTime means pacman 
        can move around with more freedom. That means the weight of scaredTimeScore should be positive. The weight of 
        capsuleScore is 1.

    """
    "*** YOUR CODE HERE ***"
    currenPacmanPosition = currentGameState.getPacmanPosition()
    currentFoodsPosition = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentGhostsPosition = [ghostState.getPosition()
                             for ghostState in currentGhostStates if ghostState.scaredTimer == 0]
    currentCapsulePosition = currentGameState.getCapsules()

    currenctScore = currentGameState.getScore()

    foodScore = 0
    if len(currentFoodsPosition):
        neareastFoodDistance = min(util.manhattanDistance(
            food, currenPacmanPosition) for food in currentFoodsPosition)
        if neareastFoodDistance == 0:
            foodScore -= 10
        foodScore = -1 * neareastFoodDistance

    ghostScore = 0
    if len(currentGhostsPosition):
        neareastGhostDistance = min(util.manhattanDistance(
            ghost, currenPacmanPosition) for ghost in currentGhostsPosition)
        if neareastGhostDistance == 0:
            return float("-inf")
        ghostScore = 2 * neareastGhostDistance

    capsuleScore = 0
    if len(currentCapsulePosition):
        nearestCapsulesDistance = min(util.manhattanDistance(
            currenPacmanPosition, capsule) for capsule in currentCapsulePosition)
        capsuleScore = -2 * nearestCapsulesDistance

    scaredTimeScore = 1 * sum([
        ghostState.scaredTimer for ghostState in currentGhostStates])

    newScore = currenctScore + foodScore + \
        ghostScore + capsuleScore + scaredTimeScore

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
