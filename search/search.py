# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.total_cost = 0
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
            self.total_cost = parent.total_cost + path_cost

    def expand(self, problem):
        return [Node(sucessor[0], self, sucessor[1], sucessor[2]) for sucessor in problem.getSuccessors(self.state)]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def actions(self):
        return [node.action for node in self.path()][1:]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    startState = problem.getStartState()
    startNode = Node(startState, None, [], 0)
    fringe.push(startNode)
    closed = {}
    while not fringe.isEmpty():
        currentNode = fringe.pop()
        if problem.isGoalState(currentNode.state):
            return currentNode.actions()
        elif currentNode.state not in closed:
            closed[currentNode.state] = 1
            for nextNode in currentNode.expand(problem):
                fringe.push(nextNode)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    startState = problem.getStartState()
    startNode = Node(startState, None, [], 0)
    fringe.push(startNode)
    closed = {}
    while not fringe.isEmpty():
        currentNode = fringe.pop()
        if problem.isGoalState(currentNode.state):
            return currentNode.actions()
        elif currentNode.state not in closed:
            closed[currentNode.state] = 1
            for nextNode in currentNode.expand(problem):
                fringe.push(nextNode)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = Node(startState, None, [], 0)
    fringe.push(startNode, 0)
    closed = {}
    while not fringe.isEmpty():
        currentNode = fringe.pop()
        if problem.isGoalState(currentNode.state):
            return currentNode.actions()
        elif currentNode.state not in closed:
            closed[currentNode.state] = 1
            for nextNode in currentNode.expand(problem):
                fringe.push(nextNode, nextNode.total_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    startState = problem.getStartState()
    startHeuristic = heuristic(startState, problem)
    startNode = Node(startState, None, [], 0+startHeuristic)
    fringe.push(startNode, 0)
    closed = {}
    while not fringe.isEmpty():
        currentNode = fringe.pop()
        if problem.isGoalState(currentNode.state):
            return currentNode.actions()
        elif currentNode.state not in closed:
            closed[currentNode.state] = 1
            for nextNode in currentNode.expand(problem):
                fringe.push(nextNode, nextNode.total_cost +
                            heuristic(nextNode.state, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
