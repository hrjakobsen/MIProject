from games.hexagon import HexagonGame
import copy

def generateStates(state: HexagonGame):
    states = [state]
    terminalStates = []
    seen = {}
    toSearch = [state]
    while toSearch:
        s = toSearch.pop()

        if s.hash() in seen:
            continue

        seen[s.hash()] = True

        if s.gameEnded():
            terminalStates.append(s)
        else:
            states.append(s)

        for p in [1, 2]:
            for a in s.getActions():
                newS = copy.deepcopy(s)
                newS.makeMove(p, a)
                if newS.hash() not in seen:
                    toSearch.append(newS)

    print("found {} states".format(len(seen)))
    return list(set(states)), list(set(terminalStates))


class Node(object):
    def __init__(self, state: HexagonGame, player, gamma):
        self.state = state
        self.player = player
        self.terminal = state.gameEnded()
        self.q = None
        if self.terminal:
            self.q = state.getReward(1)
        self.subTrees = {}
        self.hash = state.hash()
        self.possibleActions = []
        self.gamma = gamma

    def build(self, nodes):
        nextPlayer = self.nextPlayer()
        hashes = []
        for a in self.state.getActions():
            newS = copy.deepcopy(self.state)
            newS.makeMove(self.player, a)
            newHash = newS.hash()
            # don't do a move we cannot do
            if self.player == 1:
                if newHash == self.hash:
                    continue
            self.possibleActions.append(a)
            foundNode = nodes[newHash, nextPlayer]
            self.subTrees[a] = foundNode
            hashes.append(newHash)
        return

    def nextPlayer(self):
        return 1 if self.player == 2 else 2

    def calculateQ(self, later):
        if self.q is not None:
            return self.q

        f = max if self.player == 1 else min
        q = None
        for subTree in self.subTrees.values():
            if subTree.hash == self.hash:
                if subTree.q is None:
                    later[subTree.hash, self.nextPlayer()] = subTree
                continue

            subTreeQ = subTree.calculateQ(later)
            if q is None:
                q = subTreeQ
            else:
                q = f(subTreeQ, q)

        if q is None:
            state2 = copy.deepcopy(self.state)
            state2.makeMove(self.nextPlayer(), 0)
            self.q = state2.getReward(1)
            return self.q

        self.q = self.state.getReward(1) + self.gamma * q
        return q

    def calculateRestOfQ(self, nodes):
        for a in self.state.getActions():
            if a not in self.subTrees:
                self.subTrees[a] = nodes[self.hash, self.nextPlayer()]

    def createQTable(self, Q):
        for a in self.subTrees:
            subTree = self.subTrees[a]
            if subTree.hash == self.state.hash():
                # discount moves which doesn't do anything
                Q[self.hash, a] = self.state.getReward(1) + self.gamma * subTree.q
            else:
                Q[self.hash, a] = subTree.q

class HexagonBruteforce(object):
    def __init__(self, state: HexagonGame, player, gamma=1):
        states, terminal = generateStates(state)
        self.nodes = {}
        for s in states + terminal:
            for p in [1, 2]:
                self.nodes[s.hash(), p] = Node(s, p, gamma)

        # now build the tree
        for node in self.nodes.values():
            node.build(self.nodes)
        self.player = player

        for node in self.nodes.values():
            if node.q is None:
                node.calculateQ({})

        for node in self.nodes.values():
            node.calculateRestOfQ(self.nodes)

        self.Q = {}
        for node in self.nodes.values():
            if node.player == 1:
                node.createQTable(self.Q)



    def getMove(self, state: HexagonGame, reward, actions):

        maxQ = None
        maxA = None
        """
        node = self.nodes[state.hash(), 1]

        for a in state.getActions():
            subTree = node.subTrees[a]
            Q = subTree.q
            if maxQ is None:
                maxQ = Q
                maxA = a
            elif maxQ < Q:
                maxQ = Q
                maxA = a

        """
        node = self.nodes[state.hash(), 1]
        for a in state.getActions():
            q = self.Q[state.hash(), a]
            if maxQ is None:
                maxQ = q
                maxA = a
            elif maxQ < q:
                maxQ = q
                maxA = a
        return maxA

    def finalize(self, state, r, actions):
        pass  # game = HexagonGame(3, 3)

    def save(self):
        fileName = "Bruteforce.txt"
        import pickle
        with open(fileName, 'wb') as handle:
            pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
# q = HexagonBruteForce2(game, 1)  # generateStates(game)
# q.getMove(game, 0)
