import numpy as np
from MCTSNode import MCTSNode

class GameStateMove():
    def __init__(self):
        pass

class GameState():
    def __init__(self, state, depth=0):
        self.state = state
        self.depth = depth


    def move(self, move):
        new_config = np.copy(self.config)
        new_config[self.depth] = move
        return GameState(new_config, self.depth+1)

    # def get_legal_actions(self):
