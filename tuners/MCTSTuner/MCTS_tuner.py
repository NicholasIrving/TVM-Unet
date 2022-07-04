import numpy as np
import mctspy
from tvm.autotvm.tuner import Tuner
from MCTSNode import TVMMCTSNode
from MCTS import MCTS

class MCTSTuner(Tuner):
    def __init__(self, task):
        super(MCTSTuner, self).__init__(task)

        self.space = task.config_space
        self.dim_keys = []
        self.dims = []
        for k, v in self.space.space_map.items():
            self.dim_keys.append(k)
            self.dims.append(len(v))

        # self.state = self.dims #TODO: set a init state
        self.config = np.ones_like(self.dims)*-1
        self.depth = 0

        root = TVMMCTSNode(self.config, self.dims, depth=self.depth)
        mcts = MCTS(root)
        best_node = mcts.best_action(10000)




