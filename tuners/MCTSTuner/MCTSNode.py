import numpy as np
from collections import defaultdict

class MCTSNode():
    def __init__(self, state, dims, depth=0, parent=None):
        self.state = state
        self.dims = dims
        self.depth = depth
        self.parent = parent
        self.children = []


    def is_terminal_node(self):
        return self.state.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_action) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


class TVMMCTSNode(MCTSNode):

    def __init__(self, state, dims, depth=0, parent=None):
        super(TVMMCTSNode, self).__init__(state, dims, depth, parent)
        self.number_of_visit = 0.
        self.result = defaultdict(int)
        self.untried_action = None
        if self.untried_action is None:
            self.untried_action = np.arange(state[depth])

    def expand(self):
        action = list(self.untried_action.pop())
        next_state = self.state.move(action)
        child_node = TVMMCTSNode(next_state, self.dims, depth=next_state.depth, parent=self)

        return child_node

    def rollout(self, dims, depth):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_move = np.arange(self.dims[self.depth])
            action = self.rollout_policy(possible_move)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.result #TODO:get time result

    def backpropagate(self, result):
        self.number_of_visit += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)



