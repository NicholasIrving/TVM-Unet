import numpy as np
import torch.nn as nn
import torch

NUM_EPOCHS = 50
ALPHA = 5e-3  # learning rate
BATCH_SIZE = 3  # how many episodes we want to pack into an epoch
HIDDEN_SIZE = 64  # number of hidden nodes we have in our dnn
BETA = 0.1  # the entropy bonus multiplier
INPUT_SIZE = 3
ACTION_SPACE = 3
NUM_STEPS = 4
GAMMA = 0.99

class RLtuner(nn.Module):
    def __init__(self):

        self.NUM_EPOCHS = NUM_EPOCHS
        self.ALPHA = ALPHA
        self.BATCH_SIZE = BATCH_SIZE # number of models to generate for each action
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.INPUT_SIZE = INPUT_SIZE
        self.NUM_STEPS = NUM_STEPS
        self.ACTION_SPACE = ACTION_SPACE

        self.agent = self.Agent()

    def solve_environment(self):

        epoch = 0

        while epoch < self.NUM_EPOCHS:
            # init the epoch arrays
            # used for entropy calculation
            epoch_logits = torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

            # Sample BATCH_SIZE models and do average
            for i in range(self.BATCH_SIZE):
                # play an episode of the environment
                (episode_weighted_log_prob_trajectory,
                 episode_logits,
                 sum_of_episode_rewards) = self.play_episode()

            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)
            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

    def play_episode(self):

        # Init state
        init_state = config

        # get the action logits from the agent - (preferences)
        episode_logits = self.agent(torch.tensor(init_state).float().to(self.DEVICE))

        # sample an action according to the action distribution
        action_index = Categorical(logits=episode_logits).sample().unsqueeze(1)

        mask = one_hot(action_index, num_classes=self.ACTION_SPACE)

        episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

        # Get action actions
        # #TODO:get action
        action_space = torch.tensor([[3, 5, 7], [8, 16, 32], [3, 5, 7], [8, 16, 32]], device=self.DEVICE)

        action = torch.gather(action_space, 1, action_index).squeeze(1)




if __name__ == '__main__':
    RLtuner(None)