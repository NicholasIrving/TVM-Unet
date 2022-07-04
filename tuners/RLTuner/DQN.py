import torch
import torch.nn as nn
from .RLTuner import RLTuner
from .agent import Agent
from tvm.autotvm.measure import MeasureInput, create_measure_batch

START_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 1000000
MAX_EPISODE = 10
MAX_EPOCH = 100

class DQN(RLTuner):
    def __init__(self, tasks, tune_option, device):
        super(DQN, self).__init__(tasks, tune_option, device)
        self.tune_number = 3
        self.action_space = [-1, 0, 1]
        self.GAMMA = 0.99
        self.agent = Agent(1, device=self.device)

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
        rltuner = RLTuner
        for i in range(MAX_EPISODE):
            state = rltuner.init_state(self)
            total_reward = 0

            print('No.',i,'epoch')

            for j in range(MAX_EPOCH):
                action = rltuner.get_action(self, state, self.tune_number)
                # a = []
                # import numpy
                # for i in actions:
                #     a.append(i.numpy().tolist())
                # actions = a
                next_state = rltuner.get_next_state(self, state, action)
                reward = rltuner.get_reward(self, self.measure_option, state)


                if j == MAX_EPOCH:
                    done = 0
                else:
                    done = 1
                self.agent.learn(self.tune_number, state, action, reward, next_state, done)
                total_reward += reward
                print(j, '-----------', state, '------------', total_reward, '----------')
                state = next_state

            if EPSILON > FINAL_EPSILON:
                EPSILON -= (START_EPSILON - FINAL_EPSILON) / EXPLORE




