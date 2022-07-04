import numpy as np
import torch
import torch.nn as nn
import logging
import tempfile
from tvm.autotvm.tuner.tuner import Tuner
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.utils import format_si_prefix
from tvm.autotvm.env import GLOBAL_SCOPE
logger = logging.getLogger("autotvm")

from .agent import Agent
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical

class RLTuner(Tuner):
    def __init__(self, task, tune_option, device):
        super(RLTuner, self).__init__(task)

        self.device = torch.device(device)

        # space info
        # self.ACTION_SPACE = [[-1], [0], [1]]
        self.zero_ACTION_SPACE = [[0], [1]]
        self.space = task.config_space
        self.measure_option = tune_option['measure_option']
        self.dim_keys = []
        self.dims = []
        for k, v in self.space.space_map.items():
            self.dim_keys.append(k)
            self.dims.append(len(v))

        self.visited = []


        init_state = self.init_state()
        self.avg_cost = self.get_init_cost(self.measure_option, init_state)
        self.loss_func = nn.MSELoss()

    def next_batch(self, batch_size):
        return self.get_action()

    def update(self, inputs, results):
        pass

    # def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
    #     measure_batch = create_measure_batch(self.task, measure_option)
    #     n_parallel = getattr(measure_batch, "n_parallel", 1)
    #     early_stopping = early_stopping or 1e9
    #     self.n_trial = n_trial
    #     self.early_stopping = early_stopping
    #
    #     # Validate si_prefix arg
    #     format_si_prefix(0, si_prefix)
    #
    #     old_level = logger.level
    #
    #     GLOBAL_SCOPE.in_tuning = True
    #     i = error_ct = 0
    #     errors = []
    #     while i < n_trial:
    #         if not self.has_next():
    #             break
    #
    #         configs = self.next_batch(min(n_parallel, n_trial - i))
    #
    #         inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
    #         results = measure_batch(inputs)
    #
    #         reward = self.avg_cost / results.costs
    #
    #         # keep best config
    #         for k, (inp, res) in enumerate(zip(inputs, results)):
    #             config = inp.config
    #             if res.error_no == 0:
    #                 flops = inp.task.flop / np.mean(res.costs)
    #                 error_ct = 0
    #             else:
    #                 flops = 0
    #                 error_ct += 1
    #                 error = res.costs[0]
    #                 if isinstance(error, str):
    #                     errors.append(error)
    #                 else:
    #                     errors.append(str(error))
    #
    #             if flops > self.best_flops:
    #                 self.best_flops = flops
    #                 self.best_config = config
    #                 self.best_measure_pair = (inp, res)
    #                 self.best_iter = i + k
    #
    #             logger.debug(
    #                 "No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
    #                 i + k + 1,
    #                 si_prefix,
    #                 format_si_prefix(flops, si_prefix),
    #                 format_si_prefix(self.best_flops, si_prefix),
    #                 res,
    #                 config,
    #             )
    #
    #         i += len(results)
    #         self.ttl = min(early_stopping + self.best_iter, n_trial) - i
    #
    #         self.update(inputs, results)
    #         for callback in callbacks:
    #             callback(self, inputs, results)
    #
    #         if i >= self.best_iter + early_stopping:
    #             logger.debug("Early stopped. Best iter: %d.", self.best_iter)
    #             break
    #
    #         if error_ct > 150:
    #             logging.basicConfig()
    #             logger.warning("Too many errors happen in the tuning. Switching to debug mode.")
    #             logger.setLevel(logging.DEBUG)
    #         else:
    #             logger.setLevel(old_level)
    #
    #     if error_ct == i:
    #         _, f = tempfile.mkstemp(prefix="tvm_tuning_errors_", suffix=".log", text=True)
    #         with open(f, "w") as file:
    #             file.write("\n".join(errors))
    #         logging.warning(
    #             "Could not find any valid schedule for task %s. "
    #             "A file containing the errors has been written to %s.",
    #             self.task,
    #             f,
    #         )
    #     GLOBAL_SCOPE.in_tuning = False
    #     del measure_batch

    def init_state(self):
        state = point2knob(0, self.dims)
        return state

    def get_action(self, state, tune_number):
        state = state[:tune_number]
        actions = []
        # get the action logits from the agent - (preferences)
        for i in range(tune_number):
            state[i]
            episode_logits = self.agent.network(torch.tensor([[state[i]]]).float().to(self.device))

            # sample an action according to the action distribution
            action_index = Categorical(logits=episode_logits).sample().unsqueeze(1)
            # mask = one_hot(action_index, num_classes=len(self.ACTION_SPACE))
            # episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
            # action_space = torch.tensor(self.zero_ACTION_SPACE)
            # action = torch.gather(action_space, 0, action_index).squeeze(1)
            # if state[i] > 0:
            #     action = self.ACTION_SPACE[action_index]
            # if state[i] == 0:
            action = self.zero_ACTION_SPACE[action_index]
            actions.append(action)

        return actions

    def get_next_state(self, state, actions):
        next_state = state.copy()
        for i in range(len(actions)):
            next_state[i] = state[i] + actions[i][0]

        # state = self.space.get(knob2point(next_state, self.dims))
        return next_state

    def get_init_cost(self, measure_option, init_state):

        builder = measure_option["builder"]
        runner = measure_option["runner"]

        attach_objects = runner.set_task(self.task)
        build_kwargs = runner.get_build_kwargs()
        builder.set_task(self.task, build_kwargs)


        # measure_batch = create_measure_batch(self.task, measure_option)
        init_state = self.space.get(knob2point(init_state, self.dims))
        inputs = [MeasureInput(self.task.target, self.task, init_state)]

        # results = measure_batch(inputs)

        results = runner.run(inputs, builder.build(inputs))
        cost = results[0].costs
        return cost

    def get_reward(self, measure_option, state):
        measure_batch = create_measure_batch(self.task, measure_option)
        state = self.space.get(knob2point(state, self.dims))
        inputs = [MeasureInput(self.task.target, self.task, state)]
        results = measure_batch(inputs)
        try:
            reward = self.avg_cost[0] / results[0].costs[0]
        except:
            return 0
        return reward
