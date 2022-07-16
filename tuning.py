import tvm
from tvm import autotvm

import tuners.RLTuner as rl
from tuners.adaboost import AdaboostTuner


def tuning(tasks, tune, tune_option, tmp_log, log_name, device):
    print('Tuning.........')
    for i, task in enumerate(tasks):
        if tune == "xgb":
            tuner = autotvm.tuner.XGBTuner(task, loss_type="rank", plan_size=8)
        elif tune == "ga":
            tuner = autotvm.tuner.GATuner(task, pop_size=100)
        elif tune == "random":
            tuner = autotvm.tuner.RandomTuner(task)
        elif tune == "grid":
            tuner = autotvm.tuner.GridSearchTuner(task)
        elif tune == 'rl':
            tuner = rl.nas(task, tune_option, device)
        elif tune == 'adaboost':
            tuner = AdaboostTuner(task, plan_size=8)


        prefix = '[Task %2d / %2d]' % (i + 1, len(tasks))
        n_trial = min(tune_option['n_trial'], len(task.config_space))
        tuner.tune(
            n_trial=n_trial,
            early_stopping=tune_option['early_stopping'],
            measure_option=tune_option['measure_option'],
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tune_option['tmp_log'])
            ]
        )

    autotvm.record.pick_best(tmp_log, log_name)
