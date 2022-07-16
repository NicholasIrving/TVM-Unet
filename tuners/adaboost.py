# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tuner that uses xgboost as cost model"""

from tvm.autotvm.tuner.model_based_tuner import ModelBasedTuner, ModelOptimizer
# from .xgboost_cost_model import XGBoostCostModel
from tvm.autotvm.tuner.sa_model_optimizer import SimulatedAnnealingOptimizer
from .adaboost_cost_model import AdaboostCostModel
import numpy as np


class AdaboostTuner(ModelBasedTuner):

    def __init__(
        self,
        task,
        plan_size=64,
        feature_type="itervar",
        loss_type="rank",
        num_threads=None,
        optimizer="sa",
        diversity_filter_ratio=None,
        log_interval=50,
    ):
        cost_model = AdaboostCostModel(task, num_threads)

        if optimizer == "sa":
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), (
                "Optimizer must be " "a supported name string" "or a ModelOptimizer object."
            )

        super(AdaboostTuner, self).__init__(
            task, cost_model, optimizer, plan_size, diversity_filter_ratio
        )

    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(AdaboostTuner, self).tune(*args, **kwargs)

        # manually close pool to avoid multiprocessing issues
        # self.cost_model._close_pool()
