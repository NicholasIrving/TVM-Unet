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
# pylint: disable=invalid-name
"""XGBoost as cost model"""

import logging
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from tvm.autotvm.tuner.model_based_tuner import CostModel, FeatureCache
from tvm.contrib.popen_pool import PopenPoolExecutor, StatusKind
import tvm.autotvm.feature as feature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor

rf = None
logger = logging.getLogger("autotvm")


class AdaboostCostModel(CostModel):

    def __init__(self, task, num_threads=None):
        super(AdaboostCostModel, self).__init__()

        self.task = task
        self.target = task.target
        self.space = task.config_space

        self.num_threads = num_threads
        self.feature_cache = FeatureCache()
        self.feature_extract_func = _extract_itervar_feature_index
        self.feature_extra_ct = 0
        self.fea_type = "itervar"
        self.pool = None
        self._reset_pool(self.space, self.target, self.task)

    def fit(self, xs, ys, plan_size):
        x_train = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)
        stdscaler = StandardScaler().fit(x_train)
        train = stdscaler.transform(x_train)
        clf = AdaBoostRegressor()
        self.rf = clf.fit(train, y_train)

    def predict(self, xs, output_margin=False):
        feas = self._get_feature(xs)
        y_pred = self.rf.predict(feas)
        # print(y_pred)
        return y_pred

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            feas = pool.map_with_error_catching(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea.value if fea.status == StatusKind.COMPLETE else None

        feature_len = None
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = fea_cache[idx].shape[-1]
                break

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            ret[i, :] = t if t is not None else 0
        return ret

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""
        self._close_pool()

        self.pool = PopenPoolExecutor(
            max_workers=self.num_threads,
            initializer=_extract_popen_initializer,
            initargs=(space, target, task),
        )

    def _close_pool(self):
        if self.pool:
            self.pool = None

    def _get_pool(self):
        return self.pool


# Global variables for passing arguments to extract functions.
_extract_space = None
_extract_target = None
_extract_task = None


def _extract_popen_initializer(space, target, task):
    global _extract_space, _extract_target, _extract_task
    _extract_space = space
    _extract_target = target
    _extract_task = task


def _extract_itervar_feature_index(args):
    """extract iteration var feature for an index in extract_space"""
    config = _extract_space.get(args)
    with _extract_target:
        sch, fargs = _extract_task.instantiate(config)

    fea = feature.get_itervar_feature_flatten(sch, fargs, take_log=True)
    fea = np.concatenate((fea, list(config.get_other_option().values())))
    return fea


def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    inp, res = arg
    config = inp.config
    with inp.target:
        sch, args = inp.task.instantiate(config)
    fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
    x = np.concatenate((fea, list(config.get_other_option().values())))

    if res.error_no == 0:
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0.0
    return x, y

