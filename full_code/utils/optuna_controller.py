# ==========================================================================
# Copyright (c) 2012-2024 Anonymous Authors

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==========================================================================

import copy
from typing import Any, List

import optuna
from configs.utils_config import config_add
from utils.log_config import get_my_logger

logger = get_my_logger(__name__)


class OptunaController():
    def __init__(self, name_sampler: str, name_pruner: str,
                 kwargs_sampler: dict, kwargs_pruner: dict,
                 path_db: str, study_name: str) -> None:
        self.name_sampler = name_sampler
        self.name_pruner = name_pruner
        self.kwargs_sampler = kwargs_sampler
        self.kwargs_pruner = kwargs_pruner
        self.path_db = path_db
        self.study_name = study_name

        self.storage_name = "sqlite:///" + path_db
        self.sampler = self._prepare_sampler(name_sampler, kwargs_sampler)
        self.pruner = self._prepare_pruner(name_pruner, kwargs_pruner)

    def _prepare_sampler(self, name_sampler: str, kwargs_sampler: dict) -> Any:
        sampler: Any
        if name_sampler == "TPESampler":
            sampler = optuna.samplers.TPESampler(
                **kwargs_sampler[name_sampler])
        elif name_sampler == "CmaEsSampler":
            sampler = optuna.samplers.CmaEsSampler(
                **kwargs_sampler[name_sampler])
        elif name_sampler == "RandomSampler":
            sampler = optuna.samplers.RandomSampler(
                **kwargs_sampler[name_sampler])
        else:
            raise ValueError(f"name_sampler={name_sampler} is invalid.")

        return sampler

    def _prepare_pruner(self, name_pruner: str, kwargs_pruner: dict) -> Any:
        pruner: Any
        if name_pruner == "NopPruner":
            pruner = optuna.pruners.NopPruner()
        elif name_pruner == "PercentilePruner":
            pruner = optuna.pruners.PercentilePruner(
                **kwargs_pruner[name_pruner])
        elif name_pruner == "SuccessiveHalvingPruner":
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                **kwargs_pruner[name_pruner])
        elif name_pruner == "HyperbandPruner":
            pruner = optuna.pruners.HyperbandPruner(
                **kwargs_pruner[name_pruner])
        elif name_pruner == "MedianPruner":
            pruner = optuna.pruners.MedianPruner(
                **kwargs_pruner[name_pruner])
        elif name_pruner == "ThresholdPruner":
            pruner = optuna.pruners.ThresholdPruner(
                **kwargs_pruner[name_pruner])
        else:
            raise ValueError(f"name_pruner={name_pruner} is invalid.")

        return pruner

    def get_sampler(self):
        return self.sampler

    def get_pruner(self):
        return self.pruner

    def get_study_name(self):
        return self.study_name

    def get_storage_name(self):
        return self.storage_name

    def _proc_SS(self, name: str):
        type_: str = name[name.find("_")+1:name.find("_")+4]
        type_what: str = name[name.find(type_):]
        what: str = type_what[4:]
        return type_, what

    def _proc_SS_KWARGS1(self, name: str):
        what: str = name[10:]
        return what

    def _proc_SS_KWARGS2(self, name: str):
        """
        # Args
        - name: E.g., "lr_SGDFLL".

        # Returns
        - arg, method, type_: E.g, arg="lr", method="SGD", and type_="FLL".
        """
        method_type: str = name[name.rfind("_")+1:]
        arg: str = name[:name.rfind("_")]
        method: str = method_type[:-3]
        type_: str = method_type[-3:]
        return arg, method, type_

    def _suggest_with_type(self, type_: str, list_: List, name: str, trial: optuna.trial.Trial) -> Any:
        """
        # Args
        - type_: "CAT", "FLT", "INT", "FLL", or "INL".
        - list_: Usef for search space.
        - name: Name of hparam.
        - trial: Trial.

        # Returns:
        - sug: Suggested value. If len(list_) == 1, sug = list_[0] (no suggestion by Optuna).
        """
        if len(list_) == 1:
            sug = list_[0]
            logger.info("Optuna: No suggestion.")

        elif len(list_) > 1:
            logger.info(f"Optuna: Suggested!: {name}")
            if type_ == "CAT":
                sug = trial.suggest_categorical(name=name, choices=list_)
            elif type_ == "FLT":
                sug = trial.suggest_float(
                    name=name, low=list_[0], high=list_[1], log=False)
            elif type_ == "FLL":
                sug = trial.suggest_float(
                    name=name, low=list_[0], high=list_[1], log=True)
            elif type_ == "INT":
                sug = trial.suggest_int(
                    name=name, low=list_[0], high=list_[1], log=False)
            elif type_ == "INL":
                sug = trial.suggest_int(
                    name=name, low=list_[0], high=list_[1], log=True)
            else:
                raise ValueError(f"type_={type_} is invalid.")

        else:
            raise ValueError(
                f"Must be len(list_) >= 1. Got {len(list_)}. type_={type_}, name={name}.")

        return sug

    def suggest_params(self, config: dict, trial: optuna.trial.Trial) -> dict:
        for key in copy.deepcopy(config):
            # First round SS_{CAT, FLT, INT, FLL, INL}_{WHAT}
            if ("SS_" in key and key.rfind("SS_") == 0 and not "SS_KWARGS_" in key) or key == "SS_CAT_NAME_LOSS_REWEIGHT":
                type_, what = self._proc_SS(
                    key)  # E.g., "CAT", "NAME_SCHEDULER"
                assert type_ in ["CAT", "INT", "FLT",
                                 "INL", "FLL",], f"type_={type_}, what={what}, key={key}."
                list_ = config[key]  # E.g., ["Constant", "LinearLR"]
                sug = self._suggest_with_type(  # E.g., "LinearLR"
                    type_=type_, list_=list_, name=what, trial=trial)
                config = config_add(
                    config, what, sug)

            # Second round: SS_KWARGS_{WHAT}, WHAT != LOSSES
            if "SS_KWARGS_" in key and key.rfind("SS_KWARGS_") == 0 and key != "SS_KWARGS_LOSSES":
                what = self._proc_SS_KWARGS1(key)  # E.g., "NAME_OPTIMIZER"
                sug_what = config[what]  # E.g., "SGD"

                if len(copy.deepcopy(config)[key][sug_what].keys()) != 0:
                    # E.g., config[key][sug_what] =
                    # {'lr_SGDFLL': FLL_LR, 'momentum_SGDFLT': [0., 0.999], 'dampening_SGDFLT': [0.],
                    # 'weight_decay_SGDFLL': FLL_WEIGHT_DECAY, 'nesterov_SGDCAT': [True]}
                    for k in copy.deepcopy(config)[key][sug_what].keys():
                        # E.g., "lr", "SGD", "FLL"
                        arg, method, type_ = self._proc_SS_KWARGS2(k)
                        assert method == sug_what, f"{method}, {sug_what}. Invalid naming rule?"
                        assert type_ in ["CAT", "INT", "FLT",
                                         "INL", "FLL",], f"type_={type_}"
                        # E.g., [1e-8, 1e-1]
                        ss: List = config[key][sug_what][k]
                        # E.g., 0.002434
                        sug = self._suggest_with_type(
                            type_=type_,
                            list_=ss,
                            name=k,
                            trial=trial)
                        config[f"KWARGS_{what}"][method] = config_add(
                            # E.g., config["KWARGS_NAME_OPTIMIZER"]["SGD"]
                            config[f"KWARGS_{what}"][method],
                            # E.g., "lr", 0.002434
                            arg, sug)
                else:  # = Search space is empty.
                    pass

            # Third round: SS_KWARGS_LOSSES
            if key == "SS_KWARGS_LOSSES":
                what = self._proc_SS_KWARGS1(key)  # "LOSSES"
                for itr_loss in copy.deepcopy(config)["LIST_LOSSES"]:
                    if len(copy.deepcopy(config)[key][itr_loss].keys()) != 0:
                        # E.g., config[key="SS_KWARGS_LOSSES"][itr_loss="LpLoss"]
                        # = {"p_LpLossCAT": [1, 2, float("inf")]}
                        for k in copy.deepcopy(config)[key][itr_loss].keys():
                            # E.g., "p", "LpLoss", "CAT"
                            arg, method, type_ = self._proc_SS_KWARGS2(k)
                            # E.g., [1, 2, float("inf")]
                            ss = config[key][itr_loss][k]
                            # E.g., float("inf")
                            sug = self._suggest_with_type(
                                type_=type_,
                                list_=ss,
                                name=k,
                                trial=trial)
                            config[f"KWARGS_{what}"][method] = config_add(
                                # E.g., config["KWARGS_LOSSES"]["LpLoss"]
                                config[f"KWARGS_LOSSES"][method],
                                # E.g., "p", float("inf")
                                arg, sug)
                    else:  # = Search space is empty.
                        pass

        return config
