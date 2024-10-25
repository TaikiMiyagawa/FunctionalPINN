# ==========================================================================
# Copyright (c) 2012-2024 Taiki Miyagawa

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

import datetime
import glob
import os
import random
import shutil
import socket
import sys
import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from configs.utils_config import config_add
from torch import Tensor
from utils.log_config import get_my_logger

logger = get_my_logger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters of a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_sequence(output: Tensor, y: Tensor, num_classes: int) -> Tuple[Tensor, Tensor]:
    """
    Merges batch and temporal axes.

    # Args
    - output: [B,T,num_classes]
    - y: [B,]

    # Returns
    - output: [B*T, num_classes]
    - y: [B*T,]
    """
    duration = output.shape[1]
    output = output.reshape(-1, num_classes)  # [B*T,num_classes]
    y = torch.tile(y.unsqueeze(1), [1, duration]).reshape(-1)  # [B*T,]
    return output, y


def remove_padded_timestamps(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    """
    # Args
    - logits: [B*T, num_classes]
    - targets: [B*T,]

    # Returns
    - logits: [<B*T, num_classes]
    - targets: [<B*T,]. Length = logits.shape[0].

    # Caution
    If for some i, logits[i, 0] is accidentally = 0., while logits[i,1:] are all non-zero
    (i.e., the i-th component is not a padded timestamp), remove_padded_timestamps wrongly
    removes the i-th component from both logits and targets. This is not desirable even though
    that case is rare. Modification needed anyway.
    """
    # Find indices of zero-padded batches
    non_zero_indices = torch.nonzero(logits[:, 0])[:, 0]  # [*,]

    # Remove zero-padded batches from logits and targets
    logits = logits[non_zero_indices]
    targets = targets[non_zero_indices]

    return logits, targets


def initial_processes1(config: dict) -> dict:
    """
    Rank-independent initial processes are summarized.

    # Args
    - config: A config dictionary.

    # Returns
    - config: A config dictionary. Keywards, DEVICE and WORLD_SIZE, are added.
    """
    # Accelerate training
    if not config["FLAG_FIX_SEED"]:
        torch.backends.cudnn.benchmark = True  # no reproducibility, but fast

    # Default dtype
    torch.set_default_dtype(config["DEFAULT_DTYPE"])

    # Restrict visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    config = config_add(config, "DEVICE",
                        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = config_add(config, "WORLD_SIZE", torch.cuda.device_count())
    config = config_add(config, "PORT", find_free_port())

    return config


def initial_processes2(
        config: dict, rank: int,
        single_trial: Union[None, optuna.trial.Trial]) -> Tuple[optuna.trial.Trial, dict]:
    """
    Rank- and trial-dependent initial processes are summarized.

    # Args
    - config: A config dictionary.
      NOW, NAME_TRIAL, DIR_TBLOG, DIR_CKPT, DIR_TXLOG, DIR_CONFIG is to be added here.
    - rank: Local rank.
    - single_trial: Trial for rank = 0; otherwise None.

    # Returns
    - trial: Optuna trial.
    - config: A config dictionary.
    """
    # torch.distributed: initialize process group
    if config["FLAG_MULTIGPU"]:
        # TorchDistributedTrial:
        # 1. Set trial object in rank-0 node and set None in the other rank node.
        # 2. The methods of TorchDistributedTrial are expected to be called by all workers at once.
        #    They invoke synchronous data transmission to share processing results and synchronize timing.
        trial = optuna.integration.TorchDistributedTrial(
            single_trial)
    else:
        trial = single_trial

    # Fix seed
    if config["FLAG_FIX_SEED"]:
        fix_seed(config["SEED"])

    # Set num threads
    torch.set_num_threads(config["NUM_THREADS_PER_PROCESS"])

    # Add config items
    now = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S%f")[:-3]
    if config["FLAG_MULTIGPU"]:
        tensor: Tensor = torch.tensor(
            int(now.replace("_", ""))).to(rank)  # type: ignore
        torch.distributed.broadcast(tensor, src=0)
        now = str(tensor.cpu().numpy())
        now = now[:8] + "_" + now[8:]

    config = config_add(config, "NOW", now)
    config = config_add(config, "NAME_TRIAL",
                        f"{config['COMMENT']}_{config['NOW']}")
    config = config_add(config, "DIR_TBLOG",
                        config["ROOTDIR_TBLOGS"] + f"/{config['NAME_SUBPROJECT']}/{config['NAME_TRIAL']}")
    config = config_add(config, "DIR_CKPT",
                        config["ROOTDIR_CKPTS"] + f"/{config['NAME_SUBPROJECT']}/{config['NAME_TRIAL']}")
    config = config_add(config, "DIR_TXLOG",
                        config["ROOTDIR_TXLOGS"] + f"/{config['NAME_SUBPROJECT']}/{config['NAME_TRIAL']}")
    config = config_add(config, "DIR_CONFIG",
                        config["ROOTDIR_CONFIGS"] + f"/{config['NAME_SUBPROJECT']}/{config['NAME_TRIAL']}")

    # Make directories
    os.makedirs(config["DIR_TBLOG"], exist_ok=True)
    os.makedirs(config["DIR_TXLOG"], exist_ok=True)
    os.makedirs(config["DIR_CONFIG"], exist_ok=True)
    if (config["FLAG_SAVE_CKPT_TRYTUNING"] and config["EXP_PHASE"] in ["try", "tuning"]) or config["EXP_PHASE"] == "stat":
        os.makedirs(config["DIR_CKPT"], exist_ok=True)

    # Logger settings # TODO: new logger does not output logs to stout... See output.log.
    sys.stdout = Logger(config["DIR_TXLOG"]+"/output.log",
                        config["FLAG_OUTPUT_CONSOLE"])
    if rank == 0:
        logger.info(f"Printing config...\n======================= Print INITIAL config start =======================\n" +
                    f"{config}\n======================= Print INITIAL config end =======================\nPrinting config done.")

    # Save config file
    shutil.copy2(config["PATH_CONFIG"],
                 config["DIR_CONFIG"] + f"/config_{config['NOW']}.py")
    np.save(config["DIR_CONFIG"] + f"/config_{config['NOW']}.npy",
            config)

    # Stat phase: Load the best (or a) trial config used tuning phase
    if config["EXP_PHASE"] == "stat":
        logger.info(
            "EXP_PHASE is stat. Loading a config file from one of tuning trials...")
        _tmp = config["DIR_DBFILE"].replace(
            "dbfiles", "configs") + "/*"
        tuning_trial = sorted(glob.glob(_tmp))[trial.number]
        path_config_npy = glob.glob(tuning_trial+"/*.npy")[0]
        config = load_config_stat(path_config_npy, config)
        logger.info(
            f"tuning_trial={tuning_trial}, path_config_npy={path_config_npy}")

        if rank == 0:
            logger.info(f"Printing config...\n======================= Print LOADED config start =======================\n" +
                        f"{config}\n======================= Print LOADED config end =======================\nPrinting config done.")

        shutil.copy2(config["PATH_CONFIG"],
                     config["DIR_CONFIG"] + f"/config_{config['NOW']}_after_loading_tuning_config.py")
        np.save(config["DIR_CONFIG"] + f"/config_{config['NOW']}_after_loading_tuning_config.npy",
                config)

    # Verbose
    logger.info(f"Rank={rank}")
    logger.info(
        f"Using {config['DEVICE']} device: {config['CUDA_VISIBLE_DEVICES']} " +
        f"with world_size: {config['WORLD_SIZE']}.")
    logger.info(f"FLAG_MULTIGPU: {config['FLAG_MULTIGPU']}")
    logger.info(f"Trial name: {config['NAME_TRIAL']}")
    logger.info(f"with DIR_TBLOG={config['DIR_TBLOG']}")
    logger.info(f"with DIR_CKTP={config['DIR_CKPT']}")
    logger.info(f"with DIR_TXLOG={config['DIR_TXLOG']}")
    logger.info(f"with DIR_CONFIG={config['DIR_CONFIG']}")
    logger.info(f"Model={config['NAME_MODEL']}\n")

    return trial, config


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(('', 0))            # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def ddp_setup(rank: int, world_size: int, port: int) -> None:
    """
    For torch.distributed (multi-GPU).
    # Args
    - rank: Unique identifier of each process
    - world_size:
    # Remark
    - world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size)


def fix_seed(seed=777):
    """
    Fix random seeds.
    Note that torch.backends.cudnn.benchmark=True breaks reproduciblity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class Logger(object):
    def __init__(self, filename, flag_output_console):
        logger.info(f"Logging to file {filename}...")
        self.flag = flag_output_console
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        if self.flag:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def final_processes1(rank: int, study: optuna.study.Study) -> None:
    """ Do this in rank = 0. """
    assert rank == 0

    # Optuna results
    assert study is not None
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logger.info("\nStudy statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    logger.info(f"Current best trial:")
    try:
        trial = study.best_trial
        logger.info(f"  Value: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))
    except ValueError as e:
        logger.error(f"{e}")


def final_processes2(rank: int, config: dict) -> None:
    # torch.distributed: destroy process group and open ports
    if config["FLAG_MULTIGPU"]:
        if rank == 0:
            logger.info("Destroying all the processes. Wait 10 secs...")
            time.sleep(10)
        logger.info("\n==== Destroy process_group for rank {} ====\n".format(
            torch.distributed.get_rank()))
        torch.distributed.destroy_process_group()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    img = torch.sigmoid(img)  # approx. unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(model: torch.nn.Module, x: torch.Tensor):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images

    # Args
    - model: logits, feat = model(x)
    - x: A batch of images after preprocesses.
    '''
    output, _ = model(x)

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model, x: torch.Tensor, y: torch.Tensor, global_step: int) -> plt.figure:
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.

    # Args
    - x: A batch of images after preprocesses.
    - y: A batch of labels.

    # Returns
    - fig:
    '''
    preds, probs = images_to_probs(model, x)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 30))
    for idx in np.arange(min(4, x.shape[0])):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(
            x[idx], one_channel=True if x.shape[1] == 1 else False)
        ax.set_title("Pred {0}, {1:.1f}%\n(label: {2})\nStep {3}".format(
            preds[idx],
            probs[idx] * 100.0,
            y[idx],
            global_step),
            color=("green" if preds[idx] == y[idx].item() else "red"))
    return fig


def rand_with_min_max(size: List, min_: float, max_: float) -> Tensor:
    random_tensor = torch.rand(size)
    random_tensor = (max_ - min_) * random_tensor + min_
    return random_tensor


def rand_with_mins_maxes(dim_input: int, mins: Tensor,  maxes: Tensor) -> Tensor:
    """
    # Args
    - dim_input: An int. Input dimension.
    - mins: Shape = [dim_input,].
    - maxes: Shape = [dim_input,].

    # Returns
    - random_tensor: Shape = [dim_input,].
    """
    random_tensor = torch.rand(dim_input)  # [dim_input,]
    random_tensor = (maxes - mins) * random_tensor + mins
    return random_tensor  # [dim_input,]


def rand_with_lengths_mins(dim_input: int, lengths: Tensor, mins: Tensor) -> Tensor:
    """
    # Args
    - dim_input: An int. Input dimension.
    - lengths: Shape = [dim_input,].
    - mins: Shape = [dim_input,].

    # Returns
    - random_tensor: Shape = [dim_input,].
    """
    random_tensor = torch.rand(dim_input)  # [dim_input,]
    random_tensor = lengths * random_tensor + mins
    return random_tensor  # [dim_input,]


def latin_hypercube_sampling(num_samples: int, dimension: int) -> Tensor:
    """
    Generates Latin Hypercube Sampling with the given number of samples and dimensions.
    TODO: Acceleration by parallel processing.

    # Args
    - num_samples: The number of samples.
    - dimension: The number of dimensions.

    # Returns
    - samples: The Latin Hypercube Sampling result as a tensor of shape [num_samples, dimension].

    # Dependencies
    latin_hypercube_sampling -> latin_hypercube_sampling_with_min_max -> latin_hypercube_sampling_cp -> latin_hypercube_sampling_cp_BBBatch
    """
    # Calculate the interval for each dimension
    interval = 1.0 / num_samples

    # Generate samples
    rand_eps = torch.rand(
        dimension * num_samples).reshape(dimension, num_samples)
    samples_ls: List[Tensor] = []
    samples_ls_append = samples_ls.append
    for i in range(dimension):  # TODO: how to parallel-process?
        # Generate samples for a dimension
        samples = (torch.arange(0, num_samples) + rand_eps[i]) * interval

        # Shuffle samples for a dimension
        permutation = torch.randperm(num_samples)
        samples = samples[permutation]
        samples_ls_append(samples)

    samples = torch.stack(samples_ls, dim=1)  # [num_samples, dimension]

    return samples


def latin_hypercube_sampling_with_min_max(
        num_samples: int, dimension: int, min_: float, max_: float) -> Tensor:
    """
    # Dependencies
    latin_hypercube_sampling -> latin_hypercube_sampling_with_min_max -> latin_hypercube_sampling_cp -> latin_hypercube_sampling_cp_BBBatch
    """
    samples = latin_hypercube_sampling(num_samples, dimension)
    samples = (max_ - min_) * samples + min_
    return samples


def get_study_name(name_subproject_stat: str) -> str:
    study_name = name_subproject_stat[:-4] + "tuning"
    return study_name


def load_config_stat(path_config: str, config: dict) -> dict:
    """
    Config items below will NOT be restored (will not be overwritten).

    # Args
    - path_config: A str that points to the tuning config file.
    - config: A dict. Current config dictionary.
    """
    # Get config from tuning log
    config_restore: dict = np.load(path_config, allow_pickle=True).item()

    # Error handling
    for k in config_restore.keys():
        assert k in config.keys(), k

    # Not all items are restored from the tuning config (config_restore)
    # Overwrite some restored values with the current config
    config_restore["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    config_restore["EXP_PHASE"] = config["EXP_PHASE"]
    config_restore["NUM_TRIALS"] = config["NUM_TRIALS"]
    config_restore["INDEX_TRIAL_STAT"] = config["INDEX_TRIAL_STAT"]
    config_restore["NUM_ITERATIONS"] = config["NUM_ITERATIONS"]
    config_restore["LOG_INTERVAL"] = config["LOG_INTERVAL"]
    config_restore["VALIDATION_INTERVAL"] = config["VALIDATION_INTERVAL"]
    config_restore["FLAG_FIX_SEED"] = config["FLAG_FIX_SEED"]
    config_restore["SEED"] = config["SEED"]
    config_restore["FLAG_SHUFFLE"] = config["FLAG_SHUFFLE"]
    config_restore["FLAG_PIN_MEMORY"] = config["FLAG_PIN_MEMORY"]
    config_restore["NUM_WORKERS"] = config["NUM_WORKERS"]
    config_restore["NUM_THREADS_PER_PROCESS"] = config["NUM_THREADS_PER_PROCESS"]
    config_restore["NAME_OPTUNA_PRUNER"] = config["NAME_OPTUNA_PRUNER"]
    config_restore["PATH_DBFILE"] = config["PATH_DBFILE"]
    config_restore["DEVICE"] = config["DEVICE"]
    config_restore["WORLD_SIZE"] = config["WORLD_SIZE"]
    config_restore["PORT"] = config["PORT"]
    config_restore["COMMENT"] = config["COMMENT"]
    config_restore["FLAG_MULTIGPU"] = config["FLAG_MULTIGPU"]
    config_restore["ROOTDIR_DBFILES"] = config["ROOTDIR_DBFILES"]
    config_restore["NAME_TRIAL"] = config["NAME_TRIAL"]
    config_restore["DIR_TBLOG"] = config["DIR_TBLOG"]
    config_restore["DIR_CKPT"] = config["DIR_CKPT"]
    config_restore["DIR_TXLOG"] = config["DIR_TXLOG"]
    config_restore["DIR_CONFIG"] = config["DIR_CONFIG"]
    config_restore["DIR_DBFILE"] = config["DIR_DBFILE"]
    config_restore["NOW"] = config["NOW"]
    config_restore["DEFAULT_DTYPE"] = config["DEFAULT_DTYPE"]
    config_restore["DEBUG_MODE"] = config["DEBUG_MODE"]

    return config_restore
