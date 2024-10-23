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

import datetime
import os

import optuna
import torch
from dataprocesses.data_controller import DataController
from losses.loss_controller import LossController
from models.model_controller import ModelController
from optimizers.optimizer_controller import (OptimizerController,
                                             SchedulerController)
from training_controller import TrainingController
from utils.log_config import get_my_logger
from utils.metrics import RelativeErrorCalculator
from utils.misc import (count_parameters, ddp_setup, final_processes1,
                        final_processes2, get_study_name, initial_processes1,
                        initial_processes2)
from utils.optuna_controller import OptunaController

logger = get_my_logger(__name__)

os.environ["OMP_NUM_THREADS"] = "1"


def objective(single_trial, config: dict, rank: int, optuna_controller: OptunaController):
    # Initial processes for each trial
    # Config file is copied and saved here.
    # Config dict is overwritten when EXP_PHASE=stat.
    trial, config = initial_processes2(config, rank, single_trial)
    flag_debug = config["DEBUG_MODE"]

    # Optuna suggest
    # =========================================================
    if config["EXP_PHASE"] in ["tuning", "stat"]:
        logger.info(
            f"EXP_PHASE={config['EXP_PHASE']}.\nStart hparam suggestions or loading...")
        config = optuna_controller.suggest_params(config, trial)

        if rank == 0:
            logger.info(f"Printing SUGGESTED config...\n======================= Print SUGGESTED config start =======================\n" +
                        f"{config}\n======================= Print SUGGESTED config end =======================\nPrinting SUGGESTED config done.")

    # Model
    # =========================================================
    model_controller = ModelController(
        name_model=config["NAME_MODEL"],
        num_classes=config["KWARGS_NAME_DATASETS"][config["NAME_DATASET"]
                                                   ]["num_classes"],
        device=config["DEVICE"],
        flag_pretrained=config["FLAG_PRETRAINED"],
        flag_init_last_linear=config["FLAG_INIT_LAST_LINEAR"],
        kwargs_dataset=config["KWARGS_NAME_DATASETS"][config["NAME_DATASET"]],
        **config["KWARGS_NAME_MODEL"][config["NAME_MODEL"]],)
    model = model_controller.get_model()
    preprocess = model_controller.get_preprocess()
    if rank == 0:
        logger.info(f"Num of trainable params: {count_parameters(model)}")

    # Dataloader
    # =========================================================
    kwargs_dataset = config["KWARGS_NAME_DATASETS"][config["NAME_DATASET"]]
    data_controller = DataController(
        name_dataset=config["NAME_DATASET"],
        name_sampler=config["NAME_SAMPLER"],
        batch_size=config["BATCH_SIZE"],
        flag_shuffle=config["FLAG_SHUFFLE"],
        pin_memory=config["FLAG_PIN_MEMORY"],
        num_workers=config["NUM_WORKERS"],
        flag_multigpu=config["FLAG_MULTIGPU"],
        world_size=config["WORLD_SIZE"],
        rank=rank,
        rootdir_ptdatasets=config["ROOTDIR_PTDATASETS"],
        transform=preprocess,
        target_transform=None,
        kwargs_dataset=kwargs_dataset,
        task=kwargs_dataset["task"],
    )
    dataloader_tr, dataloader_va, dataloader_te = data_controller.get_dataloaders()

    # torch.distributed: synchronize
    if config["FLAG_MULTIGPU"]:
        torch.distributed.barrier()

    # Loss
    # =========================================================
    # Get loss controller
    loss_controller = LossController(
        list_name_losses=config["LIST_LOSSES"],
        model=model,
        list_kwargs=[config["KWARGS_LOSSES"][key]
                     for key in config["LIST_LOSSES"]],
        kwargs_dataset=kwargs_dataset,
        task=kwargs_dataset["task"],
        name_loss_reweight=config["NAME_LOSS_REWEIGHT"],
        device=rank,
    )

    # For PINN trainings
    if kwargs_dataset["task"] == "pinn":
        loss_controller.wrap4pinns()
        degree = config[
            "KWARGS_NAME_DATASETS"][config["NAME_DATASET"]]["degree"]
        pde = loss_controller.pde
        relerr_calculator = RelativeErrorCalculator(
            model=model, dl_test=dataloader_va, pde=pde, degree=degree,
            num_data=len(dataloader_va.dataset), device=rank)
    else:
        pde = None
        degree = None
        relerr_calculator = None

    list_losses = loss_controller.get_list_losses()

    # Optimizer & Scheduler
    # =========================================================
    optimizer_controller = OptimizerController(
        name_optimizer=config["NAME_OPTIMIZER"],
        params=model.parameters(),
        **config["KWARGS_NAME_OPTIMIZER"][config["NAME_OPTIMIZER"]])
    optimizer = optimizer_controller.get_optimizer()
    scheduler_controller = SchedulerController(
        name_scheduler=config["NAME_SCHEDULER"],
        optimizer=optimizer,
        **config["KWARGS_NAME_SCHEDULER"][config["NAME_SCHEDULER"]])
    scheduler = scheduler_controller.get_scheduler()

    # Start training
    # =========================================================
    trainer = TrainingController(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=config["BATCH_SIZE"],
        num_classes=config[
            "KWARGS_NAME_DATASETS"][config["NAME_DATASET"]]["num_classes"],
        top_k=config["TOP_K"],
        list_losses=list_losses,
        list_name_losses=config["LIST_LOSSES"],
        sampler_tr=dataloader_tr.sampler,
        task=config["KWARGS_NAME_DATASETS"][config["NAME_DATASET"]]["task"],
        num_iterations=config["NUM_ITERATIONS"],
        log_interval=config["LOG_INTERVAL"],
        validation_interval=config["VALIDATION_INTERVAL"],
        start_pruning=config["START_PRUNING"],
        tblog_interval=config["TBLOG_INTERVAL"],
        dir_tblog=config["DIR_TBLOG"],
        dir_ckpt=config["DIR_CKPT"],
        flag_save_ckpt_trytuning=config["FLAG_SAVE_CKPT_TRYTUNING"],
        exp_phase=config["EXP_PHASE"],
        rank=rank,
        world_size=config["WORLD_SIZE"],
        flag_multigpu=config["FLAG_MULTIGPU"],
        now=config["NOW"],
        device=config["DEVICE"],
        dataloader_tr=dataloader_tr,
        dataloader_va=dataloader_va,
        dataloader_te=dataloader_te,
        trial=trial,
        pde=pde,
        degree=degree,
        relerr_calculator=relerr_calculator,)

    tic = datetime.datetime.now()
    if flag_debug:
        best_value = trainer.training()
    else:
        try:
            best_value = trainer.training()
        except RuntimeError as e:
            raise optuna.exceptions.TrialPruned(
                f"Trial pruned due to RuntimeError:\n{e}")
    tac = datetime.datetime.now()

    # torch.distributed: synchronize
    if config["FLAG_MULTIGPU"]:
        torch.distributed.barrier()

    if rank == 0:
        logger.info(f"Training runtime: {tac-tic}")

    return best_value


def main(rank: int, config: dict):
    """
    Main function.
    # Args
    - rank: Rank. If FLAG_MULTIGPU=False, set rank=0.
    """
    # torch.distributed: initialize process group
    if config["FLAG_MULTIGPU"]:
        ddp_setup(rank, config["WORLD_SIZE"], config["PORT"])

    # Get sampler and pruner
    optuna_controller = OptunaController(
        name_sampler=config["NAME_OPTUNA_SAMPLER"],
        name_pruner=config["NAME_OPTUNA_PRUNER"],
        kwargs_sampler=config["KWARGS_OPTUNA_SAMPLER"],
        kwargs_pruner=config["KWARGS_OPTUNA_PRUNER"],
        path_db=config["PATH_DBFILE"],
        study_name=config["NAME_SUBPROJECT"],)

    # Load or create study, and start optimization or stat training
    if rank == 0:
        if config["EXP_PHASE"] != "stat":
            study = optuna.create_study(
                storage=optuna_controller.get_storage_name(),
                sampler=optuna_controller.get_sampler(),
                pruner=optuna_controller.get_pruner(),
                study_name=optuna_controller.get_study_name(),
                load_if_exists=True,
                direction="maximize",)

            study.optimize(
                lambda trial: objective(
                    trial, config, rank, optuna_controller),
                n_trials=config["NUM_TRIALS"],
                timeout=None, n_jobs=1,
                gc_after_trial=True,  # Set True if you observe memory leak over trials
                show_progress_bar=True,)

        elif config["EXP_PHASE"] == "stat":
            # Get study
            study_name = get_study_name(
                name_subproject_stat=config["NAME_SUBPROJECT"])
            study = optuna.load_study(
                study_name=study_name,
                storage="sqlite:///"+config["PATH_DBFILE"])

            # Start training
            if config["INDEX_TRIAL_STAT"] is None:
                for _ in range(config["NUM_TRIALS"]):
                    objective(
                        study.best_trial, config,
                        rank, optuna_controller)
            else:
                for _ in range(config["NUM_TRIALS"]):
                    objective(
                        study.trials[config["INDEX_TRIAL_STAT"]], config,
                        rank, optuna_controller)

        else:
            raise ValueError

    else:  # rank != 0
        for _ in range(config["NUM_TRIALS"]):
            try:
                objective(None, config, rank, optuna_controller)
            except optuna.TrialPruned:
                pass

    # Final processes
    # =========================================================
    if rank == 0:
        final_processes1(rank, study)
    final_processes2(rank, config)


if __name__ == "__main__":
    from configs.config_train import config
    config = initial_processes1(config)

    if config["FLAG_MULTIGPU"]:
        torch.multiprocessing.spawn(
            main, args=[config,], nprocs=torch.cuda.device_count())
    else:
        main(rank=0, config=config)
