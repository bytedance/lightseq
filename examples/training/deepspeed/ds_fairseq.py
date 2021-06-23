import math

import torch
from torch import nn
import torch.distributed as dist
import deepspeed
from deepspeed.utils import log_dist
from fairseq import tasks, distributed_utils
from fairseq.logging import metrics

from examples.training.deepspeed.ds_fairseq_data import BatchIterator
from examples.training.deepspeed.ds_fairseq_argument import gen_ds_fairseq_arg


best_bleu = 0.0


def torch_reduce_sum(
    device,
    logging_outputs,
    *extra_stats_to_sum,
    ignore=False,
):
    """
    Sync logging outputs across workers. fast_stat_sync_sum is
    faster than all_gather_list_sync, but is only suitable when
    logging outputs are scalars and can be summed. Note that
    *logging_outputs* cannot contain any nested dicts/lists.
    """
    data = {}
    for i, stat in enumerate(extra_stats_to_sum):
        data["extra_stats_" + str(i)] = stat
    if len(logging_outputs) > 0:
        log_keys = list(logging_outputs[0].keys())
        for k in log_keys:
            if not ignore:
                v = sum(log[k] for log in logging_outputs if k in log)
            else:
                v = logging_outputs[0][k]
                v = torch.zeros_like(v) if torch.is_tensor(v) else 0
            data["logging_outputs_" + k] = v
    else:
        log_keys = None

    data = distributed_utils.all_reduce_dict(data, device=device, group=None)

    extra_stats_to_sum = [
        data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
    ]
    if log_keys is not None:
        logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
    else:
        logging_outputs = []
    return logging_outputs, extra_stats_to_sum


def view_log(log_dict):
    if "bleu" in log_dict:
        global best_bleu
        best_bleu = max(best_bleu, log_dict["bleu"])
        log_dict["best_bleu"] = best_bleu
    tmp = [f"({k}, {v})" for k, v in log_dict.items()]
    return " ".join(tmp)


class DsFairseqModel(nn.Module):
    def __init__(self, model, criterion):
        super(DsFairseqModel, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, sample):
        loss, sample_size, logging_output = self.criterion(self.model, sample)
        return loss, sample_size, logging_output


class DsFairseqTrainer(object):
    def __init__(self, fs_args, ds_config, task):
        self.fs_args = fs_args
        self.ds_config = ds_config
        self.task = task
        model = task.build_model(fs_args)
        self.criterion = task.build_criterion(fs_args)
        model = DsFairseqModel(model, self.criterion)
        self.prepare_model_optimizer(model)

    def prepare_model_optimizer(self, model):
        # Initialize torch distributed
        deepspeed.init_distributed(dist_backend="nccl")

        # FIXME
        from dataclasses import dataclass

        @dataclass
        class TmpClass:
            local_rank: int

        fake_arg = TmpClass(self.fs_args.device_id)
        # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            args=fake_arg,
            model=model,
            model_parameters=model.parameters(),
            config_params=self.ds_config,
        )

    def reduce_log(self, logging_outputs, sample_size):
        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.criterion)
                del logging_outputs
        logging_output = agg.get_smoothed_values()
        logging_output["sample_size"] = sample_size
        return logging_output

    @metrics.aggregate("train_inner")
    def train_step(self, sample, is_dummy_batch):
        self.model.train()
        self.model.zero_grad()

        loss, sample_size, logging_output = self.model(sample)

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0
            loss *= 0.0
        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        logging_outputs, (sample_size,) = torch_reduce_sum(
            self.model.device, [logging_output], sample_size, ignore=is_dummy_batch
        )

        final_loss = loss * (dist.get_world_size() / sample_size)
        self.model.backward(final_loss)
        self.model.step()

        logging_output = self.reduce_log(logging_outputs, sample_size)

        if self.model.global_steps % self.model.steps_per_print() != 0:
            return

        log_dist(
            f'Step: {self.model.global_steps}, \
            {view_log(metrics.get_smoothed_values("train_inner"))}',
            [0],
        )
        metrics.reset_meters("train_inner")

    def valid_step(self, batch_itr):
        if self.model.global_steps % self.fs_args.validate_interval_updates != 0:
            return
        with torch.no_grad():
            self.model.eval()
            for subset in batch_itr.valid_dataset():
                with metrics.aggregate(new_root=True) as agg:
                    for batch, is_dummy_batch in batch_itr.valid_batch():
                        _, sample_size, logging_output = self.task.valid_step(
                            batch, self.model.module.model, self.model.module.criterion
                        )
                        logging_outputs = [logging_output]
                        if is_dummy_batch:
                            if torch.is_tensor(sample_size):
                                sample_size.zero_()
                            else:
                                sample_size *= 0.0
                        logging_outputs, (sample_size,) = torch_reduce_sum(
                            self.model.device,
                            logging_outputs,
                            sample_size,
                            ignore=is_dummy_batch,
                        )
                        logging_output = self.reduce_log(logging_outputs, sample_size)
                log_dist(
                    "Valid on step: {}, dataset: {}. {}".format(
                        self.model.global_steps,
                        subset,
                        view_log(agg.get_smoothed_values()),
                    ),
                    ranks=[0],
                )


@metrics.aggregate("train")
def train(batch_itr, trainer):
    for batch, is_dummy_batch in batch_itr.train_batch():
        trainer.train_step(batch, is_dummy_batch)
        trainer.valid_step(batch_itr)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tmp():
    fs_args, ds_config = gen_ds_fairseq_arg()
    set_seed(fs_args.seed)
    task = tasks.setup_task(fs_args)
    trainer = DsFairseqTrainer(fs_args, ds_config, task)
    batch_itr = BatchIterator(fs_args, task)
    for epoch in batch_itr.train_epoch():
        train(batch_itr, trainer)
        log_dist(
            f'Finish epoch {epoch}, \
            {view_log(metrics.get_smoothed_values("train"))}',
            [0],
        )
        metrics.reset_meters("train")


if __name__ == "__main__":
    tmp()
