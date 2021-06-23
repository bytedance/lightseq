import math
import logging

import torch.distributed as dist

from fairseq import utils

logger = logging.getLogger(__name__)


class BatchIterator(object):
    """A wrapper of fairseq data iterator, usage:
    batch_itr = BatchIterator(fairseq_args, fairseq_task)
    for epoch in batch_itr.train_epoch():
        for batch in batch_itr.train_batch():
            train(batch)
            if should_valid():
                for sub in batch_itr.valid_dataset():
                    for sample in batch_itr.valid_batch():
                        valid(sample)
                    pirnt(f"Finish valid {sub}")
        print(f"Finish train epoch {epoch}")
    """

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.dummy_batch = None

    def _train_epoch(self, epoch):
        args, task = self.args, self.task
        if epoch == 1 or task.has_sharded_data("train"):
            task.load_dataset(
                args.train_subset, epoch=epoch, combine=True, data_selector=None
            )

        self.itr = task.get_batch_iterator(
            dataset=task.dataset(args.train_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                args.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=dist.get_world_size(),
            shard_id=dist.get_rank(),
            num_workers=args.num_workers,
            epoch=epoch,
            data_buffer_size=args.data_buffer_size,
            disable_iterator_cache=task.has_sharded_data("train"),
        )

        self.itr = self.itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch > args.curriculum),
        )

    def _empty_batch(self, batch):
        # Fairseq dataloader may produce "DUMMY"
        if batch == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )
        return batch is None or len(batch) == 0

    def _post_process_batch(self, batch):
        if self._empty_batch(batch) and self._empty_batch(self.dummy_batch):
            raise Exception("First batch is empty!")
        if self._empty_batch(batch):
            batch = self.dummy_batch
            is_dummy_batch = True
        else:
            if self._empty_batch(self.dummy_batch):
                self.dummy_batch = batch
            is_dummy_batch = False
        batch = utils.move_to_cuda(batch)
        return batch, is_dummy_batch

    def train_epoch(self):
        args, task = self.args, self.task
        epoch = 1  # epoch of fairseq starts from 1
        max_epoch = args.max_epoch or math.inf
        while epoch < max_epoch:
            self._train_epoch(epoch)
            yield epoch
            epoch += 1

    def train_batch(self):
        self.dummy_batch = None
        for batch in self.itr:
            yield self._post_process_batch(batch)

    def valid_dataset(self):
        args, task = self.args, self.task
        for subset in args.valid_subset.split(","):
            task.load_dataset(subset, combine=False, epoch=1)
            self.valid_itr = task.get_batch_iterator(
                dataset=task.dataset(subset),
                max_tokens=args.max_tokens_valid,
                max_sentences=args.batch_size_valid,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_shards=dist.get_world_size(),
                shard_id=dist.get_rank(),
                num_workers=args.num_workers,
                data_buffer_size=args.data_buffer_size,
                disable_iterator_cache=False,
            ).next_epoch_itr(shuffle=False)
            yield subset

    def valid_batch(self):
        self.dummy_batch = None
        for batch in self.valid_itr:
            yield self._post_process_batch(batch)
