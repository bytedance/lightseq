# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from argparse import Namespace

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
import torch.distributed as dist
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

logger = logging.getLogger(__name__)


def set_dir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


def split_exists(split, src, tgt, lang, data_path, dataset_impl):
    filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
    return indexed_dataset.dataset_exists(filename, impl=dataset_impl)


def is_main_worker():
    if dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def sync_all_workers():
    if dist.is_initialized():
        dist.barrier()


def checkout_subprocess(proc, only_main=True):
    if only_main and not is_main_worker():
        return

    # if list of subprocess
    if isinstance(proc, list):
        for p in proc:
            checkout_subprocess(p)
        return

    # if subprocess
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    proc.kill()
    if s_return == 1:
        raise FileNotFoundError(s_err)
    return s_output


def set_subprocess(s):
    return subprocess.Popen(
        s.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    shard_iterator=None,
    is_train=False,
):
    src_datasets = []
    tgt_datasets = []
    datasets_size = 0
    cur_pace, total_pace, itr = 1, 1, (0,)
    if is_train:
        cur_pace, total_pace = shard_iterator.pace()
        itr = shard_iterator.next()
    for k in itr:
        split_k = split + (str(k) if k > 0 else "")
        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path, dataset_impl):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path, dataset_impl):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split_k, data_path)
            )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)
        datasets_size += len(src_datasets[-1])
        if not combine:
            break

    logger.info(
        "{} {}/{} {}-{} {} examples".format(
            data_path, cur_pace, total_pace, src, tgt, datasets_size
        )
    )
    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("ls_translation")
class LSTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--npick', default=1, type=int,
                            help='pick n files as a shard')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, shard_itr):
        super().__init__(
            args,
            src_dict,
            tgt_dict,
        )
        self.shard_iterator = shard_itr
        self.local_hdfs_path = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries and ShardIterator).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)
        shard_itr = ShardIterator.build(args)
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict, shard_itr)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        is_train = split == getattr(self.args, "train_subset", None)
        assert len(paths) > 0
        if not is_train:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[0]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
            shard_iterator=self.shard_iterator,
            is_train=is_train,
        )


class ShardIterator:
    """
    Responsible for providing the indices of all files for current shard.
    If it is the path of hdfs, there is also responsible for
    downloading files and organizing resources.
    """

    def __init__(self, args, total, npick, ftype, data_para):
        self.args = args
        self.total = total
        self.npick = npick
        # hdfs
        self.ftype = ftype
        self.data_para = data_para
        self.pre_procs = None
        self.cur_shard = None
        self.init_iterator()

    @staticmethod
    def _prepare_databin_from_hdfs(hdfs_path, local_path):
        """Download all databin except the training set"""
        if is_main_worker():
            set_dir(local_path)
            proc = set_subprocess(f"hadoop fs -get {hdfs_path}/valid* {local_path}")
            checkout_subprocess(proc)
            proc = set_subprocess(f"hadoop fs -get {hdfs_path}/test* {local_path}")
            checkout_subprocess(proc)
            proc = set_subprocess(f"hadoop fs -get {hdfs_path}/dict* {local_path}")
            checkout_subprocess(proc)
        sync_all_workers()

    @classmethod
    def build(cls, args):
        # get data path
        if not args.data.endswith(":"):
            raise ValueError("Paths must end with a colon (:)")
        data_path = args.data[:-1]

        k = 0
        data_para = None
        ftype = "hdfs" if data_path.startswith("hdfs") else "local"
        if ftype == "hdfs":
            local_path = "/tmp/streaming_load_databin/"
            cls._prepare_databin_from_hdfs(data_path, local_path)
            proc = set_subprocess(f"hadoop fs -ls {data_path}/train*")
            out = str(checkout_subprocess(proc, only_main=False))
            for k in itertools.count():
                split_k = "train" + (str(k) if k > 0 else "")
                if not split_k in out:
                    break
            # change path
            args.data = local_path + ":"
            data_para = {
                "real_path": data_path,
                "local_path": local_path,
            }
        else:
            src, tgt = args.source_lang, args.target_lang
            dataset_impl = args.dataset_impl
            for k in itertools.count():
                split_k = "train" + (str(k) if k > 0 else "")
                exist1 = split_exists(split_k, src, tgt, src, data_path, dataset_impl)
                exist2 = split_exists(split_k, tgt, src, src, data_path, dataset_impl)
                if not exist1 and not exist2:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError(
                            "Dataset not found: train ({})".format(data_path)
                        )

        shard_itr = cls(args, k, args.npick, ftype, data_para)
        if ftype == "hdfs":
            shard_itr._prepare_sharded_trainset_from_hdfs(0)
        return shard_itr

    def _prepare_sharded_trainset_from_hdfs(self, i):
        """Download all training set for i-th shard from hdfs"""
        if is_main_worker():
            itr = self.shard_ids[i]
            # Not downloading already downloaded files
            if self.cur_shard is not None and i == 0:
                itr = set(itr) - set(self.cur_shard)
            lpath, rpath = self.data_para["local_path"], self.data_para["real_path"]
            procs = []
            for k in itr:
                split_k = "train" + (str(k) if k > 0 else "")
                procs.append(
                    set_subprocess(f"hadoop fs -get {rpath}/{split_k}.* {lpath}")
                )
            self.pre_procs = procs
        sync_all_workers()

    def next(self):
        """Returns the indices of all files for current shard"""
        if self.ftype == "hdfs":
            checkout_subprocess(self.pre_procs)

        self.handle_last()
        self.cur_shard = self.shard_ids[self.cur_id]
        self.prepare_next()
        return self.cur_shard

    def pace(self):
        return self.cur_id + 1, len(self.shard_ids)

    def end_of_shards(self):
        return self.cur_id >= len(self.shard_ids)

    def prepare_next(self):
        self.cur_id += 1
        if self.end_of_shards():
            self.init_iterator()

        if self.ftype == "hdfs":
            self._prepare_sharded_trainset_from_hdfs(self.cur_id)

    def handle_last(self):
        if self.ftype == "hdfs":
            last_shard = self.cur_shard
            if last_shard is not None and is_main_worker():
                if self.cur_id == 0:
                    last_shard = set(last_shard) - set(self.shard_ids[self.cur_id])
                for k in last_shard:
                    split_k = "train" + (str(k) if k > 0 else "")
                    rm = f'rm -f {self.data_para["local_path"]}/{split_k}.*'
                    os.system(rm)

    def init_iterator(self):
        self.shard_ids = self.shuffle_shards()
        self.cur_id = 0

    def shuffle_shards(self):
        indices = np.arange(self.total)
        np.random.shuffle(indices)
        shuffled_indices = list(indices)
        iner = list(range(0, self.total, self.npick))
        if iner[-1] < self.total:
            iner.append(self.total)
        res = [
            tuple(shuffled_indices[iner[i] : iner[i + 1]])
            for i in range(0, len(iner) - 1)
        ]
        return res
