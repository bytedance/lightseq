import json
import logging
from argparse import Namespace

from fairseq import criterions
from fairseq.data import encoders
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

from examples.fairseq.ls_label_smoothed_cross_entropy import (
    LSLabelSmoothedCrossEntropyCriterion,
)
from examples.fairseq.ls_transformer import LSTransformerModel

logger = logging.getLogger(__name__)


@register_task("ls_translation")
class LSTranslationTask(TranslationTask):
    def build_model(self, args):
        # create custom transformer model
        model = LSTransformerModel.build_model(args, self)

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def build_criterion(self, args):
        if args.criterion and args.criterion == "label_smoothed_cross_entropy":
            criterion = LSLabelSmoothedCrossEntropyCriterion.build_criterion(args, self)
            logger.info("Enable lightseq cross entropy")
        else:
            criterion = criterions.build_criterion(args, self)
        return criterion
