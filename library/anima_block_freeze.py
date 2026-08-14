from __future__ import annotations

from typing import Dict, List, Sequence

import torch.nn as nn

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


INSERTED_40_BLOCK_INDICES: Sequence[int] = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)
SUPPORTED_BLOCK_COUNTS = (28, 40)


def _count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def apply_inserted_only_training_freeze(dit: nn.Module) -> Dict[str, object]:
    if not hasattr(dit, "blocks"):
        raise ValueError("freeze_inserted_only_training requires an Anima DiT model with a blocks attribute.")

    block_count = len(dit.blocks)
    if block_count == 28:
        raise ValueError(
            "freeze_inserted_only_training requires the 40-layer expanded checkpoint. "
            "The loaded checkpoint only has 28 blocks."
        )
    if block_count != 40:
        raise ValueError(
            f"freeze_inserted_only_training only supports 28- or 40-block Anima checkpoints today. "
            f"Detected {block_count} blocks."
        )

    inserted_block_indices = list(INSERTED_40_BLOCK_INDICES)
    inserted_block_index_set = set(inserted_block_indices)
    inherited_block_indices = [index for index in range(block_count) if index not in inserted_block_index_set]

    # Freeze the whole DiT first, then selectively unfreeze only the inserted blocks.
    dit.requires_grad_(False)
    for block_index in inserted_block_indices:
        dit.blocks[block_index].requires_grad_(True)

    non_block_trainable_names = [
        name for name, parameter in dit.named_parameters() if parameter.requires_grad and not name.startswith("blocks.")
    ]

    summary = {
        "block_count": block_count,
        "inserted_block_indices": inserted_block_indices,
        "inherited_block_indices": inherited_block_indices,
        "trainable_parameter_count": _count_trainable_parameters(dit),
        "non_block_trainable_names": non_block_trainable_names,
    }

    logger.info("Applied inserted-only training freeze for 40-layer Anima DiT.")
    logger.info(f"  Inserted trainable blocks: {inserted_block_indices}")
    logger.info(f"  Frozen inherited blocks: {inherited_block_indices}")
    logger.info(f"  Trainable parameters after freeze: {summary['trainable_parameter_count']:,}")
    if non_block_trainable_names:
        logger.warning(
            "  Non-block parameters remained trainable after inserted-only freeze: "
            f"{non_block_trainable_names[:10]}{'...' if len(non_block_trainable_names) > 10 else ''}"
        )

    return summary
