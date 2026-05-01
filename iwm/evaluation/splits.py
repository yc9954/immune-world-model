"""Documentation-style helpers; the real split logic lives in the
DataModule and the synthetic generator. Keeping this file as a single
source of truth for WHAT each split means, so evaluation reports stay
honest.
"""

from __future__ import annotations

from enum import Enum


class Split(str, Enum):
    IID = "iid"
    OOD_GENE = "ood_gene"
    OOD_TEMPORAL = "ood_temporal"
    OOD_MULTICELL = "ood_multicell"


SPLIT_DESCRIPTIONS = {
    Split.IID: "Random 80/20 holdout over (cell, step).",
    Split.OOD_GENE: "Held-out perturbation genes; tests compositional action generalisation.",
    Split.OOD_TEMPORAL: "Train t<split_at; eval t>=split_at given z0; pillar-1 test.",
    Split.OOD_MULTICELL: "Novel neighbourhood compositions; pillar-2 test.",
}
