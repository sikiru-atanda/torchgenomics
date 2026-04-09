"""Shared framework for multi-locus iterative GWAS loops (FarmCPU, BLINK).

Both FarmCPU and BLINK iterate between a scan step and a pseudo-QTN
selection step until the selected SNP set stabilizes.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class IterativeGWASLoop:
    """Base class for FEM/REM iterative multi-locus GWAS.

    Subclassed by FarmCPU and BLINK.  Manages pseudo-QTN selection,
    convergence, and iteration control.
    """

    def select_pseudo_qtns(
        self,
        p_values: Tensor,
        threshold: float = 0.01,
        max_qtns: int = 20,
    ) -> Tensor:
        """Select pseudo-QTNs from significant SNPs.

        Parameters
        ----------
        p_values : (m,) — p-values for all tested SNPs
        threshold : float — significance threshold
        max_qtns : int — maximum number of pseudo-QTNs to retain

        Returns
        -------
        Tensor of selected SNP indices (shape: [n_selected])
        """
        significant = torch.where(p_values < threshold)[0]
        if len(significant) == 0:
            return torch.tensor([], dtype=torch.long, device=p_values.device)

        # Sort by p-value and take top max_qtns
        sorted_pvals, sorted_idx = p_values[significant].sort()
        top_idx = significant[sorted_idx[:max_qtns]]
        return top_idx

    def check_convergence(self, prev_qtns: Tensor, curr_qtns: Tensor) -> bool:
        """Check if pseudo-QTN set has stabilized.

        Convergence = same set of indices (regardless of order).
        """
        if len(prev_qtns) != len(curr_qtns):
            return False
        if len(prev_qtns) == 0:
            return True
        prev_set = set(prev_qtns.tolist())
        curr_set = set(curr_qtns.tolist())
        return prev_set == curr_set
