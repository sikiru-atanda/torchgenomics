"""UnifiedScanner: reader -> model adapter -> stats -> output."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from ..config import TorchGWASConfig
from ..io.base import GenotypeReader
from ..models.base import BaseModel, NullFit, ScanResult, VariantMeta
from ..preprocess.qc import QCFilterConfig, apply_qc_filters, compute_variant_qc
from .prefetch import PrefetchIterator, move_nullfit_to_device

logger = logging.getLogger(__name__)


def merge_scan_results(results: list[ScanResult]) -> ScanResult:
    """Concatenate a list of per-chunk ScanResults into a single result."""
    if not results:
        raise ValueError("No scan results to merge.")
    if len(results) == 1:
        return results[0]

    # Merge list fields
    chr_all: list[str] = []
    pos_all: list[int] = []
    snp_all: list[str] = []
    a1_all: list[str] = []
    a2_all: list[str] = []
    for r in results:
        chr_all.extend(r.chr)
        pos_all.extend(r.pos)
        snp_all.extend(r.snp)
        a1_all.extend(r.a1)
        a2_all.extend(r.a2)

    # Merge tensor fields
    af = torch.cat([r.af for r in results])
    beta = torch.cat([r.beta for r in results])
    se = torch.cat([r.se for r in results])
    stat = torch.cat([r.stat for r in results])
    p = torch.cat([r.p for r in results])

    n_obs = None
    if all(r.n_obs is not None for r in results):
        n_obs = torch.cat([r.n_obs for r in results])  # type: ignore[arg-type]

    merged = ScanResult(
        chr=chr_all, pos=pos_all, snp=snp_all, a1=a1_all, a2=a2_all,
        af=af, beta=beta, se=se, stat=stat, p=p,
        test=results[0].test,
        n_obs=n_obs,
        inference_type=results[0].inference_type,
    )

    # Preserve the dynamic `_conditional` attribute that ConditionalLMM
    # attaches to each per-chunk ScanResult. Without this the R wrapper's
    # gwas_conditional() silently loses block metadata for scans that
    # straddle a chunk boundary (> chunk_size SNPs).
    if all(hasattr(r, "_conditional") for r in results):
        merged._conditional = _merge_conditional_payloads(
            [r._conditional for r in results], merged,
        )
    return merged


def _merge_conditional_payloads(payloads, merged_marginal):
    """Concatenate a list of ConditionalScanResult payloads in-order."""
    first = payloads[0]
    cls = type(first)

    cond_beta = torch.cat([p.conditional_beta for p in payloads])
    cond_se = torch.cat([p.conditional_se for p in payloads])
    cond_stat = torch.cat([p.conditional_stat for p in payloads])
    cond_p = torch.cat([p.conditional_p for p in payloads])
    persistence = torch.cat([p.persistence for p in payloads])
    r2_to_lead = torch.cat([p.r2_to_lead for p in payloads])

    block_ids: list[str] = []
    lead_snps: list[str] = []
    for p in payloads:
        block_ids.extend(p.ld_block_id)
        lead_snps.extend(p.lead_snp)

    return cls(
        marginal=merged_marginal,
        conditional_beta=cond_beta,
        conditional_se=cond_se,
        conditional_stat=cond_stat,
        conditional_p=cond_p,
        persistence=persistence,
        ld_block_id=block_ids,
        r2_to_lead=r2_to_lead,
        lead_snp=lead_snps,
    )


class UnifiedScanner:
    """Streams genotype chunks through any model via adapters.

    Users switch methods by changing one argument, not learning a new tool.
    The scanner handles chunking, GPU transfer, QC filtering, and result assembly.

    GPU acceleration (Phase 10):
    - Null-fit tensors are moved to the target device before scanning
    - Genotype chunks are prefetched with pinned memory for async GPU transfer
    - AMP is NOT applied to the scan loop (statistical inference must stay FP64)
    - AMP is only appropriate for GRM construction (see kinship.py)
    """

    def __init__(
        self,
        reader: GenotypeReader,
        model: BaseModel,
        config: Optional[TorchGWASConfig] = None,
    ) -> None:
        self.reader = reader
        self.model = model
        self.config = config or TorchGWASConfig()

    def scan(
        self,
        null_fit: NullFit,
        test: str = "wald",
        qc_config: Optional[QCFilterConfig] = None,
    ) -> ScanResult:
        """Run the full genome scan: iterate chunks, score each, collect results.

        Parameters
        ----------
        null_fit : NullFit
            Cached null-model quantities from model.fit_null().
        test : str
            Test type: "wald", "score", or "lrt".
        qc_config : QCFilterConfig, optional
            If provided, variants failing QC are excluded from results.

        Returns
        -------
        ScanResult
            Merged per-variant association statistics for all chunks.
        """
        chunk_size = self.config.chunk_size
        device = self.config.device

        # Move null-fit tensors to compute device (GPU if available)
        null_fit = move_nullfit_to_device(null_fit, device)

        chunk_results: list[ScanResult] = []
        total_variants = 0
        total_tested = 0

        # Wrap iterator with async prefetch for GPU
        raw_iter = self.reader.iter_chunks(chunk_size=chunk_size)
        chunk_iter = PrefetchIterator(raw_iter, device)

        for G_chunk, vmeta in chunk_iter:
            m_chunk = G_chunk.shape[1]
            total_variants += m_chunk

            # Ensure on target device (PrefetchIterator handles GPU case,
            # but for CPU or non-prefetched paths, do it explicitly)
            if G_chunk.device != device:
                G_chunk = G_chunk.to(device)

            # Optional per-chunk QC filtering
            if qc_config is not None:
                G_chunk, vmeta = self._apply_chunk_qc(
                    G_chunk, vmeta, qc_config,
                )
                if G_chunk.shape[1] == 0:
                    continue

            # Score this chunk (always FP64 — no AMP here)
            result = self.model.score_chunk(G_chunk, null_fit, vmeta, test=test)
            chunk_results.append(result)
            total_tested += len(result)

        if not chunk_results:
            raise RuntimeError(
                f"No variants passed QC filters out of {total_variants} total."
            )

        merged = merge_scan_results(chunk_results)
        logger.info(
            "Scan complete: %d/%d variants tested (%d chunks, test=%s, device=%s)",
            total_tested, total_variants, len(chunk_results), test, device,
        )
        return merged

    def _apply_chunk_qc(
        self,
        G_chunk: Tensor,
        vmeta: VariantMeta,
        qc_config: QCFilterConfig,
    ) -> tuple[Tensor, VariantMeta]:
        """Apply QC filters to a chunk and return filtered (G, vmeta)."""
        ploidy = self.config.ploidy
        stats = compute_variant_qc(G_chunk, vmeta, ploidy=ploidy)
        passes = apply_qc_filters(stats, qc_config)

        if passes.all():
            return G_chunk, vmeta

        keep = passes.nonzero(as_tuple=True)[0].tolist()
        G_filt = G_chunk[:, keep]
        vmeta_filt = VariantMeta(
            snp=[vmeta.snp[i] for i in keep],
            chr=[vmeta.chr[i] for i in keep],
            pos=[vmeta.pos[i] for i in keep],
            a1=[vmeta.a1[i] for i in keep],
            a2=[vmeta.a2[i] for i in keep],
        )
        return G_filt, vmeta_filt
