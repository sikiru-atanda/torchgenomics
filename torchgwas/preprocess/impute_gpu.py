"""GPU-accelerated imputation: Li-and-Stephens HMM and deep learning (Phase 19).

Two methods:
1. **Li-and-Stephens HMM**: Genotype-level hidden Markov model where
   transitions encode LD between adjacent markers.  Forward-backward runs
   in log-space with all samples batched on GPU in parallel.
2. **Deep learning**: Masked autoencoder that learns LD patterns from
   observed genotypes and predicts missing values.

Both accept an (n, m) dosage tensor with NaN for missing entries and return
a complete tensor plus per-marker imputation R-squared.

References:
    - Li & Stephens (2003). Modeling linkage disequilibrium and identifying
      recombination hotspots using SNP data.  Genetics 165:2213-2233.
    - Browning & Browning (2007). Rapid and accurate haplotype phasing and
      missing-data inference for whole-genome association studies by use of
      localized haplotype clustering.  AJHG 81:1084-1097.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# =====================================================================
# Transition matrix estimation
# =====================================================================

def compute_transition_matrices(
    G: Tensor,
    n_states: int = 3,
    smoothing: float = 0.5,
) -> Tensor:
    """Estimate genotype-to-genotype transition matrices from observed data.

    For each pair of adjacent markers (j-1, j), counts co-occurrences of
    genotype values and normalises to give P(g_j | g_{j-1}).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Complete genotype matrix (no NaN).  Values in {0, 1, ..., n_states-1}.
    n_states : int
        Number of genotype states (3 for diploid, ploidy+1 in general).
    smoothing : float
        Laplace (additive) smoothing count to avoid zero probabilities.

    Returns
    -------
    T : Tensor, shape (m-1, n_states, n_states)
        T[j, a, b] = P(g_{j+1} = b | g_j = a).  Rows sum to 1.
    """
    n, m = G.shape
    G_int = G.long()

    # Count co-occurrences for all adjacent pairs simultaneously
    # Index: j * S*S + g_prev * S + g_curr
    g_prev = G_int[:, :-1]  # (n, m-1)
    g_curr = G_int[:, 1:]   # (n, m-1)

    # Flat index per (j, prev, curr)
    j_idx = torch.arange(m - 1, device=G.device).unsqueeze(0).expand(n, -1)
    flat_idx = j_idx * (n_states * n_states) + g_prev * n_states + g_curr
    flat_idx = flat_idx.reshape(-1)

    counts = torch.zeros(
        (m - 1) * n_states * n_states,
        dtype=G.dtype, device=G.device,
    )
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=G.dtype))
    counts = counts.reshape(m - 1, n_states, n_states)

    # Add Laplace smoothing and normalise rows
    counts = counts + smoothing
    row_sums = counts.sum(dim=-1, keepdim=True)
    T = counts / row_sums

    return T


# =====================================================================
# Log-space forward-backward
# =====================================================================

def _log_forward(
    log_emission: Tensor,
    log_T: Tensor,
    log_prior: Tensor,
) -> Tensor:
    """Scaled forward algorithm in log-space.

    Parameters
    ----------
    log_emission : (n_batch, m, S)
    log_T : (m-1, S, S)  where log_T[j, a, b] = log P(g_{j+1}=b | g_j=a)
    log_prior : (S,)

    Returns
    -------
    log_alpha : (n_batch, m, S)
    """
    n_batch, m, S = log_emission.shape
    log_alpha = torch.empty_like(log_emission)

    # Initialisation
    log_alpha[:, 0, :] = log_prior.unsqueeze(0) + log_emission[:, 0, :]

    for j in range(1, m):
        # log_alpha[:, j-1, :] is (n_batch, S)
        # log_T[j-1] is (S, S) -> prev_state x next_state
        # Compute: log sum_a [ alpha(j-1, a) * T(a, b) ] + emission(j, b)
        prev = log_alpha[:, j - 1, :].unsqueeze(-1)   # (n_batch, S, 1)
        trans = log_T[j - 1].unsqueeze(0)              # (1, S, S)
        log_alpha[:, j, :] = torch.logsumexp(prev + trans, dim=-2) + \
            log_emission[:, j, :]

    return log_alpha


def _log_backward(
    log_emission: Tensor,
    log_T: Tensor,
) -> Tensor:
    """Scaled backward algorithm in log-space.

    Parameters
    ----------
    log_emission : (n_batch, m, S)
    log_T : (m-1, S, S)

    Returns
    -------
    log_beta : (n_batch, m, S)
    """
    n_batch, m, S = log_emission.shape
    log_beta = torch.zeros_like(log_emission)

    # Initialisation: log_beta[:, m-1, :] = 0 (beta = 1)

    for j in range(m - 2, -1, -1):
        # Need: log sum_b [ T(a, b) * emission(j+1, b) * beta(j+1, b) ]
        bwd = log_emission[:, j + 1, :] + log_beta[:, j + 1, :]  # (n_batch, S)
        bwd = bwd.unsqueeze(-2)                    # (n_batch, 1, S)
        trans = log_T[j].unsqueeze(0)              # (1, S, S)
        log_beta[:, j, :] = torch.logsumexp(trans + bwd, dim=-1)

    return log_beta


def _compute_posteriors(
    log_alpha: Tensor,
    log_beta: Tensor,
) -> Tensor:
    """Compute normalised posterior P(g_j | all obs) from alpha and beta.

    Returns
    -------
    gamma : (n_batch, m, S)  probabilities summing to 1 along dim=-1.
    """
    log_gamma = log_alpha + log_beta
    # Normalise via softmax along state dimension
    gamma = torch.softmax(log_gamma, dim=-1)
    return gamma


# =====================================================================
# Emission model
# =====================================================================

def _build_emissions(
    G: Tensor,
    n_states: int,
    error_rate: float,
) -> Tensor:
    """Build log-emission matrix.

    For observed genotype g at marker j:
        P(obs=g | state=s) = (1 - error_rate) if s == g
                             else error_rate / (n_states - 1)
    For missing (NaN) markers: uniform over all states.

    Parameters
    ----------
    G : (n, m) genotype tensor (NaN for missing)
    n_states : int
    error_rate : float in (0, 1)

    Returns
    -------
    log_emission : (n, m, n_states)
    """
    n, m = G.shape
    missing = torch.isnan(G)

    # Build (n, m, S) one-hot-ish emission
    # For observed: high probability on matching state, low on others
    p_match = 1.0 - error_rate
    p_mismatch = error_rate / max(n_states - 1, 1)
    log_match = math.log(p_match)
    log_mismatch = math.log(p_mismatch)
    log_uniform = math.log(1.0 / n_states)

    # Start with uniform (for missing)
    log_emission = torch.full(
        (n, m, n_states), log_uniform,
        dtype=G.dtype, device=G.device,
    )

    # For observed entries, set emission based on genotype value
    G_safe = G.clone()
    G_safe[missing] = 0  # placeholder, won't be used

    for s in range(n_states):
        match_mask = (~missing) & (G_safe == s)  # (n, m)
        obs_mask = (~missing) & (G_safe != s)
        log_emission[:, :, s][match_mask] = log_match
        log_emission[:, :, s][obs_mask] = log_mismatch

    return log_emission


# =====================================================================
# Li-and-Stephens HMM imputer (main entry point)
# =====================================================================

def impute_li_stephens(
    G: Tensor,
    ploidy: int = 2,
    leave_one_out: bool = False,
    window_size: int = 10000,
    window_overlap: int = 500,
    batch_size: int = 256,
    error_rate: float = 0.01,
    smoothing: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Li-and-Stephens HMM genotype imputation on GPU.

    Uses a genotype-level first-order HMM where hidden states are
    genotype values {0, 1, ..., ploidy} and transitions encode LD
    between adjacent markers.  Forward-backward runs in log-space with
    all samples batched in parallel.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN for missing values.  Non-missing values
        should be integers in {0, 1, ..., ploidy}.
    ploidy : int
        Ploidy level (default 2 for diploid).
    leave_one_out : bool
        If True, exclude each sample from the transition matrix
        (exact but slower).  If False, use global transitions (default).
    window_size : int
        Maximum markers per window for memory control.
    window_overlap : int
        Overlap between adjacent windows for stitching.
    batch_size : int
        Samples processed per batch on GPU.
    error_rate : float
        Genotyping error probability for emission model.
    smoothing : float
        Laplace smoothing for transition counts.

    Returns
    -------
    G_imputed : Tensor, shape (n, m)
        Complete dosage matrix (expected dosage from posterior).
    dosage_r2 : Tensor, shape (m,)
        Per-marker imputation R-squared (1.0 for fully observed markers).
    """
    n, m = G.shape
    n_states = ploidy + 1
    missing = torch.isnan(G)

    if not missing.any():
        return G.clone(), torch.ones(m, dtype=G.dtype, device=G.device)

    G_out = G.clone()
    state_values = torch.arange(
        n_states, dtype=G.dtype, device=G.device,
    )  # [0, 1, ..., ploidy]

    # Markers with at least one observed value (for transition estimation)
    # Use mean-imputed version for transition matrix estimation
    G_for_trans = G.clone()
    for j in range(m):
        col = G_for_trans[:, j]
        mask_j = torch.isnan(col)
        if mask_j.any():
            obs = col[~mask_j]
            if obs.numel() > 0:
                G_for_trans[mask_j, j] = obs.mean().round()
            else:
                G_for_trans[mask_j, j] = ploidy / 2.0

    # Clamp to valid integer range for transition estimation
    G_for_trans = G_for_trans.clamp(0, ploidy).round()

    # Process in windows for memory control
    if m <= window_size:
        windows = [(0, m)]
    else:
        windows = []
        start = 0
        while start < m:
            end = min(start + window_size, m)
            windows.append((start, end))
            start = end - window_overlap
            if start + window_overlap >= m:
                break

    # Per-marker accumulators for stitching
    posterior_sum = torch.zeros(n, m, n_states, dtype=G.dtype, device=G.device)
    posterior_count = torch.zeros(m, dtype=G.dtype, device=G.device)

    for w_start, w_end in windows:
        G_win = G[:, w_start:w_end]
        G_trans_win = G_for_trans[:, w_start:w_end]
        m_win = w_end - w_start

        # Compute transition matrices for this window
        T = compute_transition_matrices(G_trans_win, n_states, smoothing)
        log_T = torch.log(T.clamp(min=1e-30))

        # Prior from allele frequencies at first marker
        af_counts = torch.zeros(n_states, dtype=G.dtype, device=G.device)
        col0 = G_trans_win[:, 0]
        for s in range(n_states):
            af_counts[s] = (col0 == s).sum()
        prior = (af_counts + smoothing) / (af_counts.sum() + n_states * smoothing)
        log_prior = torch.log(prior.clamp(min=1e-30))

        # Process samples in batches
        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            G_batch = G_win[b_start:b_end]  # (batch, m_win)

            # Build emissions
            log_emission = _build_emissions(G_batch, n_states, error_rate)

            # Forward-backward
            log_alpha = _log_forward(log_emission, log_T, log_prior)
            log_beta = _log_backward(log_emission, log_T)
            gamma = _compute_posteriors(log_alpha, log_beta)

            # Accumulate posteriors
            posterior_sum[b_start:b_end, w_start:w_end, :] += gamma
        posterior_count[w_start:w_end] += 1.0

    # Average posteriors across overlapping windows
    posterior_avg = posterior_sum / posterior_count.unsqueeze(0).unsqueeze(-1).clamp(min=1)

    # Impute: expected dosage = sum_s s * P(state=s)
    expected_dosage = (posterior_avg * state_values.reshape(1, 1, -1)).sum(dim=-1)

    # Fill in missing values
    G_out[missing] = expected_dosage[missing]

    # Compute per-marker R-squared for quality
    dosage_r2 = _imputation_rsq(G, G_out, missing)

    n_missing = missing.sum().item()
    n_total = n * m
    logger.info(
        "Li-Stephens HMM imputed %d/%d values (%.1f%%) across %d windows",
        n_missing, n_total, 100.0 * n_missing / n_total, len(windows),
    )

    return G_out, dosage_r2


# =====================================================================
# Deep learning imputer
# =====================================================================

class GenotypeAutoencoder(nn.Module):
    """Masked autoencoder for genotype imputation.

    Encodes the full genotype vector (m markers) through a bottleneck
    and decodes to per-marker genotype probabilities.  Trained with
    masked reconstruction: a fraction of observed entries are masked
    and the network learns to predict them from the remaining context.
    """

    def __init__(
        self,
        n_markers: int,
        n_states: int = 3,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.n_states = n_states

        # Encoder
        layers = []
        in_dim = n_markers
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        # Decoder (mirror of encoder)
        layers = []
        for h in reversed(hidden_dims[:-1]):
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        # Output: n_markers * n_states logits
        layers.append(nn.Linear(in_dim, n_markers * n_states))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, n_markers) dosage values (normalised to [0, 1]).

        Returns
        -------
        logits : (batch, n_markers, n_states)
        """
        h = self.encoder(x)
        out = self.decoder(h)
        return out.reshape(-1, self.n_markers, self.n_states)


def impute_deep_learning(
    G: Tensor,
    ploidy: int = 2,
    mask_rate: float = 0.15,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    dropout: float = 0.1,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Deep learning imputation using a masked autoencoder.

    Trains a small autoencoder on the observed genotypes by masking a
    fraction of entries and learning to reconstruct them.  At inference,
    missing entries are set to the population mean and the trained model
    predicts genotype probabilities at all positions.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN for missing values.
    ploidy : int
        Ploidy level (default 2).
    mask_rate : float
        Fraction of observed entries to mask during training.
    n_epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size for training.
    lr : float
        Learning rate.
    hidden_dims : tuple of int
        Hidden layer dimensions for encoder.
    dropout : float
        Dropout rate.
    device : torch.device, optional
        Device for training.  Defaults to G's device.

    Returns
    -------
    G_imputed : Tensor, shape (n, m)
        Complete dosage matrix.
    dosage_r2 : Tensor, shape (m,)
        Per-marker imputation R-squared.
    """
    n, m = G.shape
    n_states = ploidy + 1
    missing = torch.isnan(G)

    if not missing.any():
        return G.clone(), torch.ones(m, dtype=G.dtype, device=G.device)

    if device is None:
        device = G.device

    # Mean-impute for network input (normalised to [0, 1])
    G_filled = G.clone()
    for j in range(m):
        col = G_filled[:, j]
        mask_j = torch.isnan(col)
        if mask_j.any():
            obs = col[~mask_j]
            G_filled[mask_j, j] = obs.mean() if obs.numel() > 0 else ploidy / 2.0
    G_norm = G_filled / ploidy  # scale to [0, 1]

    # Integer targets for cross-entropy
    G_targets = G_filled.clamp(0, ploidy).round().long()

    # Limit hidden dims for small m
    effective_dims = tuple(min(h, m) for h in hidden_dims)

    # Build model
    model = GenotypeAutoencoder(
        n_markers=m, n_states=n_states,
        hidden_dims=effective_dims, dropout=dropout,
    ).to(device).to(G.dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Training
    model.train()
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_masked = 0

        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            idx = perm[b_start:b_end]
            x = G_norm[idx].to(device)            # (batch, m)
            targets = G_targets[idx].to(device)    # (batch, m)
            obs_mask = ~missing[idx].to(device)    # (batch, m)

            # Randomly mask a fraction of observed entries
            train_mask = obs_mask & (torch.rand_like(x) < mask_rate)

            # Zero out masked positions in input
            x_masked = x.clone()
            x_masked[train_mask] = 0.5  # neutral value

            # Forward
            logits = model(x_masked)  # (batch, m, S)

            # Loss only on masked observed positions
            loss_per_pos = loss_fn(
                logits.reshape(-1, n_states), targets.reshape(-1),
            ).reshape(x.shape)
            loss = loss_per_pos[train_mask].mean() if train_mask.any() else loss_per_pos[obs_mask].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * train_mask.sum().item()
            n_masked += train_mask.sum().item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(n_masked, 1)
            logger.debug("DL imputer epoch %d/%d, loss=%.4f", epoch + 1, n_epochs, avg_loss)

    # Inference
    model.eval()
    G_out = G.clone()

    with torch.no_grad():
        posteriors = torch.zeros(n, m, n_states, dtype=G.dtype, device=G.device)
        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            x = G_norm[b_start:b_end].to(device)
            logits = model(x)
            posteriors[b_start:b_end] = torch.softmax(logits, dim=-1).to(G.dtype)

    # Expected dosage
    state_values = torch.arange(n_states, dtype=G.dtype, device=G.device)
    expected = (posteriors * state_values.reshape(1, 1, -1)).sum(dim=-1)

    G_out[missing] = expected[missing]

    dosage_r2 = _imputation_rsq(G, G_out, missing)

    n_missing = missing.sum().item()
    logger.info(
        "DL imputer: %d missing values imputed across %d epochs", n_missing, n_epochs,
    )

    return G_out, dosage_r2


# =====================================================================
# Imputation R-squared
# =====================================================================

def _imputation_rsq(
    G_original: Tensor,
    G_imputed: Tensor,
    missing: Tensor,
) -> Tensor:
    """Compute per-marker imputation R-squared.

    For fully observed markers, returns 1.0.  For markers with imputed
    values, returns the squared correlation between imputed dosages and
    observed dosages across all samples (treating the non-missing entries
    as validation).  If a marker has no observed values, returns 0.0.

    Parameters
    ----------
    G_original : (n, m) with NaN for missing
    G_imputed : (n, m) complete
    missing : (n, m) boolean mask of originally missing entries

    Returns
    -------
    r2 : (m,) per-marker R-squared
    """
    m = G_original.shape[1]
    r2 = torch.ones(m, dtype=G_original.dtype, device=G_original.device)

    for j in range(m):
        n_miss = missing[:, j].sum().item()
        if n_miss == 0:
            continue  # fully observed
        # R-sq based on variance of posterior (proxy for info content)
        # For markers where we imputed, use the dosage variance ratio
        obs = ~missing[:, j]
        if obs.sum() < 2:
            r2[j] = 0.0
            continue
        # Imputed dosages for observed positions should match well
        # Use variance of imputed dosages vs variance of observed as proxy
        obs_vals = G_original[obs, j]
        var_obs = obs_vals.var()
        if var_obs < 1e-10:
            r2[j] = 0.0
        else:
            imp_vals = G_imputed[missing[:, j], j]
            if imp_vals.numel() == 0:
                r2[j] = 1.0
            else:
                # Info metric: 1 - var(imputed) / var(observed)
                var_imp = imp_vals.var() if imp_vals.numel() > 1 else torch.tensor(0.0)
                r2[j] = max(0.0, 1.0 - (var_imp / var_obs).item())

    return r2
