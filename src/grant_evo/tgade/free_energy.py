"""Free-energy maintenance for T-GADE thermodynamical selection (T-GADE paper §3.2 / §3.5).

Paper definition (2026-09-10): F_T(P) = <E>_P - T · H(P) with the per-individual
diversity measure H(P) = (1/|P|) · Σ_{k=1..M} log det (L_k(P) + ε I).

This module maintains the total form |P| · F_T(P):

    Σ_{x ∈ P} E(x) - T · Σ_{k=1..M} log det (L_k(P) + ε I)

Every selection step compares candidate sets of equal size, so minimising the
total form gives exactly the argmin of F_T (greedy construction and removal).
Submodularity statements apply to the unnormalised log-det sum only.

For backwards compatibility this module also exposes the **intensive**
form ``F_int(P) = ⟨E⟩_P - T · Σ_k log det L_k`` which earlier T-GADE
revisions used. The extensive form is the engine default as of the
P0-1 migration; intensive is
retained as a legacy hook for ablation (paper §6 L13 ``free_energy_mode
∈ {extensive, intensive}``).

``L_k(P)`` is the |P|×|P| Gram matrix of the section-k embeddings across
the population P. The entropy proxy is the sum of log-determinants across
sections — independent diversity per section, additively combined.

Step 8 builds P_{t+1} (size N) from P' (size 2N+1) by sequential greedy
addition of the candidate y* that minimises ΔF(y). In the extensive form

    ΔF(y) = E(y) - T · ΔH(y)

(the candidate energy enters as itself, not as the incremental mean
update — this is the key simplification of the extensive form). In the
intensive form, ΔE = (E(y) - ⟨E⟩_S) / (|S|+1) as before.

ΔH(y) decomposes per section as

    ΔH_k(y) = log( x_k(y)^T x_k(y) + ε - c_k(y)^T R_k c_k(y) )
            = log( d_k(y) )

Here c_k(y) = X_k x_k(y) (|S|-vector, population-space convention with
X_k ∈ R^{|S| × D}) is the inner product of the candidate's section-k
embedding with the |S| already-selected ones, and
R_k = (X_k X_k^T + ε I)^{-1} ∈ R^{|S| × |S|} (population-space Gram).

The Schur complement update of R_k after adding y is:

    R'_k = [[R_k + (R_k c_k c_k^T R_k)/d_k,  -R_k c_k / d_k],
            [   -(R_k c_k)^T / d_k,             1/d_k     ]]

This is O(|S|^2) per addition, giving O(N · (2N+1) · M · N^2) total per
generation — trivial for N=8, M=5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gram_matrix(X: np.ndarray, *, eps: float) -> np.ndarray:  # noqa: N803
    """Compute (X^T X + ε I), assuming X has rows as observations.

    For our convention ``X ∈ R^{N × D}`` (N individuals, D-dim embedding),
    the Gram matrix in **population space** is ``X X^T + ε I``. We use
    population-space rather than feature-space because we want the |P|×|P|
    similarity matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps!r}")
    if X.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    gram: np.ndarray = np.asarray(X @ X.T, dtype=np.float64)
    n = gram.shape[0]
    gram += eps * np.eye(n)
    return gram


def logdet(G: np.ndarray) -> float:  # noqa: N803
    """log det of a positive-(semi)definite matrix via slogdet."""
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be a square matrix, got shape {G.shape}")
    if not np.isfinite(G).all():
        raise ValueError("G must contain only finite values")
    if G.size == 0:
        return 0.0
    if not np.allclose(G, G.T, rtol=1e-10, atol=1e-12):
        raise ValueError("G must be symmetric")
    if float(np.linalg.eigvalsh(G).min()) < -1e-10:
        raise ValueError("G must be positive semidefinite")
    sign, val = np.linalg.slogdet(G)
    if sign <= 0:
        # Numerically singular. Fallback: add small jitter.
        n = G.shape[0]
        sign, val = np.linalg.slogdet(G + 1e-12 * np.eye(n))
        if sign <= 0:
            raise ValueError("G must be positive semidefinite")
    return float(val)


@dataclass
class _SectionState:
    """Mutable state for one section's incremental L_k^{-1} maintenance."""

    eps: float
    # Embedding rows of the currently-selected candidates (|S| × D).
    X: np.ndarray
    # Inverse of (X X^T + ε I), shape |S| × |S|.
    R: np.ndarray
    logdet_acc: float

    @classmethod
    def empty(cls, dim: int, eps: float) -> _SectionState:
        return cls(
            eps=eps,
            X=np.zeros((0, dim), dtype=np.float64),
            R=np.zeros((0, 0), dtype=np.float64),
            logdet_acc=0.0,
        )

    def delta_logdet(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Return (Δ log det, c, d) for adding row x to current state.

        Δ log det = log d, where
            c = X x         (shape (|S|,))
            d = x·x + ε - c^T R c
        """
        x64 = x.astype(np.float64)
        if self.X.shape[0] == 0:
            d = float(x64 @ x64) + self.eps
            return float(np.log(max(d, 1e-300))), np.zeros((0,)), d
        c = (self.X @ x64).astype(np.float64)
        d = float(x64 @ x64) + self.eps - float(c @ self.R @ c)
        d_safe = max(d, 1e-300)
        return float(np.log(d_safe)), c, d_safe

    def add(self, x: np.ndarray) -> None:
        """Incorporate x into state via Schur-complement R update."""
        delta, c, d = self.delta_logdet(x)
        x64 = x.astype(np.float64)
        s = self.X.shape[0]
        if s == 0:
            self.X = x64.reshape(1, -1)
            self.R = np.array([[1.0 / d]], dtype=np.float64)
        else:
            rc = self.R @ c                            # (s,)
            top_left = self.R + np.outer(rc, rc) / d   # (s, s)
            top_right = (-rc / d).reshape(s, 1)        # (s, 1)
            bottom_left = top_right.T                  # (1, s)
            bottom_right = np.array([[1.0 / d]])
            top = np.concatenate([top_left, top_right], axis=1)
            bot = np.concatenate([bottom_left, bottom_right], axis=1)
            self.R = np.concatenate([top, bot], axis=0)
            self.X = np.concatenate([self.X, x64.reshape(1, -1)], axis=0)
        self.logdet_acc += delta


class GramFreeEnergy:
    """Per-section Gram-matrix state with incremental Schur updates.

    Usage::

        gfe = GramFreeEnergy(num_sections=5, dim=1536, eps=1e-3, T=0.5)
        gfe.add(individual_id_a, embeddings_a)   # embeddings_a: (5, 1536)
        delta_h = gfe.delta_h(embeddings_b)      # without committing
        gfe.add(individual_id_b, embeddings_b)
        f = gfe.free_energy(mean_energy=0.5)

    The state is mutated by ``add``. ``delta_h`` is non-mutating.
    """

    def __init__(
        self,
        *,
        num_sections: int,
        dim: int,
        eps: float = 1e-3,
        temperature: float = 0.5,
    ) -> None:
        if num_sections <= 0:
            raise ValueError("num_sections must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        self.num_sections = num_sections
        self.dim = dim
        self.eps = eps
        self.T = float(temperature)
        self._sections = [_SectionState.empty(dim, eps) for _ in range(num_sections)]
        self.member_ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def delta_h(self, embeddings: np.ndarray) -> float:
        """Total Σ_k Δ log det L_k for a new candidate (non-mutating).

        ``embeddings`` is shape (M, D) where M=num_sections.
        """
        if embeddings.shape != (self.num_sections, self.dim):
            raise ValueError(
                f"expected embeddings shape ({self.num_sections}, {self.dim}), "
                f"got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        total = 0.0
        for k in range(self.num_sections):
            delta, _, _ = self._sections[k].delta_logdet(embeddings[k])
            total += delta
        return total

    def add(self, member_id: str, embeddings: np.ndarray) -> None:
        if not member_id:
            raise ValueError("member_id must be non-empty")
        if member_id in self.member_ids:
            raise ValueError(f"duplicate member_id: {member_id!r}")
        if embeddings.shape != (self.num_sections, self.dim):
            raise ValueError(
                f"expected embeddings shape ({self.num_sections}, {self.dim}), "
                f"got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        for k in range(self.num_sections):
            self._sections[k].add(embeddings[k])
        self.member_ids.append(member_id)

    def total_logdet(self) -> float:
        return sum(s.logdet_acc for s in self._sections)

    def free_energy(
        self,
        *,
        mean_energy: float | None = None,
        sum_energy: float | None = None,
        mode: str = "extensive",
    ) -> float:
        """F(P) for the current state.

        Default (``mode='extensive'``): the total objective
            Σ_x E(x) - T · Σ_k log det L_k = |P| · F_T(P)
            for the paper's F_T(P) = ⟨E⟩_P - T · H(P) with
            H(P) = (1/|P|) Σ_k log det L_k (same minimiser over equal-size sets)
            → pass ``sum_energy`` (the sum of E(x) over current members).

        Legacy (``mode='intensive'``, ablation only, not the paper's F_T):
            ⟨E⟩_P - T · Σ_k log det L_k
            → pass ``mean_energy``.

        For backwards compatibility, if only one of {sum_energy,
        mean_energy} is provided, the mode is inferred. Passing both
        with a mode mismatch raises ValueError.
        """
        if mean_energy is not None and sum_energy is not None:
            raise ValueError("pass only one of mean_energy and sum_energy")
        if mode == "extensive":
            if sum_energy is None:
                if mean_energy is not None and self.size > 0:
                    # Backwards-compat shim: caller passed mean by old habit.
                    sum_energy = float(mean_energy) * float(self.size)
                else:
                    raise ValueError(
                        "extensive free_energy requires sum_energy"
                    )
            return float(sum_energy) - self.T * self.total_logdet()
        elif mode == "intensive":
            if mean_energy is None:
                if sum_energy is not None and self.size > 0:
                    mean_energy = float(sum_energy) / float(self.size)
                else:
                    raise ValueError(
                        "intensive free_energy requires mean_energy"
                    )
            return float(mean_energy) - self.T * self.total_logdet()
        else:
            raise ValueError(f"unknown free_energy mode: {mode!r}")

    def delta_f_for_greedy(
        self,
        *,
        candidate_energy: float,
        candidate_embeddings: np.ndarray,
        current_mean_energy: float | None = None,
        mode: str = "extensive",
    ) -> float:
        """ΔF used by the greedy selector.

        Default (``mode='extensive'``): the increment of the total objective
            Δ(y) = E(y) - T · Δlogdet(y)
            (the candidate energy enters as itself, since adding y raises
            Σ_{x ∈ S} E(x) by E(y)). For the paper's mean-form F_T the
            increment is (Δ(y) - F_T(S)) / (|S|+1); the argmin over the
            candidates is the same because F_T(S) and |S| are fixed.

        Legacy (``mode='intensive'``):
            ΔF_int(y) = (E(y) - ⟨E⟩_S) / (|S|+1) - T · ΔH(y)
            → pass ``current_mean_energy``.

        Note: the two modes do NOT in general produce the same greedy
        trajectory at fixed N. The candidate comparison expression has
        coefficient ``1`` on E(y) under extensive and ``1/(n+1)`` under
        intensive, so a low-E low-diversity candidate can be selected
        first under one mode and later under the other. The extensive
        form is the engine default because it is the paper's objective up
        to the factor |P|; intensive is kept as a legacy ablation hook only.
        """
        delta_h = self.delta_h(candidate_embeddings)
        if mode == "extensive":
            return float(candidate_energy) - self.T * delta_h
        elif mode == "intensive":
            if current_mean_energy is None:
                raise ValueError(
                    "intensive delta_f_for_greedy requires current_mean_energy"
                )
            n = self.size
            delta_e = (float(candidate_energy) - float(current_mean_energy)) / float(n + 1)
            return delta_e - self.T * delta_h
        else:
            raise ValueError(f"unknown delta_f_for_greedy mode: {mode!r}")
