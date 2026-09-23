"""
Diagnostic instrumentation for LOZO: effective-rank statistics of the low-rank
gradient estimate, logged once per lazy-sampling interval (every `step_interval` steps).

Nothing here changes the LOZO update. The probe only reads the `u` matrices and the
scalar `c = (F+ - F-) / 2eps` that the update already computes, and it uses its own
RNG so the global numpy / torch random streams used by LOZO are untouched.

Sources logged per (interval, layer):
  lozo       singular values of (1/nu) sum_t c^t U^t V^T  -- the accumulated update direction.
             E[c U] = G V for fixed V, so this estimates G V V^T.
  null       same, but with c^t multiplied by an independent random sign. This kills the
             gradient signal and keeps the noise, so it is the "pure random-matrix" baseline
             that `lozo` has to be compared against.
  grad_proj  (optional, needs backprop) singular values of G V V^T for the true minibatch
             gradient G at the start of the interval -- the target that `lozo` estimates.
  grad_full  (optional, needs backprop) top singular values / rank stats of the full G.
"""

import csv
import os
import re

import numpy as np
import torch


DELTAS = (0.01, 0.05, 0.10)


def rank_stats(s):
    """Rank estimators from a 1-D array of singular values (any order)."""
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    s = np.clip(s, 0, None)
    s2 = s ** 2
    fro2 = s2.sum()
    out = {"fro2": fro2}
    if fro2 <= 0 or s[0] <= 0:
        out.update({"r_eff": np.nan, "r_part": np.nan, "gap_rank": np.nan})
        out.update({f"r_{d:.2f}": np.nan for d in DELTAS})
        return out
    out["r_eff"] = fro2 / s2[0]
    out["r_part"] = s.sum() ** 2 / fro2
    cum = np.cumsum(s2) / fro2
    for d in DELTAS:
        out[f"r_{d:.2f}"] = int(np.searchsorted(cum, 1 - d - 1e-12) + 1)
    # gap rank: argmax_i (sigma_i - sigma_{i+1}), with sigma_{r+1} = 0
    gaps = s - np.append(s[1:], 0.0)
    out["gap_rank"] = int(np.argmax(gaps) + 1)
    return out


def lowrank_svdvals(A, V):
    """Nonzero singular values of A @ V.T (A: m x r, V: n x r) without forming the m x n matrix."""
    _, Ra = torch.linalg.qr(A.float())
    _, Rv = torch.linalg.qr(V.float())
    return torch.linalg.svdvals(Ra @ Rv.T)


class RankProbe:
    def __init__(self, args, output_path):
        self.args = args
        self.nu = args.step_interval
        self.r = args.rank_r
        self.layer_re = re.compile(args.rank_probe_layers) if args.rank_probe_layers else None
        self.grad_re = re.compile(args.rank_probe_grad_layers) if args.rank_probe_grad_layers else None
        self.grad_topk = args.rank_probe_grad_topk
        self.n_sv = max(self.r, self.grad_topk if self.grad_re else 0)
        # Independent RNG for the null signs: must not consume np.random (LOZO's seed stream)
        self.rng = np.random.default_rng(args.seed + 12345)

        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.fields = (
            ["step", "interval", "layer", "source", "rank_r", "step_interval", "n_steps", "loss", "c_rms"]
            + ["fro2", "r_eff", "r_part"] + [f"r_{d:.2f}" for d in DELTAS] + ["gap_rank", "energy_frac"]
            + [f"sv_{i + 1}" for i in range(self.n_sv)]
        )
        with open(self.output_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

        self._reset()

    def _reset(self):
        self.acc, self.null = {}, {}
        self.n_steps = 0
        self.loss_sum = 0.0
        self.c_sq_sum = 0.0
        self.sign = 1.0

    def tracks(self, name):
        return self.layer_re is None or self.layer_re.search(name) is not None

    # ---- called from the LOZO update ----

    def begin_step(self, c, loss):
        self.sign = float(self.rng.choice((-1.0, 1.0)))
        self.c = c
        self.n_steps += 1
        self.loss_sum += loss
        self.c_sq_sum += c ** 2

    def accumulate(self, name, u):
        if not self.tracks(name):
            return
        u = u.float()
        if name not in self.acc:
            self.acc[name] = torch.zeros_like(u)
            self.null[name] = torch.zeros_like(u)
        self.acc[name].add_(u, alpha=self.c)
        self.null[name].add_(u, alpha=self.c * self.sign)

    def end_interval(self, step, interval, v_dict):
        """Called after the last update of an interval, before V is resampled."""
        if self.n_steps == 0:
            return
        common = {
            "step": step, "interval": interval, "rank_r": self.r, "step_interval": self.nu,
            "n_steps": self.n_steps, "loss": self.loss_sum / self.n_steps,
            "c_rms": (self.c_sq_sum / self.n_steps) ** 0.5,
        }
        rows = []
        for name, A in self.acc.items():
            V = v_dict[name]
            for source, M in (("lozo", A), ("null", self.null[name])):
                s = lowrank_svdvals(M / self.n_steps, V).cpu().numpy()
                rows.append(self._row(common, name, source, s))
        self._write(rows)
        self._reset()

    # ---- optional true-gradient probe ----

    @torch.no_grad()
    def grad_stats(self, step, interval, grads, v_dict):
        """grads: dict name -> true gradient G (m x n) at the start of the interval."""
        common = {"step": step, "interval": interval, "rank_r": self.r, "step_interval": self.nu, "n_steps": 0}
        rows = []
        for name, G in grads.items():
            G = G.float()
            s_full = torch.linalg.svdvals(G).cpu().numpy()
            rows.append(self._row(common, name, "grad_full", s_full))
            if name in v_dict:
                V = v_dict[name].float()
                GV = G @ V
                s_proj = lowrank_svdvals(GV, V).cpu().numpy()
                row = self._row(common, name, "grad_proj", s_proj)
                # fraction of gradient energy inside the current row subspace span(V)
                Qv, _ = torch.linalg.qr(V)
                row["energy_frac"] = float((G @ Qv).pow(2).sum() / G.pow(2).sum().clamp_min(1e-30))
                rows.append(row)
        self._write(rows)

    def wants_grad(self, name):
        return self.grad_re is not None and self.grad_re.search(name) is not None

    # ---- helpers ----

    def _row(self, common, name, source, s):
        s = np.sort(s)[::-1]
        row = dict(common, layer=name, source=source)
        row.update(rank_stats(s))
        for i, x in enumerate(s[: self.n_sv]):
            row[f"sv_{i + 1}"] = float(x)
        return row

    def _write(self, rows):
        with open(self.output_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            for row in rows:
                w.writerow({k: row.get(k, "") for k in self.fields})
