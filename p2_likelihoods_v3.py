# -*- coding: utf-8 -*-
"""v3 似然 (prereg v1.4): 逐格模拟器误差底图 + 精确闭式 E0/E1/E2 — 替换 rank-K Woodbury
根因 (integrity I9): GP 偏置局域化于洪峰格 (p99≈0.8m), rank-K 逐 PC 标量方差无法表达。
模型: 残差 e ~ N(0, Σ), Σ_i = D_i R_i D_i (+ τ² 11ᵀ for E2), D_i = diag(V_ti) = diag(s2 + W_ti)
  - W: 逐格模拟器误差方差 (24 离设计验证运行实测, θ 无关保守底图)
  - R_i: AR(1) 相关矩阵 (E1/E2); E0: R=I
  - E2: 站点偏置 rank-1, τ 在 TAU_GRID 上 profile (SM 解析, matrix-determinant lemma)
全部闭式; Q=R⁻¹ 三对角 → scipy solve_banded O(T) 求积。
布局: e (n_grid, T, n_st); 掩码 M (n_st, T)。
"""
import numpy as np
from scipy.linalg import solve_banded


def ar_bands(p, T):
    """AR(1) 精度矩阵 (stationary) 的 banded 表示: [[0, off, off...],[diag],[off...,0]]
    端点 1/(1−p²), 内部 (1+p²)/(1−p²), 次对角 −p/(1−p²)"""
    p = float(p)
    c = 1.0 / (1 - p * p)
    d = np.full(T, (1 + p * p) * c)
    d[0] = d[-1] = c
    off = np.full(T - 1, -p * c)
    ab = np.zeros((3, T))
    ab[0, 1:] = off
    ab[1, :] = d
    ab[2, :-1] = off
    return ab


class DiagLikV3:
    def __init__(self, W_map, M, T, n_st, tau_grid):
        self.T, self.n_st = T, n_st
        self.MT = M.T.copy()                    # (T,n_st)
        self.W = np.where(self.MT, W_map, 0.0)  # (T,n_st)
        self.tau_grid = list(tau_grid)
        self.phi = None
        self._ab = None

    def set_phi(self, phi):
        self.phi = np.asarray(phi, dtype=np.float64)
        # 每站 AR(1) 精度按该站有效时段子序列长度构建 (idx 连续段上平稳 AR)
        self._ab = []
        for i in range(self.n_st):
            n_i = len(self._station_idx(i))
            self._ab.append(ar_bands(max(self.phi[i], 1e-3), n_i) if n_i >= 2 else None)

    def _station_idx(self, i):
        return np.where(self.MT[:, i])[0]

    def _Qv(self, i, vi):
        """vᵀQv 逐 (grid) 行: Q 三对角直接乘 (O(T)), quad=Σ v·(Qv)"""
        ab = self._ab[i]
        d, off = ab[1], ab[0, 1:]
        qv = d[None, :] * vi
        qv[:, :-1] += off[None, :] * vi[:, 1:]
        qv[:, 1:] += off[None, :] * vi[:, :-1]
        return (vi * qv).sum(1)

    def _Qmul(self, i, x):
        ab = self._ab[i]
        d, off = ab[1], ab[0, 1:]
        out = d * x
        out[:-1] += off * x[1:]
        out[1:] += off * x[:-1]
        return out

    def _Qw(self, i, wi):
        return float(wi @ self._Qmul(i, wi))

    # ---------- E0: 独立 + 逐格方差 ----------
    def e0_ll(self, e, s2_grid):
        MT = self.MT
        best = np.full(e.shape[0], -np.inf)
        for s2 in s2_grid:
            V = self.W + s2
            Vm = np.where(MT, V, 1.0)
            ldet = (np.log(2 * np.pi * Vm) * MT).sum()   # 标量 (掩码同)
            ll = -0.5 * (ldet + (e ** 2 / Vm * MT).sum((1, 2)))
            best = np.maximum(best, ll)
        return best

    # ---------- E1: AR(1) 精确 + 逐格方差 ----------
    def e1_ll(self, e, s2_grid):
        MT = self.MT
        best = np.full(e.shape[0], -np.inf)
        for s2 in s2_grid:
            V = self.W + s2
            Vm = np.where(MT, V, 1.0)
            v = e / np.sqrt(Vm)[None]          # (b,T,n_st)
            quad = np.zeros(e.shape[0])
            ldet = (np.log(2 * np.pi * Vm) * MT).sum()
            for i in range(self.n_st):
                idx = self._station_idx(i)
                if len(idx) < 5:
                    continue
                p = max(self.phi[i], 1e-3)
                ldet += (len(idx) - 1) * np.log(1 - p * p)
                quad += self._Qv(i, v[:, idx, i])
            best = np.maximum(best, -0.5 * (ldet + quad))
        return best

    # ---------- E2: AR + 站点偏置 rank-1 (精确 SM) ----------
    def e2_ll(self, e, s2_grid):
        """quad_i = vᵀQv − τ²(vᵀQw)²/(1+τ²wᵀQw); logdet += log(1+τ²wᵀQw)"""
        MT = self.MT
        best = np.full(e.shape[0], -np.inf)
        for s2 in s2_grid:
            V = self.W + s2
            Vm = np.where(MT, V, 1.0)
            v = e / np.sqrt(Vm)[None]
            w = np.where(MT, 1.0 / np.sqrt(Vm), 0.0)
            base_quad = np.zeros(e.shape[0])
            ldet0 = (np.log(2 * np.pi * Vm) * MT).sum()
            a_wQw = np.zeros(self.n_st)
            c_vQw = np.zeros((e.shape[0], self.n_st))
            for i in range(self.n_st):
                idx = self._station_idx(i)
                if len(idx) < 5:
                    continue
                p = max(self.phi[i], 1e-3)
                ldet0 += (len(idx) - 1) * np.log(1 - p * p)
                base_quad += self._Qv(i, v[:, idx, i])
                a_wQw[i] = self._Qw(i, w[idx, i])
                # vᵀQw: Qw 向量先算 (θ 无关), 再与 v 内积
                Qw = self._Qmul(i, w[idx, i])
                c_vQw[:, i] = v[:, idx, i] @ Qw
            for tau in self.tau_grid:
                t2 = tau ** 2
                quad = base_quad - (t2 * c_vQw ** 2 / (1 + t2 * a_wQw[None, :])).sum(1)
                ldet = ldet0 + np.log(1 + t2 * a_wQw).sum()
                best = np.maximum(best, -0.5 * (ldet + quad))
        return best

    def e2_ll_b(self, e, s2_grid):
        """e2_ll 同时返回逐 (θ行) 在 argmax(s2,τ) 处的条件 b 后验参数:
        mean_i = τ²·vᵀQw/(1+τ²wᵀQw), var_i = τ²/(1+τ²wᵀQw)
        返回 (ll (rows,), b_mean (rows,n_st), b_var (rows,n_st))"""
        MT = self.MT
        best = np.full(e.shape[0], -np.inf)
        bm = np.zeros((e.shape[0], self.n_st))
        bv = np.zeros((e.shape[0], self.n_st))
        for s2 in s2_grid:
            V = self.W + s2
            Vm = np.where(MT, V, 1.0)
            v = e / np.sqrt(Vm)[None]
            w = np.where(MT, 1.0 / np.sqrt(Vm), 0.0)
            base_quad = np.zeros(e.shape[0])
            ldet0 = (np.log(2 * np.pi * Vm) * MT).sum()
            a_wQw = np.zeros(self.n_st)
            c_vQw = np.zeros((e.shape[0], self.n_st))
            for i in range(self.n_st):
                idx = self._station_idx(i)
                if len(idx) < 5:
                    continue
                p = max(self.phi[i], 1e-3)
                ldet0 += (len(idx) - 1) * np.log(1 - p * p)
                base_quad += self._Qv(i, v[:, idx, i])
                a_wQw[i] = self._Qw(i, w[idx, i])
                Qw = self._Qmul(i, w[idx, i])
                c_vQw[:, i] = v[:, idx, i] @ Qw
            for tau in self.tau_grid:
                t2 = tau ** 2
                quad = base_quad - (t2 * c_vQw ** 2 / (1 + t2 * a_wQw[None, :])).sum(1)
                ldet = ldet0 + np.log(1 + t2 * a_wQw).sum()
                ll = -0.5 * (ldet + quad)
                upd = ll > best
                if upd.any():
                    denom = 1.0 + t2 * a_wQw[None, :]
                    best[upd] = ll[upd]
                    # b 条件后验: mean 逐 (θ,站); var 与 θ 无关 (依赖 s2,τ,w)
                    bm[upd] = t2 * c_vQw[upd] / denom
                    bv[upd] = np.where(a_wQw[None, :] > 0, t2 / np.where(denom > 0, denom, 1.0), 0.0)
        return best, bm, bv
