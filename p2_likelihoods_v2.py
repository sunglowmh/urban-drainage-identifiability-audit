# -*- coding: utf-8 -*-
"""v2 似然: 代理误差 rank-K 相关场解析边缘化 (Woodbury) — preregistration v1.2 B2
残差协方差 = s2·I (+ 定标/白化) + Σ_l var_l(θ)·g_l g_lᵀ, g_l = Vt_l·Ys (θ 无关 basis)
var_l(θ) = GP 克里金预测方差 (θ 依赖, 调用方逐批提供); s2 对数网格 profile (取 max)。
布局: e (n_grid, T, n_st); 展平 N=T*n_st, 行序 (t*n_st+i) 与 Vt 列序一致; 掩码零化。
E1: 逐格 1/h 定标 + 逐站 φ 白化 (与现行管线同序);
E2: E1 域 + 站点偏置广义 SM + 正交化代理 basis 修正 (SᵀGo=0 ⇒ 精确分解)。
"""
import numpy as np


def whiten_t(X, phi):
    """X: (b,T,n_st) 或 (b,T,n_st,K) — 沿 axis=1 (t) 白化, phi 对齐 axis=2 (站)"""
    out = X.copy()
    if X.ndim == 3:
        out[:, 1:, :] -= phi[None, None, :] * X[:, :-1, :]
    elif X.ndim == 4:
        out[:, 1:, :, :] -= phi[None, None, :, None] * X[:, :-1, :, :]
    else:
        raise ValueError(f"unsupported ndim {X.ndim}")
    return out


class EmuLik:
    def __init__(self, Vt, Ys, n_pc, T, n_st, M):
        self.T, self.n_st, self.K, self.N = T, n_st, int(n_pc), T * n_st
        Ys_flat = np.asarray(Ys, dtype=np.float64).ravel()       # (N,) t-major
        self.mask = M.T.ravel().astype(np.float64)               # (N,)
        G = Vt[:self.K].T * Ys_flat[:, None]                     # (N, K)
        self.G0 = G * self.mask[:, None]
        self.G3 = self.G0.reshape(T, n_st, self.K)               # (T, n_st, K)
        self.GG0 = self.G0.T @ self.G0                           # (K, K)
        self.phi = None

    def set_phi(self, phi):
        """逐案例设置白化系数并重算 phi 相关量 (站点偏置 basis, 仅含有效站)"""
        T, n_st = self.T, self.n_st
        self.phi = np.asarray(phi, dtype=np.float64)
        mT = (self.mask > 0).reshape(T, n_st)
        self.active = np.where(mT.sum(0) > 0)[0]        # 全零站 (QC 排除) 不进 basis
        scol = np.stack([np.concatenate([[1.0], np.full(T - 1, 1.0 - self.phi[i])])
                         for i in self.active], axis=1)               # (T, n_active)
        self.scol_full = np.zeros((T, n_st))
        self.scol_full[:, self.active] = scol
        self.scol_T = self.scol_full
        n_act = len(self.active)
        Sflat = np.zeros((self.N, n_act))
        t_arr = np.arange(T)
        for j, i in enumerate(self.active):
            col = np.zeros(self.N)
            col[t_arr * n_st + i] = scol[:, j]      # 站 i 的所有 t 格
            Sflat[:, j] = col * self.mask
        self.Sflat = Sflat
        self.StS = self.Sflat.T @ self.Sflat
        self.ss_i = np.zeros(n_st)
        self.ss_i[self.active] = (scol ** 2 * mT[:, self.active]).sum(0)

    def _h(self, Yg):
        d_p90 = np.nanpercentile(Yg, 90, axis=(0, 1)) + 1e-6
        return 0.1 + 0.9 * np.clip(Yg / d_p90[None, None], 0, 2)

    # ---------------- E0 ----------------
    def e0_stats(self, e, batch=4000):
        ef = e.reshape(e.shape[0], self.N)
        ee = (ef ** 2).sum(1)
        Gte = ef @ self.G0
        return ee, Gte

    def e0_ll(self, ee, Gte, var_l, s2_grid, N_eff):
        b, K = Gte.shape
        eye = np.eye(K)
        best = np.full(b, -np.inf)
        for s2 in s2_grid:
            A = eye[None] / var_l[:, :, None] + self.GG0[None] / s2
            rhs = (Gte / s2)[..., None]
            X = np.linalg.solve(A, rhs)[..., 0]
            corr = np.einsum('bk,bk->b', rhs[..., 0], X)
            _, ldet = np.linalg.slogdet(eye[None] + var_l[:, :, None] * self.GG0[None] / s2)
            best = np.maximum(best, -0.5 * (ee / s2 - corr + N_eff * np.log(s2) + ldet))
        return best

    # ---------------- E1/E2 公共统计 (分批) ----------------
    def _stats(self, e, Yg, batch=200, orth_S=False):
        """返回 uu (b,), Gtu (b,K), BuBu (b,K,K); orth_S 时另返回 (C_i, S1_i, GoGtu, GoBuBu)"""
        b_all = e.shape[0]
        D = 1.0 / self._h(Yg)
        uu = np.empty(b_all); Gtu = np.empty((b_all, self.K))
        BuBu = np.empty((b_all, self.K, self.K))
        C_i = np.empty((b_all, self.n_st)); S1_i = np.empty((b_all, self.n_st))
        GoGtu = np.empty((b_all, self.K)); GoBuBu = np.empty((b_all, self.K, self.K))
        StS_inv = np.linalg.inv(self.StS)
        for s in range(0, b_all, batch):
            sl = slice(s, min(s + batch, b_all))
            eD = e[sl] * D[sl]
            u3 = whiten_t(eD, self.phi)
            u = u3.reshape(sl.stop - sl.start, self.N)
            Bu3 = whiten_t(self.G3[None] * D[sl][..., None], self.phi)
            Bu = Bu3.reshape(-1, self.N, self.K)
            uu[sl] = (u ** 2).sum(1)
            Gtu[sl] = np.einsum('bn,bnk->bk', u, Bu)
            BuBu[sl] = np.einsum('bnk,bnl->bkl', Bu, Bu)
            if orth_S:
                uT = u3.reshape(sl.stop - sl.start, self.T, self.n_st)
                S1_i[sl] = (uT ** 2).sum(1)                       # (b, n_st)
                C_i[sl] = np.einsum('bti,ti->bi', uT, self.scol_T)
                coef = np.linalg.solve(self.StS,
                                       np.einsum('ns,bnk->bsk', self.Sflat, Bu))
                Go = Bu - np.einsum('ns,bsk->bnk', self.Sflat, coef)
                GoBuBu[sl] = np.einsum('bnk,bnl->bkl', Go, Go)
                GoGtu[sl] = np.einsum('bn,bnk->bk', u, Go)
        if orth_S:
            return uu, Gtu, BuBu, (C_i, S1_i, GoGtu, GoBuBu)
        return uu, Gtu, BuBu

    def _emu_corr(self, BB, Gt, var_l, s2_grid):
        b, K = Gt.shape
        eye = np.eye(K)
        corr = np.empty((b, len(s2_grid))); ldet = np.empty((b, len(s2_grid)))
        for j, s2 in enumerate(s2_grid):
            A = eye[None] / var_l[:, :, None] + BB / s2
            rhs = (Gt / s2)[..., None]
            X = np.linalg.solve(A, rhs)[..., 0]
            corr[:, j] = np.einsum('bk,bk->b', rhs[..., 0], X)
            _, ldet[:, j] = np.linalg.slogdet(eye[None] + var_l[:, :, None] * BB / s2)
        return corr, ldet

    def e1_ll(self, uu, Gtu, BuBu, var_l, s2_grid, N_eff):
        corr, ldet = self._emu_corr(BuBu, Gtu, var_l, s2_grid)
        best = np.full(len(uu), -np.inf)
        for j, s2 in enumerate(s2_grid):
            best = np.maximum(best, -0.5 * (uu / s2 - corr[:, j] +
                                            N_eff * np.log(s2) + ldet[:, j]))
        return best

    def e2_ll(self, uu, C_i, S1_i, GoGtu, GoBuBu, var_l, s2_grid, tau_grid, N_eff):
        corr, ldet = self._emu_corr(GoBuBu, GoGtu, var_l, s2_grid)
        best = np.full(len(uu), -np.inf)
        n_i = (self.mask > 0).reshape(self.T, self.n_st).sum(0)
        for tau in tau_grid:
            v_i = tau ** 2 * self.ss_i
            for j, s2 in enumerate(s2_grid):
                Vi = s2 + v_i
                # 广义 SM (rank-1 per station): quad_i = S1_i/s2 − τ²C_i²/(s2·Vi)
                quad_sm = (S1_i / s2).sum(1) - \
                          (tau ** 2 * C_i ** 2 / (s2 * Vi[None])).sum(1)
                ldet_sm = ((n_i - 1) * np.log(s2) + np.log(Vi)).sum()
                best = np.maximum(best, -0.5 * (quad_sm - corr[:, j] + ldet_sm + ldet[:, j]))
        return best
