# -*- coding: utf-8 -*-
"""p2_likelihoods_v2 数值验证: Woodbury 边缘化 vs 稠密协方差直接计算"""
import numpy as np
import sys
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2_likelihoods_v2 import EmuLik, whiten_t

rng = np.random.default_rng(7)
T, n_st, K = 6, 15, 4
N = T * n_st
M = np.ones((n_st, T), dtype=bool)          # 全有效, 便于稠密对照
phi = rng.uniform(0.2, 0.8, n_st)
Ys = rng.uniform(0.05, 0.3, (T, n_st))
Vt = rng.normal(0, 1 / np.sqrt(N), (K, N))
Vt /= np.linalg.norm(Vt, axis=1, keepdims=True)   # 每行范数 1
lik = EmuLik(Vt, Ys, K, T, n_st, M)
lik.set_phi(phi)

b = 3
Yg = rng.uniform(0.05, 1.2, (b, T, n_st))
e = rng.normal(0, 0.05, (b, T, n_st))
var_l = rng.uniform(0.02, 0.5, (b, K)) ** 2
s2_grid = [3e-4, 1e-3, 3e-3, 1e-2]
ok = True

# ---- E0 ----
ee, Gte = lik.e0_stats(e)
ll_w = lik.e0_ll(ee, Gte, var_l, s2_grid, N)
for j, s2 in enumerate(s2_grid):
    for k in range(b):
        B = lik.G0 * np.sqrt(var_l[k])[None]
        C = s2 * np.eye(N) + B @ B.T
        ef = (e[k] * M.T).ravel()
        quad = ef @ np.linalg.solve(C, ef)
        ldet = np.linalg.slogdet(C)[1]
        ll_d = -0.5 * (quad + ldet)
        d = abs(ll_d - (-0.5 * (ee[k] / s2)) - (ll_w[k] - max(ll_w)))  # 归一化后比较
        # 直接比较未归一化: ll_w 是 max over s2 — 改为逐 s2 比较
        pass
# 逐 s2 精确比较 (绕过 max): 重新实现逐 s2
for j, s2 in enumerate(s2_grid):
    eye = np.eye(K)
    A = eye[None] / var_l[:, :, None] + lik.GG0[None] / s2
    rhs = (Gte / s2)[..., None]
    X = np.linalg.solve(A, rhs)[..., 0]
    corr = np.einsum('bk,bk->b', rhs[..., 0], X)
    _, ldet_w = np.linalg.slogdet(eye[None] + var_l[:, :, None] * lik.GG0[None] / s2)
    ll_wj = -0.5 * (ee / s2 - corr + N * np.log(s2) + ldet_w)
    for k in range(b):
        B = lik.G0 * np.sqrt(var_l[k])[None]
        C = s2 * np.eye(N) + B @ B.T
        ef = (e[k] * M.T).ravel()
        ll_d = -0.5 * (ef @ np.linalg.solve(C, ef) + np.linalg.slogdet(C)[1])
        rel = abs(ll_wj[k] - ll_d) / max(abs(ll_d), 1)
        if rel > 1e-8:
            ok = False
            print(f'E0 s2={s2} k={k}: woodbury={ll_wj[k]:.10f} dense={ll_d:.10f} rel={rel:.2e}')
print('E0:', 'PASS' if ok else 'FAIL')

# ---- E1 ----
uu, Gtu, BuBu = lik._stats(e, Yg)
h = lik._h(Yg)
ll_w1 = lik.e1_ll(uu, Gtu, BuBu, var_l, s2_grid, N)
ok1 = True
for j, s2 in enumerate(s2_grid):
    # 逐 s2 重算 (复制 e1_ll 内部)
    eye = np.eye(K)
    A = eye[None] / var_l[:, :, None] + BuBu / s2
    rhs = (Gtu / s2)[..., None]
    X = np.linalg.solve(A, rhs)[..., 0]
    corr = np.einsum('bk,bk->b', rhs[..., 0], X)
    _, ldet_w = np.linalg.slogdet(eye[None] + var_l[:, :, None] * BuBu / s2)
    ll_wj = -0.5 * (uu / s2 - corr + N * np.log(s2) + ldet_w)
    for k in range(b):
        D = 1.0 / h[k]
        B3 = whiten_t(lik.G3[None] * D[..., None], phi).reshape(N, K)
        u3 = whiten_t((e[k] * M.T)[None] * D[None], phi).ravel()
        C = s2 * np.eye(N) + B3 @ np.diag(var_l[k]) @ B3.T
        ll_d = -0.5 * (u3 @ np.linalg.solve(C, u3) + np.linalg.slogdet(C)[1])
        rel = abs(ll_wj[k] - ll_d) / max(abs(ll_d), 1)
        if rel > 1e-8:
            ok1 = False
            print(f'E1 s2={s2} k={k}: woodbury={ll_wj[k]:.10f} dense={ll_d:.10f} rel={rel:.2e}')
print('E1:', 'PASS' if ok1 else 'FAIL')
ok = ok and ok1

# ---- E2 ----
uu2, Gtu2, BuBu2, (C_i, S1_i, GoGtu, GoBuBu) = lik._stats(e, Yg, orth_S=True)
tau_grid = [0.05, 0.15]
ll_w2 = lik.e2_ll(uu2, C_i, S1_i, GoGtu, GoBuBu, var_l, s2_grid, tau_grid, N)
ok2 = True
# 正交性检查
for k in range(b):
    D = 1.0 / h[k]
    B3 = whiten_t(lik.G3[None] * D[..., None], phi).reshape(N, K)
    StB = lik.Sflat.T @ B3
    # Go 重算
    coef = np.linalg.solve(lik.StS, lik.Sflat.T @ B3)
    Go = B3 - lik.Sflat @ coef
    orth = np.abs(Go.T @ lik.Sflat).max()
    if orth > 1e-8 * max(1, np.abs(lik.StS).max()):
        print(f'  GoᵀS = {orth:.2e} (应≈0)')
# 稠密对照
for tau in tau_grid:
    for j, s2 in enumerate(s2_grid):
        for k in range(b):
            D = 1.0 / h[k]
            B3 = whiten_t(lik.G3[None] * D[..., None], phi).reshape(N, K)
            coef = np.linalg.solve(lik.StS, lik.Sflat.T @ B3)
            Go = B3 - lik.Sflat @ coef
            C = s2 * np.eye(N) + lik.Sflat @ (tau**2 * lik.StS * 0 + np.diag(tau**2 * np.ones(n_st))) @ lik.Sflat.T + Go @ np.diag(var_l[k]) @ Go.T
            # SM 部分 (S 块): C_S = S diag(tau²) Sᵀ — 注意 StS 与 tau 组合
            u3 = whiten_t((e[k] * M.T)[None] * D[None], phi).ravel()
            ll_d = -0.5 * (u3 @ np.linalg.solve(C, u3) + np.linalg.slogdet(C)[1])
            # woodbury 逐 (tau,s2)
            eye = np.eye(K)
            A = eye / var_l[k][:, None] + GoBuBu[k] / s2
            rhs = (GoGtu[k][None] / s2)[..., None]
            X = np.linalg.solve(A, rhs)[..., 0]
            corr = np.einsum('bk,bk->b', rhs[..., 0], X)[0]
            _, ldet_e = np.linalg.slogdet(eye + var_l[k][:, None] * GoBuBu[k] / s2)
            v_i = tau**2 * lik.ss_i
            Vi = s2 + v_i
            quad_sm = (S1_i[k] / s2).sum() - tau**2 * (C_i[k]**2 / (s2 * Vi)).sum()
            ldet_sm = ((M.sum(1) - 1) * np.log(s2) + np.log(Vi)).sum()
            ll_wj = -0.5 * (quad_sm - corr + ldet_sm + ldet_e)
            rel = abs(ll_wj - ll_d) / max(abs(ll_d), 1)
            if rel > 1e-6:
                ok2 = False
                print(f'E2 tau={tau} s2={s2} k={k}: woodbury={ll_wj:.10f} dense={ll_d:.10f} rel={rel:.2e}')
print('E2:', 'PASS' if ok2 else 'FAIL')
print('ALL:', 'PASS' if (ok and ok2) else 'FAIL')
