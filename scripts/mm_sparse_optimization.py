# mm_sparse_slaith.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# -----------------------------
# Helper: AS1 / ASu solvers (Propositions 1 & 2 in paper)
# -----------------------------
def as1_solver(q: np.ndarray) -> np.ndarray:
    """
    Solve min_w ||w||^2 + q^T w s.t. w >= 0, sum(w)=1  (u = 1 case)
    Closed form as described in Proposition 1 (AS1).
    """
    N = len(q)
    # Sort by q ascending because active set A = {i | mu + q_i < 0}
    idx = np.argsort(q)
    q_sorted = q[idx]

    # cumulative sums to find A efficiently
    cumsum_q = np.cumsum(q_sorted)
    # we need find largest k such that mu + q_i < 0 for i in A
    # mu = - (sum_{i in A} q_i) + 2 / card(A)
    # rearrange condition => for candidate A of size k, compute mu and check if mu + q_sorted[k-1] < 0
    best_k = 0
    mu = 0.0
    for k in range(1, N + 1):
        sum_q_A = cumsum_q[k - 1]
        mu_k = - (sum_q_A) + 2.0 * k
        # check smallest q in A (which is q_sorted[k-1])
        if mu_k + q_sorted[k - 1] < 0:
            best_k = k
            mu = mu_k
        else:
            break

    if best_k == 0:
        # no active set => all zeros except constraint forces? fallback to put everything on smallest q?
        # We'll project to simplex by standard method: minimize ||w + q/2||^2 s.t. simplex, but closed form simple:
        # Use standard Euclidean projection onto simplex for -q/2
        v = -q / 2.0
        # projection onto simplex:
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, N+1) > (cssv - 1))[0]
        if len(rho) == 0:
            theta = 0.0
        else:
            rho = rho[-1]
            theta = (cssv[rho] - 1.0) / (rho + 1.0)
        w = np.maximum(v - theta, 0.0)
        return w
    # construct w in sorted order: w_i = max(-0.5*(mu + q_i), 0)
    w_sorted = np.maximum(-0.5 * (mu + q_sorted), 0.0)
    # reorder to original indices
    w = np.zeros(N, dtype=float)
    w[idx] = w_sorted
    # normalize for numerical stability (should already sum to 1)
    s = w.sum()
    if s > 0:
        w /= s
    return w

def asu_solver(q: np.ndarray, u: float) -> np.ndarray:
    """
    Solve min_w ||w||^2 + q^T w s.t. 0 <= w <= u, sum(w) = 1
    Uses Proposition 2 (ASu). We'll implement a simple O(N log N) procedure.
    """
    N = len(q)
    # If u >= 1, reduce to AS1
    if u >= 1.0 - 1e-12:
        return as1_solver(q)

    # Sort q ascending
    idx = np.argsort(q)
    q_sorted = q[idx]

    # We'll search for sets B1 and B2 as in paper:
    # naive O(N) scan that computes mu for each candidate partition.
    # For stability, we implement a binary-search style over mu values:
    # We need mu that satisfies sum_i clip(-0.5*(mu + q_i), 0, u) == 1
    def weight_sum(mu):
        w = np.clip(-0.5 * (mu + q_sorted), 0.0, u)
        return w.sum()

    # find bracketing mu where weight_sum crosses 1
    mu_low, mu_high = -1e6, 1e6
    # shrink interval by sampling
    for _ in range(80):
        mu_mid = 0.5 * (mu_low + mu_high)
        s = weight_sum(mu_mid)
        if s > 1.0:
            mu_low = mu_mid
        else:
            mu_high = mu_mid
    mu = 0.5 * (mu_low + mu_high)
    w_sorted = np.clip(-0.5 * (mu + q_sorted), 0.0, u)
    w = np.zeros(N, dtype=float)
    w[idx] = w_sorted
    # normalize (to exact 1)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        # fallback uniform feasible
        w = np.ones(N) / N
        w = np.minimum(w, u)
        w /= w.sum()
    return w

# -----------------------------
# SLAIT/SLAITH style MM optimizer (specialized Wu case)
# -----------------------------
def mm_sparse_slaith(X: np.ndarray,
                     rb: np.ndarray,
                     lambda_pen: float = 1e-2,
                     u: float = 0.05,
                     p: float = 1e-3,
                     gamma: float = None,
                     max_iter: int = 200,
                     tol: float = 1e-6,
                     use_holdings: bool = False,
                     verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Specialized SLAIT/SLAITH implementation for W = {w >=0, sum(w)=1, w <= u}.
    Returns:
        w_final: (N,) weight vector
        info: dict with 'selected_idx', 'n_iter', 'converged', 'objective_history'
    Notes:
      - We implement the iterative MM updates using the linearization of rho_p,gamma.
      - dp,γ and cp,γ are implemented as in equations (9),(10) in the paper.
      - We use the closed-form projection ASu / AS1 for the inner step.
    Citations: dp,γ / cp,γ definitions and AS1/ASu closed-form steps. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
    """
    T, N = X.shape
    if gamma is None:
        gamma = u  # paper sets gamma = u
    # Standardize X along columns (paper works with returns directly; standardization optional)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    rb_c = rb.copy()

    # Precompute
    L1 = (Xs.T @ Xs) / T
    Xtr_rb = (Xs.T @ rb_c) / T

    # Lipschitz surrogate parameter lambda(L1)_max = max eigenvalue of L1
    L1_eigs = np.linalg.eigvals(L1)
    lambda_L1_max = np.max(L1_eigs).real + 1e-12

    # initialization
    w = np.ones(N) / N

    obj_hist = []
    converged = False

    # constants for rho_p,gamma majorizer
    kappa1 = np.log(1 + gamma / p)

    for k in range(max_iter):
        # dp,γ(w(k)) per component: dp = 1 / (kappa * (p + w_k))
        dp = 1.0 / (kappa1 * (p + w))
        # cp not needed in SLAIT (linear term only)
        # Build q^(k)_1 as in equation (25) / (37) (for SLAITH we would add extra b terms)
        q1 = (1.0 / lambda_L1_max) * (2.0 * (L1 - lambda_L1_max * np.eye(N)) @ w + lambda_pen * dp - 2.0 * Xtr_rb)

        # inner problem: minimize w^T w + q1^T w subject to Wu
        if u >= 1.0 - 1e-12:
            w_new = as1_solver(q1)
        else:
            w_new = asu_solver(q1, u=u)

        # objective (ETE + lambda * sum rho_p,u(w))
        res = Xs @ w_new - rb_c
        ete = (res ** 2).mean()
        # approximate sparsity penalty via sum rho_p,γ(w)
        rho_vals = np.log(1.0 + w_new / p) / np.log(1.0 + gamma / p)
        obj = ete + lambda_pen * rho_vals.sum()
        obj_hist.append(obj)

        if verbose and (k % 20 == 0 or k < 5):
            print(f"iter {k:3d} obj {obj:.6e} selected {np.sum(w_new > 1e-8)}")

        if np.linalg.norm(w_new - w) < tol:
            converged = True
            w = w_new
            break
        w = w_new

    # Final allocation refinement on selected support (paper suggests re-optimizing over selected assets)
    selected_idx = np.where(w > 1e-8)[0]
    if len(selected_idx) == 0:
        w_final = np.zeros(N)
    else:
        X_sel = Xs[:, selected_idx]
        A = (2.0 / T) * (X_sel.T @ X_sel)
        b_sel = (2.0 / T) * (X_sel.T @ rb_c)
        # pseudo-inverse solution then projection onto positive simplex
        try:
            w_ref = np.linalg.pinv(A) @ b_sel
        except np.linalg.LinAlgError:
            w_ref = np.maximum(np.linalg.lstsq(A + 1e-8*np.eye(A.shape[0]), b_sel, rcond=None)[0], 0.0)
        w_ref = np.clip(w_ref, 0.0, None)
        if w_ref.sum() == 0.0:
            w_final = np.zeros(N)
        else:
            w_ref /= w_ref.sum()
            w_final = np.zeros(N)
            w_final[selected_idx] = w_ref

    info = {
        'selected_idx': np.where(w_final > 1e-8)[0],
        'n_iter': k + 1,
        'converged': converged,
        'objective_history': obj_hist
    }
    return w_final, info

# -----------------------------
# Backtester implementing 2 training / 2 testing windows and the "no-more-than-half-change" rule
# -----------------------------
def run_backtest_mm_sparse(ret_df: pd.DataFrame,
                           index_name: str,
                           train_months: int = 6,
                           test_months: int = 1,
                           lambda_pen: float = 1e-2,
                           u: float = 0.05,
                           p: float = 1e-3,
                           verbose: bool = False) -> Dict:
    """
    ret_df: DataFrame with datetime index and columns = asset tickers + index_name column (daily returns)
    index_name: column name for benchmark returns (e.g. '^FCHI' or 'CAC40')
    train_months: months used per training window (we will concatenate two training windows)
    test_months: months used per test window (we will evaluate over two consecutive test windows)
    Returns dict with results DataFrame, weights history, diagnostics.
    """
    # convert months -> approx trading days (21 days per month)
    days_per_month = 21
    window_days = int(train_months * days_per_month)
    step_days = int(test_months * days_per_month)

    dates = ret_df.index
    n_obs = len(dates)
    if window_days < 10:
        raise ValueError("training window too small")

    results = []
    weights_history = []
    prev_selected = None
    prev_w = None

    # We'll move start so that each iteration uses: training = last 2*window_days, testing = next 2*step_days
    for start in range(0, n_obs - 2*window_days - 2*step_days + 1, step_days):
        # two training windows concatenated: [start : start + 2*window_days)
        train_start = start
        train_end = start + 2 * window_days
        test_start = train_end
        test_end = train_end + 2 * step_days

        X_train = ret_df.drop(columns=[index_name]).iloc[train_start:train_end].values
        rb_train = ret_df[index_name].iloc[train_start:train_end].values
        X_test = ret_df.drop(columns=[index_name]).iloc[test_start:test_end].values
        rb_test = ret_df[index_name].iloc[test_start:test_end].values

        if X_train.shape[0] < 20 or X_test.shape[0] == 0:
            continue

        # fit
        w_new, info = mm_sparse_slaith(X_train, rb_train, lambda_pen=lambda_pen, u=u, p=p, verbose=False)

        # Enforce rule: do not change more than half the assets from prev_selected
        if prev_selected is not None and len(prev_selected) > 0:
            prev_set = set(prev_selected.tolist())
            new_set = set(info['selected_idx'].tolist())
            changed = new_set.symmetric_difference(prev_set)
            max_changes = int(np.floor(len(prev_selected) / 2))
            if len(changed) > max_changes:
                # keep the top prev_keep assets by previous weights, allow only limited new additions
                prev_keep = int(np.ceil(len(prev_selected) / 2))
                # sort previous assets by prev weight
                prev_sorted_idx = np.argsort(prev_w)[::-1]
                kept = []
                for idxp in prev_sorted_idx:
                    if idxp in prev_set and len(kept) < prev_keep:
                        kept.append(idxp)
                # candidate set = kept U new selected (but limit new ones so that total changes <= max_changes)
                allowed_new = max_changes
                new_candidates = [i for i in info['selected_idx'] if i not in kept]
                new_candidates = new_candidates[:allowed_new]
                candidate_set = sorted(list(set(kept) | set(new_candidates)))
                # re-optimize weights only over candidate_set using nonnegative least squares with simplex constraint
                if len(candidate_set) == 0:
                    w_new = prev_w.copy()
                else:
                    # Solve min ||X_candidate w_c - rb_train||^2 s.t. w_c >=0, sum=1
                    from scipy.optimize import minimize

                    Xc = (X_train - X_train.mean(axis=0))[:, candidate_set]  # rough standardization not necessary here
                    # objective
                    def obj_wc(wc):
                        r = Xc @ wc - rb_train
                        return (r**2).mean()
                    # initial guess: uniform
                    x0 = np.ones(len(candidate_set)) / len(candidate_set)
                    cons = ({'type': 'eq', 'fun': lambda z: np.sum(z) - 1.0})
                    bounds = [(0.0, 1.0) for _ in range(len(candidate_set))]
                    res = minimize(obj_wc, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol':1e-9,'maxiter':200})
                    if res.success:
                        w_tmp = np.zeros_like(prev_w)
                        w_tmp[candidate_set] = np.clip(res.x, 0.0, None)
                        if w_tmp.sum() > 0:
                            w_new = w_tmp / w_tmp.sum()
                        else:
                            w_new = prev_w.copy()
                    else:
                        # fallback to previous weights
                        w_new = prev_w.copy()

        # evaluate OOS
        rp_test = X_test @ w_new
        active = rp_test - rb_test
        ete = ((X_test @ w_new - rb_test)**2).mean()
        tracking_error_annual = np.std(active) * np.sqrt(252)
        vol_portfolio = np.std(rp_test) * np.sqrt(252)
        vol_index = np.std(rb_test) * np.sqrt(252)
        turnover = 0.0
        if prev_w is not None:
            turnover = np.sum(np.abs(w_new - prev_w))

        results.append({
            'date': dates[test_start],
            'portfolio_ret': float(np.mean(rp_test)),
            'index_ret': float(np.mean(rb_test)),
            'active_ret': float(np.mean(active)),
            'ETE': float(ete),
            'tracking_error_annual': float(tracking_error_annual),
            'vol_portfolio_annual': float(vol_portfolio),
            'vol_index_annual': float(vol_index),
            'n_selected': int(np.sum(w_new > 1e-6)),
            'turnover': float(turnover)
        })
        weights_history.append(w_new)
        prev_selected = np.where(w_new > 1e-8)[0]
        prev_w = w_new.copy()

    results_df = pd.DataFrame(results).set_index('date')
    # cumulative returns
    results_df['cum_portfolio'] = (1.0 + results_df['portfolio_ret']).cumprod()
    results_df['cum_index'] = (1.0 + results_df['index_ret']).cumprod()

    return {
        'results_df': results_df,
        'weights_history': weights_history
    }

# -----------------------------
# Plotting helper
# -----------------------------
def plot_backtest_results(bt_out: Dict, figsize=(12, 8)):
    """
    bt_out: dict returned by run_backtest_mm_sparse
    Produces:
      - cumulative performance plot (portfolio vs index)
      - rolling volatility (portfolio vs index)
      - tracking error / ETE over time
    """
    df = bt_out['results_df']
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cum_portfolio'], label='Sparse Portfolio (MM)')
    plt.plot(df.index, df['cum_index'], label='Index')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['vol_portfolio_annual'], label='Portfolio vol')
    plt.plot(df.index, df['vol_index_annual'], label='Index vol')
    plt.title('Annualized Volatility (rolling test windows)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['tracking_error_annual'], label='Annualized Tracking Error')
    plt.title('Tracking Error (annualized)')
    plt.legend()
    plt.show()