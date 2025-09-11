"""
Complete Schur-HRP Portfolio Optimizer Web App - FIXED VERSION
Save this as app.py and deploy to Streamlit Cloud
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit Cloud
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import squareform
from dataclasses import dataclass
from typing import Optional, List
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Schur-HRP Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CORE UTILITY FUNCTIONS
# ========================================
EPS = 1e-12

def safe_pinv(M, rcond=1e-8):
    return np.linalg.pinv(M, rcond=rcond)

def symmetrize(M):
    return 0.5 * (M + M.T)

def safe_outer(u):
    return np.outer(u, u)

def elementwise_divide(A, u):
    denom = safe_outer(u)
    tiny = np.abs(denom) < EPS
    if np.any(tiny):
        denom[tiny] = np.sign(denom[tiny]) * EPS
    return A / denom

def cov_to_corr(cov: pd.DataFrame):
    d = np.sqrt(np.maximum(np.diag(cov), EPS))
    D = np.outer(d, d)
    corr = cov / D
    corr = np.clip(corr, -1, 1)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def correl_dist(corr: np.ndarray):
    return np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))

def get_quasi_diag_order(cov: pd.DataFrame):
    corr = cov_to_corr(cov)
    dist = correl_dist(corr.values)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method='single')
    order = leaves_list(Z)
    return list(order)

# ========================================
# DATA & RETURNS FUNCTIONS
# ========================================
def fetch_prices_yf(tickers, period="1y", interval="1d", min_valid_frac=0.9):
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if isinstance(t, str) and len(t.strip())]))
    if not tickers:
        return pd.DataFrame(), [], []
    df = yf.download(tickers, period=period, interval=interval, progress=False, group_by='column', auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            prices = df['Adj Close'].copy()
        elif 'Close' in df.columns.get_level_values(0):
            prices = df['Close'].copy()
        else:
            raise ValueError("yfinance: no 'Adj Close' or 'Close' found.")
    else:
        if 'Adj Close' in df.columns:
            prices = df['Adj Close'].copy()
        elif 'Close' in df.columns:
            prices = df['Close'].copy()
        else:
            raise ValueError("yfinance: no 'Adj Close' or 'Close' found (flat columns).")
    present = [c for c in prices.columns if c in tickers]
    prices = prices[present].copy()
    valid_counts = prices.notna().sum()
    min_required = int(np.ceil(len(prices) * min_valid_frac))
    kept = [t for t in present if valid_counts.get(t, 0) >= min_required]
    dropped = [t for t in tickers if t not in kept]
    prices = prices[kept].ffill().bfill()
    return prices, kept, dropped

def compute_returns(prices: pd.DataFrame, log=True):
    if log: 
        rets = np.log(prices / prices.shift(1))
    else:   
        rets = prices.pct_change()
    return rets.dropna(how='all').dropna()

# ========================================
# HRP FUNCTIONS
# ========================================
def cluster_variance(cov: pd.DataFrame, idx):
    sub = cov.values[np.ix_(idx, idx)]
    iv = 1.0 / np.maximum(np.diag(sub), EPS)
    w = iv / iv.sum()
    var = float(w @ sub @ w)
    return max(var, EPS)

def hrp_weights(cov: pd.DataFrame):
    if cov.shape[0] == 1:
        return pd.Series([1.0], index=cov.index)
    order = get_quasi_diag_order(cov)
    items = order.copy()
    def _bisect(items):
        n = len(items)
        if n == 1: 
            return {items[0]: 1.0}
        L = items[: n // 2]
        R = items[n // 2 :]
        vL = cluster_variance(cov, L)
        vR = cluster_variance(cov, R)
        aL = vR / (vL + vR)
        aR = 1.0 - aL
        wl = _bisect(L)
        wr = _bisect(R)
        for k in wl: 
            wl[k] *= aL
        for k in wr: 
            wr[k] *= aR
        wl.update(wr)
        return wl
    wmap = _bisect(items)
    w = pd.Series([wmap[i] for i in range(cov.shape[0])], index=cov.index)
    w = w.clip(lower=0)
    s = w.sum()
    return w / (s if s > 0 else 1.0)

# ========================================
# HMV FUNCTIONS
# ========================================
def hmv_weights(cov: pd.DataFrame, gamma: float = 1.0):
    gamma = float(np.clip(gamma, 0.0, 1.0))
    n = cov.shape[0]
    if n == 1: 
        return pd.Series([1.0], index=cov.index)
    order = get_quasi_diag_order(cov)
    Sigma0 = cov.values[np.ix_(order, order)]
    
    def _hmv(Sigma):
        m = Sigma.shape[0]
        if m == 1: 
            return np.array([1.0], dtype=float)
        k = m // 2
        A = Sigma[:k, :k]
        D = Sigma[k:, k:]
        B = Sigma[:k, k:]
        C = B.T
        A = symmetrize(A)
        D = symmetrize(D)
        Ainv = safe_pinv(A)
        Dinv = safe_pinv(D)
        A_c = symmetrize(A - gamma * (B @ Dinv @ C))
        D_c = symmetrize(D - gamma * (C @ Ainv @ B))
        one_L = np.ones((k, 1))
        one_R = np.ones((m - k, 1))
        bA = (one_L - gamma * (B @ Dinv @ one_R)).reshape(-1)
        bD = (one_R - gamma * (C @ Ainv @ one_L)).reshape(-1)
        A_c_inv = safe_pinv(A_c)
        D_c_inv = safe_pinv(D_c)
        qA = float(bA.T @ A_c_inv @ bA)
        qA = max(qA, EPS)
        qD = float(bD.T @ D_c_inv @ bD)
        qD = max(qD, EPS)
        nuA = 1.0 / qA
        nuD = 1.0 / qD
        aL = nuD / (nuA + nuD)
        aR = 1.0 - aL
        A_dd = symmetrize(elementwise_divide(A_c, bA))
        D_dd = symmetrize(elementwise_divide(D_c, bD))
        wl = _hmv(A_dd)
        wr = _hmv(D_dd)
        return np.concatenate([aL * wl, aR * wr])
    
    w_ord = _hmv(Sigma0)
    w_ord = np.clip(w_ord, 0, None)
    s = w_ord.sum()
    w_ord = w_ord / (s if s > 0 else 1.0)
    w = np.zeros(n)
    for pos, idx in enumerate(order): 
        w[idx] = w_ord[pos]
    return pd.Series(w, index=cov.index)

# ========================================
# CAPPED-SIMPLEX PROJECTION
# ========================================
def project_to_capped_simplex(base_nonneg: np.ndarray, total: float, cap: float, tol: float = 1e-12):
    """Project nonnegative vector onto {x >= 0, sum x = total, x_i <= cap} via water-filling."""
    base = np.maximum(base_nonneg.astype(float), 0.0)
    n = base.size
    if total > n * cap + 1e-12:
        raise ValueError(f"Infeasible cap: need sum {total:.6f} across {n} names with cap {cap:.6f} (max feasible {n*cap:.6f}).")
    if base.sum() <= tol:
        x = np.full(n, min(cap, total / n))
        x *= total / max(x.sum(), tol)
    else:
        x = base * (total / base.sum())
    active = np.ones(n, dtype=bool)
    while True:
        over = (x > cap + 1e-15) & active
        if not over.any(): 
            break
        x[over] = cap
        active[over] = False
        remain = total - x.sum()
        if remain <= tol or active.sum() == 0: 
            break
        w = base[active]
        if w.sum() <= tol: 
            x[active] += remain / active.sum()
        else:              
            x[active] = np.minimum(cap, x[active] + remain * (w / w.sum()))
    diff = total - x.sum()
    if abs(diff) > 1e-10:
        free = np.where((x < cap - 1e-12))[0]
        if free.size > 0:
            x[free] += diff / free.size
            x = np.clip(x, 0.0, cap)
        else:
            x += diff / n
            x = np.clip(x, 0.0, cap)
    return x

# ========================================
# RISK CONTRIBUTIONS
# ========================================
def compute_risk_contributions(cov: pd.DataFrame,
                               weights: pd.Series,
                               long_tickers: List[str],
                               short_tickers: List[str]) -> pd.DataFrame:
    Sigma = cov.values
    w_vec = weights.reindex(cov.index).fillna(0.0).astype(float).values
    port_var = float(w_vec @ Sigma @ w_vec)
    mrc = Sigma @ w_vec
    rc  = w_vec * mrc
    pct = rc / port_var if port_var > 0 else np.zeros_like(rc)
    side_map = {t: ("Long" if t in long_tickers else ("Short" if t in short_tickers else "Unknown"))
                for t in cov.index}
    df = pd.DataFrame({
        "Ticker": cov.index,
        "Side": [side_map[t] for t in cov.index],
        "Weight": w_vec,
        "MRC": mrc,
        "RC": rc,
        "PctRC": pct
    }, index=cov.index)
    df["AbsWeight"] = np.abs(df["Weight"])
    df["AbsRC"] = np.abs(df["RC"])
    df["AbsPctRC"] = np.abs(df["PctRC"])
    return df

# ========================================
# SCHUR HMV CONFIG & ALLOCATION
# ========================================
@dataclass
class SchurHMVConfig:
    between_group_mode: str = 'schur_conditional_risk'
    gross_target: float = 1.0
    net_target: float = 0.0
    respect_net_target: bool = True
    annualize_factor: int = 252
    gamma: float = 1.0
    max_long_abs_weight: Optional[float] = None
    max_short_abs_weight: Optional[float] = None

def schur_hmv_allocate(cov: pd.DataFrame, long_idx, short_idx, cfg: SchurHMVConfig, intra_method='hmv'):
    A = cov.iloc[long_idx, long_idx].copy().astype(float)
    C = cov.iloc[short_idx, short_idx].copy().astype(float)
    B = cov.iloc[long_idx, short_idx].copy().astype(float)

    # Intra-sleeve weights (sum=1, >=0)
    if intra_method == 'hmv':
        l = hmv_weights(A, gamma=cfg.gamma).values
        s = hmv_weights(C, gamma=cfg.gamma).values
    elif intra_method == 'hrp':
        l = hrp_weights(A).values
        s = hrp_weights(C).values
    else:
        raise ValueError("intra_method must be 'hmv' or 'hrp'.")

    # Sleeve scalar risks
    vL = float(l @ A.values @ l)
    vS = float(s @ C.values @ s)
    cLS = float(l @ B.values @ s)

    G = float(cfg.gross_target)
    eps = 1e-14

    # γ-aware conditional risks (for between-sleeve budgeting)
    Ainv = safe_pinv(A.values)
    Cinv = safe_pinv(C.values)
    A_c = symmetrize(A.values - cfg.gamma * (B.values @ Cinv @ B.values.T))
    C_c = symmetrize(C.values - cfg.gamma * (B.values.T @ Ainv @ B.values))
    var_cond_L = float(l @ A_c @ l)
    var_cond_L = max(var_cond_L, eps)
    var_cond_S = float(s @ C_c @ s)
    var_cond_S = max(var_cond_S, eps)

    # α, β from mode
    if cfg.between_group_mode == 'dollar_neutral':
        alpha = G / 2.0
        beta = G / 2.0
    elif cfg.between_group_mode == 'min_var_fixed_gross':
        denom = (vL + vS - 2.0 * cLS)
        if denom <= eps:
            alpha = G / 2.0
        else:
            alpha = G * max(vS - cLS, 0.0) / max(denom, eps)
        alpha = float(np.clip(alpha, 0.0, G))
        beta = G - alpha
    elif cfg.between_group_mode == 'schur_conditional_risk':
        invrisk_L = 1.0 / np.sqrt(var_cond_L)
        invrisk_S = 1.0 / np.sqrt(var_cond_S)
        alpha = G * invrisk_L / (invrisk_L + invrisk_S)
        beta = G - alpha
    else:
        raise ValueError("Unknown between_group_mode.")

    # Enforce net target if requested (overrides prior α/β)
    if cfg.respect_net_target:
        alpha = (G + cfg.net_target) / 2.0
        beta  = (G - cfg.net_target) / 2.0
        if alpha < 0 or beta < 0:
            alpha = max(alpha, 0.0)
            beta = max(beta, 0.0)
            s_ab = alpha + beta
            if s_ab > 0:
                alpha = G * alpha / s_ab
                beta = G * beta / s_ab
            else:
                alpha = beta = G / 2.0

    # Asymmetric per-name caps
    capL = cfg.max_long_abs_weight
    capS = cfg.max_short_abs_weight

    if (capL is not None) or (capS is not None):
        nL = len(long_idx)
        nS = len(short_idx)
        if capL is not None and alpha > nL * capL + 1e-12:
            raise ValueError(
                f"Infeasible long cap {capL:.4f}: α={alpha:.4f} but max feasible is {nL*capL:.4f} with nL={nL}. "
                f"Reduce gross_target / increase long cap / add names."
            )
        if capS is not None and beta > nS * capS + 1e-12:
            raise ValueError(
                f"Infeasible short cap {capS:.4f}: β={beta:.4f} but max feasible is {nS*capS:.4f} with nS={nS}. "
                f"Reduce gross_target / increase short cap / add names."
            )

        if capL is not None:
            w_long = project_to_capped_simplex(l, total=alpha, cap=float(capL))
        else:
            w_long = alpha * l

        if capS is not None:
            w_short_abs = project_to_capped_simplex(s, total=beta, cap=float(capS))
            w_short = -w_short_abs
        else:
            w_short = -beta * s
    else:
        w_long  = +alpha * l
        w_short = -beta  * s

    # Assemble final vector
    w = np.zeros(cov.shape[0])
    w[long_idx]  = w_long
    w[short_idx] = w_short

    # Diagnostics
    port_var_daily = float(w @ cov.values @ w)
    ann_vol = np.sqrt(max(port_var_daily, 0.0)) * np.sqrt(cfg.annualize_factor)
    gross = float(np.sum(np.abs(w)))
    net   = float(np.sum(w))

    diag = {
        "vL": vL, "vS": vS, "cLS": cLS,
        "alpha_long_gross": float(alpha), "beta_short_gross": float(beta),
        "gross_exposure": gross, "net_exposure": net,
        "annualized_vol": ann_vol,
        "gamma": cfg.gamma,
        "var_cond_L_gamma": var_cond_L,
        "var_cond_S_gamma": var_cond_S,
        "max_long_abs_weight": capL,
        "max_short_abs_weight": capS
    }
    return w, diag, A, C

# ========================================
# MAIN ORCHESTRATION FUNCTION
# ========================================
def build_schur_hmv_from_tickers(long_tickers, short_tickers,
                                 gamma=1.0,
                                 between_group_mode='schur_conditional_risk',
                                 gross_target=1.0,
                                 net_target=0.0,
                                 respect_net_target=True,
                                 min_valid_frac=0.9,
                                 intra_method='hmv',
                                 max_long_abs_weight=None,
                                 max_short_abs_weight=None):
    all_tickers = list(dict.fromkeys([*long_tickers, *short_tickers]))
    prices, kept, dropped = fetch_prices_yf(all_tickers, period="1y", interval="1d", min_valid_frac=min_valid_frac)
    if prices.empty:
        raise ValueError("No price data fetched. Check tickers or network.")

    rets = compute_returns(prices, log=True).dropna(axis=1, how='any')
    names = list(rets.columns)

    long_ticks = [t for t in long_tickers if t in names]
    short_ticks = [t for t in short_tickers if t in names]
    if not long_ticks or not short_ticks:
        raise ValueError("Need at least one valid ticker on EACH side after data cleaning.")

    idx = {t: i for i, t in enumerate(names)}
    long_idx = [idx[t] for t in long_ticks]
    short_idx = [idx[t] for t in short_ticks]

    cov_all = rets.cov()

    cfg = SchurHMVConfig(
        between_group_mode=between_group_mode,
        gross_target=gross_target,
        net_target=net_target,
        respect_net_target=respect_net_target,
        gamma=float(np.clip(gamma, 0.0, 1.0)),
        max_long_abs_weight=max_long_abs_weight,
        max_short_abs_weight=max_short_abs_weight
    )

    w_vec, diag, cov_long, cov_short = schur_hmv_allocate(cov_all, long_idx, short_idx, cfg, intra_method=intra_method)
    ordered_names = long_ticks + short_ticks
    weights = pd.Series(w_vec, index=names).loc[ordered_names]

    # With caps active, do NOT rescale globally (caps would break).
    if (max_long_abs_weight is None) and (max_short_abs_weight is None):
        if weights.abs().sum() > 0:
            weights *= (gross_target / weights.abs().sum())

    # Recompute diagnostics using final weights
    w_full = weights.reindex(cov_all.index).fillna(0).values
    ann_vol = np.sqrt(max(float(w_full @ cov_all.values @ w_full), 0.0)) * np.sqrt(cfg.annualize_factor)
    diag["annualized_vol"] = ann_vol
    diag["gross_exposure"] = float(np.sum(np.abs(w_full)))
    diag["net_exposure"] = float(np.sum(w_full))

    # Per-name risk contributions (for used names, using combined cov)
    cov_used = cov_all.loc[ordered_names, ordered_names].copy()
    risk_contrib_df = compute_risk_contributions(
        cov=cov_used,
        weights=weights,
        long_tickers=long_ticks,
        short_tickers=short_ticks
    )

    info = {
        "dropped_for_data_insufficiency": dropped,
        "used_tickers": ordered_names,
        "final_weights": weights.sort_values(ascending=False),
        "diagnostics": diag,
        "risk_contributions": risk_contrib_df.sort_values("PctRC", ascending=False),
        "cov_all": cov_used,
        "cov_long": cov_long,
        "cov_short": cov_short
    }
    return info

# ========================================
# PRESET TICKER LISTS
# ========================================
PRESET_LONG_TICKERS = [
    "JPM","6758.T","PEP","SHECF","MSFT","GOOGL","ASML","CPRT","COST",
    "TSM","OR.PA","AMZN","ABNB","MC.PA","DDOG","MCO","IBKR","MCD","WFC",
    "BKNG","ABBN.SW","SPGI","FTNT","SYK","SAF.PA"
]

PRESET_SHORT_TICKERS = [
    "HEIA.AS","JCI","CBK.DE","CTAS","SAP.DE","BTI","HWM","GEV","SCHW","SBUX","BUD",
    "WSM","CEG","HEI.DE","RL","RCL","GLE.PA","DBK.DE","FTE.MU","DIS","MAR","IBM",
    "AAPL","DLTR","INFY","MMC"
]

# ========================================
# TICKER VALIDATION
# ========================================
@st.cache_data(ttl=3600)
def validate_ticker(ticker: str) -> tuple:
    """Validate if ticker exists in yfinance"""
    try:
        info = yf.Ticker(ticker.upper()).info
        if info and 'symbol' in info:
            return True, info.get('longName', ticker)
        return False, "Ticker not found"
    except:
        return False, "Validation error"

# ========================================
# MAIN STREAMLIT APP
# ========================================
def main():
    st.title("📊 Schur-HRP Portfolio Optimizer")
    st.markdown("### Hierarchical Risk Parity with Long/Short Allocation")
    
    # ========================================
    # SIDEBAR: PARAMETERS
    # ========================================
    with st.sidebar:
        st.header("⚙️ Portfolio Parameters")
        
        st.subheader("Risk Parameters")
        
        gamma = st.slider(
            "Gamma (γ)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Controls conditional risk adjustment. 0=ignore correlations, 1=full Schur adjustment"
        )
        
        between_mode = st.selectbox(
            "Between-Group Mode",
            options=['schur_conditional_risk', 'dollar_neutral', 'min_var_fixed_gross'],
            index=0,
            help="Method for allocating between long and short sleeves"
        )
        
        st.subheader("Exposure Targets")
        
        gross_target = st.number_input(
            "Gross Target",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Total portfolio leverage (sum of |weights|)"
        )
        
        net_target = st.number_input(
            "Net Target",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Target net exposure (long - short)"
        )
        
        respect_net = st.checkbox(
            "Enforce Net Target",
            value=False,
            help="If checked, overrides between-group mode to achieve exact net target"
        )
        
        st.subheader("Position Limits")
        
        use_caps = st.checkbox("Use Position Caps", value=True)
        
        if use_caps:
            max_long_weight = st.number_input(
                "Max Long Weight (%)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Maximum weight per long position"
            ) / 100.0
            
            max_short_weight = st.number_input(
                "Max Short Weight (%)",
                min_value=1.0,
                max_value=20.0,
                value=4.0,
                step=0.5,
                help="Maximum |weight| per short position"
            ) / 100.0
        else:
            max_long_weight = None
            max_short_weight = None
        
        st.subheader("Data & Method")
        
        intra_method = st.selectbox(
            "Intra-Sleeve Method",
            options=['hmv', 'hrp'],
            index=0,
            help="Method for within-sleeve allocation"
        )
        
        min_valid_frac = st.slider(
            "Min Valid Data Fraction",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Minimum fraction of valid price data required"
        )
    
    # ========================================
    # MAIN AREA: TICKER SELECTION
    # ========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Long Tickers")
        
        use_preset_long = st.checkbox("Use preset long list", value=True)
        
        if use_preset_long:
            long_tickers_str = st.text_area(
                "Long tickers (comma-separated)",
                value=", ".join(PRESET_LONG_TICKERS),
                height=100,
                help="Enter tickers separated by commas"
            )
        else:
            long_tickers_str = st.text_area(
                "Long tickers (comma-separated)",
                value="",
                height=100,
                placeholder="e.g., MSFT, GOOGL, NVDA",
                help="Enter tickers separated by commas"
            )
    
    with col2:
        st.subheader("📉 Short Tickers")
        
        use_preset_short = st.checkbox("Use preset short list", value=True)
        
        if use_preset_short:
            short_tickers_str = st.text_area(
                "Short tickers (comma-separated)",
                value=", ".join(PRESET_SHORT_TICKERS),
                height=100,
                help="Enter tickers separated by commas"
            )
        else:
            short_tickers_str = st.text_area(
                "Short tickers (comma-separated)",
                value="",
                height=100,
                placeholder="e.g., AAPL, DIS, IBM",
                help="Enter tickers separated by commas"
            )
    
    # Parse tickers
    long_tickers = [t.strip().upper() for t in long_tickers_str.split(",") if t.strip()]
    short_tickers = [t.strip().upper() for t in short_tickers_str.split(",") if t.strip()]
    
    # Validation option
    if st.checkbox("Validate tickers before optimization", value=False):
        with st.spinner("Validating tickers..."):
            invalid_tickers = []
            for ticker in long_tickers + short_tickers:
                valid, _ = validate_ticker(ticker)
                if not valid:
                    invalid_tickers.append(ticker)
            
            if invalid_tickers:
                st.warning(f"⚠️ Invalid tickers found: {', '.join(invalid_tickers)}")
    
    # ========================================
    # CALCULATE BUTTON
    # ========================================
    if st.button("🚀 Calculate Portfolio", type="primary", use_container_width=True):
        
        if not long_tickers or not short_tickers:
            st.error("Please enter at least one ticker for both long and short lists")
            return
        
        with st.spinner("Fetching data and optimizing portfolio..."):
            try:
                # Run optimization
                result = build_schur_hmv_from_tickers(
                    long_tickers=long_tickers,
                    short_tickers=short_tickers,
                    gamma=gamma,
                    between_group_mode=between_mode,
                    gross_target=gross_target,
                    net_target=net_target,
                    respect_net_target=respect_net,
                    min_valid_frac=min_valid_frac,
                    intra_method=intra_method,
                    max_long_abs_weight=max_long_weight,
                    max_short_abs_weight=max_short_weight
                )
                
                # Store in session state
                st.session_state['result'] = result
                st.session_state['long_tickers'] = long_tickers
                st.session_state['short_tickers'] = short_tickers
                st.session_state['parameters'] = {
                    'gamma': gamma,
                    'between_mode': between_mode,
                    'gross_target': gross_target,
                    'net_target': net_target,
                    'respect_net': respect_net,
                    'min_valid_frac': min_valid_frac,
                    'intra_method': intra_method,
                    'max_long_weight': max_long_weight,
                    'max_short_weight': max_short_weight
                }
                
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                return
    
    # ========================================
    # DISPLAY RESULTS
    # ========================================
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Portfolio Weights", 
            "🌳 Dendrogram", 
            "📈 Risk Analysis",
            "🔗 Correlation Matrix",
            "📥 Export"
        ])
        
        with tab1:
            st.subheader("Portfolio Weights")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Gross Exposure", f"{result['diagnostics']['gross_exposure']:.1%}")
            with col2:
                st.metric("Net Exposure", f"{result['diagnostics']['net_exposure']:.1%}")
            with col3:
                st.metric("Annual Volatility", f"{result['diagnostics']['annualized_vol']:.1%}")
            with col4:
                st.metric("Used Tickers", len(result['used_tickers']))
            
            # Weights table
            weights_df = pd.DataFrame({
                'Ticker': result['final_weights'].index,
                'Weight (%)': result['final_weights'].values * 100,
                'Side': ['Long' if w > 0 else 'Short' for w in result['final_weights'].values],
                'Abs Weight (%)': np.abs(result['final_weights'].values) * 100
            })
            
            # Interactive plot
            fig = px.bar(
                weights_df.sort_values('Weight (%)', ascending=True),
                x='Weight (%)',
                y='Ticker',
                orientation='h',
                color='Side',
                color_discrete_map={'Long': 'green', 'Short': 'red'},
                title="Portfolio Weights",
                height=max(400, len(weights_df) * 20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(
                weights_df.sort_values('Abs Weight (%)', ascending=False),
                use_container_width=True
            )
            
            # Dropped tickers
            if result['dropped_for_data_insufficiency']:
                st.warning(f"⚠️ Dropped tickers (insufficient data): {', '.join(result['dropped_for_data_insufficiency'])}")
        
        with tab2:
            st.subheader("Hierarchical Clustering Dendrogram")
            
            try:
                # Create dendrogram
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Build dendrogram data
                cov = result['cov_all']
                
                # Check if we have enough data
                if len(cov) < 2:
                    st.error("Need at least 2 assets to create a dendrogram")
                else:
                    corr = cov_to_corr(cov)
                    dist = correl_dist(corr.values)
                    Z = linkage(squareform(dist, checks=False), method='single')
                    
                    # Add weights to labels
                    weights = result['final_weights']
                    labels = [f"{t} ({weights[t]*100:+.2f}%)" for t in cov.index]
                    
                    # Plot
                    dendro = dendrogram(
                        Z,
                        labels=labels,
                        ax=ax,
                        orientation='top',
                        leaf_font_size=9
                    )
                    
                    # Color labels by side
                    for i, tick in enumerate(ax.get_xticklabels()):
                        ticker = tick.get_text().split(' ')[0]
                        if ticker in st.session_state.get('long_tickers', []):
                            tick.set_color('green')
                        else:
                            tick.set_color('red')
                        tick.set_rotation(35)
                        tick.set_rotation_mode("anchor")
                        tick.set_horizontalalignment("right")
                    
                    ax.set_title("Portfolio Dendrogram (Single Linkage)")
                    ax.set_ylabel("Distance")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)  # Important: close the figure
                    
                    # Cluster analysis
                    st.subheader("Cluster Analysis")
                    cut_distance = st.slider("Cut Distance", 0.2, 0.8, 0.45, 0.05)
                    clusters = fcluster(Z, t=cut_distance, criterion="distance")
                    
                    cluster_df = pd.DataFrame({
                        'Ticker': cov.index,
                        'Cluster': clusters,
                        'Weight (%)': weights.values * 100
                    })
                    
                    st.write(f"Number of clusters at cut distance {cut_distance:.2f}: {len(np.unique(clusters))}")
                    
                    # Show clusters
                    for cluster_id in np.unique(clusters):
                        cluster_tickers = cluster_df[cluster_df['Cluster'] == cluster_id]['Ticker'].tolist()
                        cluster_weights = cluster_df[cluster_df['Cluster'] == cluster_id]['Weight (%)'].sum()
                        st.write(f"**Cluster {cluster_id} (Total Weight: {cluster_weights:.2f}%):** {', '.join(cluster_tickers)}")
                    
            except Exception as e:
                st.error(f"Error creating dendrogram: {str(e)}")
        
        with tab3:
            st.subheader("Risk Analysis")
            
            # Get and clean risk contributions
            risk_df = result['risk_contributions'].copy()
            risk_df = risk_df.reset_index(drop=True)
            
            # Ensure numeric columns are numeric
            numeric_cols = ['Weight', 'MRC', 'RC', 'PctRC', 'AbsWeight', 'AbsRC', 'AbsPctRC']
            for col in numeric_cols:
                if col in risk_df.columns:
                    risk_df[col] = pd.to_numeric(risk_df[col], errors='coerce').fillna(0)
            
            # Try to create treemap, with fallback options
            chart_created = False
            
            # Try sunburst first (more stable than treemap)
            try:
                fig = px.sunburst(
                    risk_df,
                    path=['Side', 'Ticker'],
                    values='AbsPctRC',
                    title="Risk Contribution by Position",
                    color='PctRC',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                chart_created = True
            except:
                pass
            
            # If sunburst fails, try bar chart
            if not chart_created:
                try:
                    fig = px.bar(
                        risk_df.sort_values('AbsPctRC', ascending=True),
                        x='AbsPctRC',
                        y='Ticker',
                        color='Side',
                        orientation='h',
                        title="Risk Contribution by Position (%)",
                        color_discrete_map={'Long': 'green', 'Short': 'red'},
                        labels={'AbsPctRC': 'Risk Contribution (%)'}
                    )
                    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    chart_created = True
                except:
                    st.warning("Could not create risk contribution chart")
            
            # Risk metrics table (always show this)
            st.write("**Per-Position Risk Metrics:**")
            display_df = risk_df[['Ticker', 'Side', 'Weight', 'MRC', 'RC', 'PctRC']].copy()
            display_df['Weight'] = display_df['Weight'] * 100
            display_df['PctRC'] = display_df['PctRC'] * 100
            display_df.columns = ['Ticker', 'Side', 'Weight (%)', 'Marginal Risk', 'Risk Contrib', 'Risk (%)']
            
            st.dataframe(
                display_df.sort_values('Risk (%)', ascending=False, key=abs),
                use_container_width=True
            )
            
            # Diagnostics
            st.write("**Portfolio Diagnostics:**")
            diag_df = pd.DataFrame(list(result['diagnostics'].items()), columns=['Metric', 'Value'])
            st.dataframe(diag_df, use_container_width=True)
        
        with tab4:
            st.subheader("Correlation Matrix")
            
            # Calculate correlation
            corr_matrix = cov_to_corr(result['cov_all'])
            
            # Interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
                hoverongaps=False,
                hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                height=700,
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation statistics
            st.write("**Correlation Statistics:**")
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Correlation", f"{np.mean(corr_values):.3f}")
            with col2:
                st.metric("Median Correlation", f"{np.median(corr_values):.3f}")
            with col3:
                st.metric("Max Correlation", f"{np.max(corr_values):.3f}")
            with col4:
                st.metric("Min Correlation", f"{np.min(corr_values):.3f}")
        
        with tab5:
            st.subheader("Export Results")
            
            # Prepare Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Weights sheet
                weights_df.to_excel(writer, sheet_name='Weights', index=False)
                
                # Risk contributions
                risk_df.to_excel(writer, sheet_name='RiskContrib', index=False)
                
                # Diagnostics
                diag_df = pd.DataFrame(list(result['diagnostics'].items()), columns=['Metric', 'Value'])
                diag_df.to_excel(writer, sheet_name='Diagnostics', index=False)
                
                # Parameters
                params = st.session_state.get('parameters', {})
                params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                
                # Dropped tickers
                dropped_df = pd.DataFrame({'Dropped Tickers': result['dropped_for_data_insufficiency']})
                dropped_df.to_excel(writer, sheet_name='Dropped', index=False)
            
            # Download button
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # CSV option for weights
            csv = weights_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Weights CSV",
                data=csv,
                file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # JSON option for complete results
            import json
            json_data = {
                'weights': result['final_weights'].to_dict(),
                'diagnostics': result['diagnostics'],
                'dropped_tickers': result['dropped_for_data_insufficiency'],
                'parameters': st.session_state.get('parameters', {})
            }
            json_str = json.dumps(json_data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON Results",
                data=json_str,
                file_name=f"portfolio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
