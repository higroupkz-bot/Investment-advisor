"""Риск-движок MAC3/Barra: факторная модель, ковариация, метрики портфеля.

Арх-3.10: выделен из `finance/investment_logic.py`, который остался ФАСАДОМ —
ни один существующий импорт не изменился. Здесь живут ВСЕ модульные константы и
хелперы движка, поэтому `portfolio_manager` берёт нужное отсюда ОТНОСИТЕЛЬНЫМ
импортом: `tests/test_layering.py` такие импорты намеренно не считает
нарушением («относительный импорт — внутри пакета, это норма»), и приватные
имена переезжают без переименования.

Паспорт конвейера и порядок стадий — в `portfolio_manager.py`.
"""

import os
import hashlib
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf

from finance.broker_api import RealPortfolioRequired
from freedom_portfolio import TradernetClient, get_history_frame
from freedom_portfolio.history import HistoryResult
# Арх-2: SSOT ФОРМЫ возврата `analyze_all` (35 ключей, 8 потребителей).
# Слой L0 — только typing, без логики и без обратных импортов, цикла нет.
# Новый ключ добавляется СНАЧАЛА туда: `tests/test_contracts_results.py`
# сверяет множества по AST и падает, если они разошлись.
from finance.contracts import AnalyzeResults

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Antigravity_RiskEngine")


def _safe_num(v, default=None):
    """Число или `default` — NaN/None/мусор не должны ронять конверсию (−37)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default        # NaN != NaN

# Sprint-5 Task 4 — mandate → Black-Litterman constraint table.  Maps the
# canonical 3-state risk mandate to the optimiser's risk-aversion (δ), turnover
# cap (active share) and per-name weight cap.  Conservative ⇒ less aggressive
# tilts, less turnover and tighter single-name caps; Aggressive ⇒ the reverse.
_MANDATE_BL_CONSTRAINTS: dict[str, dict[str, float]] = {
    "CONSERVATIVE": {"risk_aversion": 4.0, "max_active_share": 0.15, "max_single_weight": 0.10},
    "MODERATE":     {"risk_aversion": 2.5, "max_active_share": 0.25, "max_single_weight": 0.20},
    "AGGRESSIVE":   {"risk_aversion": 2.0, "max_active_share": 0.35, "max_single_weight": 0.30},
}

# Multi-period (1м / 3м / 6м / 12м / YTD) returns live in a separate module
# so they can be unit-tested without pulling in sklearn / engine dependencies.
from finance.period_returns import (
    MIN_OVERLAP_TDAYS as _MIN_OVERLAP_TDAYS,
    compute_period_returns_table as _compute_period_returns_table,
    build_portfolio_log_returns as _build_portfolio_log_returns,
    aggregate_log_returns as _aggregate_log_returns,
    apply_margin_financing as _apply_margin_financing,
    compute_benchmark_stats as _compute_benchmark_stats,
)
# Stress engine (parametric factor shocks) — also sklearn-free.
from finance.stress import (
    run_stress_scenarios as _run_stress_scenarios,
    STRESS_TEST_DISCLAIMER as _STRESS_TEST_DISCLAIMER,
)
# Base-currency layer (H2): reporting-currency-aware FX transformation +
# currency-matched RFR with geometric daily compounding (H3).
from finance.currency import (
    ReportingCurrency,
    convert_price_matrix,
    daily_rfr_geometric,
    get_rfr_for_currency,
    infer_currencies_for_tickers,
)
# Simulator (Black-Litterman "after-plan" portfolio metrics) — sklearn-free.
from finance.simulate import simulate_after_plan as _simulate_after_plan
# FRED macro feed (yield curve / HY spread / PMI / VIX / breakeven) — also
# sklearn-free; gracefully degrades when FRED_API_KEY is missing or network
# is unreachable.
from services.macro_data import MacroFeed as _MacroFeed
# P-5 (audit risk-methodology): small-window confidence intervals — χ² for
# the sample σ, Fisher z for correlations.  Dependency-light (stdlib only).
from finance.inference import (
    fisher_rho_ci as _fisher_rho_ci,
    sigma_ci_multiplier as _sigma_ci_multiplier,
)

# P-3 (audit A2/D2): historical VaR/CVaR quantiles need ~100 observations for
# a stable 95% tail; below this the point estimates are published with an
# explicit "insufficient_history" reliability verdict instead of silently.
MIN_RELIABLE_TAIL_OBS = 100

# P-4 (audit B2, roadmap METHODOLOGY_SPARSE_AND_LEVERAGED §2.2): Vasicek
# shrinkage of regression betas toward a prior — GATED, default OFF, because
# it changes Σ on staggered books and needs its own validation pass before
# production.  w = n/(n+n₀); prior: Market → 1.0 (or the ETP's leverage L for
# registry names), style/EM factors → 0.0.
BETA_SHRINKAGE_ENV = "BETA_SHRINKAGE"
BETA_SHRINKAGE_PRIOR_OBS_ENV = "BETA_SHRINKAGE_PRIOR_OBS"


def beta_shrinkage_enabled() -> bool:
    return str(os.getenv(BETA_SHRINKAGE_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on")


# ── BLOCK 3.5 — hierarchical factor orthogonalization (gated) ────────────────
# The factor set mixes a CORE macro axis (Market, Rates, Commodities) with
# long-only style ETFs (Momentum/Value/Quality/Size/Volatility) and EM proxies
# that all carry the SAME market/rates beta — so the raw factor covariance F is
# near-collinear (condition number κ ≈ 62 in production), the Ridge betas get
# unstable, and B·F·Bᵀ double-counts the shared market risk.
#
# Rather than PCA (which would destroy the NAMED factors the report depends on),
# we residualize each style/EM factor against its CORE PARENT via OLS — a
# hierarchical Gram-Schmidt: «основные макро-факторы → суб-факторы стилей».
# Each factor keeps its name and its own mean; only the redundant shared beta is
# removed, so "Momentum" becomes MARKET-NEUTRAL momentum, etc.  κ drops sharply
# and the style premia stop being counted twice.
#
# BLOCK 3 (2026-06-26): multicollinearity fix promoted to DEFAULT-ON.  The
# production factor set carried κ≈60.8 (max|corr|≈0.89) — the style/EM betas
# were near-interpolating noise because their shared market/rates beta was
# counted twice.  The hierarchical residualisation below is the effective math
# solution (chosen over PCA precisely BECAUSE it preserves the named factors the
# report depends on): it removes only the redundant shared beta, dropping κ
# below the 30 collinearity threshold while every factor keeps its name (and,
# since H-1 `§−134`, its OWN market-neutral premium rather than the raw mean).
# The env var now ONLY exists as an escape hatch — set FACTOR_ORTHOGONALIZE=0 to
# restore the legacy raw-factor decomposition; any other value (incl. unset)
# orthogonalises.
FACTOR_ORTHOGONALIZE_ENV = "FACTOR_ORTHOGONALIZE"
_FACTOR_CORE = ("Market", "Rates", "Commodities")
# Each non-core factor → the core parent(s) to neutralise its shared beta against.
_FACTOR_PARENTS = {
    "Momentum":   ("Market",),
    "Value":      ("Market",),
    "Quality":    ("Market",),
    "Size":       ("Market",),
    "Volatility": ("Market",),
    "EM_Equity":  ("Market",),
    "EM_Bond":    ("Rates",),
}


def factor_orthogonalize_enabled() -> bool:
    # DEFAULT-ON (BLOCK 3): unset → enabled.  Only an explicit off-switch
    # (0/false/no/off) restores the legacy collinear decomposition.
    return str(os.getenv(FACTOR_ORTHOGONALIZE_ENV, "on")).strip().lower() not in (
        "0", "false", "no", "off", "")


# ── F-23 / P-1 — daily-reset leveraged ETPs: structural variance drag ────────
# A k×-daily-reset product's log return is ≈ k·r_u − ½k(k−1)σ_u² per day: the
# −½k(k−1)σ_u² variance drag is CONTRACTUAL (rebalancing cost), not luck.  In
# the factor regression that drag lands in the intercept α̂ — which the F-14
# forward panel rightly EXCLUDES for ordinary names (realised idiosyncratic
# drift is not a persistent premium).
#
# P-1 (2026-07-13, audit risk-methodology): the name-only registry moved to
# `finance/leveraged.py` and now carries the leverage multiplier L, the
# underlying and the expense ratio.  For names with a KNOWN L the forward
# drag is the CONTRACTUAL −½L(L−1)σ_u² (σ_u = σ_ETP/|L| from the ETP's own
# daily history) plus the fee −ER/252, disclosed separately (C5).  Names in
# the registry WITHOUT parameters keep the F-23 empirical fallback
# min(α̂, 0).  Either way the adjustment can only LOWER a forecast (the
# anti-218% guard is preserved).  Extend via LEVERAGED_ETP_EXTRA (names) or
# LEVERAGED_ETP_PARAMS (full parameters) without a redeploy.
from finance.leveraged import (
    contractual_drag_daily as _contractual_drag_daily,
    etp_info as _etp_info,
    is_leveraged_etp as _is_leveraged_etp_impl,
)


def _is_leveraged_etp(ticker: str) -> bool:
    """True when the BASE symbol (suffix stripped) is a known daily-reset
    leveraged/inverse ETP (registry finance/leveraged.py or env extras).
    Kept as the historical name — tests and callers import it from here."""
    return _is_leveraged_etp_impl(ticker)


def apply_leveraged_drag(expected_log_daily: "np.ndarray",
                         tickers,
                         alpha_daily) -> "np.ndarray":
    """F-23: add min(α̂_daily, 0) to the forward daily log expectation of
    leveraged-ETP tickers only.  Pure function so the policy is unit-testable:
    ordinary names and positive alphas pass through untouched.  Retained as
    the EMPIRICAL fallback for registry names without a known multiplier —
    `apply_leveraged_forward` below routes between this and the contractual
    formula."""
    out = np.asarray(expected_log_daily, dtype=float).copy()
    alpha = pd.to_numeric(pd.Series(list(alpha_daily)), errors="coerce")\
              .fillna(0.0).values
    for i, t in enumerate(tickers):
        if i < len(out) and i < len(alpha) and _is_leveraged_etp(t):
            out[i] += min(float(alpha[i]), 0.0)
    return out


def apply_leveraged_forward(expected_log_daily: "np.ndarray",
                            tickers,
                            alpha_daily,
                            sigma_daily_map: "dict | None" = None,
                            ) -> "tuple[np.ndarray, dict]":
    """P-1: forward adjustment for daily-reset leveraged ETPs, with the
    contractual/empirical split disclosed per ticker.

    For each registry ticker:
      • L known AND σ_ETP available → contractual: drag = −½L(L−1)(σ_ETP/|L|)²,
        fee = −ER/252 — applied INSTEAD of the α̂ fallback (α̂ already contains
        the realised drag; applying both would double-count);
      • otherwise → empirical F-23 fallback min(α̂, 0).
    Ordinary names pass through untouched.  Every adjustment is ≤ 0.

    Returns (adjusted_array, details) where details maps ticker →
    {method, L, underlying, drag_daily, fee_daily, drag_ann_pct, fee_ann_pct}
    for the QC panel (audit C5: drag and cost-of-leverage are separate rows).
    """
    out = np.asarray(expected_log_daily, dtype=float).copy()
    alpha = pd.to_numeric(pd.Series(list(alpha_daily)), errors="coerce")\
              .fillna(0.0).values
    sigma_daily_map = sigma_daily_map or {}
    details: dict[str, dict] = {}
    for i, t in enumerate(tickers):
        if i >= len(out) or i >= len(alpha):
            continue
        info = _etp_info(t)
        if info is None:
            continue
        sigma = sigma_daily_map.get(str(t))
        if info.leverage is not None and sigma is not None and \
                np.isfinite(float(sigma)) and float(sigma) > 0:
            drag, fee = _contractual_drag_daily(
                info.leverage, float(sigma), info.expense_ratio)
            out[i] += drag + fee
            details[str(t)] = {
                "method":       "contractual",
                "L":            float(info.leverage),
                "underlying":   info.underlying,
                "drag_daily":   round(drag, 8),
                "fee_daily":    round(fee, 8),
                # Annualised display twins (log→simple on each component so
                # the QC panel reads in pp/yr without re-deriving).
                "drag_ann_pct": round((np.exp(drag * 252) - 1.0) * 100, 2),
                "fee_ann_pct":  round((np.exp(fee * 252) - 1.0) * 100, 2),
            }
        else:
            emp = min(float(alpha[i]), 0.0)
            out[i] += emp
            details[str(t)] = {
                "method":       "empirical_alpha",
                "L":            (float(info.leverage)
                                 if info.leverage is not None else None),
                "underlying":   info.underlying,
                "drag_daily":   round(emp, 8),
                "fee_daily":    0.0,
                "drag_ann_pct": round((np.exp(emp * 252) - 1.0) * 100, 2),
                "fee_ann_pct":  0.0,
            }
    return out, details


def orthogonalize_factors_hierarchical(f_data: "pd.DataFrame",
                                       return_betas: bool = False):
    """
    Residualize each style/EM factor against its core macro parent(s).

    Pure + deterministic.  Returns a NEW DataFrame with identical columns/index:
    core factors pass through untouched; each child factor is replaced by the
    HEDGED series child − Σ β̂·parent (H-1, `§−134`): the shared parent beta is
    removed and the series keeps only its OWN, market-neutral premium — its
    mean is the OLS intercept.  Before H-1 the RAW child mean was added back
    (`resid + mean(child)`), so the parent's premium stayed inside the child:
    the forward counted it twice — once through the asset's parent beta, once
    through the child's.  Betas are unaffected (the asset regression has an
    intercept, a constant shift of a regressor only moves that intercept).
    Falls back to the input unchanged when the core parents are absent or
    history is too short (<10 rows).

    F-1 (2026-07-10): with ``return_betas=True`` the function ALSO returns the
    fitted child→parent OLS coefficients ``{child: {parent: beta}}``.  The
    stress engine needs them to transform its RAW-factor shock catalog into
    the SAME residual space these betas live in — otherwise a raw «Momentum
    −15%» shock (which historically INCLUDED MTUM's market component) is
    applied to a market-neutral residual beta and the market leg of the shock
    is double-counted.  See ``finance.stress.residualize_shocks``.
    """
    if f_data is None or f_data.empty or len(f_data) < 10:
        return (f_data, {}) if return_betas else f_data
    cols = list(f_data.columns)
    out = f_data.copy()
    ortho_betas: dict[str, dict[str, float]] = {}
    for child, parents in _FACTOR_PARENTS.items():
        if child not in cols:
            continue
        par = [p for p in parents if p in cols]
        if not par:
            continue
        y = f_data[child].values.astype(float)
        X = np.column_stack(
            [np.ones(len(f_data))] + [f_data[p].values.astype(float) for p in par])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            # H-1 (`§−134`): хеджированный ряд ребёнок − Σβ̂·родитель = остаток
            # + интерсепт; его среднее — СОБСТВЕННАЯ (рыночно-нейтральная)
            # премия фактора.  Прежнее `resid + mean(child)` оставляло внутри
            # премию родителя, и форвард считал её второй раз.
            out[child] = resid + float(beta[0])
            # beta[0] is the intercept; beta[1:] align with `par`.
            ortho_betas[child] = {p: float(b) for p, b in zip(par, beta[1:])}
        except Exception:
            continue
    return (out, ortho_betas) if return_betas else out


def _nearest_psd(mat: "np.ndarray", floor: float = 1e-12) -> "np.ndarray":
    """
    Project a symmetric matrix onto the nearest positive-semi-definite matrix
    (Higham-style eigenvalue clipping).

    Sprint-3 #9: the factor covariance F is a blend (0.7·EWMA + 0.3·Ledoit-Wolf)
    and is NOT guaranteed PSD; a non-PSD F can produce slightly negative w'Σw and
    biased per-asset MCTR/ERC.  This symmetrises F and clamps negative eigenvalues
    to `floor`, then rebuilds it.  When F is already PSD the operation is a no-op
    within float tolerance, so production decompositions are unchanged in the
    common case — it only repairs the rare degenerate blend.
    """
    M = np.asarray(mat, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.size == 0:
        return M
    M = 0.5 * (M + M.T)                      # symmetrise
    try:
        w, V = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return M
    if np.all(w >= -floor):                  # already PSD → leave untouched
        return M
    # C-6 (2026-08-02): починка ОСТАЁТСЯ молчаливой для математики, но факт
    # записывается — `PHASE_03 §3a.4`: «чекеры не должны ПОВТОРЯТЬ то, что
    # движок уже чинит, они должны СООБЩАТЬ об этом».  Отрицательное
    # собственное значение блендa означает вырожденную/почти вырожденную
    # факторную ковариацию, а значит неустойчивые беты и ERC.
    _nearest_psd.last_repair = {                     # type: ignore[attr-defined]
        "min_eigenvalue": float(np.min(w)),
        "n_clipped": int(np.sum(w < floor)),
        "size": int(M.shape[0]),
    }
    w_clipped = np.clip(w, floor, None)
    return (V * w_clipped) @ V.T

class MAC3RiskEngine:
    """
    Институциональный движок рисков (OMBRIOS Style).
    Внедрены: Структурная ковариация (Barra/MAC3), Euler Risk Decomposition, CVaR.
    """
    # Все тикеры используют формат Tradernet "SYMBOL.EXCHANGE".
    # KSPI → KSPI.KZ потому что Tradernet нативно котирует KZ-биржу.
    # KAP/HSBK → .IL (Лондон) — там их основная ликвидность.
    # SSOT синонимов тикера (B-1). ЕДИНСТВЕННЫЙ словарь распознавания имён:
    # парсер ручного ввода (Фаза 4) обязан звать `resolve_tickers`, а не заводить
    # свою копию — два словаря неизбежно разъедутся, и пользователь, написавший
    # «каспи», попадёт не туда, куда брокер, приславший `KSPI`.
    #
    # Инвариант, закреплённый `test_manual_ticker_synonyms.py`: синоним и канон
    # обязаны резолвиться ОДИНАКОВО. Он же ловит запись, ведущую на проксируемую
    # бумагу (см. порядок шагов в `resolve_tickers`).
    #
    # Кириллица здесь намеренно: `resolve_tickers` делает `.upper()`, а ручной
    # ввод по-русски — основной сценарий Фазы 4. Держать её в парсере значило бы
    # завести тот самый второй словарь.
    TICKER_MAP = {
        'KAZATOMPROM':   'KAP.IL',
        'КАЗАТОМПРОМ':   'KAP.IL',
        'KASPI':         'KSPI.KZ',
        'KSPI':          'KSPI.KZ',
        'КАСПИ':         'KSPI.KZ',
        'HALYK':         'HSBK.IL',
        'HSBK':          'HSBK.IL',
        'ХАЛЫК':         'HSBK.IL',
        'KAZAKHTELECOM': 'KZTK.KZ',
        'КАЗАХТЕЛЕКОМ':  'KZTK.KZ',
        # KMG: канон подтверждён по коду — `asset_taxonomy` использует `KMGZ.KZ`
        # как пример KZ-бумаги.
        'KMG':           'KMGZ.KZ',
        'KAZMUNAYGAS':   'KMGZ.KZ',
        'КМГ':           'KMGZ.KZ',
        'КАЗМУНАЙГАЗ':   'KMGZ.KZ',
        # 🔴 KEGOC: символ KASE в коде НЕ встречается, взят как общеизвестный.
        # Перед включением ручного ввода в проде (Фаза 6) его надо сверить с
        # живым листингом: неверный символ не роняет отчёт (бумага уйдёт в
        # sparse-guard и оценится ценой покупки), но МОЛЧА исключит её из
        # фактор-модели. Проверка — чекпойнт ЧК-08.1.
        'KEGOC':         'KEGC.KZ',
        'КЕГОК':         'KEGC.KZ',
        'BITCOIN':       'BTC-USD',
        'ETHEREUM':      'ETH-USD',
    }

    # Инструменты нулевого риска (или которые не прогоняются через регрессию)
    NON_RISK_ASSETS = ['USD', 'EUR', 'CASH', 'RUB', 'KZT']

    # Proxy-ETF для облигаций (формат Tradernet — все .US).
    # Класс подбирается по КРЕДИТНОМУ РИСКУ эмитента, а не по слову «bond»:
    # казахстанский долг живёт в EM-спреде, а не в кривой US Treasury/US IG.
    BOND_PROXIES = {
        'RF': 'BIL.US',    # Risk Free (1-3 Month T-Bill ETF)
        'IG': 'LQD.US',    # Investment Grade (US Corporate Bond ETF)
        'HY': 'HYG.US',    # High Yield (Junk Bond ETF)
        'EM': 'VWOB.US',   # EM-долг (Vanguard EM Government Bond, USD)
    }

    # Маркеры казахстанского/EM-происхождения в свободно введённом тикере.
    # Нужны ветке-эвристике: «BOND» в имени сам по себе не говорит, ЧЕЙ это долг,
    # а для KZ-бумаги прокси на US IG занижает спред и волатильность.
    # 'OVD' — облигации внутреннего долга (KZ), поэтому это тоже KZ-маркер.
    _EM_BOND_MARKERS = frozenset({
        'KZ', 'KAZ', 'AIX', 'OVD', 'KASE', 'KASPI', 'KSPI', 'HALYK', 'HSBK',
        'FREEDOM', 'FFIN', 'TENGE', 'KZT',
    })

    # Маппинг известных облигаций к категориям (для автоопределения).
    # KZ-суверен и KZ-корпорат — EM-класс: прежние 'RF' (BIL.US ≈ нулевой риск)
    # и 'IG' (LQD.US, развитый рынок) занижали риск казахстанского долга.
    BOND_CLASSIFICATION_MAP = {
        'KZ_GOV_BOND': 'EM',
        'US_TREASURY': 'RF',
        'KASPI_BOND':  'EM',
    }

    # Прокси для инструментов, у которых либо нет ценовой истории, либо
    # история слишком короткая/иллквидная для надёжного факторного анализа.
    # Применяется ТОЛЬКО к ковариации/корреляции — текущая цена и P&L
    # берутся из реального тикера (брокера).
    # Логика подбора прокси: похожая duration + кредит-рейтинг + валюта.
    #
    # Замысел слоя: у неликвидной бумаги нет пригодного ценового ряда (нет
    # истории, либо цена «сглажена» редкими сделками — appraisal-smoothing), и
    # оценивать по ней риск нельзя.  Вместо неё в ФАКТОРНУЮ МОДЕЛЬ подставляется
    # ликвидный индекс с похожим профилем; цена и P&L при этом берутся из
    # реального тикера (брокер), так что стоимость портфеля не искажается.
    #
    # Что изменено: AIX-ноты Freedom проксировались на LQD.US — американский
    # investment-grade корпоративный кредит.  Это развитый рынок, и он занижает
    # риск казахстанской структурной ноты: у EM-долга шире спред, выше
    # волатильность и другая реакция на risk-off.  Прокси переведены на
    # EM-инструменты, что честнее отражает природу бумаги.
    #
    # 🔴 ЖЁСТКОЕ ОГРАНИЧЕНИЕ: прокси НЕ ИМЕЕТ ПРАВА совпадать с факторным
    # тикером.  `calculate_structural_risk` строит `needed_cols` как
    # `[*factor_tickers, *resolved_assets]`; совпадение даёт ДУБЛИРОВАННУЮ
    # колонку, и присвоение факторных имён падает с
    # `ValueError: Length mismatch` — то есть отчёт не строится вообще.
    # Поэтому взяты EM-ETF, которых нет в факторной панели: EMB/EEM
    # зарезервированы под факторы EM_Bond/EM_Equity.  Инвариант закреплён
    # тестом `ProxyNeverCollidesWithFactorTest`.
    #
    # Вторая причина не брать сам фактор: тогда бета ноты к нему была бы ровно
    # 1.0 при нулевом остаточном риске — модель заявила бы, что у неликвидной
    # структурной ноты НЕТ специфического риска, то есть прямо обратное правде.
    INSTRUMENT_PROXY_MAP = {
        # Структурные ноты Astana Exchange (Freedom SPC): USD-номинал, короткая
        # duration, кредит квазисуверенного/корпоративного EM-эмитента.
        # VWOB — Vanguard EM Government Bond (USD), не входит в факторную панель.
        'AIX_SPC':           'VWOB.US',
        # Любой .AIX-тикер с иным префиксом — тот же класс риска, а не T-Bill:
        # прежний BIL.US (US T-Bills, риск ≈ 0) занижал риск неизвестной
        # AIX-бумаги до нуля.
        'AIX_DEFAULT':       'VWOB.US',
        # KZ корпоративные бонды без прямого Tradernet-маппинга.
        'KZ_CORPORATE_BOND': 'VWOB.US',
        # Неликвидные EM-акции (KAP.IL/HSBK.IL/KSPI.KZ при слишком короткой
        # серии).  VWO — Vanguard FTSE EM, вне факторной панели (там EEM).
        'EM_EQUITY_FALLBACK': 'VWO.US',
    }

    # Sector ETF index map — used for sector exposure analysis in reports.
    # Format: sector_name -> Tradernet ETF ticker.
    SECTOR_ETF_MAP = {
        'Technology':     'XLK.US',
        'Semiconductors': 'SOXX.US',
        'Finance':        'XLF.US',
        'Energy':         'XLE.US',
        'Healthcare':     'XLV.US',
        'Commodities':    'DBC.US',
        'Gold':           'GLD.US',
        'Silver':         'SLV.US',
        'Oil':            'USO.US',
        'Industrials':    'XLI.US',
        'Consumer':       'XLP.US',
        'Materials':      'XME.US',
    }

    # Ticker-to-sector classification for automatic sector detection.
    # KZ/KASE tickers are classified as EM_Kazakhstan — they are benchmarked
    # against the MSCI EM index (EEM.US), not the S&P 500.
    TICKER_SECTOR = {
        # US Technology
        'AAPL':  'Technology',
        'MSFT':  'Technology',
        'GOOGL': 'Technology',
        'META':  'Technology',
        'AMZN':  'Technology',
        'ORCL':  'Technology',
        # US Semiconductors
        'NVDA':  'Semiconductors',
        'AVGO':  'Semiconductors',
        'AMD':   'Semiconductors',
        'INTC':  'Semiconductors',
        'QBTS':  'Semiconductors',
        'SOXX':  'Semiconductors',
        'TSM':   'Semiconductors',
        # US Finance
        'JPM':   'Finance',
        'GS':    'Finance',
        'BAC':   'Finance',
        # US Energy
        'XOM':   'Energy',
        'CVX':   'Energy',
        'COP':   'Energy',
        # US Healthcare
        'JNJ':   'Healthcare',
        'UNH':   'Healthcare',
        'PFE':   'Healthcare',
        # Commodities / Precious Metals
        'GLD':   'Gold',
        'SLV':   'Silver',
        'GDX':   'Gold',
        'USO':   'Oil',
        # Kazakhstan / KASE / AIX — EM_Kazakhstan (benchmarked vs MSCI EM)
        'KSPI':  'EM_Kazakhstan',
        'HSBK':  'EM_Kazakhstan',
        'KAP':   'EM_Kazakhstan',
        'KZTK':  'EM_Kazakhstan',
        'KCEL':  'EM_Kazakhstan',
        'BAST':  'EM_Kazakhstan',
        'HRGL':  'EM_Kazakhstan',
        'KZAP':  'EM_Kazakhstan',
        # US Bond ETFs
        'TLT':   'Bonds',
        'AGG':   'Bonds',
        'BND':   'Bonds',
        'LQD':   'Bonds',
        'HYG':   'Bonds',
        'IEF':   'Bonds',
        'BIL':   'Bonds',
        'EMB':   'Bonds',
        'SHY':   'Bonds',
        'VCIT':  'Bonds',
        # EM-долг: используется прокси неликвидных AIX/KZ-бумаг, но может и
        # держаться напрямую — тогда сектор нужен для секторной панели.
        # (Прокси сам в sector_map НЕ попадает: он строится по ОРИГИНАЛЬНЫМ
        # тикерам портфеля, поэтому сектор AIX-ноты остаётся EM_Kazakhstan.)
        'VWOB':  'Bonds',
    }

    # Benchmark / catalogue ETFs fetched alongside factor ETFs.
    # NOT used as regression factors — only for benchmark_comparison + TE.
    # Includes all tickers from profile_manager.BENCHMARK_LIST so they are
    # pre-loaded in a single batch (avoids a second download pass).
    BENCHMARK_EXTRA = [
        'QQQ.US', 'AGG.US', 'URTH.US',  # broad indices
        # Factor ETFs already in factor_tickers (SPY, MTUM, VLUE, QUAL, IWM, DBC, IEF, EEM, EMB)
    ]

    def __init__(self, trading_days=252, ewma_halflife=63,
                 reporting_currency: ReportingCurrency | str | None = None,
                 fx_provider=None, risk_mandate: str = "MODERATE",
                 price_source: str = "freedom",
                 tradernet_client: TradernetClient | None = None):
        self.trading_days = trading_days
        # Источник ЦЕН (не портфеля): 'freedom' — живой Tradernet, 'demo' —
        # локальная детерминированная витрина (finance/demo_portfolio.py).
        # Дефолт сохраняет поведение всех существующих вызовов.
        self.price_source = str(price_source or "freedom").lower()
        # halflife=63 торговых дня (~3 мес) ⇒ дневной decay λ = 0.5^(1/63) ≈ 0.989.
        # Это СОЗНАТЕЛЬНО плавнее RiskMetrics λ=0.94 (что соответствовало бы
        # halflife ≈ 11 дн): факторная ковариация стабильнее, меньше шумовых
        # разворотов весов. (Sprint-5.1: прежний коммент «≈ λ=0.94» был неверен.)
        self.ewma_halflife = ewma_halflife
        # H4: investor risk mandate (CONSERVATIVE/MODERATE/AGGRESSIVE) — drives
        # the composite-risk-score weighting + CVaR base divisor.
        from finance.scoring import normalize_risk_mandate as _nrm
        self.risk_mandate = _nrm(risk_mandate)
        # Факторные прокси (формат Tradernet — все .US ETF).
        # EM_Equity (EEM.US) и EM_Bond (EMB.US) добавлены для корректного
        # моделирования казахстанских и EM-активов (.KZ/.IL тикеры).
        self.factor_tickers = {
            'Market':      'SPY.US',    # S&P 500 — глобальный рыночный фактор
            'Momentum':    'MTUM.US',
            'Value':       'VLUE.US',
            'Quality':     'QUAL.US',
            'Size':        'IWM.US',
            'Volatility':  'SPLV.US',   # Invesco S&P 500 Low Volatility — low-vol factor
            'Commodities': 'DBC.US',
            'Rates':       'IEF.US',    # 7-10y Treasury — фактор процентных ставок
            'EM_Equity':   'EEM.US',    # MSCI Emerging Markets — EM equity premium
            'EM_Bond':     'EMB.US',    # JP Morgan EM Bond ETF — EM credit/FX premium
        }
        # ── H2: Base-currency / RFR layer ────────────────────────────────────
        # Reporting currency drives BOTH the FX transformation of the price
        # matrix and the RFR applied in Sharpe/Sortino — they must agree to
        # avoid the cross-currency premium leak (USD assets vs KZ-RFR bug).
        if reporting_currency is None:
            self.reporting_currency = ReportingCurrency.from_env()
        elif isinstance(reporting_currency, ReportingCurrency):
            self.reporting_currency = reporting_currency
        else:
            self.reporting_currency = ReportingCurrency(str(reporting_currency).upper())
        self.current_rfr_annual, self.rfr_source = get_rfr_for_currency(self.reporting_currency)
        # Geometric daily RFR (H3) — cached once, used by both Sharpe and
        # Sortino's downside filter so they stay consistent.
        self.current_rfr_daily = daily_rfr_geometric(self.current_rfr_annual,
                                                     trading_days)
        # FX provider hook: callable (base_ccy, quote_ccy) -> pd.Series, or
        # None to bypass FX (USD-only portfolios stay on the fast path).
        # When None AND a real FRED_API_KEY is set in env, we auto-wire the
        # FredFxProvider so Cloud Run picks up DEXKZUS without any caller
        # change.  Import is local to avoid pulling `requests` into the
        # sklearn-free unit-test path.
        if fx_provider is None:
            try:
                from services.fx_feed import default_fx_provider
                fx_provider = default_fx_provider()
            except Exception as exc:                  # pragma: no cover
                logger.warning("Auto FX provider unavailable: %s", exc)
                fx_provider = None
        self.fx_provider = fx_provider
        # Кешируемый клиент Tradernet — переиспользуется между запросами одного
        # вызова (внутри analyze_all). Без явного клиента ключи читаются из env
        # (Cloud Run secret). Явный клиент — ключи САМОГО клиента брокера (веб);
        # для `manual`/`demo` он не используется: провайдер цен его не просит (I-12).
        self._tradernet_client: TradernetClient | None = tradernet_client
        # Провайдер цен — ленивый, выбирается по self.price_source (Фаза 2).
        self._price_provider = None
        # Audit trail filled by `_apply_fx_conversion` — surfaced into the
        # report's QC panel so the user can verify what was converted.
        self._last_fx_records: list = []
        # −37: кэш курсов «валюта → базовая» на один прогон отчёта (курс обязан
        # быть ОДИН для всех строк, иначе цена покупки и стоимость разъедутся).
        self._fx_rate_cache: dict = {}
        # Даты наблюдения тех же курсов — для подписи под конверсией
        # (`fx_quote_to_base`).  Живут рядом с кэшем курсов и заполняются в том
        # же месте: разъехавшиеся курс и дата были бы хуже отсутствующей даты.
        self._fx_rate_dates: dict = {}
        # C-6: заполняется в `calculate_structural_risk`, если PSD-проекция
        # реально чинила факторную ковариацию (вырожденный бленд).
        self._last_psd_repair = None
        self._last_asset_currencies: dict[str, str] = {}

    def math_firewall(self, df):
        """
        Защита от 'битых' данных (галлюцинаций API).

        Replaces ±Inf with NaN BEFORE the directional fill so a single
        infinite price tick can't survive the fill and poison the covariance
        matrix or Sharpe/CVaR downstream.  ffill then carries the last finite
        value across interior gaps and up to the frame's end.

        F-6 (2026-07-10): the old trailing ``.bfill()`` is REMOVED — it filled
        LEADING NaNs by copying an asset's first quote backwards over its
        pre-listing period.  Those flat phantom prices are a look-ahead bias:
        zero log-returns that shrink a young asset's σ/β toward zero and
        distort its correlations.  Leading NaNs now survive the firewall by
        design; the covariance path handles them via the row-level dropna
        (honest common-date window) plus the sparse-asset guard in
        ``calculate_structural_risk``.
        """
        cleaned = df.replace([np.inf, -np.inf], np.nan)
        return cleaned.ffill()

    # ── Bootstrap CVaR (stationary block bootstrap, Politis-Romano 1994) ─────
    @staticmethod
    def _bootstrap_cvar(returns: np.ndarray, n_boot: int = 2000,
                        alpha: float = 0.05, seed: int | None = None) -> dict:
        """
        Compute CVaR(alpha) point estimate plus 95% bootstrap CI.

        Stationary block-bootstrap with geometric block lengths (mean ≈ √N)
        preserves daily-return autocorrelation — naive iid resampling
        underestimates tail risk on serially-dependent series.

        H5 — deterministic-yet-data-driven seed
        ────────────────────────────────────────
        When `seed is None`, the seed is derived from the returns vector
        itself (`int(abs(hash(tuple(round(r, 6)))) % 1e8)`).  This keeps
        the call deterministic for unit tests (same input → same CI) but
        makes the CI move when the input window rolls forward — fixing
        the H5 "CI never widens between runs" UX bug while preserving
        reproducibility.  Tests that explicitly pin `seed=N` keep working.

        Returns: {'point': float, 'lo95': float, 'hi95': float}
        """
        if seed is None:
            # Sprint-3 #10: derive the seed with hashlib.sha256, NOT the builtin
            # hash().  Python's hash() of containers is salted per-process for
            # some types and its algorithm is an implementation detail — so the
            # same return window could seed differently across interpreters /
            # versions, breaking the "same input → same CI" contract.  sha256 of
            # the rounded byte buffer is stable on every Python/OS.  Round to 6
            # decimals first so float noise on identical inputs can't move it.
            try:
                arr = np.round(np.asarray(returns, dtype=float), 6)
                digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).digest()
                seed = int.from_bytes(digest[:8], "big") % 100_000_000
            except (TypeError, ValueError):
                seed = 42       # fall back if returns is exotic (object dtype)
        rng = np.random.default_rng(seed)
        N = len(returns)
        if N == 0:
            return {"point": 0.0, "lo95": None, "hi95": None}
        block = max(1, int(np.sqrt(N)))
        cvars = np.empty(n_boot, dtype=float)
        ret_arr = np.asarray(returns, dtype=float)
        for b in range(n_boot):
            idx = np.empty(N, dtype=np.int64)
            filled = 0
            while filled < N:
                start = int(rng.integers(0, N))
                L = int(rng.geometric(1.0 / block))
                take = min(L, N - filled)
                for k in range(take):
                    idx[filled + k] = (start + k) % N
                filled += take
            sample = ret_arr[idx]
            var_b = np.percentile(sample, alpha * 100)
            tail = sample[sample <= var_b]
            cvars[b] = float(tail.mean()) if tail.size else float(var_b)
        return {
            "point": float(np.mean(cvars)),
            "lo95":  float(np.percentile(cvars, 2.5)),
            "hi95":  float(np.percentile(cvars, 97.5)),
        }

    # ── Marginal VaR (numerical sensitivity dVaR/dw_i) ───────────────────────
    @staticmethod
    def _marginal_var(a_data: pd.DataFrame, weights: np.ndarray,
                      var_p: float = 0.05, h: float = 0.005) -> pd.Series:
        """
        Symmetric finite-difference marginal VaR per asset, BUDGET-NEUTRAL.
        dVaR/dw_i ≈ (VaR(w⁺) − VaR(w⁻)) / (2h), where w± bumps w_i by ±h and
        rescales the REST of the book so Σw stays constant — i.e. the +1%
        is funded by selling the other names pro-rata, not by adding leverage.

        (F-1 fix: the previous version bumped w_i alone, which changed gross
        exposure and measured a leveraged sensitivity — systematically
        inflating |M-VaR| versus the reallocation the report describes.)

        Result is in the same units as one-day return (decimal).  A negative
        Marginal VaR for a long position means reallocating INTO that position
        would REDUCE 1-day downside (rare, usually negatively-correlated hedges).
        """
        cols = list(a_data.columns)
        ret_mat = a_data.values
        total_w = float(weights.sum())
        out: dict[str, float] = {}

        def _bumped(i: int, dh: float) -> np.ndarray:
            w = weights.copy()
            rest = total_w - w[i]
            if rest > 1e-12:
                # Fund the bump pro-rata from the rest of the book: Σw unchanged.
                w_other_scale = (rest - dh) / rest
                w *= w_other_scale
                w[i] = weights[i] + dh
            else:
                # Single-asset book — nothing to fund from; fall back to the
                # raw bump (documented leveraged sensitivity for this edge).
                w[i] = weights[i] + dh
            return w

        for i, ticker in enumerate(cols):
            v_plus  = float(np.percentile(
                _aggregate_log_returns(ret_mat, _bumped(i, +h)), var_p * 100))
            v_minus = float(np.percentile(
                _aggregate_log_returns(ret_mat, _bumped(i, -h)), var_p * 100))
            out[ticker] = (v_plus - v_minus) / (2.0 * h)
        return pd.Series(out)

    # ── Composite Risk Score (0..100) ────────────────────────────────────────
    # Thin delegate to the single source of truth in finance.scoring.
    # Kept as a staticmethod so existing callers / tests that reference
    # MAC3RiskEngine._composite_risk_score keep working unchanged.
    @staticmethod
    def _composite_risk_score(volatility: float, cvar: float,
                              max_erc_pct: float, mandate: str = "MODERATE",
                              *, max_drawdown: float | None = None,
                              leverage_ratio: float | None = None,
                              sector_top_pct: float | None = None) -> int:
        from finance.scoring import composite_risk_score as _crs
        return _crs(volatility, cvar, max_erc_pct, mandate=mandate,
                    max_drawdown=max_drawdown, leverage_ratio=leverage_ratio,
                    sector_top_pct=sector_top_pct)

    def _get_tradernet_client(self) -> TradernetClient:
        """Lazy-init Tradernet client.  Reused across calls inside analyze_all.

        Остаётся фабрикой клиента — её использует `TradernetProvider`.  Убирать
        её при переходе на провайдеров нельзя: это единственная точка, где
        читаются сервисные ключи.
        """
        if self._tradernet_client is None:
            api_key    = (os.getenv("FREEDOM_API_KEY")    or "").strip()
            secret_key = (os.getenv("FREEDOM_API_SECRET") or "").strip()
            self._tradernet_client = TradernetClient(api_key, secret_key)
        return self._tradernet_client

    def _get_price_provider(self):
        """Провайдер цен для ЭТОГО отчёта — по источнику портфеля (Фаза 2).

        Клиент создаётся ЛЕНИВО и только для брокерского провайдера: витрина
        и ручной портфель не должны инстанцировать `TradernetClient` вовсе
        (I-12 проверяется в том числе по факту его создания).
        """
        if self._price_provider is None:
            from finance.price_providers import provider_for_source
            src = self.price_source
            client = self._get_tradernet_client() if src not in ("demo", "manual") else None
            self._price_provider = provider_for_source(src, client=client)
        return self._price_provider

    @classmethod
    def _looks_emerging_market(cls, ticker: str) -> bool:
        """True, если в имени бумаги есть маркер KZ/EM-происхождения.

        Вызывается ТОЛЬКО из облигационной ветки `resolve_tickers`, поэтому
        подстрочный поиск безопасен: коллизия потребовала бы US-инструмента, в
        имени которого одновременно есть и «BOND»/«OVD», и KZ-маркер.
        """
        t = str(ticker).upper()
        return any(marker in t for marker in cls._EM_BOND_MARKERS)

    @classmethod
    def proxy_for(cls, ticker) -> str | None:
        """SSOT: возвращает прокси-тикер, если бумага ПОДМЕНЯЕТСЯ, иначе None.

        Отличает подмену инструмента от нормализации имени.  Это разные вещи, и
        путать их нельзя:
          • `AAPL` → `AAPL.US`, `KSPI` → `KSPI.KZ`, `BITCOIN` → `BTC-USD` —
            ТА ЖЕ бумага, просто в формате провайдера.  Прокси НЕТ.
          • `FFSPC6.1028.AIX` → `VWOB.US` — ДРУГАЯ бумага, взятая вместо
            неликвидной для оценки риска.  Это прокси.

        Три потребителя обязаны читать одну функцию, иначе слои разъедутся:
          1. `resolve_tickers` — что скачивать и что класть в фактор-модель;
          2. цена позиции (`analyze_all`) — прокси НЕ ИМЕЕТ ПРАВА давать цену;
          3. раскрытие пользователю (`prefetch_market_data.proxy_map`, экран
             подтверждения ручного ввода) — подмену обязаны показать.
        """
        t = str(ticker).upper().strip()
        if t in cls.NON_RISK_ASSETS:
            return None
        # 1. Известные облигации → ETF-прокси по кредитному классу
        if t in cls.BOND_CLASSIFICATION_MAP:
            category = cls.BOND_CLASSIFICATION_MAP[t]
            return cls.BOND_PROXIES.get(category, cls.BOND_PROXIES['RF'])
        # 2. Эвристика по имени: слово «BOND» не говорит, ЧЕЙ это долг —
        #    KZ-бумага должна проксироваться на EM-долг, а не на US IG.
        if 'BOND' in t or 'OVD' in t:
            return cls.BOND_PROXIES['EM' if cls._looks_emerging_market(t) else 'IG']
        # 3. AIX (Astana International Exchange) — структурные ноты Freedom
        if t.endswith('.AIX') or 'FFSPC' in t:
            return cls.INSTRUMENT_PROXY_MAP.get(
                'AIX_SPC', cls.INSTRUMENT_PROXY_MAP['AIX_DEFAULT'])
        return None

    @classmethod
    def _canon_name(cls, ticker) -> str | None:
        """Шаг 1 нормализации: синоним → канон. `None` — это кэш.

        Отдельный шаг нужен потому, что `proxy_for` спрашивается ИМЕННО на этом
        имени: его первая ветка — точное совпадение с `BOND_CLASSIFICATION_MAP`
        (`US_TREASURY`, `KZ_GOV_BOND`), а после форматирования шага 2 ключ стал
        бы `US_TREASURY.US` и совпадение пропало бы — молча, с подменой на None
        вместо `BIL.US`.
        """
        t_str = str(ticker).upper().strip()
        if t_str in cls.NON_RISK_ASSETS:
            return None
        # Синоним → канонический тикер (B-1). Шаг идёт ПЕРВЫМ, и это не
        # косметика: пока маппинг стоял ПОСЛЕ прокси, синоним минова́л
        # проверку неликвидности, и одна и та же бумага резолвилась
        # по-разному в зависимости от написания. Замерено на живом коде:
        # синоним, ведущий на `FFSPC6.1028.AIX`, давал сам тикер, тогда
        # как канон давал прокси `VWOB.US` — то есть в фактор-модель
        # попадала бумага без ценовой истории. Сегодня ни одна запись
        # `TICKER_MAP` на проксируемую бумагу не ведёт, поэтому разрыв
        # был ЛАТЕНТНЫМ; ручной ввод (Фаза 4) его бы и активировал —
        # именно там AIX-ноты приходят по имени. `AUDIT §−70`.
        return cls.TICKER_MAP.get(t_str, t_str)

    @classmethod
    def _provider_format(cls, canon: str) -> str:
        """Шаг 2 нормализации: канон → формат провайдера (`SYM.EXCHANGE`)."""
        # Крипто (Tradernet их не торгует напрямую — оставляем *-USD формат
        # как маркер; отсутствующие колонки пропускаются без падения).
        #
        # 2026-07-28: ветка ловила ТОЛЬКО голый символ, поэтому уже
        # канонический `BTC-USD` (именно в таком виде он лежит в демо-
        # портфеле и приходит из ручного ввода) проваливался в последнюю ветку
        # и получал суффикс: `BTC-USD` → `BTC-USD.US`.  Такого инструмента нет
        # ни у одного провайдера → цена не находилась → строка молча
        # удалялась `dropna(subset=['Current_Price'])` в analyze_all, и
        # демо-витрина из 3 позиций показывала 2.  Суффикс `-USD` — тот же
        # маркер крипто, что использует `broker_api.classify_instrument`
        # (`t.endswith("-USD")` → «Крипто»), поэтому проверки согласованы.
        if canon.endswith('-USD'):
            return canon                        # уже канонический вид
        if canon in ('BTC', 'ETH', 'SOL', 'BNB'):
            return f"{canon}-USD"
        # Уже в формате SYM.EXCHANGE — оставляем.
        if '.' in canon:
            return canon
        # Голый US-тикер → добавить .US
        return f"{canon}.US"

    @classmethod
    def canonical_ticker(cls, ticker) -> str | None:
        """Имя бумаги в формате провайдера — БЕЗ подмены на прокси.

        Отделено от `resolve_tickers` для потребителя, которому нужна ТА ЖЕ
        бумага, а не её факторный заменитель: слой ввода портфеля
        (`finance/manual_portfolio.py`). Разница не косметическая — она
        решает, ЧТО именно окажется в портфеле:

          `canonical_ticker('FFSPC6.1028.AIX')` → `'FFSPC6.1028.AIX'`  (бумага)
          `resolve_tickers(['FFSPC6.1028.AIX'])` → `['VWOB.US']`       (прокси)

        Пока ручной ввод строил позицию по второму, нота ИСЧЕЗАЛА: в портфель
        попадал `VWOB` с его рыночной ценой, `Asset_Type` становился «ETF», а
        две разные ноты с одним прокси склеивались в одну позицию — то есть
        ровно тот дефект `§−38`, который прокси-слой и был призван убрать
        (инвариант «прокси влияет ТОЛЬКО на риск», `src/finance/CLAUDE.md`).

        `None` — это кэш (`NON_RISK_ASSETS`): у валюты нет тикера, и вызывающий
        обязан обработать её отдельно, а не получить строку-подделку.

        Реализация ОДНА на обе функции: обе собраны из `_canon_name` и
        `_provider_format`, и различаются ровно вопросом к `proxy_for` между
        шагами. Второй экземпляр правил нормализации разъехался бы с этим при
        первой же правке (ловушка Т-4).
        """
        canon = cls._canon_name(ticker)
        return None if canon is None else cls._provider_format(canon)

    def resolve_tickers(self, tickers):
        """
        Преобразует пользовательский тикер в формат Tradernet ("SYMBOL.EXCHANGE").

        Спецслучаи:
          - cash currencies → пропускаются (NON_RISK_ASSETS)
          - неликвидные бумаги (KZ-облигации, AIX-ноты) → ликвидный прокси
            (`proxy_for`): подмена ТОЛЬКО для фактор-модели, цена и P&L
            остаются за реальным тикером
          - крипто → '*-USD' (исторический формат, для совместимости с
            форком CCXT/Yahoo, не Tradernet — но если тикер известен через
            Tradernet, его можно подменить позже)

        Нормализация имени живёт в `_canon_name`/`_provider_format`; здесь
        остаётся ровно одно решение — брать бумагу или её прокси.
        """
        resolved = []
        proxied: list[str] = []
        for t in tickers:
            t_raw = str(t).upper().strip()
            canon = self._canon_name(t_raw)
            if canon is None:
                continue   # Cash отдельно

            # Неликвидная бумага → прокси (единая точка решения). Спрашиваем на
            # ИМЕНИ, а не на отформатированном тикере: см. `_canon_name`.
            proxy = self.proxy_for(canon)
            if proxy is not None:
                # В лог идёт то, что НАПИСАЛ пользователь, а не канон: иначе
                # провенанс подмены теряется на первом же синониме.
                proxied.append(f"{t_raw}→{proxy}")
                resolved.append(proxy)
                continue

            resolved.append(self._provider_format(canon))
        if proxied:
            logger.info("Illiquid instruments proxied for factor model only "
                        "(pricing stays on the real ticker): %s", ", ".join(proxied))
        return resolved

    def get_market_data(self, tickers, period_days: int | None = None):
        """
        Загрузка дневных закрытий через Tradernet API (replaces yfinance).

        period_days (Фаза 4 · Data Resilience §4-б): default читается из env
        ``HISTORY_LOOKBACK_DAYS``.  С 2026-07-07 дефолт = **1825 КАЛЕНДАРНЫХ
        дней ≈ 1260 ТОРГОВЫХ ≈ 5 лет** (ровно целевое окно §4-б: «1260 торговых
        ≈ 1825 кал.»).  Параметр `days` в `history.py` — КАЛЕНДАРНЫЙ
        (`today − timedelta(days=days)`), поэтому 5 торговых лет = 1825 кал., а
        НЕ 1260 кал. (то было бы ~3.45 года).  Расширение окна меняет все
        метрики; молодые активы не роняют матрицу — sparse-history дроп +
        row-level `dropna` дают общее пересечение дат (эффективное окно =
        min-история среди активов и фактор-ETF), а `min_obs` гейт защищает
        от вырожденной регрессии.  Откат на прежнее 2-летнее окно —
        `HISTORY_LOOKBACK_DAYS=730`.

        Returns (data, history_result) tuple where history_result contains
        loaded/failed/retried ticker details for user-facing messages.

        H2: After Tradernet loads native-currency prices, we transform
        every column whose native currency differs from
        `self.reporting_currency` by multiplying with the matching FX
        series.  USD-only portfolios short-circuit (no FX calls, no copy).
        """
        if period_days is None:
            try:
                # Дефолт 1825 кал.дн ≈ 1260 торговых ≈ 5 лет (§4-б Data Resilience).
                period_days = int(os.getenv("HISTORY_LOOKBACK_DAYS", "1825"))
            except (TypeError, ValueError):
                period_days = 1825
            period_days = max(90, min(3650, period_days))   # sanity clamp

        valid_tickers = self.resolve_tickers(tickers)
        all_req = list(dict.fromkeys(
            valid_tickers + list(self.factor_tickers.values()) + self.BENCHMARK_EXTRA
        ))

        # Загрузка идёт ЧЕРЕЗ ПРОВАЙДЕРА (Фаза 2): движок больше не знает, откуда
        # берутся цены.  Провайдер выбирается по источнику портфеля — там же
        # охраняется юридическая граница I-12 (данные брокера недопустимы для
        # ручного портфеля).  `ProviderResult` duck-совместим с `HistoryResult`,
        # поэтому шесть модулей-потребителей ниже не правились (I-11).
        provider = self._get_price_provider()
        logger.info("Загрузка цен через '%s' (%s) для %d инструментов",
                    provider.name, provider.convention.value, len(all_req))
        history_result = provider.fetch(all_req, days=period_days)

        if history_result.data.empty:
            logger.error("Tradernet вернул пустой набор цен — все тикеры провалились")
            return history_result.data, history_result

        firewalled = self.math_firewall(history_result.data)
        # ── H2: FX conversion to reporting currency ──────────────────────────
        converted = self._apply_fx_conversion(firewalled)
        return converted, history_result

    def _apply_fx_conversion(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the price matrix into the reporting currency (H2).

        Safety:
          • If every column is already in `reporting_currency`, returns the
            input frame unchanged (no allocation).
          • Never mutates the caller's frame: convert_price_matrix copies
            internally before writing.
          • Updates `self._last_fx_records` and `self._last_asset_currencies`
            so the report's QC panel can show what was actually converted.
        """
        asset_currencies = infer_currencies_for_tickers(prices.columns)
        # Factor ETFs are all .US → USD; explicitly pin them so an exotic
        # symbol that sneaks in doesn't trigger a spurious FX call.
        for factor_tkr in self.factor_tickers.values():
            asset_currencies.setdefault(factor_tkr, "USD")
        result = convert_price_matrix(
            prices            = prices,
            asset_currencies  = asset_currencies,
            reporting         = self.reporting_currency,
            fx_provider       = self.fx_provider,
            lag_one_day       = True,
        )
        self._last_fx_records       = result.fx_records
        self._last_asset_currencies = result.asset_currencies
        if not result.no_op:
            logger.info(
                "FX → %s applied to %d pair(s): %s",
                self.reporting_currency.value, len(result.fx_records),
                ", ".join(r.pair for r in result.fx_records),
            )
        return result.prices_base

    def fx_rate_to_base(self, currency: str) -> Optional[float]:
        """Курс `currency → reporting_currency` (МУЛЬТИПЛИКАТИВНЫЙ), либо None.

        −37 (2026-08-02).  Тот же курс и та же конвенция, что у матрицы цен
        (`convert_price_matrix`: `цена × rate`), поэтому стоимость позиции и её
        цена покупки пересчитываются ОДНИМ курсом — решение владельца
        «единый текущий курс, движение с момента покупки не учитываем».

        Зачем отдельный метод, если есть `_last_fx_records`: там лежат только
        валюты, встретившиеся в МАТРИЦЕ ЦЕН.  Тенговый кэш при портфеле из
        одних US-бумаг в матрицу не попадает, а конвертировать его обязательно —
        иначе 2 500 000 ₸ входят в портфель как $2 500 000 (замерено).
        """
        return self.fx_quote_to_base(currency)[0]

    def fx_quote_to_base(self, currency: str) -> tuple[Optional[float], Optional[str]]:
        """Тот же курс и ДАТА наблюдения (ISO), либо `(None, None)`.

        Дата нужна ровно одному потребителю — экрану подтверждения ручного
        ввода (`PHASE_05 §4`): «2 500 000 KZT по курсу 520.0 на 2026-07-24».
        Курс без даты не даёт читателю оценить лаг, а лаг здесь настоящий:
        матрица цен намеренно берёт курс с задержкой в день (`lag_one_day`,
        защита от look-ahead).

        Реализация ОДНА на оба метода — `fx_rate_to_base` стал обёрткой.  Два
        независимых прохода за курсом рано или поздно дали бы РАЗНЫЕ числа в
        подписи и в расчёте, а это ровно тот класс расхождения, который
        конверсия кэша уже однажды породила (F-2).

        Дата известна не всегда: когда курс приходит из `_last_fx_records`
        (валюта уже встречалась в матрице цен), у записи `FxConversion` даты
        нет.  Тогда возвращается `(rate, None)` — и подпись обязана честно
        обойтись без даты, а не подставить сегодняшнюю.
        """
        ccy = str(currency or "").upper().strip()
        rep = self.reporting_currency.value
        if not ccy or ccy == rep:
            return 1.0, None
        if ccy in self._fx_rate_cache:
            return self._fx_rate_cache[ccy], self._fx_rate_dates.get(ccy)

        rate: Optional[float] = None
        as_of: Optional[str] = None
        # 1. Уже посчитано при конверсии матрицы цен — берём ровно то значение.
        for rec in (self._last_fx_records or []):
            if getattr(rec, "pair", "") == f"{ccy}{rep}":
                lv = getattr(rec, "last_value", None)
                if lv is not None and np.isfinite(lv) and lv > 0:
                    rate = float(lv)
                break
        # 2. Валюты не было в матрице (типичный случай — кэш) → спросить провайдера.
        if rate is None and self.fx_provider is not None:
            try:
                from finance.currency import PENCE_SCALE
                lookup, unit_scale = PENCE_SCALE.get(ccy, (ccy, 1.0))
                if lookup == rep:
                    rate = float(unit_scale)
                else:
                    series = self.fx_provider(lookup, rep)
                    if series is not None and len(series) > 0:
                        clean = pd.Series(series).dropna()
                        last = float(clean.iloc[-1])
                        if np.isfinite(last) and last > 0:
                            rate = last * unit_scale
                            as_of = self._observation_date(clean)
            except Exception as exc:
                logger.warning("FX %s→%s недоступен: %s", ccy, rep, exc)
        self._fx_rate_cache[ccy] = rate
        self._fx_rate_dates[ccy] = as_of
        return rate, as_of

    @staticmethod
    def _observation_date(series) -> Optional[str]:
        """Дата последнего наблюдения ряда в ISO, либо None.

        Индекс FX-ряда — дата у FRED, но контракт `FxProvider` этого НЕ требует
        (провайдер вправе отдать ряд с любым индексом).  Поэтому дата
        извлекается защитно: непонятный индекс — это `None`, а не выдуманный
        день в подписи под курсом.
        """
        try:
            stamp = pd.Timestamp(series.index[-1])
        except Exception:                                # pragma: no cover
            return None
        return None if pd.isna(stamp) else stamp.date().isoformat()

    def get_ticker_sector(self, ticker: str) -> str:
        """Возвращает сектор для тикера (или 'Other').

        KZ-тикеры с суффиксами .KZ или .IL автоматически классифицируются как
        EM_Kazakhstan, если они не найдены в TICKER_SECTOR по базовому имени.
        """
        base = ticker.split('.')[0].upper() if '.' in ticker else ticker.upper()
        suffix = ticker.upper().rsplit('.', 1)[-1] if '.' in ticker else ''
        sector = self.TICKER_SECTOR.get(base)
        if sector:
            return sector
        # KZ/IL exchange suffix → treat as EM Kazakhstan
        if suffix in ('KZ', 'IL') or ticker.upper().endswith('.AIX'):
            return 'EM_Kazakhstan'
        return 'Other'

    def get_sector_exposure(self, tickers: list, weights: dict) -> dict:
        """
        Рассчитывает секторное распределение портфеля.

        Returns dict mapping sector name to total weight %.

        R-1 (2026-08-02): кэш и маржинальный остаток СЮДА НЕ ВХОДЯТ.
        `get_ticker_sector("USD")` → `Other`, поэтому отрицательный вес маржи
        раньше СКЛАДЫВАЛСЯ с акциями сектора Other: в живом отчёте 30.07 сектор
        27.03% рисовался как 7% (27.03 − 20.20 маржи), и с диаграммы исчезал
        главный источник риска (CRCL, 17.5% всего риска). Кэш — не сектор;
        плечо раскрывается СВОЕЙ панелью (leverage badge), а не искажением
        секторного разреза.
        """
        sectors: dict[str, float] = {}
        for t in tickers:
            if str(t).upper().strip() in self.NON_RISK_ASSETS:
                continue
            sector = self.get_ticker_sector(t)
            sectors[sector] = sectors.get(sector, 0) + weights.get(t, 0)
        # Sort by weight descending
        return dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True))

    def calculate_structural_risk(self, data, asset_tickers, weights_dict):
        """
        Ядро MAC3: Построение факторной модели и декомпозиция Эйлера.
        Облигации и Кэш в asset_tickers не передаются!
        Но weights_dict содержит их доли от ОБЩЕГО портфеля, поэтому сумма весов asset_tickers < 1.0.
        """
        # Reset the additive-analytics slot up-front: the early returns below
        # (no data / no factors / short history / zero weights) must never
        # leave a STALE decomposition from a previous call on the engine.
        self._last_factor_decomposition = {}
        # `§−128`: ряд книги — тоже. С D-5 его читает стадия бенчмарков (TE/IR,
        # таблица периодов), и ранний выход ниже не вправе оставить ей ряд
        # ПРОШЛОГО вызова на том же движке.
        self._last_port_log_returns = None
        # F-1: reset the orthogonalization betas alongside — a stale map from a
        # previous call would residualize stress shocks that were fitted on
        # different data (or when this run skips orthogonalization entirely).
        self._last_ortho_betas = {}
        # F-14: number of common observations the Ridge regression actually
        # used — BLOCK 5 gates the FORWARD expected-return panel on it (an
        # annualised forecast off a sub-year window is not publishable).
        self._last_regression_nobs = 0
        # F-20: reset the sparse-exclusion list alongside (same staleness
        # contract as the ortho betas above).
        self._last_sparse_dropped = []
        resolved_assets = self.resolve_tickers(asset_tickers)

        # Restrict to columns we actually need and drop all-NaN columns BEFORE
        # the row-level dropna. Without this, a single broker-only ticker with
        # no Tradernet history (e.g. FFSPC6.1028.AIX on Astana exchange) leaves
        # an all-NaN column whose NaNs propagate through the row-level dropna
        # and erase every row — Ridge then gets shape (0, K) and raises.
        # H-1 (2026-07-29): дедупликация ОБЯЗАТЕЛЬНА.  `data[available]` с
        # повторяющимся именем даёт DataFrame с дублированными колонками, и
        # дальше ломается всё: `data[c]` в sparse-гейте возвращает DataFrame
        # (`TypeError: cannot convert the series to int`), а присвоение
        # `f_data.columns = present_factors.keys()` падает с
        # `ValueError: Length mismatch: Expected axis has 11 elements,
        # new values have 10`.  Дубликат возникает в двух реальных случаях:
        #   1. пользователь ДЕРЖИТ факторный ETF (SPY, IWM, EEM, IEF, DBC…) —
        #      его resolved-тикер совпадает с факторной колонкой.  До этого
        #      фикса портфель с SPY не строил отчёт ВООБЩЕ;
        #   2. две неликвидные бумаги делят один прокси (AIX-нота +
        #      KZ_GOV_BOND → VWOB.US).
        # Дедуп только в ВЫБОРКЕ колонок; `valid_resolved` ниже дубликаты
        # сохраняет намеренно — две проксированные бумаги должны получить
        # одинаковые беты по общему прокси.
        needed_cols = [*self.factor_tickers.values(), *resolved_assets]
        available = list(dict.fromkeys(c for c in needed_cols if c in data.columns))
        data = data[available].dropna(axis=1, how='all')

        # F-6: sparse-history guard.  With leading NaNs no longer bfilled (the
        # look-ahead fix in math_firewall / history.py), the row-level dropna
        # below shrinks the common window to the YOUNGEST surviving column.  A
        # thinly-listed asset with a few weeks of quotes must not collapse the
        # whole book's 5-year window — drop it from the STRUCTURAL model
        # instead (same graceful contract as a missing ticker: its weight in
        # weights_dict still dilutes portfolio risk; price/PnL keep coming
        # from the broker).  Factor ETFs are exempt — long histories by
        # construction, and losing one would silently change the model.
        _factor_cols = set(self.factor_tickers.values())
        sparse_assets = [
            c for c in data.columns
            if c not in _factor_cols
            and int(data[c].notna().sum()) < _MIN_OVERLAP_TDAYS
        ]
        if sparse_assets:
            logger.warning(
                "Структурная модель: исключены %d актив(ов) с историей "
                "< %d торговых дней (защита общего окна дат): %s",
                len(sparse_assets), _MIN_OVERLAP_TDAYS, ", ".join(sparse_assets))
            data = data.drop(columns=sparse_assets)
        # F-20: expose the exclusions so the Action Plan can annotate WHY a
        # name has no beta / BL target / quantity instead of printing a bare
        # directional call (live report: «SELL SPCX» with qty=null).
        self._last_sparse_dropped = list(sparse_assets)

        # Логарифмические доходности для агрегации факторов
        returns = np.log(data / data.shift(1)).dropna()

        if returns.empty:
            logger.warning("Нет общих ценовых данных для факторной модели — пропускаем структурный риск.")
            return pd.DataFrame(), pd.DataFrame(), {}

        # 1. Выделяем факторы и активы
        present_factors = {k: v for k, v in self.factor_tickers.items() if v in returns.columns}
        if not present_factors:
            logger.warning("Ни одна фактор-серия не доступна — пропускаем структурный риск.")
            return pd.DataFrame(), pd.DataFrame(), {}
        f_data = returns[list(present_factors.values())]
        f_data.columns = list(present_factors.keys())

        # BLOCK 3.5: optionally orthogonalize style/EM factors against the core
        # macro factors to kill multicollinearity (κ≈62) BEFORE fitting betas +
        # covariance — so B and F are built on independent, still-named factors.
        # Gated; default OFF leaves the production decomposition unchanged.
        _orthogonalized = factor_orthogonalize_enabled()
        if _orthogonalized:
            # F-1: capture the child→parent OLS betas so the stress engine can
            # transform its RAW-space shock catalog into this residual space
            # (shock_resid = shock_raw − Σ β̂·shock_parent).  Without the
            # transform the market leg of a style shock is double-counted.
            f_data, self._last_ortho_betas = orthogonalize_factors_hierarchical(
                f_data, return_betas=True)

        # F-3: underdetermined-regression guard.  Ridge fits K factor betas per
        # asset and LedoitWolf/EWMA estimate a KxK factor covariance — with
        # fewer observations than ~2K the betas are near-interpolating noise
        # and the covariance is degenerate/rank-deficient, silently poisoning
        # structural vol, every ERC% and the Black-Litterman prior.  Skip the
        # structural model entirely (same graceful contract as the guards
        # above); callers already tolerate the empty return.
        k_factors = len(present_factors)
        min_obs   = max(2 * k_factors, 30)
        if len(returns) < min_obs:
            logger.warning(
                "Структурный риск пропущен: %d общих наблюдений < %d "
                "(нужно ≥2×%d факторов; короткая общая история).",
                len(returns), min_obs, k_factors,
            )
            return pd.DataFrame(), pd.DataFrame(), {}
        # F-14: record the effective regression window (common date rows) so
        # analyze_all can gate the forward expected-return panel on it.
        self._last_regression_nobs = int(len(returns))
        
        # Защита от отсутствующих тикеров:
        existing_resolved = [r for r in resolved_assets if r in returns.columns]
        # Сопоставляем обратно с оригинальными (чтобы отчет был красивым)
        valid_originals = []
        valid_resolved = []
        for orig, res in zip(asset_tickers, resolved_assets):
            if res in returns.columns:
                valid_originals.append(orig)
                valid_resolved.append(res)
                
        if not valid_resolved:
            logger.warning("Нет рисковых активов для расчета.")
            return pd.DataFrame(), pd.DataFrame(), {}
            
        a_data = returns[valid_resolved]
        a_data.columns = valid_originals # Возвращаем оригинальные имена
        
        # Вектор весов (БЕЗ нормализации к 1.0! Если вложено 50% в акции и 50% в Кэш, сумма здесь будет 0.5)
        # Это обеспечит правильное разбавление риска
        weights = np.array([weights_dict.get(ticker, 0) for ticker in valid_originals])
        
        # 2. Факторная регрессия (Поиск Бета коэффициентов)
        B_matrix = [] # Экспозиции
        specific_variances = [] # Специфический риск
        exposures_report = {}
        # P-4 (audit B1): SE(β_k) ≈ σ_ε / (σ_{f_k}·√T) — точная OLS-формула
        # для ОРТОГОНАЛЬНЫХ регрессоров; факторы ортогонализованы по умолчанию
        # (BLOCK 3.5), для legacy raw-режима это нижняя граница SE.  Кэшируем
        # входы один раз на весь цикл.
        _f_std_daily = f_data.std(ddof=1)
        _reg_T = int(len(f_data))
        beta_se_map: dict[str, dict[str, float]] = {}
        # P-4b: Vasicek-сжатие (default OFF — см. beta_shrinkage_enabled).
        _shrink_on = beta_shrinkage_enabled()
        try:
            _shrink_n0 = max(1, int(os.getenv(BETA_SHRINKAGE_PRIOR_OBS_ENV, "120")))
        except (TypeError, ValueError):
            _shrink_n0 = 120

        for asset in valid_originals:
            y = a_data[asset]
            X = f_data

            # alpha=1.0 shrinks daily-return betas by ~98% (penalty >> RSS on
            # 0.01-scale returns).  Use 0.001 — enough to stabilise correlated
            # factors while keeping betas economically meaningful.
            model = Ridge(alpha=0.001, fit_intercept=True).fit(X, y)
            betas = model.coef_
            alpha = model.intercept_

            # P-4b: Vasicek shrinkage toward the prior (Market → 1.0, либо L
            # для реестровых ETP; стили/EM → 0).  Интерсепт и остатки
            # ПЕРЕсчитываются с сжатыми бетами, чтобы D в Σ = B·F·Bᵀ + D
            # остался согласованным.  Выключено по умолчанию — прод не
            # меняется, пока сжатие не пройдёт отдельную валидацию.
            if _shrink_on:
                _w_shr = _reg_T / (_reg_T + _shrink_n0)
                _lev = None
                try:
                    from finance.leveraged import leverage_of as _lev_of
                    _lev = _lev_of(asset)
                except Exception:
                    _lev = None
                prior = np.array([
                    (float(_lev) if (_lev is not None and k == "Market") else
                     1.0 if k == "Market" else 0.0)
                    for k in f_data.columns], dtype=float)
                betas = _w_shr * np.asarray(betas, dtype=float) \
                    + (1.0 - _w_shr) * prior
                alpha = float(np.mean(y)
                              - np.asarray(X.mean().values, dtype=float) @ betas)
                residuals = np.asarray(
                    y - (X.values @ betas + alpha), dtype=float)
            else:
                residuals = np.asarray(y - model.predict(X), dtype=float)

            # Guard: ddof=1 variance needs ≥2 finite residuals, else it
            # divides by (n-1)=0 → NaN/Inf that would poison D and the
            # structural covariance.  A 1-sample (or degenerate) regression
            # carries no measurable specific risk → treat as 0.
            finite_res = residuals[np.isfinite(residuals)]
            if finite_res.size < 2:
                spec_var = 0.0
                resid_vol_ann = 0.0
            else:
                spec_var = float(np.var(finite_res, ddof=1))
                resid_vol_ann = float(np.std(finite_res, ddof=1)) * np.sqrt(self.trading_days)

            B_matrix.append(betas)
            specific_variances.append(spec_var)

            # P-4 (audit B1): standard errors of the fitted betas.
            # SE(β_k) = σ_ε / (σ_{f_k}·√T); NaN-safe on degenerate factors.
            if finite_res.size >= 2 and _reg_T > 0:
                _res_sd = float(np.std(finite_res, ddof=1))
                _se_row: dict[str, float] = {}
                for _k in f_data.columns:
                    _fsd = float(_f_std_daily.get(_k, 0.0) or 0.0)
                    if _fsd > 0 and np.isfinite(_res_sd):
                        _se_row[str(_k)] = round(
                            _res_sd / (_fsd * np.sqrt(_reg_T)), 6)
                if _se_row:
                    beta_se_map[str(asset)] = _se_row

            exposures_report[asset] = {f'Beta_{k}': b for k, b in zip(f_data.columns, betas)}
            exposures_report[asset]['Specific_Alpha_Daily'] = alpha
            # ddof=1 for self-consistency with specific_variances above
            # (both are estimators of residual dispersion; biased ddof=0
            # under-reports the per-asset residual vol by ~√((n-1)/n)).
            exposures_report[asset]['Residual_Vol_Ann'] = resid_vol_ann
            exposures_report[asset]['Weight_Pct'] = weights_dict.get(asset, 0) * 100

        B = np.array(B_matrix) # (N_assets, K_factors)
        D = np.diag(specific_variances) # (N_assets, N_assets)
        
        # 3. Структурная Ковариационная Матрица
        # EWMA-ковариация: halflife=63 дн ⇒ λ_daily ≈ 0.989 (плавнее
        # RiskMetrics λ=0.94 ≈ hl 11 дн — выбрано для стабильности факторной
        # матрицы).  Недавние дни влияют сильнее, чем старые.
        ewma_cov = f_data.ewm(halflife=self.ewma_halflife).cov()
        # Берём последний срез (самый актуальный)
        last_date = f_data.index[-1]
        cov_factors_ewma = ewma_cov.loc[last_date].values
        
        # Ledoit-Wolf Shrinkage (стабилизация при малых N)
        try:
            lw = LedoitWolf().fit(f_data.values)
            cov_factors_shrunk = lw.covariance_
            # Блендим EWMA (70%) + Shrinkage (30%) для баланса адаптивности и стабильности
            cov_factors = 0.7 * cov_factors_ewma + 0.3 * cov_factors_shrunk
        except Exception:
            cov_factors = cov_factors_ewma
        # Sprint-3 #9: the EWMA+LW blend is not guaranteed PSD — project it onto
        # the nearest PSD matrix so Σ = B·F·Bᵀ + D stays PSD (well-defined ERC,
        # no negative w'Σw).  No-op within tolerance when the blend is already PSD.
        _nearest_psd.last_repair = None              # type: ignore[attr-defined]
        cov_factors = _nearest_psd(cov_factors)
        # C-6: результат — на движке, чтобы блок C мог о нём доложить.
        self._last_psd_repair = getattr(_nearest_psd, "last_repair", None)

        # ── BLOCK 4.6: factor-multicollinearity diagnostic (read-only) ───────
        # The structural model ALREADY avoids double-counting correlated risk:
        # Σ = B·F·Bᵀ + D carries the FULL off-diagonal factor covariance F, so
        # two correlated factors contribute their covariance exactly once.  A
        # PCA / orthogonalisation would only DESTROY the named-factor
        # interpretability the report depends on — so instead of forcing it we
        # MEASURE collinearity and surface a diagnostic:
        #   • max |off-diagonal correlation| across the factor set
        #   • condition number κ of the factor CORRELATION matrix
        # |corr| > 0.95 or κ > 30 ⇒ near-collinear factors make the Ridge betas
        # unstable; we log a warning and feed the numbers to a CoVe row so the
        # operator can prune/merge a redundant factor (e.g. SPY vs QQQ).
        factor_diagnostic: dict = {}
        try:
            _F = np.asarray(cov_factors, dtype=float)
            if _F.ndim == 2 and _F.shape[0] >= 2:
                _d    = np.sqrt(np.clip(np.diag(_F), 1e-18, None))
                _corr = _F / np.outer(_d, _d)
                _off  = _corr - np.eye(_corr.shape[0])
                _max_off = float(np.max(np.abs(_off))) if _off.size else 0.0
                _cond    = float(np.linalg.cond(_corr))
                # P-5 (audit D4): Fisher z-ДИ на максимальную |off-diag|
                # корреляцию — на коротком окне регрессии знак/величина ρ̂
                # статистически не определены, и ДИ делает это видимым.
                _corr_ci = _fisher_rho_ci(_max_off, int(len(returns)))
                factor_diagnostic = {
                    "n_factors":        int(_F.shape[0]),
                    "factors":          list(f_data.columns),
                    "max_abs_corr":     round(_max_off, 4),
                    "max_corr_ci95":    ([round(x, 4) for x in _corr_ci]
                                         if _corr_ci else None),
                    "corr_window_days": int(len(returns)),
                    "condition_number": round(_cond, 2),
                    "near_collinear":   bool(_max_off > 0.95 or _cond > 30.0),
                    # BLOCK 3.5: whether the hierarchical orthogonalization was
                    # applied (κ above is then the POST-orthogonalization value).
                    "orthogonalized":   bool(_orthogonalized),
                }
                if factor_diagnostic["near_collinear"]:
                    logger.warning(
                        "Факторная мультиколлинеарность: max|corr|=%.2f, κ=%.1f "
                        "(порог 0.95 / 30 → беты Ridge нестабильны; рассмотрите "
                        "слияние коррелирующих факторов).",
                        _max_off, _cond,
                    )
        except Exception as exc:
            logger.debug("Factor diagnostic skipped: %s", exc)
        self._last_factor_diagnostic = factor_diagnostic

        structural_cov = B @ cov_factors @ B.T + D
        structural_cov_ann = structural_cov * self.trading_days 
        
        cov_df = pd.DataFrame(structural_cov_ann, index=valid_originals, columns=valid_originals)
        
        # Если портфель пустой по рисковым активам
        if weights.sum() == 0:
            return cov_df, pd.DataFrame(exposures_report).T, {}
            
        # 4. Риск портфеля
        port_variance = weights.T @ structural_cov_ann @ weights
        # BLK-3: the blended factor covariance (EWMA + Ledoit-Wolf) is not
        # guaranteed positive-semi-definite, so w'Σw can land marginally below
        # zero from float noise / a non-PSD matrix.  Clamp at 0 before the
        # sqrt so port_volatility is never NaN (a NaN here silently poisons
        # Sharpe/Sortino and zeroes every Euler risk contribution).
        port_variance = float(max(port_variance, 0.0))
        port_volatility = np.sqrt(port_variance)
        
        # 5. Декомпозиция Эйлера
        # MCTR = (Sigma * w) / Port_Vol
        mctr = (structural_cov_ann @ weights) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
        # ERC (%) = w * MCTR / Port_Vol
        erc_pct = (weights * mctr) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
        
        for i, asset in enumerate(valid_originals):
            exposures_report[asset]['Euler_Risk_Contribution_Pct'] = erc_pct[i] * 100

        # ── Additive analytics layer: factor-VARIANCE decomposition (по
        # источникам риска, а не по активам) + marginal overlap («факторные
        # двойники» + unique-risk share).  Pure add-on из отдельного модуля —
        # читает ТЕ ЖЕ B/F/D/w, что структурная модель уже построила, ничего
        # не мутирует; любая ошибка деградирует в {} (тот же graceful-паттерн,
        # что и κ-диагностика BLOCK 4.6 выше).
        try:
            from finance.factor_decomposition import build_factor_decomposition
            self._last_factor_decomposition = build_factor_decomposition(
                B=B,
                F=cov_factors,
                D=np.asarray(specific_variances, dtype=float),
                weights=weights,
                factor_names=list(f_data.columns),
                asset_names=list(valid_originals),
            )
        except Exception as exc:
            logger.debug("Factor decomposition skipped: %s", exc)
            self._last_factor_decomposition = {}

        # 6. Хвостовые метрики
        # `port_returns_daily` carries LOG returns in the REPORTING CURRENCY
        # (H2 pre-converted the price matrix → a_data is base-currency log
        # returns).  Sum(weights) < 1 by design when cash sits in the
        # portfolio — that dilutes risk correctly without renormalisation.
        # `§−121`: ln(1 + Σ wᵢ·Rᵢ), not Σ wᵢ·rᵢ — the log of the sum, not the
        # sum of logs (the latter drops the diversification return).
        port_returns_daily = _aggregate_log_returns(a_data.values, weights)
        port_series_index  = a_data.index
        # F-22 (2026-07-11): realized-metrics basis — masked composite series.
        # The intersection window `a_data` shrinks to the YOUNGEST kept listing
        # (honest windows after F-6), silently re-basing Sharpe / Annualised
        # Return / MaxDD / VaR / CVaR onto a sub-year window when the book
        # holds one young name.  Rebuild the portfolio series over the FULL
        # price panel with per-day renormalised weights of the names actually
        # trading that day (the same composite-backfill convention as the
        # period-returns table and the equity curve), re-scaled by Σw so cash
        # dilution is preserved: on full-coverage days the composite equals
        # `a_data @ weights` EXACTLY, and when it adds no history the legacy
        # path is kept bit-for-bit (β/Σ regression stays on the intersection
        # window — only the REALIZED panel gains history).
        try:
            _w_by_res = {res: float(weights_dict.get(orig, 0.0) or 0.0)
                         for orig, res in zip(valid_originals, valid_resolved)}
            _total_w  = float(sum(_w_by_res.values()))
            _comp, _comp_info = _build_portfolio_log_returns(
                data[valid_resolved], _w_by_res)
            if _comp is not None and _total_w > 0 and len(_comp) > len(a_data):
                # Cash dilution in SIMPLE space (cash earns 0): the book's day
                # is Σw · R_comp, then back to log (`§−121`).
                port_returns_daily = np.log1p(np.expm1(_comp.values) * _total_w)
                port_series_index  = _comp.index
                logger.info(
                    "Realized-metrics basis: composite %d дн "
                    "(окно пересечения %d дн, покрыто весов %.0f%%).",
                    len(_comp), len(a_data),
                    float(_comp_info.get("covered_weight", 0.0)) * 100)
        except Exception as _exc:
            logger.warning(
                "Composite series unavailable — realized metrics fall back "
                "to the intersection window: %s", _exc)
        # `§−121`: a margin loan (negative cash row in weights_dict) accrues
        # financing at the reporting-currency rf — never free.  The threshold
        # (−0.1%) is the same float-noise guard as `leverage_metrics`.
        _cash_w = sum(float(w or 0.0) for t, w in weights_dict.items()
                      if str(t).upper() in self.NON_RISK_ASSETS)
        port_returns_daily = _apply_margin_financing(
            port_returns_daily, _cash_w, self.current_rfr_daily)
        # H3 (Phase-3): expose the date-indexed portfolio log-return series so
        # the Telegram equity-curve SVG consumes it instead of recomputing
        # `log(prices/prices.shift) @ weights` in the bot layer.  Stored on
        # the engine (same pattern as _last_fx_records); analyze_all copies it
        # into results["port_log_returns"].
        try:
            self._last_port_log_returns = pd.Series(port_returns_daily,
                                                    index=port_series_index)
        except Exception:
            self._last_port_log_returns = None
        # H3: geometric daily RFR — `(1+r)^(1/252)-1` rather than `r/252`.
        # Keeps Sortino's downside filter consistent with Sharpe's annual
        # excess-return numerator.
        risk_free_rate_daily = self.current_rfr_daily

        # H-8 (`§−130`): окно Sharpe/Sortino — последние 252 торговых дня ряда
        # книги, ТО ЖЕ, что у `Return_12M` обложки.  Короче года — окна нет.
        _win_12m = (port_returns_daily[-self.trading_days:]
                    if len(port_returns_daily) >= self.trading_days else None)
        vol_12m: Optional[float] = None
        downside_vol = 0.0
        if _win_12m is not None:
            # Реализованная σ дневных доходностей книги за те же 252 дня
            # (ddof=1 — как у replay «Эффекта» и спарклайнов).
            vol_12m = float(np.std(_win_12m, ddof=1) * np.sqrt(self.trading_days))
            excess_returns = _win_12m - risk_free_rate_daily
            # H-4: industry-standard downside deviation (Sortino/Satchell).
            # Take the shortfalls min(excess, 0) against the MAR (=0), square
            # them, and average over the TOTAL number of observations N — not
            # just the losing days.  The previous `np.std(excess[excess<0])`
            # used the wrong denominator (count of negatives) and de-meaned
            # around the downside mean instead of the target, making the ratio
            # non-comparable to the standard definition.
            downside_shortfall = np.minimum(excess_returns, 0.0)
            downside_vol = float(np.sqrt(np.mean(downside_shortfall ** 2))
                                 * np.sqrt(self.trading_days))

        # H1: geometric annualisation of log-returns.
        # Arithmetic `mean·252` over-states annual return by ~σ²/2 (it
        # ignores the Itô correction).  For 24% annual vol that's ~+2.9
        # pp of phantom return per year — directly inflates Sharpe.
        # `exp(mean_log·252) - 1` is the exact equivalent simple return.
        ann_return = float(np.exp(np.mean(port_returns_daily) * self.trading_days) - 1.0)

        # D-5 (`§−128`): ТЕКУЩАЯ годовая доходность — ФАКТ, а не прогноз:
        # накопленная простая доходность за последние 252 торговых дня по ТОМУ
        # ЖЕ ряду книги, что и обложка (кэш 0%, маржа по rf, композит F-22).
        # Короче торгового года — None: годовую цифру с неполного окна не
        # печатаем (тот же принцип, что гейт F-14 у форварда).
        return_12m = (
            float(np.expm1(np.sum(port_returns_daily[-self.trading_days:])))
            if len(port_returns_daily) >= self.trading_days else None)

        var_95 = np.percentile(port_returns_daily, 5) if len(port_returns_daily)>0 else 0
        cvar_95 = port_returns_daily[port_returns_daily <= var_95].mean() if len(port_returns_daily)>0 else 0

        # Bootstrap CVaR with 95% CI (stationary block bootstrap, Politis-Romano).
        # The point CVaR above is just the empirical mean of the bottom-5%
        # observations on a finite window — its standard error is ~15-20% of
        # the magnitude.  Bootstrap gives a confidence interval the user can
        # weight against the lower bound when sizing tail-risk hedges.
        if len(port_returns_daily) >= 60:
            cvar_boot = self._bootstrap_cvar(port_returns_daily, n_boot=2000, alpha=0.05)
        else:
            cvar_boot = {"point": float(cvar_95), "lo95": None, "hi95": None}

        # Real Max Drawdown (peak-to-trough on equity curve).
        # port_returns_daily are daily LOG returns → equity curve = exp(cumsum).
        # MaxDD ≠ VaR_95: VaR is a 1-day quantile, MaxDD is the worst peak-to-trough
        # of the realised equity path. Both are needed for a complete risk picture.
        if len(port_returns_daily) > 0:
            eq_curve     = np.exp(np.cumsum(port_returns_daily))
            running_max  = np.maximum.accumulate(eq_curve)
            drawdowns    = eq_curve / running_max - 1.0
            max_drawdown = float(drawdowns.min())
        else:
            max_drawdown = 0.0

        # Marginal VaR per asset — sensitivity dVaR/dw_i.
        #
        # Useful complement to Euler TRC: TRC says "what fraction of CURRENT
        # portfolio risk is driven by asset i", M-VaR says "by how many bps
        # would 1-day VaR move if I add 1% weight to asset i".  M-VaR is in
        # the same units as VaR itself (decimal daily return per unit weight).
        if len(port_returns_daily) >= 60 and a_data.shape[1] > 0:
            mvar_series = self._marginal_var(a_data, weights, var_p=0.05, h=0.005)
            for asset, mvar_val in mvar_series.items():
                if asset in exposures_report:
                    exposures_report[asset]['Marginal_VaR_Daily'] = float(mvar_val)

        # Composite Risk Score 0..100 — blends the three independent signals
        # (volatility, tail, concentration) into a single user-facing gauge.
        # Replaces the prior gauge that looked at vol alone.
        max_erc = 0.0
        if len(weights) > 0 and port_volatility > 0:
            max_erc = float(np.max(np.abs(erc_pct))) * 100
        composite_risk = self._composite_risk_score(
            volatility=port_volatility, cvar=cvar_95, max_erc_pct=max_erc,
            mandate=getattr(self, "risk_mandate", "MODERATE"),
        )

        # P-4 (audit B1): сводный вердикт надёжности бет — максимальная
        # ОТНОСИТЕЛЬНАЯ SE по Market-оси; > 0.5 на окне регрессии короче
        # торгового года ⇒ беты статистически шумные (β=1.8 при SE=1.0
        # неотличима от 1.0) — warning в лог + вердикт в metrics.
        _beta_reliability = "ok"
        try:
            _rel_ses = []
            for _a, _ses in beta_se_map.items():
                _b_mkt = float(exposures_report.get(_a, {}).get("Beta_Market", 0.0) or 0.0)
                _se_mkt = float(_ses.get("Market", 0.0) or 0.0)
                if abs(_b_mkt) > 1e-9 and _se_mkt > 0:
                    _rel_ses.append(_se_mkt / abs(_b_mkt))
            if _rel_ses and max(_rel_ses) > 0.5 and _reg_T < self.trading_days:
                _beta_reliability = "noisy_short_window"
                logger.warning(
                    "Беты статистически шумные: max SE(β_Market)/|β| = %.2f "
                    "на окне %d торг. дней (< %d).",
                    max(_rel_ses), _reg_T, self.trading_days)
        except Exception:
            _beta_reliability = "ok"

        # Sharpe / Sortino with currency-matched, geometrically-compounded RFR
        # (H1+H3).
        #
        # H-8 (`§−130`, решение владельца): базис — 12 МЕСЯЦЕВ, как у факта
        # «Доходность за 12 мес» на обложке.  Числитель — `Return_12M` − rf
        # (простая годовая против простой годовой ставки валюты отчёта);
        # знаменатель — РЕАЛИЗОВАННАЯ σ дневных доходностей книги за те же
        # 252 дня (Sortino — нижнее отклонение там же).  Прежде (F-4) в одной
        # дроби стояли два горизонта: среднегодовая за ВСЁ окно (до 5 лет) и
        # структурная σ EWMA(63)⊕LW с весом к последним ~3 мес — Sharpe нельзя
        # было сверить ни с одним числом страницы.  Структурная σ остаётся
        # карточкой «Волатильность»: это прогноз риска, а не знаменатель
        # факта.  Короче торгового года — NaN («—»): годовой коэффициент с
        # обрывка истории не печатаем, как и `Return_12M`.
        _excess_12m = (return_12m - self.current_rfr_annual
                       if return_12m is not None else None)
        sharpe  = (_excess_12m / vol_12m
                   if _excess_12m is not None and vol_12m and vol_12m > 0
                   else np.nan)
        sortino = (_excess_12m / downside_vol
                   if _excess_12m is not None and downside_vol > 0
                   else np.nan)

        portfolio_metrics = {
            "Total_Volatility_Ann":  port_volatility,
            "Annualised_Return":     ann_return,            # H1: geometric simple return
            "Return_12M":            return_12m,            # D-5: факт за 252 торг. дня
            "Volatility_12M":        vol_12m,               # H-8: знаменатель Sharpe
            "Sharpe_Ratio":          sharpe,
            "Sortino_Ratio":         sortino,
            "VaR_95_Daily":          var_95,
            "CVaR_95_Daily":         cvar_95,
            "CVaR_95_Bootstrap":     cvar_boot,
            "Max_Drawdown":          max_drawdown,
            "Max_Euler_Risk_Pct":    max_erc,
            "Composite_Risk_Score":  composite_risk,
            "Positive_Days_Pct":     (port_returns_daily > 0).mean() * 100 if len(port_returns_daily) > 0 else 0,
            # F-22: how many trading days the REALIZED panel (Sharpe numerator,
            # Annualised_Return, VaR/CVaR, MaxDD) was computed on — composite
            # window when it extends the intersection, else the intersection.
            "realized_window_days":  int(len(port_returns_daily)),
            # P-3 (audit A2/D2): explicit reliability verdict for the
            # historical VaR/CVaR quantiles — ~100 obs is the floor for a
            # stable 95% tail; below it the numbers print WITH the flag.
            "var_reliability":       ("ok"
                                      if len(port_returns_daily) >= MIN_RELIABLE_TAIL_OBS
                                      else "insufficient_history"),
            # P-5 (audit A3): χ²-ДИ множители на выборочную σ реализованного
            # окна — «σ известна с точностью ×[lo, hi]».  None на окнах < 2.
            "volatility_ci":         (
                {"lo_mult": round(_vci[0], 3), "hi_mult": round(_vci[1], 3),
                 "window_days": int(len(port_returns_daily)),
                 "confidence": 0.95}
                if (_vci := _sigma_ci_multiplier(int(len(port_returns_daily))))
                else None),
            # P-4 (audit B1): SE(β) per asset/factor + сводный флаг
            # надёжности: max относительная SE по Market-оси > 0.5 на окне
            # короче года ⇒ беты статистически шумные.
            "beta_standard_errors":  beta_se_map,
            "beta_reliability":      _beta_reliability,
            "beta_shrinkage_applied": bool(_shrink_on),
            # H2 audit trail — surfaced to the report's QC panel.
            "reporting_currency":    self.reporting_currency.value,
            "risk_free_rate_annual": self.current_rfr_annual,
            "risk_free_rate_daily":  self.current_rfr_daily,
            "risk_free_rate_source": self.rfr_source,
            "fx_conversion": [
                {"pair": r.pair, "coverage_pct": r.coverage_pct,
                 "last": r.last_value, "fallback_used": r.fallback_used}
                for r in self._last_fx_records
            ],
            # H4 disclaimer is always present — even on portfolios where
            # no scenario hits the convex cap, the user sees the policy.
            "stress_test_disclaimer": _STRESS_TEST_DISCLAIMER,
            # H-8 (`§−130`): базис Sharpe/Sortino — 12 месяцев (прежде F-4:
            # среднегодовая за всё окно / структурная σ).  Surfaced to the
            # QC/Integrity panel so the basis is auditable.
            "sharpe_basis_note": (
                "Sharpe/Sortino за последние 12 мес: числитель — доходность "
                "книги за 252 торговых дня (то же число, что «Доходность за "
                "12 мес» на обложке) минус безрисковая ставка валюты отчёта; "
                "знаменатель — реализованная σ дневных доходностей за те же "
                "252 дня (у Sortino — нижнее отклонение). Короче года истории "
                "коэффициенты не считаются. Карточка «Волатильность» — "
                "структурная σ (прогноз риска), это другое число."
            ),
            # BLOCK 4.6: factor-multicollinearity diagnostic (κ + max|corr|),
            # surfaced to the CoVe panel so collinear factors are visible.
            "factor_diagnostics": getattr(self, "_last_factor_diagnostic", {}),
            # Additive layer (finance/factor_decomposition): дисперсия по
            # ИСТОЧНИКАМ риска + факторные двойники; {} когда не определена.
            "factor_decomposition": getattr(self, "_last_factor_decomposition", {}),
        }

        return cov_df, pd.DataFrame(exposures_report).T, portfolio_metrics

    def fit_factor_betas(self, data, target_ticker: str) -> "dict[str, float] | None":
        """
        Факторные беты ПРОИЗВОЛЬНОГО тикера (B1 2026-07-17: мандатный бенчмарк
        в секции «Факторное разложение»).

        Прогоняет дневные лог-доходности `target_ticker` через ТОТ ЖЕ
        Ridge-пайплайн, что и активы портфеля в `calculate_structural_risk`:
        общая факторная панель `factor_tickers`, та же иерархическая
        ортогонализация (BLOCK 3.5, gated `FACTOR_ORTHOGONALIZE`), тот же
        guard на недоопределённую регрессию (≥ max(2K, 30) наблюдений) и тот
        же `Ridge(alpha=0.001)` — иначе беты бенчмарка несопоставимы с бетами
        активов в таблице.

        Возвращает {factor_name: beta} или None (нет данных / короткое
        перекрытие истории) — потребители обязаны переживать None (фолбэк на
        S&P-константу в `tg_bot._build_factor_betas_table`).  Чистая
        read-only функция: модель Σ = B·F·Bᵀ + D и её downstream-потребители
        (стресс, BL, Euler, κ) НЕ затрагиваются (ADR Вариант A).
        """
        try:
            if data is None or getattr(data, "empty", True):
                return None
            if target_ticker not in data.columns:
                return None
            factor_cols = [v for v in self.factor_tickers.values()
                           if v in data.columns]
            if not factor_cols:
                return None
            cols = list(dict.fromkeys([*factor_cols, target_ticker]))
            sub = data[cols].dropna(axis=1, how="all")
            if target_ticker not in sub.columns:
                return None
            returns = np.log(sub / sub.shift(1)).dropna()
            present = {k: v for k, v in self.factor_tickers.items()
                       if v in returns.columns}
            if not present or target_ticker not in returns.columns:
                return None
            f_data = returns[list(present.values())].copy()
            f_data.columns = list(present.keys())
            if factor_orthogonalize_enabled():
                f_data = orthogonalize_factors_hierarchical(f_data)
            min_obs = max(2 * len(present), 30)
            if len(returns) < min_obs:
                logger.info(
                    "Беты бенчмарка %s пропущены: %d наблюдений < %d "
                    "(короткое перекрытие истории).",
                    target_ticker, len(returns), min_obs)
                return None
            y = returns[target_ticker]
            model = Ridge(alpha=0.001, fit_intercept=True).fit(f_data, y)
            return {str(k): float(b)
                    for k, b in zip(f_data.columns, model.coef_)}
        except Exception as exc:                      # noqa: BLE001
            logger.warning("fit_factor_betas(%s) failed: %s",
                           target_ticker, exc)
            return None

    def calculate_atr(self, data, tickers, ohlc_data=None, period=14):
        """
        True ATR (Wilder’s Average True Range) with OHLC when available.

        True Range = max(H-L, |H-Cp|, |L-Cp|) where Cp is previous Close.
        Falls back to Close-only approximation (|ΔClose|) when OHLC is unavailable.

        Args:
            data: Close-price DataFrame (columns = resolved tickers)
            tickers: list of original (unresolved) tickers
            ohlc_data: optional dict {resolved_ticker: DataFrame[Open,High,Low,Close]}
            period: ATR lookback window (default: 14 days, Wilder standard)
        """
        resolved = self.resolve_tickers(tickers)
        atr_results = {}

        for orig, res in zip(tickers, resolved):
            if res not in data.columns:
                continue

            # Prefer True ATR from OHLC if available
            if ohlc_data and res in ohlc_data:
                ohlc = ohlc_data[res]
                if all(c in ohlc.columns for c in ('High', 'Low', 'Close')):
                    high = ohlc['High']
                    low = ohlc['Low']
                    close_prev = ohlc['Close'].shift(1)
                    tr = pd.concat([
                        high - low,
                        (high - close_prev).abs(),
                        (low - close_prev).abs(),
                    ], axis=1).max(axis=1)
                    # Wilder's RMA (α = 1/period). NOT pandas span (which gives
                    # α = 2/(span+1) — that's EMA, ~2× faster than Wilder, and
                    # would under-smooth True Range. SEC 34-105226 / Wilder 1978.
                    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
                    last_close = ohlc['Close'].iloc[-1]
                    atr_pct = (atr / last_close) * 100 if last_close > 0 else 0
                    atr_results[orig] = {'ATR_Absolute': atr, 'ATR_Pct': atr_pct}
                    continue

            # Fallback: Close-only ATR approximation (Wilder RMA on |ΔClose|)
            prices = data[res]
            daily_range = prices.diff().abs()
            atr = daily_range.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
            atr_pct = (atr / prices.iloc[-1]) * 100 if prices.iloc[-1] > 0 else 0
            atr_results[orig] = {'ATR_Absolute': atr, 'ATR_Pct': atr_pct}

        return pd.DataFrame(atr_results).T if atr_results else pd.DataFrame()
