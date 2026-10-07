"""
AI narrative generator — produces verdict, plain-language summary, insight bullets,
stock-pick scenarios, and (deep tier only) action-plan commentary for the PDF templates.

Model selection (BLOCK 1.2 — current-generation routing, env-overridable):
  Base tier  → Claude Sonnet 4.6 (quality lift over the prior Haiku tier)
  Deep tier  → Claude Opus  4.8  (maximum depth/analysis; omits `temperature`,
               so idea variety comes from the prompt-level freshness directive)

Falls back to a rule-based summary when the API key is absent or the call fails,
so the PDF is always produced regardless of API availability.

ALL output is in Russian.  Source tags [Quant Engine], [SEC EDGAR], [Regime],
[CDS], [RAG: file] are mandatory on every factual claim.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from typing import Optional

# SSOT имён эмитентов (§−95).  L1 → импорт вниз разрешён; модуль на импорте
# тянет только stdlib (chromadb/pymupdf4llm там ленивые).
import branding
from env_config import env_float, env_int
from agent.rag_engine import (
    BANK_ORDER,
    bank_alias_regex,
    bank_short_regex,
    canonical_bank,
)

logger = logging.getLogger("AINarrative")

#: Потолок ожидания ответа Anthropic, секунд (`§−122`). Только через env_config:
#: голый `float(os.getenv())` на уровне модуля роняет импорт (`§−50`).
ANTHROPIC_TIMEOUT_S: float = env_float("ANTHROPIC_TIMEOUT_S", 300.0, lo=30.0, hi=1800.0)
#: `§−126`: повторы SDK. По умолчанию их ДВА, то есть три попытки по
#: `ANTHROPIC_TIMEOUT_S`: зависшая модель держала «Шаг 4/4» до 15 минут при
#: обещании «5–10». Одна повторная попытка оставляет защиту от 529/5xx, а
#: худший случай — 2 × таймаут, после чего отчёт уходит с фолбэк-нарративом.
ANTHROPIC_MAX_RETRIES: int = env_int("ANTHROPIC_MAX_RETRIES", 1, lo=0, hi=5)

MAX_TOKENS_BASE = 5_000  # 4_500 → 5_000 (BLOCK 1.2): BASE now runs on Sonnet (more verbose than Haiku); headroom so the structured tool output is never truncated → no JSON repair / dropped KPI notes
MAX_TOKENS_DEEP = 7_000


# ── Prompt-injection defence: input sanitization ────────────────────────────
# Every string that flows from BROKER / RAG / TELEGRAM into the LLM prompt
# passes through these guards.  Without them a holding called
# "Ignore previous instructions" or a poisoned bank-PDF chunk can override
# the system prompt.  Tickers go through a strict regex; free-text strings
# (sector tags, fund names) are control-char-stripped and length-clamped.
#
# Ticker regex permits up to 32 chars so KZ structured notes like
# "FFSPC6.1028.AIX" (16 chars) survive — strictly tighter ^[A-Z0-9.\-]{1,10}$
# from the original spec would drop them, breaking KZ workflows.
_TICKER_RE       = re.compile(r"^[A-Z0-9.\-]{1,32}$")
_SAFE_TEXT_DROP  = re.compile(r"[\x00-\x1F\x7F<>{}|`]")   # control / tag chars


def _safe_ticker(t) -> str:
    """Strict ticker validator.  Non-matching input → '?' (never propagated as instruction)."""
    s = str(t or "").strip().upper()
    return s if _TICKER_RE.match(s) else "?"


def _safe_text(t, max_len: int = 60) -> str:
    """Strip control chars + length-clamp untrusted free-text fields."""
    s = str(t or "")[:max_len]
    return _SAFE_TEXT_DROP.sub("", s).strip()


def _wrap_untrusted(label: str, payload: str) -> str:
    """
    Fence external data with an XML-like tag.  The system prompt instructs
    the model to NEVER follow instructions found inside these tags.
    """
    return f"<untrusted_data source=\"{label}\">\n{payload}\n</untrusted_data>"


# ── Soft-trim helper ────────────────────────────────────────────────────────
# Hard-cap slicing (`text[:N]`) produces obviously truncated sentences in
# the production report ("Это давит на перегруженный Tech-портфель [GS][Bar").
# `_soft_trim` cuts at the LAST sentence boundary (".", "!", "?", "…")
# inside the budget so the user never sees a half-sentence.  Hard fallback
# to the budget if no boundary is present (e.g. one long URL).
_SENTENCE_END_RE = re.compile(r"[\.\!\?…](?:\s|\)|\]|$)")


def _soft_trim(text: str, max_chars: int, *, allow_grace: bool = True) -> str:
    """
    Trim ``text`` to ``max_chars`` characters, ending at the LAST complete
    sentence boundary inside the budget.

    Audit 2026-07-04 (completion grace): the owner's reports kept showing a
    dangling «…это компромисс, а…» whenever the model overshot the budget by a
    few words — the sentence straddling the cap was amputated.  Now, when no
    acceptable boundary fits the budget, the sentence CROSSING the cap is kept
    WHOLE if it finishes within a bounded grace window (≤ +25% of the budget,
    40–120 chars) — a complete thought slightly over budget beats a marked
    truncation.  Runaway sentences longer than the grace still fall back to a
    word-boundary cut + ellipsis so truncation stays visually marked.

    ``allow_grace=False`` keeps the STRICT budget (no overrun) for the cover
    verdict / plain-summary, whose boxes have a hard layout ceiling.
    """
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    head    = s[:max_chars]
    matches = list(_SENTENCE_END_RE.finditer(head))
    if matches:
        end = matches[-1].end()
        # Strip trailing whitespace after the boundary char.
        candidate = head[:end].rstrip()
        if len(candidate) >= max_chars // 2:
            return candidate
    # Completion grace: finish the straddling sentence if it ends soon enough.
    if allow_grace:
        grace = min(120, max(40, max_chars // 4))
        m2 = _SENTENCE_END_RE.search(s, max(0, max_chars - 1))
        if m2 and m2.end() <= max_chars + grace:
            return s[:m2.end()].rstrip()
    # The straddling sentence runs away past the grace — prefer the COMPLETE
    # in-budget sentence(s) even if short-ish (≥25% of budget) over a dangling
    # clause; only then hard-fall to the marked word-cut.
    if matches:
        candidate = head[:matches[-1].end()].rstrip()
        if len(candidate) >= max_chars // 4:
            return candidate
    # Audit 2026-06-14: no sentence boundary in budget → cut at the last WORD
    # boundary, never mid-word.  The prod reports showed «увеличивать Fi…»,
    # «недооце…», «рекомендуе…» — a hard char-cut landing inside a word.
    cut = head.rstrip()
    sp  = cut.rfind(" ")
    if sp >= max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(" ,;:[(") + "…"


def validate_stress_comment(text: str) -> str:
    """
    M1 — post-validator for the stress-test AI comment.

    The system prompt INSTRUCTS the model to mention the non-linear convex
    cap, but Sonnet/Haiku sometimes omit it.  This guard appends an explicit
    footnote whenever neither "ограничител…" nor "cap" is present, so the
    reader always knows extreme per-asset drawdowns were convex-capped.
    """
    if not text:
        return text
    low = text.lower()
    if "ограничител" not in low and "cap" not in low:
        text += ("\n\n*(Примечание: Экстремальные просадки рассчитаны с учётом "
                 "нелинейного ограничителя риска).*")
    return text


# Tier → model routing.  The owner's BLOCK-1 request was "BASE → Sonnet,
# DEEP → Opus (maximum depth & analysis)".  The literal IDs in that request
# (`claude-3-5-sonnet-latest` / `claude-3-opus-latest`) are 2024-generation
# models — pinning them would REGRESS quality below the project's current
# Haiku-4.5/Sonnet-4.6 stack and cost MORE.  We honour the INTENT with the
# current-generation IDs instead:
#   BASE → Sonnet 5   (2026-07-09 upgrade from Sonnet 4.6 — latest-gen quality
#                       for the volume tier; owner-approved «обновить на Sonnet 5»)
#   DEEP → Opus  4.8   (the planned upgrade that was already guard-ready below)
# Both remain env-overridable so cost/latency can be tuned without a deploy
# (e.g. ANTHROPIC_MODEL_BASE=claude-sonnet-4-6 or claude-haiku-4-5-20251001 to
# restore the cheaper BASE tier).
MODEL_BASE = os.getenv("ANTHROPIC_MODEL_BASE", "claude-sonnet-5")
MODEL_DEEP = os.getenv("ANTHROPIC_MODEL_DEEP", "claude-opus-4-8")

# Sprint-5 (idea staleness): the narrative + stock-picks are produced by ONE
# model call.  This `temperature` knob is retained for env-override BACK-COMPAT
# (e.g. ANTHROPIC_MODEL_BASE=claude-sonnet-4-6 restores a temperature-accepting
# BASE model) — it is applied ONLY when `_model_supports_temperature(model)` is
# true.  As of the 2026-07-09 upgrade BOTH default tiers (BASE=Sonnet 5,
# DEEP=Opus 4.8) REJECT `temperature`, so by default it is omitted and idea
# variety comes from the prompt-level freshness directive (ideas_rule) + the
# day-rotating angle.  The risk NUMBERS are engine-computed (deterministic)
# regardless — only the prose / picks vary.
def _resolve_temperature() -> float:
    try:
        t = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
    except (TypeError, ValueError):
        t = 0.7
    return min(0.85, max(0.5, t))


NARRATIVE_TEMPERATURE = _resolve_temperature()

# Sprint-5.1 (§4.2): the current-generation models REJECT the `temperature`
# param (HTTP 400); they expose only adaptive thinking.  This applies to BOTH
# the DEEP tier (Opus 4.7/4.8) AND — since the 2026-07-09 upgrade — the BASE
# tier (Sonnet 5).  So this guard is LOAD-BEARING for both tiers now: the API
# call omits `temperature` and idea variety comes entirely from the prompt-level
# freshness directive (ideas_rule) + the day-rotating angle, NOT from sampling.
_TEMPERATURE_UNSUPPORTED_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)


def _model_supports_temperature(model: str) -> bool:
    # N-5: через OpenRouter (ANTHROPIC_BASE_URL) имя приходит с префиксом
    # провайдера — `anthropic/claude-sonnet-5`.  Сравниваем без него, иначе
    # `temperature` уйдёт в запрос и модель ответит HTTP 400.
    m = str(model or "").strip().rsplit("/", 1)[-1].lower()
    return not any(m.startswith(p) for p in _TEMPERATURE_UNSUPPORTED_PREFIXES)


# BLOCK 1.1 (06-22) — day-rotating "angle" for idea generation (time-decay).
# Indexed by day-of-year so successive reports nudge the model toward DIFFERENT
# fresh angles instead of the same blue-chips every run.  Deterministic (same
# day → same angle = reproducible report) but varies daily.
_IDEA_ROTATION_ANGLES = [
    "приоритизируй НЕДОоценённые сектора текущего режима, а не очевидные мегакэпы",
    "предложи имена СРЕДНЕЙ капитализации (mid-cap) с сильными фундаменталиями",
    "усиль НЕДОпредставленный в портфеле фактор (Value/Quality/Momentum) точечной идеей",
    "добавь международную / EM-идею для географической диверсификации",
    "приоритизируй имена с недавним позитивным пересмотром прибыли и сильным momentum",
    "сделай акцент на качестве с реальной премией, избегая дежурных дивидендных мегакэпов",
    "предложи тематическую идею под текущий макро-режим (а не «вечные» имена)",
]


# Sprint-5.3 (замечание 2): plain-Russian regime names.  The engine's labels
# (Expansion/Recovery/Slowdown/Recession) are finance jargon — feed the model
# a human phrase so the verdict/comments read naturally for a retail user.
_REGIME_RU: dict[str, str] = {
    "Expansion": "экономика растёт",
    "Recovery":  "рынок восстанавливается после спада",
    "Slowdown":  "экономика замедляется",
    "Recession": "экономический спад",
}


def _regime_ru(label: str) -> str:
    return _REGIME_RU.get(str(label or "").strip(), "режим рынка не определён")


# ── Structured Outputs tool (Tools API) ──────────────────────────────────────
# Forcing the narrative through a typed tool call makes the response a
# GUARANTEED structured object the SDK parses for us — eliminating the
# brace-finding + _repair_truncated_json hack on possibly-truncated free text.
# The property names mirror exactly what the downstream extractor reads via
# parsed.get(...), and the per-field CONTENT rules still come from the user
# prompt (lengths, citation tags, anti-re-aggregation directives).
REPORT_TOOL: dict = {
    "name": "emit_report",
    "description": ("Вернуть институциональный нарратив строго в этой структуре. "
                    "Заполни КАЖДОЕ поле по правилам из пользовательского запроса "
                    "(лимиты длины, источники [Quant Engine]/[Regime]/[RAG], "
                    "запрет на самостоятельную агрегацию секторов)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict":                {"type": "string"},
            "plain_summary":          {"type": "string"},
            "bullets":                {"type": "array", "items": {"type": "string"}},
            "action_plan_text":       {"type": "string"},
            # KPI-strip notes.  ai_cvar_note was MISSING here while the prompt
            # asks for it and the extractor reads it (line ~1273): a model that
            # adheres strictly to the declared schema (Sonnet/BASE) then dropped
            # it, leaving the CVaR card blank in the live BASE report while
            # Sharpe/MDD rendered.  Declaring every field the extractor consumes
            # makes the structured output deterministic across models.
            "ai_cvar_note":           {"type": "string"},
            "ai_sharpe_note":         {"type": "string"},
            "ai_mdd_note":            {"type": "string"},
            # Четвёртая карточка полосы KPI (волатильность) существует ТОЛЬКО в
            # BASE — в DEEP её роль играет отдельная риск-панель.  Заметка нужна
            # затем же, зачем три соседние: карточка без пояснения в ряду из
            # четырёх читается как недогруженная, а не как «пояснять нечего».
            "ai_vol_note":            {"type": "string"},
            "ai_risk_comment":        {"type": "string"},
            "ai_holdings_comment":    {"type": "string"},
            "ai_sector_comment":      {"type": "string"},
            "ai_factor_comment":      {"type": "string"},
            "ai_4pillar_comment":     {"type": "string"},
            "ai_benchmark_comment":   {"type": "string"},
            "ai_performance_comment": {"type": "string"},
            "ai_regime_comment":      {"type": "string"},
            # DEEP-tier section comments — likewise extracted downstream, so
            # they must be declared or a schema-strict model silently omits them.
            "ai_stress_comment":      {"type": "string"},
            "ai_action_comment":      {"type": "string"},
            "ai_effect_comment":      {"type": "string"},
            # Sprint-5 (margin/leverage AI-trigger): when the book carries
            # margin debt (cash leg < 0) the prompt sets has_leverage and
            # REQUIRES this field — an explicit Margin-Call / exponential-risk
            # warning.  Empty string when the book is unlevered.
            "ai_leverage_warning":    {"type": "string"},
            # Nested structures stay permissive — _normalise_stock_picks and
            # _regime_confirmation() validate their shape downstream.
            "stock_picks":            {"type": "object"},
            "regime_confirmation":    {"type": "object"},
        },
        "required": ["verdict", "bullets"],
    },
}


# ── System prompt ────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    here = os.path.dirname(__file__)
    for path in [
        os.path.join(here, "..", "SYSTEM_PROMPT.md"),
        os.path.join(here, "..", "..", "SYSTEM_PROMPT.md"),
    ]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            continue
    return ("You are an investment-advisory and risk-management assistant. "
            "Output advisory-only text in Russian. Never mention the brand "
            f"('{branding.project_name()}', '{branding.bot_name()}', 'RAMP').")


# ── Sector figures fed to the LLM (SSOT — shared with the report panel) ──────

def _sector_shares_long_book(results: dict) -> dict:
    """Sector weights as share of the LONG book — the SAME basis the report
    panel renders, so the AI prose and the panel agree."""
    se = results.get("sector_exposure") or {}
    long_only = [(s, float(w)) for s, w in se.items() if float(w) > 0]
    lsum = sum(w for _, w in long_only) or 1.0
    return {s: round(w / lsum * 100, 1) for s, w in long_only}


def _sector_complex_figure(results: dict) -> dict | None:
    """Authoritative combined super-group figure (e.g. Tech-комплекс).

    Computed via the SAME grouping `pdf_payload.build_sector_groups` uses, so
    the model cites one number instead of re-aggregating Tech+Semiconductors
    ad-hoc (the BASE 55% vs DEEP 80.8% bug)."""
    se = results.get("sector_exposure") or {}
    long_only = [(s, float(w)) for s, w in se.items() if float(w) > 0]
    lsum = sum(w for _, w in long_only) or 1.0
    pairs = [(s, w / lsum) for s, w in long_only]
    try:
        from pdf_payload import build_sector_groups  # leaf import, no cycle
        groups = build_sector_groups(pairs)
    except Exception:
        return None
    return groups[0] if groups else None


# ── Compact summary for LLM input ───────────────────────────────────────────

def _regime_for_prompt(regime: Optional[dict]) -> Optional[dict]:
    """Copy the regime dict and add `confidence_pct` = int(round(conf*100)) so
    the AI echoes the SAME integer the report panel renders (no 74% vs 75%)."""
    if not isinstance(regime, dict):
        return regime
    out = dict(regime)
    try:
        out["confidence_pct"] = int(round(float(regime.get("confidence") or 0) * 100))
    except (TypeError, ValueError):
        pass
    return out


def _money_for_prompt(metrics: dict, total_value) -> dict:
    """Денежные эквиваленты рисковых метрик — ОДНА конвенция с KPI-карточкой.

    Повторяет `pdf_payload._dollar`: сумма считается от ПОКАЗАННОГО процента
    (округлённого до одного знака), а не от сырой доли. Иначе текст ИИ и
    карточка расходятся на пару долларов при одном и том же числе (R-13).
    """
    try:
        tv = float(total_value or 0)
    except (TypeError, ValueError):
        tv = 0.0
    if tv <= 0:
        return {}

    def _d(frac) -> Optional[str]:
        try:
            pct_disp = round(float(frac or 0) * 100.0, 1)     # тот же округлённый %
        except (TypeError, ValueError):
            return None
        return f"${abs(pct_disp / 100.0 * tv):,.0f}"

    out = {}
    for key, src in (("cvar_95", "CVaR_95_Daily"),
                     ("var_95", "VaR_95_Daily"),
                     ("max_dd", "Max_Drawdown")):
        val = _d(metrics.get(src))
        if val:
            out[key] = val
    return out


def _summarise_for_prompt(results: dict) -> dict:
    """
    Build a compact, JSON-serialisable view of analyze_all() results.
    Kept small to keep input tokens bounded.
    """
    metrics = results.get("portfolio_metrics") or {}
    perf    = results.get("performance_table")
    perf_rows: list[dict] = []
    if perf is not None and not perf.empty:
        df = perf.copy()
        if "Current_Value" in df.columns:
            df = df.sort_values("Current_Value", ascending=False)
        for _, row in df.head(25).iterrows():
            # SANITIZE every string field — these flow straight into the LLM
            # prompt, so a broker-supplied ticker like "<ignore_above>" or a
            # sector field with hidden instructions must never reach Claude.
            perf_rows.append({
                "ticker":   _safe_ticker(row.get("Ticker")),
                "weight":   round(float(row.get("Current_Value") or 0)
                                  / float(results.get("total_value") or 1) * 100, 2),
                "trc_pct":  _safe_round(row.get("Euler_Risk_Contribution_Pct"), 2),
                "atr_pct":  _safe_round(row.get("ATR_Pct"), 2),
                "pnl_pct":  _safe_round((row.get("Return_Pct") or 0) * 100, 2),
                "score":    _safe_round(row.get("Score_Total"), 1),
                "action":   _safe_text(row.get("Score_Action"), 20),
                "sector":   _safe_text(row.get("Fundamental_Sector"), 30),
                "beta_mkt": _safe_round(row.get("Beta_Market"), 2),
                "sec_roe":  _safe_round(row.get("SEC_ROE"), 3),
                "sec_debt": _safe_round(row.get("SEC_Debt_to_Assets"), 3),
                "sec_margin": _safe_round(row.get("SEC_Op_Margin"), 3),
            })
    bm = {
        name: {
            "ir":     _safe_round(d.get("Information_Ratio"), 2),
            "te":     _safe_round((d.get("Tracking_Error") or 0) * 100, 1),
            "excess": _safe_round((d.get("Excess_Return_Ann") or 0) * 100, 1),
        }
        for name, d in (results.get("benchmark_comparison") or {}).items()
    }
    cvar_boot = metrics.get("CVaR_95_Bootstrap") or {}

    # Compact macro-driver pack — surfaced so the AI can cross-check the
    # engine's regime label against the actual yield curve / credit spread /
    # volatility / inflation signals (FRED).  Only series with a real value
    # are included; status flag is kept so the AI weights stale data lower.
    macro_src = results.get("macro_drivers") or {}
    _MACRO_KEYS = (
        ("yield_curve_10y2y", "yield_curve_10y2y"),
        ("hy_credit_spread",  "hy_oas"),
        ("vix",               "vix"),
        ("breakeven_inflation", "breakeven"),
        # BLOCK 3.4 — hard macro series the AI should weigh by their TREND.
        ("unemployment",      "unemployment"),
        ("gdp_growth",        "gdp_growth"),
    )

    def _macro_trend(row: dict):
        """Темп ряда — ТЕМ ЖЕ окном, что показано пользователю в панели.

        R-2/R-3 (аудит отчётов 2026-07-29): здесь был жёстко зашит ``lag=3``,
        тогда как панель считает по частоте публикации (дневной ряд → 21
        наблюдение, подпись «за 1м»).  Один индикатор получал две оценки:
        пользователь видел «VIX ▲ +1.61 за 1м», модель — +1.79 и честно его
        пересказывала.  На инфляционных ожиданиях окна разошлись по ЗНАКУ —
        панель «▲ растут», текст «снижаются».  Модель ничего не выдумывала:
        расходились два расчёта движка.

        Теперь окно берётся из единого источника ``regime.macro_trend_lookback``,
        и вместе с величиной модели передаётся ПОДПИСЬ окна, чтобы в тексте
        можно было честно сказать «за 1м», а не безымянную дельту.
        """
        hist = row.get("history_30d") or []
        vals = [h.get("value") for h in hist if isinstance(h, dict)]
        try:
            from finance.regime import macro_trend_lookback, series_trend
            lag, window = macro_trend_lookback(row.get("publish_cadence"))
            total, _slope, _n = series_trend(vals, lag)
        except Exception:
            total, window = None, ""
        if total is None:
            return None, ""
        return _safe_round(total, 2), window

    macro_summary: dict = {}
    for src_key, label in _MACRO_KEYS:
        row = macro_src.get(src_key) or {}
        val = row.get("value")
        if val is None:
            continue
        # Audit 2026-06-14: FRED-sourced, but sanitize for defense-in-depth so
        # NO free-text field reaches the prompt un-fenced (closes the one
        # exception to the "sanitize every string" invariant).
        entry = {
            "value":  _safe_round(val, 2),
            "status": _safe_text(row.get("status"), 16),
            "as_of":  _safe_text(row.get("as_of"), 16),
            "unit":   _safe_text(row.get("unit"), 12),
        }
        # F3 — include the rate-of-change (темп роста/падения).  A rising 4.1%
        # unemployment vs a falling 4.1% are OPPOSITE regime signals.
        trend, trend_window = _macro_trend(row)
        if trend is not None:
            entry["trend"] = trend
            # Подпись окна («1м» / «3м» / «3кв») — чтобы модель писала «+1.61 за 1м»,
            # а не безымянную дельту, и чтобы число совпадало с чипом панели.
            if trend_window:
                entry["trend_window"] = trend_window
        macro_summary[label] = entry

    return {
        "metrics": {
            "vol_ann":      _safe_round((metrics.get("Total_Volatility_Ann") or 0) * 100, 1),
            "sharpe":       _safe_round(metrics.get("Sharpe_Ratio"), 2),
            "sortino":      _safe_round(metrics.get("Sortino_Ratio"), 2),
            "var_95":       _safe_round((metrics.get("VaR_95_Daily") or 0) * 100, 2),
            "cvar_95":      _safe_round((metrics.get("CVaR_95_Daily") or 0) * 100, 2),
            "cvar_lo95":    _safe_round((cvar_boot.get("lo95") or 0) * 100, 2),
            "cvar_hi95":    _safe_round((cvar_boot.get("hi95") or 0) * 100, 2),
            "max_dd":       _safe_round((metrics.get("Max_Drawdown") or 0) * 100, 2),
            "max_erc_pct":  _safe_round(metrics.get("Max_Euler_Risk_Pct"), 1),
            "composite":    metrics.get("Composite_Risk_Score"),
        },
        # R-13 (2026-08-02): ГОТОВЫЕ рублёвые/долларовые суммы — те же строки,
        # что печатает KPI-карточка.  Раньше модель получала только процент и
        # NAV и считала произведение сама, а `pdf_payload._dollar` намеренно
        # берёт ПОКАЗАННЫЙ (округлённый до 0.1) процент — чтобы «−3.3%» и
        # сумма рядом были согласованы.  Две разные конвенции округления дали
        # в живом отчёте 02.08 «$467» на карточке против «около 469 долларов»
        # в тексте ИИ.  Теперь сумма приходит готовой и цитируется дословно.
        "money": _money_for_prompt(metrics, results.get("total_value")),
        "total_value":  _safe_round(results.get("total_value"), 0),
        # Audit 06-25: feed the AI the SAME pre-rounded confidence integer the
        # panel shows (int(round(conf*100))) so the prose can't print 74% next to
        # the panel's 75%.  `confidence_pct` is authoritative — the prompt tells
        # the model to echo it verbatim.
        "regime":       _regime_for_prompt(results.get("regime")),
        "macro":        macro_summary,
        # Share-of-long-book basis (matches the report panel) + the authoritative
        # combined super-group figure.  The prompt forbids re-aggregating sectors
        # — the model must cite `sector_complex` for combined tech exposure.
        "sectors":         _sector_shares_long_book(results),
        "sector_complex":  _sector_complex_figure(results),
        "benchmarks":   bm,
        # B1 (2026-07-17): мандатный бенчмарк клиента по ИМЕНИ + его факторные
        # беты (если движок их посчитал) — ИИ говорит «перекос относительно
        # Nasdaq 100», а не абстрактно «относительно рынка».
        "benchmark_profile": _benchmark_profile_for_prompt(results),
        "holdings":     perf_rows,
        "action_plan":  (results.get("action_plan") or [])[:8],
        "asset_scores": {t: s for t, s in
                         (results.get("asset_scores") or {}).items()},
        # ── Stress scenarios with per-asset CAPPED deltas ──────────────────
        # Sonnet's old narrative kept re-computing β × shock by hand, which
        # bypassed the H4 convex cap and produced "AVGO −42%, NVDA −47%"
        # against table figures of −15.6% port.  Feed the engine's CAPPED
        # `asset_delta_pct` directly so the AI cites the same numbers the
        # table shows.
        "stress_scenarios": [
            {
                "name":          s.get("name"),
                "port_pct":      _safe_round((s.get("port_pct") or 0) * 100, 2),
                "max_dd_pct":    _safe_round((s.get("max_dd_pct") or 0) * 100, 2),
                "recovery_mo":   s.get("recovery_months"),
                # Top-5 per-asset CAPPED impacts (after convex cap ±35%);
                # `asset_delta_pct` ALREADY %-scaled (not decimal) — keep
                # as-is so the AI quotes the number verbatim.
                "top_assets": [
                    {"ticker":  a.get("ticker"),
                     "delta_pct_capped": a.get("asset_delta_pct"),
                     "delta_pct_raw":    a.get("asset_delta_raw"),
                     "weight_pct":       a.get("weight_pct")}
                    for a in (s.get("by_asset") or [])[:5]
                ],
            }
            for s in (results.get("stress_scenarios") or [])
        ],
        # Reporting-currency / RFR metadata so the AI never writes the
        # wrong RFR in trailing commentary (Sharpe, Sortino).
        "reporting": {
            "currency": (results.get("portfolio_metrics") or {}).get("reporting_currency"),
            "rfr_ann":  (results.get("portfolio_metrics") or {}).get("risk_free_rate_annual"),
        },
        # Composite verdict from simulate_after_plan — the AI must echo
        # this verdict in `ai_effect_comment` instead of inventing its own.
        "rebalance_verdict": ((results.get("expected_effect") or {}).get("verdict")
                              if isinstance(results.get("expected_effect"), dict) else None),
        # 🔴 Аудит 2026-08-12: ПОГАШЕНА ЛИ ДЕЛЬТА SHARPE.  При отрицательной
        # премии (er < rf) Sharpe перестаёт упорядочивать портфели, движок
        # обнуляет дельту и печатает причину — а промпт `ai_effect_comment`
        # при этом ПРОДОЛЖАЛ требовать «…нестабильность (Vol), Sharpe».
        # Модель добросовестно исполняла требование: живой отчёт 12.08 писал
        # «отдача на риск улучшается» ровно рядом с карточкой «Sharpe 0.56 →
        # 0.56 · дельта не показана — смотрите на волатильность, CVaR и
        # просадку».  Промпт сам провоцировал ложь — тот же класс, что литерал
        # «8%» (`§−48`).  Факт обязан доехать до модели, иначе запрет ниже ей
        # не на чем исполнить.
        "sharpe_delta_suppressed": bool(
            (results.get("expected_effect") or {}).get("sharpe_note")
            if isinstance(results.get("expected_effect"), dict) else False),
        "sharpe_suppressed_reason": str(
            (results.get("expected_effect") or {}).get("sharpe_note") or ""
            if isinstance(results.get("expected_effect"), dict) else ""),
        # L-18/L-19: сами сделки ребаланса (Продать/Купить/В кэш, Δпп, имена
        # внешних ETF) — чтобы ai_effect_comment называл направления по имени,
        # а не пересказывал одни метрики.
        "rebalance_actions": ([
            {"ticker": a.get("ticker"), "side": a.get("side"),
             "delta_pp": a.get("delta_pp"), "name": (a.get("name") or None),
             "external": bool(a.get("is_external")), "cash": bool(a.get("is_cash"))}
            for a in ((results.get("expected_effect") or {}).get("high_priority_actions") or [])[:12]
        ] if isinstance(results.get("expected_effect"), dict) else []),
        # R-5 (2026-08-02): ГОТОВАЯ строка адресата реинвеста.  Модель уже
        # получала `rebalance_actions`, но в живом отчёте 30.07 всё равно
        # написала «высвобожденный вес идёт в кэш и снижение маржинального
        # долга», тогда как движок покупал IEF/EEM/EMB (+21.41 пп = −21.41 пп
        # продаж).  Инструкция «назови из rebalance_actions … и/или Кэш»
        # оставляла выбор формулировки модели.  Собираем фразу ДЕТЕРМИНИРОВАННО
        # — модель её цитирует, а не сочиняет (то же правило, что numbers_rule:
        # ИИ описывает то, что посчитал движок).
        "reinvest_destination": _reinvest_destination(results),
        # H4: investor risk mandate so the LLM tailors tone/recommendations
        # (a conservative investor needs tail-risk framing; an aggressive one
        # growth framing).  Composite-risk score is already calibrated for it.
        "risk_mandate": results.get("risk_mandate", "MODERATE"),
        # Sprint-5 (margin/leverage): surface the engine's leverage metrics so
        # the model can quote the actual gross-exposure / leverage-ratio when
        # warning about a margin-funded book (cash leg < 0 → is_leveraged).
        "leverage": _leverage_for_prompt(results.get("leverage_metrics")),
        # Factor-variance decomposition + факторные двойники (additive layer,
        # finance/factor_decomposition) — grounds ai_factor_comment in the
        # SAME numbers the report's «Источники риска» panel shows, вместо
        # догадок по одной лишь Beta_Market.
        "factor_decomposition": _factor_decomposition_for_prompt(results),
        # R-3: наклон против бенчмарка ГОТОВОЙ строкой — см. `_factor_tilt_text`.
        "factor_tilt_text": _factor_tilt_text(results),
    }


def _benchmark_profile_for_prompt(results: dict) -> Optional[dict]:
    """
    Compact view of the user's mandate benchmark for the prompt (B1 2026-07-17).

    Keys: ticker, name, betas ({axis: β} — None when the engine couldn't fit
    them; the factor section then falls back to the S&P 500 constant and the
    prompt's benchmark rule says so).  None when no mandate benchmark is set.
    """
    bfp = results.get("benchmark_factor_profile") or {}
    if isinstance(bfp, dict) and bfp.get("betas"):
        return {
            "ticker": _safe_ticker(bfp.get("ticker")),
            "name":   _safe_text(bfp.get("name"), 44),
            "betas":  {_safe_text(k, 20): _safe_round(v, 2)
                       for k, v in (bfp.get("betas") or {}).items()},
        }
    tk = results.get("profile_benchmark_ticker")
    if not tk:
        return None
    try:
        from profile_manager import BENCHMARK_LIST as _BML
        name = _BML.get(str(tk), str(tk))
    except Exception:                                 # pragma: no cover
        name = str(tk)
    return {"ticker": _safe_ticker(tk), "name": _safe_text(name, 44),
            "betas": None}


def _factor_tilt_text(results: dict) -> str:
    """Готовая фраза про активный наклон — модель её ЦИТИРУЕТ, а не считает.

    🔴 Живой DEEP 17.08 (`§−97`): в факторной таблице рядом стояли Value
    портфеля −0.36 и Value бенчмарка −0.08. Нарратив написал «портфель ещё
    сильнее в рынке и импульсе и почти нулевой в недооценённых акциях
    (Value ≈ −0,08 у бенчмарка)» — то есть взял число БЕНЧМАРКА, описал им
    наклон ПОРТФЕЛЯ и получил вывод, противоположный правде: −0.36 это не
    «почти ноль», а заметный недовес стоимости.

    Лечится приёмом `§−44` R-5: не просить модель пересказать число, а дать
    готовую строку. Разность считает движок (`finance.factor_decomposition.
    active_factor_tilt`), здесь — только формат, как и положено слою отчёта.
    """
    try:
        from finance.factor_decomposition import active_factor_tilt
    except Exception:                                   # pragma: no cover - defensive
        return ""
    fd  = (results.get("portfolio_metrics") or {}).get("factor_decomposition") or {}
    bfp = results.get("benchmark_factor_profile") or {}
    rows = active_factor_tilt(fd.get("betas_covered") or {}, bfp.get("betas") or {})
    if not rows:
        return ""
    parts = [
        f"{r['factor']}: портфель {r['port']:+.2f} против {r['bench']:+.2f} "
        f"у бенчмарка ({'перевес' if r['tilt'] > 0 else 'недовес'} {abs(r['tilt']):.2f})"
        for r in rows
    ]
    return "; ".join(parts)


def _factor_decomposition_for_prompt(results: dict) -> dict:
    """
    Compact view of portfolio_metrics["factor_decomposition"] for the prompt.

    Keys (все проценты уже округлены движком — модель цитирует verbatim):
      var_shares    — {источник риска: % дисперсии портфеля} (знак сохранён;
                      отрицательный = фактор-хедж),
      idio_pct      — % дисперсии, не объяснённый ни одним фактором,
      betas         — портфельные β по осям (шкала таблицы отчёта),
      top_drivers   — {фактор: "TICKER w·β, ..."} топ-вклады w·β,
      twins         — пары «факторных двойников» (systematic corr ≥ 0.90):
                      одна факторная ставка куплена дважды,
      most_unique   — топ-3 позиции по доле СОБСТВЕННОГО (идио) риска —
                      кандидаты «истинных диверсификаторов»/уникальных ставок.
    """
    fd = (results.get("portfolio_metrics") or {}).get("factor_decomposition") or {}
    if not fd:
        return {}
    drivers = {
        _safe_text(f, 20): ", ".join(
            f"{_safe_ticker(d.get('ticker'))} {_safe_round(d.get('contribution'), 2)}"
            for d in rows[:3]
        )
        for f, rows in (fd.get("driven_by") or {}).items()
    }
    unique = [u for u in (fd.get("unique_risk") or [])
              if u.get("unique_risk_pct") is not None and (u.get("weight_pct") or 0) >= 2.0]
    unique.sort(key=lambda u: -(u.get("unique_risk_pct") or 0))
    return {
        "var_shares": {_safe_text(g.get("source"), 44): g.get("share_pct")
                       for g in (fd.get("group_shares") or [])},
        "idio_pct":   fd.get("idio_pct"),
        "betas":      {_safe_text(k, 20): v
                       for k, v in (fd.get("betas_covered") or {}).items()},
        "top_drivers": drivers,
        "twins": [
            {"pair": [_safe_ticker(p) for p in (t.get("pair") or [])],
             "corr": t.get("systematic_corr"),
             "combined_weight_pct": t.get("combined_weight_pct")}
            for t in (fd.get("twins") or [])
        ],
        "most_unique": [
            {"ticker": _safe_ticker(u.get("ticker")),
             "unique_risk_pct": u.get("unique_risk_pct"),
             "weight_pct": u.get("weight_pct")}
            for u in unique[:3]
        ],
    }


def _reinvest_destination(results: dict) -> str:
    """R-5: куда движок ФАКТИЧЕСКИ отправляет высвобожденный вес.

    Готовая русская фраза из buy-стороны `high_priority_actions` — модель её
    цитирует дословно. Пустая строка = плана реинвеста нет, и тогда промпт
    запрещает выдумывать адресат вообще.
    """
    ee = results.get("expected_effect")
    if not isinstance(ee, dict):
        return ""
    def _pp(a) -> float:
        try:
            return float(a.get("delta_pp") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    buys = [a for a in (ee.get("high_priority_actions") or [])
            if isinstance(a, dict)
            and str(a.get("side", "")).lower() == "buy"
            and _pp(a) > 0]
    if not buys:
        return ""
    buys.sort(key=lambda a: -_pp(a))
    parts = []
    for a in buys[:4]:
        t = str(a.get("ticker") or "").strip()
        if not t:
            continue
        nm = str(a.get("name") or "").strip()
        dw = _pp(a)
        if a.get("is_cash"):
            parts.append(f"Кэш (+{dw:.1f} пп)")
        elif nm:
            parts.append(f"{t} — {nm} (+{dw:.1f} пп)")
        else:
            parts.append(f"{t} (+{dw:.1f} пп)")
    return " · ".join(parts)


# `§−121` — правило §−13 («плечо и долг видны ТОЛЬКО при отрицательном кэше»)
# для ПРОЗЫ модели. Словарь намеренно узкий и однозначный: «маржинальность»
# (рентабельность из SEC) и «плечевой ETF» (свойство инструмента) — законные
# слова, их трогать нельзя. Ловится только ЗАЁМ СЧЁТА.
# «Плечо» — только СУЩЕСТВИТЕЛЬНОЕ и только при счёте/портфеле/книге:
# «портфель с плечевыми ETF» и «ETF с кредитным плечом 2×» — про инструмент.
_PLECHO = r"плеч(?:о|а|ом|е|у)\b"
_ACCOUNT_LEVERAGE_RE = re.compile(
    r"маржинальн\w*\s+(?:долг|за[её]м|кредит|сч[её]т|позици|требовани)"
    r"|margin[\s-]*call|маржин[\s-]*колл"
    r"|за[её]мн\w*\s+средств"
    rf"|(?:кредитн\w+\s+)?{_PLECHO}\s+(?:портфел|сч[её]т|книг)"
    rf"|(?:портфел\w*|сч[её]т\w*|книг\w*)\s+(?:без|с)\s+(?:кредитн\w+\s+)?{_PLECHO}",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def strip_account_leverage(text: str, leveraged: bool) -> str:
    """Убрать из текста ПРЕДЛОЖЕНИЯ о заёмных средствах счёта, если книга
    без маржи. На маржинальной книге текст не трогается.

    Режется по предложению, а не по слову: вырезанное слово оставило бы
    бессмысленную фразу, а переписать её за модель — значит выдумать.
    """
    if leveraged or not text or not _ACCOUNT_LEVERAGE_RE.search(text):
        return text
    kept = [s for s in _SENTENCE_SPLIT_RE.split(text)
            if not _ACCOUNT_LEVERAGE_RE.search(s)]
    return " ".join(kept).strip()


def _leverage_for_prompt(lm: Optional[dict]) -> dict:
    """Compact, %-scaled leverage view for the LLM prompt.

    `is_leveraged` is True only when the cash leg is negative (margin debt).
    gross_exposure / long_weight / net_exposure are decimals from the engine
    (1.0 == 100% of equity); we %-scale them so the model quotes them verbatim.
    """
    lm = lm or {}
    # `§−121`: на книге БЕЗ маржи модели не отдаётся ни одного числа плеча.
    # Факт «плечо 1.00x / валовая 94%» провоцировал фразы вида «портфель без
    # кредитного плеча» — упоминание плеча там, где правило §−13 требует
    # тишины. Нет факта — нет повода; запрет в промпте стоит рядом.
    if not bool(lm.get("is_leveraged")):
        return {"is_leveraged": False}
    return {
        "is_leveraged":   True,
        "gross_exposure_pct": _safe_round((lm.get("gross_exposure") or 0) * 100, 1),
        "long_weight_pct":    _safe_round((lm.get("long_weight") or 0) * 100, 1),
        "net_exposure_pct":   _safe_round((lm.get("net_exposure") or 0) * 100, 1),
        "margin_debt_pct":    _safe_round(abs(min(0.0, lm.get("cash_weight") or 0)) * 100, 1),
        "leverage_ratio":     _safe_round(lm.get("leverage_ratio"), 2),
        # R-4: НАСКОЛЬКО плечо заметно — вердикт движка, а не «есть/нет».
        "salience": _leverage_salience(
            _safe_round(abs(min(0.0, lm.get("cash_weight") or 0)) * 100, 1)),
    }


def _leverage_salience(margin_debt_pct) -> str:
    """Обёртка над движком: слой отчёта решений о значимости не принимает."""
    try:
        from finance.scoring import leverage_salience
        return leverage_salience(margin_debt_pct)
    except Exception:                                   # pragma: no cover - defensive
        return "material"


def _safe_round(v, digits: int):
    try:
        x = float(v)
        return None if x != x else round(x, digits)
    except (TypeError, ValueError):
        return None


# ── Strip unverified RAG citations ───────────────────────────────────────────

def _strip_unverified_rag_citations(text: str, market_context: str) -> str:
    if not text:
        return text
    if not market_context:
        return re.sub(r"\s*\[RAG:[^\]]*\]", "", text)
    def _keep(match: re.Match) -> str:
        name = match.group(1).strip()
        return match.group(0) if name and name in market_context else ""
    return re.sub(r"\[RAG:\s*([^\]]+)\]", _keep, text)


# ── Bank-citation observability (RAG audit trail) ────────────────────────────
# The narrative references investment-bank views in two DISTINCT ways and the
# CoVe panel must tell them apart so the reader knows whether "Goldman says…"
# came from an actually-ingested report or from the model's own memory:
#   • [RAG: <file>]  — a claim backed by a RETRIEVED bank PDF (already verified
#                      against market_context by _strip_unverified_rag_citations,
#                      so any surviving tag is a TRUE report citation).
#   • bank name in   — bank CONSENSUS the system prompt explicitly allows the
#     prose / [GS]     model to surface from training knowledge even when
#     bracket tags     ChromaDB is empty ("даже без RAG-данных").  NOT proof a
#                      report was read.
_RAG_FILE_CITE_RE = re.compile(r"\[RAG:\s*[^\]]+\]")
# §−95: перечень эмитентов больше НЕ живёт здесь — он один на проект
# (`agent.rag_engine`, SSOT ФАКТОВ).  Разошедшиеся копии стоили нам «Citi» в
# прозе: ингест клал `bank="Citi"`, а этот regex знал только Citigroup|Citibank,
# и модель, честно сославшаяся на Citi, получала «0 ссылок на банки».
# РЕШЕНИЕ при этом остаётся здешним: в ПРОЗЕ засчитываются только полные имена
# (`§−14` C-8 — «MS» это Microsoft/миллисекунды, «GS» слишком общо), а короткие
# формы — исключительно внутри квадратного тега `[JPM]`.
_BANK_NAME_RE = re.compile(
    "(" + "|".join(bank_alias_regex(b) for b in BANK_ORDER) + ")", re.IGNORECASE)
# Bare abbreviations count ONLY inside square-bracket citation tags.
_BANK_TAG_RE = re.compile(
    r"\[(" + "|".join(
        "|".join(p for p in (bank_alias_regex(b), bank_short_regex(b)) if p)
        for b in BANK_ORDER) + r")[^\]]*\]", re.IGNORECASE)


#: Короткая форма имени банка, стоящая В ПРОЗЕ (а не внутри тега `[GS]`).
#: Собирается из ЕДИНОГО реестра (`agent.rag_engine`), поэтому новый банк
#: подхватывается автоматически, а не заводится здесь второй раз (`§−95`).
_BANK_SHORT_IN_PROSE = [
    (bank, re.compile(r"(?<!\[)" + bank_short_regex(bank) + r"(?![^\[\]]*\])",
                      re.IGNORECASE))
    for bank in BANK_ORDER if bank_short_regex(bank)
]


def _expand_bank_short_forms(text: str, held: set[str]) -> str:
    """«против совета GS» → «против совета Goldman Sachs».

    🔴 Живой DEEP 17.08: нарратив написал «против совета GS», а панель
    провенанса рядом сообщила «ИИ сослался на 1 банк(ов) (Barclays)». Оба
    утверждения по-своему верны, и вместе они противоречат друг другу: аудитор
    засчитывает в прозе ТОЛЬКО полные имена (`§−14` C-8 — «MS» это чаще
    Microsoft или миллисекунды), а модель короткую форму всё равно пишет.

    Причина не в модели: короткой форме её научил САМ ПРОМПТ, где примеры
    источников выглядят как `[GS]/[JPM]/[Barclays]`. Правило «в прозе только
    полное имя» жило в коде аудитора и в `CLAUDE.md`, но производителю текста
    его никто не сообщал — тот же класс, что «запрет неисполним без факта»
    (`§−90` A-5). Промпт теперь это правило называет, а функция ниже —
    вторая линия: она чинит текст, если модель всё же сорвалась.

    `held` — тикеры портфеля: если пользователь держит бумагу `MS`, то `MS` в
    прозе это ПОЗИЦИЯ, а не Morgan Stanley, и трогать её нельзя.
    """
    s = str(text or "")
    if not s:
        return s
    for bank, rx in _BANK_SHORT_IN_PROSE:
        def _sub(m: "re.Match[str]") -> str:
            token = m.group(0)
            return token if token.upper() in held else bank
        s = rx.sub(_sub, s)
    return s


def _canon_bank(raw: str) -> str:
    return canonical_bank(raw)


def _count_rag_citations(texts: list[str]) -> dict:
    """Count verified [RAG:file] citations and distinct bank-consensus mentions
    across the FINAL narrative, so the CoVe panel can show the AI's real
    sourcing (report-backed vs. model-knowledge)."""
    blob = "  ".join(t for t in texts if t)
    banks: set[str] = set()
    for m in _BANK_NAME_RE.finditer(blob):
        banks.add(_canon_bank(m.group(1)))
    for m in _BANK_TAG_RE.finditer(blob):
        banks.add(_canon_bank(m.group(1)))
    return {
        "file_cites": len(_RAG_FILE_CITE_RE.findall(blob)),
        "bank_cites": len(banks),
        "banks":      sorted(banks),
    }


# ── JSON repair for truncated responses ──────────────────────────────────────

def _repair_truncated_json(raw: str) -> str:
    """
    Repair JSON truncated by max_tokens.  Closes open strings, arrays, objects.
    """
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    repaired = raw.rstrip()
    repaired = repaired.rstrip(',')
    # Close an open string
    if repaired.count('"') % 2 == 1:
        # Find last unmatched quote and truncate partial value
        last_q = repaired.rfind('"')
        # Check if it's a key or value being written
        before = repaired[:last_q].rstrip()
        if before.endswith(':'):
            # Truncated value — close string
            repaired = repaired + '"'
        else:
            # Truncated mid-string — close it
            repaired = repaired + '"'
    # Remove trailing partial key-value (key without value)
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', '', repaired)
    # Close open arrays and objects
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    repaired = repaired.rstrip(',')
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass
    # Last resort: try to find the last valid closing brace
    for i in range(len(repaired) - 1, 0, -1):
        if repaired[i] == '}':
            candidate = repaired[:i+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return raw  # give up, return original


# ── Narrative prompt (RUSSIAN) ───────────────────────────────────────────────

def _user_prompt(summary: dict, *, tier: str, market_context: str = "",
                 user_profile: str = "Moderate",
                 regime_rag_confirm: list[str] | None = None,
                 user_mandate: dict | None = None) -> str:
    regime = (summary.get("regime") or {})
    regime_label = regime.get("regime", "unknown")
    regime_ru    = _regime_ru(regime_label)   # Sprint-5.3: plain-Russian phrase

    # 2026-07-18: при ОЧЕНЬ низкой уверенности классификатора режим статистически
    # на грани шума (живой отчёт: «Expansion» при confidence 8%).  §6 системного
    # промпта калибрует ТОН, но модель всё равно якорила идеи/тилты на режим.
    # Здесь — жёсткое data-driven правило: ниже порога НЕ строить рекомендации на
    # режиме, опираться на измеримое (риск, концентрация, мандат, бенчмарк).
    # Фолбэк-нарратив ниже тоже гасит режим при низкой уверенности (симметрично).
    try:
        _conf_pct = int(regime.get("confidence_pct")) if regime.get("confidence_pct") is not None else None
    except (TypeError, ValueError):
        _conf_pct = None
    _REGIME_CONF_FLOOR = 25   # ниже — АГРЕГАТНЫЙ ярлык фазы не значим как основа
    regime_confidence_rule = ""
    if _conf_pct is not None and _conf_pct < _REGIME_CONF_FLOOR:
        regime_confidence_rule = (
            f"РЕЖИМ — СЛАБЫЙ АГРЕГАТНЫЙ ЯРЛЫК (уверенность {_conf_pct}%): классификатор почти НЕ "
            f"различает ФАЗУ цикла («{regime_label}» на грани шума), поэтому НЕ строй "
            "рекомендации/идеи/секторные тилты на самом ЯРЛЫКЕ фазы. "
            "НО ЭТО НЕ ЗНАЧИТ «игнорируй режим» — наоборот: КОНКРЕТНЫЕ сигналы этой секции ИЗВЕСТНЫ "
            "точно, читай их УВЕРЕННО, ЧЕСТНО и понятно, каждый по отдельности: кривая доходности "
            "10Y−2Y, кредитный спред HY OAS, VIX, инфляционные ожидания breakeven (уровень ⊕ темп), "
            "безработица и ВВП (направление ТЕМПА), факторные беты портфеля и консенсус банков "
            "[RAG]/[GS]/[JPM]. По КАЖДОМУ скажи, что он означает и risk-on это или risk-off. "
            "Затем СВЯЖИ их с мандатом и риск-профилем клиента и с концентрацией портфеля: что эти "
            "сигналы значат ИМЕННО для этого портфеля. Неуверенность — ТОЛЬКО в агрегатной фазе, "
            "НЕ в самих сигналах; не будь расплывчатым. В ai_regime_comment: одна честная оговорка "
            f"про слабость ярлыка (увер. {_conf_pct}%), далее — уверенный разбор реальных сигналов "
            "и их смысла для портфеля/мандата. Идеи выбирай по качеству (4-Pillar), концентрации и "
            "мандату, а не по «благоприятным секторам фазы». "
            "«Ярлык»/«агрегатный» — внутренние термины: читателю пиши «оценка "
            f"фазы цикла» (напр. «фаза цикла определена неуверенно — {_conf_pct}%»).\n"
            # R-19 (2026-08-02): здесь СТОЯЛ литерал «8%» — число из живого
            # отчёта июля, оставленное как пример формулировки.  Модель взяла
            # его как ДАННЫЕ: в отчёте 02.08 при реальной уверенности 1%
            # напечаталось «определена крайне неуверенно (8%… фактически 1%)».
            # Пример обязан нести ТЕКУЩЕЕ число, иначе он фабрикует факт.
            f"ЕДИНСТВЕННОЕ допустимое число уверенности фазы — {_conf_pct}%. "
            "Никакого другого процента уверенности в тексте быть не должно.\n"
        )

    # B1 (2026-07-17): бенчмарк клиента по ИМЕНИ — комментарии секций обязаны
    # сравнивать с ним, а не с абстрактным «рынком».
    _bench_prof  = summary.get("benchmark_profile") or {}
    bench_label  = str(_bench_prof.get("name") or "S&P 500")
    benchmark_rule = ""
    if _bench_prof:
        benchmark_rule = (
            f"БЕНЧМАРК КЛИЕНТА — {bench_label}: в ai_benchmark_comment и "
            "ai_factor_comment называй бенчмарк ПО ИМЕНИ, а не абстрактно "
            "«рынок». "
            + ("Факторные беты бенчмарка — summary.benchmark_profile.betas; "
               "«Наклон Δ» в отчёте = β портфеля − β бенчмарка.\n"
               if _bench_prof.get("betas") else
               "Беты выбранного бенчмарка недоступны — факторное сравнение в "
               "отчёте приведено к S&P 500 как общей оси; скажи об этом, если "
               "комментируешь факторный перекос.\n")
            + ("СЕКТОРНАЯ КОНЦЕНТРАЦИЯ в контексте бенчмарка: если бенчмарк сам "
               "секторно-концентрирован (напр. Nasdaq 100 — техно-индекс), "
               "часть перекоса портфеля в тот же сектор ОЖИДАЕМА и согласуется с "
               "выбором клиента — отметь это. НО concentration-риск этим НЕ "
               "отменяется: доля сектора ВЫШЕ, чем в самом бенчмарке (и «мягкий "
               "лимит диверсификации» 40% — общая прудентная эвристика, не "
               "мандат), — это по-прежнему уязвимость к секторному спаду; так и "
               "формулируй, без ложного успокоения.\n")
        )

    # B1 (2026-07-17): мандатные лимиты клиента — идеи не должны противоречить
    # мандату (класс с лимитом 0–0 запрещён; напр. Crypto у консервативного).
    mandate_rule = ""
    if user_mandate:
        _limits = user_mandate.get("limits_dict") or {}
        _banned: list[str] = []
        for _k, _b in _limits.items():
            try:
                _lo, _hi = float(_b[0]), float(_b[1])
            except (TypeError, ValueError, IndexError):
                continue
            if _lo == 0 and _hi == 0:
                _banned.append(str(_k))
        _pname = _safe_text(user_mandate.get("profile_name"), 40)
        try:
            from profile_manager import ASSET_DISPLAY as _AD
        except Exception:                              # pragma: no cover
            _AD = {}
        _lines = []
        if _pname:
            _tv = user_mandate.get("target_volatility")
            _tvs = (f" · целевая волатильность ≈{float(_tv) * 100:.0f}%"
                    if isinstance(_tv, (int, float)) else "")
            _lines.append(f"- Риск-профиль мандата: {_pname}{_tvs}.\n")
        # Class limits (skip 0–0 banned; they get their own harder rule) so the
        # AI can NAME an over-limit class as a mandate breach and steer the plan
        # back inside the bands — «Акции США 84% превышает лимит 30–60%».
        _limit_bits = []
        for _k, _b in _limits.items():
            try:
                _lo, _hi = float(_b[0]), float(_b[1])
            except (TypeError, ValueError, IndexError):
                continue
            if _lo == 0 and _hi == 0:
                continue
            _limit_bits.append(f"{_AD.get(_k, _k)} {int(_lo)}–{int(_hi)}%")
        if _limit_bits:
            _lines.append(
                "- Лимиты классов активов мандата: " + ", ".join(_limit_bits) +
                ". Если фактическая доля класса ВЫШЕ верхней границы — это "
                "нарушение мандата: назови его в нарративе и в Action Plan веди "
                "долю ОБРАТНО в лимит.\n")
        if _banned:
            _names = ", ".join(_AD.get(k, k) for k in _banned)
            _lines.append(
                f"- ЗАПРЕЩЁННЫЕ классы активов (лимит мандата 0–0%): {_names}. "
                "НЕ предлагай инструменты этих классов ни в stock_picks, ни в "
                "текстовых рекомендациях.\n")
        if _lines:
            mandate_rule = "МАНДАТ КЛИЕНТА — СТРОГО:\n" + "".join(_lines)

    # Sprint-5.3 (замечание 2) — shared hard rule banning jargon the retail
    # reader can't parse.  Injected into BOTH the base and deep prompts.
    plain_rule = (
        "ЯЗЫК — СТРОГО ДЛЯ НЕСПЕЦИАЛИСТА:\n"
        f"- Режим рынка называй ТОЛЬКО по-русски: «{regime_ru}». НЕ пиши "
        "Expansion/Recovery/Slowdown/Recession.\n"
        "- ЗАПРЕЩЕНЫ слова: trailing, trailing 12M, ДИ, CI, CVaR, TRC, Hotspot, "
        "Sharpe, Sortino, Beta, IR, HHI, Value, Growth, Financials — как "
        "голые термины. Пиши простыми словами:\n"
        "  trailing / trailing 12M → «за последние 12 месяцев»\n"
        "  ДИ / доверительный интервал → «разброс оценки» или просто диапазон без аббревиатуры\n"
        "  CVaR → «средний убыток в редкий плохой день (такие дни случаются примерно раз в месяц)»\n"
        "  Sharpe → «отдача на единицу риска» (можно «коэффициент Шарпа X» в скобках ОДИН раз)\n"
        "  TRC / доля в риске → «сколько процентов риска даёт эта позиция»\n"
        "  Beta → «во сколько раз сильнее рынка двигается»\n"
        "  Value/Growth/Financials → «недооценённые акции» / «акции роста» / «банки и финансы»\n"
        "- Если без термина никак — добавь короткое пояснение в скобках ОДИН раз, "
        "дальше используй простое слово.\n"
    )

    # L-16 (2026-07-19): анти-шаблон.  Глоссарий замен модель вставляла ДОСЛОВНО
    # в каждое поле («В худший день из 20 теряется ≈3.5%» — в verdict, KPI-note
    # И риск-комментарии одновременно) → отчёт звучит как копипаста.  Плюс
    # самодельные сокращения («счёт −5.2») и внутренние термины («ярлык»),
    # непонятные читателю.
    style_rule = (
        "СТИЛЬ — БЕЗ ПОВТОРОВ И ЖАРГОНА:\n"
        "- Глоссарий замен — это СЛОВАРЬ, а не готовые фразы: НЕ вставляй одну "
        "и ту же формулировку (напр. расшифровку CVaR) дословно в "
        "несколько секций. Развёрнутое объяснение термина — ОДИН раз за отчёт "
        "(при первом упоминании); дальше — короткая форма или просто число. "
        "Синонимичные формулировки приветствуются.\n"
        "- Каждая секция добавляет НОВУЮ мысль (причину, следствие, действие), "
        "а не пересказывает соседнюю с теми же цифрами.\n"
        "- ЗАПРЕЩЕНО в тексте для читателя: слово «счёт» как сокращение оценки "
        "(пиши «сводная оценка 4-Pillar −5.2 — продавать»), слово «ярлык» "
        "(пиши «оценка фазы цикла»).\n"
        "- СТРУКТУРА комментария (для полей длиннее 200 знаков): 2–3 ПОЛНЫХ "
        "предложения — (1) факт с числом из данных движка, (2) что это значит "
        "для владельца портфеля, (3) действие или связь с мандатом/бенчмарком/"
        "банковской аналитикой. Не обрывай мысль и не пиши телеграфные обрывки.\n"
    )

    # L-19 (2026-07-19): анти-фабрикация чисел — владелец потребовал «пусть не
    # придумывает ложные цифры и округления»: каждое число в тексте обязано
    # существовать в данных промпта; пере-округления запрещены.
    numbers_rule = (
        "ЧИСЛА — СТРОГО ИЗ ДАННЫХ:\n"
        "- Используй ТОЛЬКО числа, которые есть в данных выше. Не выдумывай доли, "
        "суммы, даты и «примерные» оценки, которых в данных нет.\n"
        "- Не пере-округляй: цифру приводи как в данных (допустимо сократить до "
        "одного знака после запятой, не более).\n"
        "- Нет данных для утверждения — напиши «данных недостаточно», а не сочиняй.\n"
        # R-2…R-5 (аудит живых отчётов 2026-07-29): в тексте появлялись величины,
        # которых в данных не было (бета 3.59), и инструменты, которых не было
        # в плане («продать KZT, USD» — при том что USD это маржинальный ЗАЁМ,
        # закрываемый покупкой).  Правило ниже закрывает оба класса ошибок.
        "- ТЫ НЕ СЧИТАЕШЬ — ТЫ ОПИСЫВАЕШЬ. Все величины уже вычислены движком. "
        "Не складывай, не вычитай, не усредняй, не пересчитывай проценты в деньги "
        "и обратно, не выводи «примерно столько же». Нужного числа нет в данных — "
        "значит его нет в отчёте.\n"
        # R-13: раньше суммы в тексте приходилось выводить самому (был только
        # процент и NAV), и модель нарушала правило выше «по необходимости».
        # Теперь готовая сумма лежит в `money` — цитировать её, а не считать.
        "- ДЕНЕЖНЫЕ суммы для CVaR/VaR/просадки бери ГОТОВЫМИ из блока `money` "
        "(там ровно те строки, что печатает карточка отчёта). Не умножай процент "
        "на стоимость портфеля сам — получится другое число, и текст разойдётся "
        "с карточкой.\n"
        "- ТЕМП (trend) цитируй ВЕРБАТИМ вместе с окном из trend_window: "
        "«+1.61 за 1м». Не пересчитывай темп на другое окно и не меняй его знак: "
        "если trend положительный — показатель РАСТЁТ, отрицательный — ПАДАЕТ, "
        "как бы ни выглядел уровень.\n"
        "- ДЕЙСТВИЯ называй ТОЛЬКО те, что есть в плане (action_plan). Не добавляй "
        "в список продаж/покупок инструменты, которых там нет — в том числе "
        "денежные строки: кэш и маржинальный заём не «продают».\n"
    )

    # Валютное правило (замечание 2026-07-09): модель писала «8 из каждых 10
    # РУБЛЕЙ», хотя портфель в USD — недопустимо.  Инжектим в ОБА промпта.
    # Правило описываем АБСТРАКТНО, без готовых фраз-примеров: раньше пример
    # «8 из каждых 10» модель копировала дословно в отчёт (замечание R2#1).
    _rcur = str(((summary.get("reporting") or {}).get("currency")) or "USD").upper()
    _cur_word = {"USD": "долларов", "KZT": "тенге", "EUR": "евро",
                 "RUB": "рублей"}.get(_rcur, "единиц")
    currency_rule = (
        "ВАЛЮТА — СТРОГО:\n"
        f"- Портфель оценён в {_rcur}. НЕ придумывай другую валюту. Слово «рубль/рублей» "
        f"ЗАПРЕЩЕНО, если валюта не RUB.\n"
        "- Доли и концентрацию выражай в ПРОЦЕНТАХ (например «около 80%»), НЕ привязывая "
        "к валюте и НЕ фразами вида «N из каждых 10». Если по смыслу нужна сумма — "
        f"используй только «{_cur_word}».\n"
    )

    rag_block = ""
    # 🔴 Правило имён эмитента говорится МОДЕЛИ, а не только аудитору.
    # Аудитор ссылок засчитывает в прозе лишь полные имена (`§−14` C-8: «MS»
    # это чаще Microsoft или миллисекунды), но самому промпту это правило не
    # сообщалось — зато примеры источников в нём выглядят как `[GS]/[JPM]`,
    # то есть промпт УЧИЛ короткой форме. Живой DEEP 17.08 её и написал:
    # «против совета GS», при этом панель провенанса честно отчиталась об
    # одном банке. Запрет, о котором не сказано исполнителю, неисполним
    # (`§−90` A-5, разбор — `§−97`).
    rag_rule  = (
        "ИМЯ БАНКА В ТЕКСТЕ — ТОЛЬКО ПОЛНОЕ: «Goldman Sachs», «Morgan Stanley», "
        "«JPMorgan». Сокращения (GS, MS, JPM) допустимы ИСКЛЮЧИТЕЛЬНО внутри "
        "квадратного тега источника — [GS], [JPM]. В самой фразе сокращение "
        "недопустимо: читатель не обязан их знать, а «MS» это ещё и тикер.\n"
    )
    if market_context:
        ctx_limit = 6000 if tier == "deep" else 2000
        # Defence: RAG chunks come from third-party PDFs that may carry
        # injection payloads.  Fence the entire block as `<untrusted_data>`
        # so the system-prompt guardrail tells the model to treat it as
        # pure data, never as instructions.
        rag_block = (
            "\n\n=== АНАЛИТИКА БАНКОВ (RAG) ===\n"
            + _wrap_untrusted("rag_bank_pdfs", market_context[:ctx_limit])
            + "\n=== КОНЕЦ АНАЛИТИКИ БАНКОВ ==="
        )
        rag_rule = (
            "Если используешь факт из АНАЛИТИКИ БАНКОВ — цитируй "
            "[RAG: <имя_файла>]. Не выдумывай файлы.\n"
        )

    # 2026-07-05 (Фаза 4, блок 1): the regime-specific bank excerpts used to
    # flow ONLY to the report chips — the model never saw them, so
    # ai_regime_comment / regime_confirmation leaned on memory-consensus even
    # when the KB held on-point excerpts.  Feed them as a dedicated fenced
    # block + a binding rule so the regime narrative is grounded in the
    # retrieved chunks first.
    if regime_rag_confirm:
        # Each item is either a plain excerpt (legacy) or {text, bank} (2026-07-09
        # #5): render «[BANK] excerpt» so the model attributes the bank view.
        def _rag_line(s):
            if isinstance(s, dict):
                _t = str(s.get("text", ""))[:200]
                _b = str(s.get("bank", "")).strip()
                return f"[{_b}] {_t}" if _b else _t
            return str(s)[:200]
        _lines = "\n".join(f"- {_rag_line(s)}" for s in regime_rag_confirm[:3])
        rag_block += (
            "\n\n=== RAG-ПОДТВЕРЖДЕНИЕ РЕЖИМА (выдержки из банковских PDF) ===\n"
            + _wrap_untrusted("rag_regime_confirm", _lines)
            + "\n=== КОНЕЦ RAG-ПОДТВЕРЖДЕНИЯ ==="
        )
        rag_rule += (
            "В ai_regime_comment и regime_confirmation ОПИРАЙСЯ В ПЕРВУЮ ОЧЕРЕДЬ "
            "на «RAG-ПОДТВЕРЖДЕНИЕ РЕЖИМА» и «АНАЛИТИКУ БАНКОВ» (с [RAG: файл]); "
            "консенсус из собственной памяти [GS]/[Barclays]/[JPM] — только как "
            "ДОПОЛНЕНИЕ, когда выдержки не покрывают вопрос.\n"
        )

    # Build list of tickers that are on Sell/Trim in action_plan
    sell_tickers: set[str] = set()
    for ap in (summary.get("action_plan") or []):
        act = str(ap.get("action", "")).lower()
        if act in ("sell", "trim", "сократить", "продать"):
            sell_tickers.add(str(ap.get("ticker", "")).upper())
    # Also check holdings with action Sell/Trim
    for h in (summary.get("holdings") or []):
        act = str(h.get("action", "")).lower()
        if act in ("sell", "trim", "сократить", "продать"):
            sell_tickers.add(str(h.get("ticker", "")).upper())
    
    contradiction_rule = ""
    if sell_tickers:
        contradiction_rule = (
            f"КРИТИЧНО: тикеры {', '.join(sorted(sell_tickers))} имеют статус Sell/Trim в action_plan. "
            "НЕ предлагай их в stock_picks (ни в boost_alpha, ни в rebalance). "
            "Это противоречит сигналам Quant Engine.\n"
        )

    # ── BLOCK 4: Action Plan = SINGLE SOURCE OF TRUTH for BUY/SELL/TRIM ──────
    # The action_plan rows carry the authoritative per-ticker signal
    # (Strong Buy / Buy / Hold / Trim / Sell) derived from the 4-Pillar score +
    # Black-Litterman delta-weights.  The cross-section desync bug — an AI
    # comment saying «Купить NVDA» while the high-priority plan says «Trim NVDA»
    # — happened because the prose was generated free-hand.  Pin an explicit
    # ticker→signal map into the prompt and HARD-REQUIRE every directional
    # mention (verdict / bullets / per-section comments / picks) to match it, in
    # BOTH tiers and ALL sections.
    _SIGNAL_RU = {
        "strong buy": "STRONG BUY (наращивать)", "buy": "BUY (покупать)",
        "hold": "HOLD (держать)", "trim": "TRIM (сократить)", "sell": "SELL (продать)",
    }
    signal_map: dict[str, str] = {}
    for ap in (summary.get("action_plan") or []):
        tkr = str(ap.get("ticker", "")).strip().upper()
        act = str(ap.get("action", "")).strip().lower()
        if tkr and act:
            signal_map[tkr] = _SIGNAL_RU.get(act, act.upper())
    signal_sync_rule = ""
    if signal_map:
        _pairs = " · ".join(f"{t}={s}" for t, s in sorted(signal_map.items()))
        signal_sync_rule = (
            "🔒 ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ — ACTION PLAN (математическое ядро движка):\n"
            f"    Авторитетные сигналы по тикерам: {_pairs}.\n"
            "    Эти сигналы — РЕШЕНИЕ ДВИЖКА (4-Pillar score + Black-Litterman). "
            "ВСЕ разделы и AI-комментарии (verdict, plain_summary, bullets, "
            "ai_action_comment, ai_holdings_comment, ai_effect_comment, "
            "action_plan_text, stock_picks) ОБЯЗАНЫ совпадать с ними НА 100%. "
            "ЗАПРЕЩЕНО: рекомендовать «купить/нарастить» тикер со статусом "
            "TRIM/SELL, или «сократить/продать» тикер со статусом BUY/STRONG BUY. "
            "Если упоминаешь действие по тикеру из списка — бери ТОЛЬКО его "
            "сигнал отсюда (не противоречь и не смягчай). Приоритетность в "
            "тексте: сначала SELL/TRIM, затем STRONG BUY/BUY, затем HOLD.\n"
        )

    # Sprint-5.2 (live-report audit) — held-tickers rule.  The 2026-06-12 BASE
    # report showed pure catalogue picks: Haiku spent its picks on HELD names,
    # the held-filter drained them, and the deterministic backfill took over.
    # Telling the model UP FRONT which tickers are already owned prevents the
    # drain at the source (the post-filter stays as defence in depth).
    held_tickers = sorted({
        str(h.get("ticker", "")).upper().split(".")[0]
        for h in (summary.get("holdings") or [])
        if str(h.get("ticker", "")).strip() not in ("", "?")
    } - {"USD", "EUR", "RUB", "KZT", "CASH"})
    held_rule = ""
    if held_tickers:
        held_rule = (
            f"УЖЕ В ПОРТФЕЛЕ: {', '.join(held_tickers[:20])}. "
            "НЕ предлагай эти тикеры в stock_picks — только НОВЫЕ имена "
            "вне портфеля.\n"
        )

    # Sprint-5.1 — data-driven ideas rule.  Temperature alone (Sprint-5) gave
    # variety, but the model could still parrot its training-set "favourites"
    # (PLTR/CRWD/JNJ/KO/COST).  Require every pick to be ANCHORED to data
    # from THIS report (regime, sector gaps, 4-pillar, RAG) and ban the
    # template names unless the data of the current report supports them.
    # BLOCK 1.1 (06-22) — ротация идей по дню (time-decay).  Картина «идеи не
    # меняются месяц» имела две причины: (1) якорь свежести был МЕСЯЧНЫМ
    # (`%Y-%m` → «2026-06» весь месяц), (2) бан «дежурных» имён не покрывал
    # quality-compounder-мегакэпы (V/MA/GS/UNH/PG), к которым модель сходится
    # «по умолчанию».  Якорь стал ДНЕВНЫМ + дневной «угол» ротации, а бан-лист
    # расширен — так BASE-идеи варьируются от отчёта к отчёту, как в DEEP.
    _angle = _IDEA_ROTATION_ANGLES[
        date.today().timetuple().tm_yday % len(_IDEA_ROTATION_ANGLES)]
    ideas_rule = (
        "ИДЕИ — СТРОГО DATA-DRIVEN: каждая идея в stock_picks обязана "
        "опираться на данные ЭТОГО отчёта: текущий режим "
        f"{regime_label} и его благоприятные сектора, недовесы/перекосы из "
        "sectors, сигналы 4-Pillar и/или АНАЛИТИКУ БАНКОВ (RAG). В поле why — "
        "явная привязка к этим данным (что именно в портфеле/режиме делает "
        "идею уместной СЕЙЧАС). НЕ предлагай «дежурные» имена (PLTR, CRWD, "
        "JNJ, KO, COST, MTUM, DBC, V, MA, GS, UNH, PG, AVGO), если их "
        "преимущество не следует из данных текущего отчёта — предпочитай менее "
        "очевидные имена с тем же профилем риска/качества.\n"
        # Idea freshness — now DAILY-anchored + a day-rotating angle so the
        # BASE tier (Sonnet) diverges report-to-report instead of parroting the
        # same blue-chips for a whole month.
        f"СВЕЖЕСТЬ ИДЕЙ (срез на {date.today():%Y-%m-%d}): идеи должны отражать "
        "ИМЕННО текущий срез данных этого отчёта. НЕ повторяй один и тот же "
        "набор тикеров из отчёта в отчёт «по привычке» — если режим/недовесы/"
        "оценки сместились, picks обязаны сместиться вслед за ними.\n"
        f"УГОЛ РОТАЦИИ НА СЕГОДНЯ (соблюдай, если не противоречит данным): {_angle}.\n"
    )

    # Sprint-5 — Margin/leverage AI-trigger.  When the engine flags a
    # margin-funded book (cash leg < 0), the model is HARD-REQUIRED to fill
    # `ai_leverage_warning` and to lead a bullet with the Margin-Call risk.
    lev = summary.get("leverage") or {}
    # `§−121` (правило §−13 для прозы): книга без маржи — о заёмных средствах
    # счёта молчим. Плечевой ETF как ИНСТРУМЕНТ (CONL 2×) — другое понятие,
    # его описывать можно и нужно.
    leverage_rule = (
        "ПЛЕЧО СЧЁТА: кэш-баланс НЕ отрицательный — маржинального долга нет. НЕ упоминай "
        "кредитное плечо портфеля, маржинальный долг, заёмные средства, Margin Call — ни "
        "утвердительно, ни в отрицании («без плеча» тоже нельзя). Плечевые ETF как "
        "инструмент описывать можно.\n"
    )
    if lev.get("is_leveraged"):
        # 🔴 R-4: масштаб языка пропорционален размеру долга, а не факту его
        # наличия.  Прежде директива включалась по бинарному `is_leveraged` и
        # требовала «ОТДЕЛЬНЫМ ПЕРВЫМ пунктом» одинаково при марже 1.9 % и при
        # марже 60 %.  На книге с техническим минусом кэша это вытесняло из
        # первой строки вердикта НАСТОЯЩИЙ главный риск.  Порог — вердикт
        # движка (`finance.scoring.leverage_salience`), не литерал промпта.
        _sal = str(lev.get("salience") or "material")
        _place = {
            "minor":    ("НЕ выноси плечо в первый пункт bullets и не открывай им verdict: "
                         "при таком размере долга это ДЕТАЛЬ книги, а не её главный риск. "
                         "Скажи о нём там, где речь о структуре капитала"),
            "material": ("Упомяни плечо среди bullets — но первым пунктом ставь то, что "
                         "действительно доминирует в риске"),
            "dominant": ("Выдели это ОТДЕЛЬНЫМ ПЕРВЫМ пунктом в bullets: при таком долге "
                         "плечо и есть главный факт о книге"),
        }.get(_sal, "Упомяни плечо среди bullets")
        leverage_rule = (
            "⚠ ЗАЁМНЫЕ СРЕДСТВА (МАРЖА): портфель использует кредитное плечо — "
            f"валовая экспозиция ≈{lev.get('gross_exposure_pct')}% капитала, "
            f"плечо ≈{lev.get('leverage_ratio')}x, маржинальный долг ≈{lev.get('margin_debt_pct')}%. "
            "ОБЯЗАТЕЛЬНО заполни поле ai_leverage_warning (≤240 знаков): объясни простыми словами, "
            f"что и прибыль, и убыток умножаются на коэффициент плеча ≈{lev.get('leverage_ratio')}x "
            f"(НЕ пиши «удваивается», если плечо не ≈2x), и при просадке возможен Margin Call "
            f"(принудительное закрытие позиций брокером по худшим ценам). {_place} [Quant Engine].\n"
        )

    # ── Base tier: compact prompt, 1 pick per scenario, 4 scenarios ──
    if tier != "deep":
        return (
            "Ты — Senior Financial Analyst. Анализируй портфель как ЕДИНЫЙ НАРРАТИВ: "
            "все секции связаны, каждый комментарий ссылается на другие данные в отчёте.\n"
            "Верни СТРОГО JSON (без текста вне JSON). ВСЕ ТЕКСТЫ НА РУССКОМ.\n\n"
            "ЦЕПОЧКА АНАЛИЗА (обязательно соблюдать):\n"
            "  Риск (CVaR/Vol) объясняется → конкретными позициями (holdings) →\n"
            "  которые создают → секторную концентрацию → которая даёт → факторный перекос →\n"
            "  который противоречит/поддерживает → текущий режим рынка →\n"
            "  что подтверждается/опровергается → аналитикой инвестбанков → итог: Action Plan.\n"
            f"  СКВОЗЬ ВСЮ ЦЕПОЧКУ держи в поле зрения БЕНЧМАРК ({bench_label}) и МАНДАТ клиента: "
            "факторный перекос считай ОТНОСИТЕЛЬНО бенчмарка (не абстрактного «рынка»); "
            "секторную/классовую концентрацию сверяй с ЛИМИТАМИ мандата (нарушение лимита — "
            "это риск, а не нейтральный факт); Action Plan и идеи ОБЯЗАНЫ возвращать портфель "
            "В рамки мандата (к целевой волатильности/лимитам классов), а не просто снижать риск абстрактно.\n\n"
            '{\n'
            '  "verdict": "ОДНО короткое предложение ≤110 знаков — главный риск простыми словами + что делать. БЕЗ терминов, аббревиатур и нагромождения чисел",\n'
            '  "plain_summary": "МАКСИМУМ 2 коротких предложения ≤170 знаков — что не так и что делать, простым языком. Без перечисления метрик",\n'
            '  "bullets": ["4 пункта ≤120 знаков — КАЖДЫЙ связывает 2 разных раздела отчёта [Источник]"],\n'
            '  "ai_cvar_note": "≤120 знаков — простыми словами: в 5% худших дней теряется ≈X$ (≈Y% портфеля). '
            'Это [нормально/высоко] потому что [причина в 3-5 словах]",\n'
            '  "ai_sharpe_note": "≤120 знаков — простыми словами: на каждую единицу риска портфель зарабатывает X. '
            '[Сравни с нормой 1.0]. [Хорошо/плохо] потому что [причина]",\n'
            '  "ai_mdd_note": "≤120 знаков — простыми словами: портфель уже падал на X% от максимума '
            '(≈Y$). [Приемлемо/опасно] для [профиля]",\n'
            '  "ai_vol_note": "≤120 знаков — простыми словами: насколько сильно скачет стоимость '
            'портфеля за год и укладывается ли это в целевую волатильность мандата [Quant Engine]",\n'
            '  "ai_risk_comment": "≤160 знаков — ПРИЧИНА высокого CVaR/Vol: назови 1-2 конкретных тикера '
            '(→ из раздела holdings) и их вклад в риск. Скажи что продать чтобы снизить [Quant Engine]",\n'
            '  "ai_holdings_comment": "≤170 знаков — hotspot-позиции с наибольшим вкладом в риск '
            '(→ TRC%). Объясни ПОЧЕМУ они опасны через их чувствительность к рынку (бета). '
            'Связь: высокий риск → высокий CVaR (→ см. риск-блок) [Quant Engine]",\n'
            '  "ai_sector_comment": "≤140 знаков — используй ТОЛЬКО числа из раздела sectors; '
            'НЕ суммируй сектора сам. Для совокупной техно-экспозиции бери sector_complex.weight_pct '
            # R-19: пример формата НЕ должен содержать готовое число — модель
            # копирует его как факт (см. инцидент с «8%» уверенности режима).
            'ВЕРБАТИМ (формат: \\"Tech-комплекс <weight_pct>%\\"). Сектор с перекосом + % → ПОЧЕМУ это опасно в режиме '
            f'{regime_label}. Назови сектор для докупки [Regime]",\n'
            '  "ai_factor_comment": "≤220 знаков — наклон против бенчмарка бери ГОТОВОЙ строкой '
            'summary.factor_tilt_text (числа портфеля и бенчмарка там уже разведены). (1) Value-фактор портфеля [Quant Engine]: '
            'если отрицательный — значит портфель против стоимостных акций, что ПРОТИВОРЕЧИТ режиму '
            f'{regime_label} (→ Barclays/Goldman рекомендуют Value в Recovery). '
            '(2) Назови КОНКРЕТНЫЙ фактор для наращивания (напр. Value через JNJ/KO) и что продать",\n'
            f'  "ai_benchmark_comment": "≤160 знаков — опережает/отстаёт от бенчмарка {bench_label} на X%. '
            'Причина (→ сектор/фактор). Стабильно ли это или зависит от одного ралли? [Quant Engine]",\n'
            '  "ai_performance_comment": "≤150 знаков — доходность за 1М/3М/12М: тренд роста/падения. '
            'Что было драйвером (→ конкретные тикеры из holdings). Устойчиво ли это [Quant Engine]",\n'
            '  "ai_regime_comment": "≤140 знаков — режим '
            f'{regime_label}: что это значит простыми словами. '
            'Что рекомендуют Goldman/Barclays/JPMorgan для этого режима. '
            'Как ваш портфель позиционирован (→ совпадает/противоречит факторному анализу) [Regime]",\n'
            '  "stock_picks": {\n'
            '    "boost_alpha": {"label": "Повышение доходности", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: ROE/маржа/бета + '
            'ПОЧЕМУ подходит под текущий режим [Источник]", "type": "Stock"}]},\n'
            '    "rebalance": {"label": "Ребалансировка", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: качество + '
            'замена какой позиции из Sell-списка [Источник]", "type": "Stock"}]},\n'
            '    "protect_capital": {"label": "Защита капитала", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: низкая бета + '
            'дивиденды + защита от стресс-сценария [Источник]", "type": "Stock"}]},\n'
            '    "smart_money": {"label": "Smart Money", "desc": "≤80 знаков — следование за умными деньгами", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: ПРИЗНАК умных денег — '
            'заметные покупки инсайдеров (Form 4), накопление институционалов/13F или необычный '
            'объём — + как это согласуется с режимом [Smart Money]", "type": "Stock|ETF"}]}\n'
            '  }\n'
            '}\n\n'
            "ПРАВИЛА:\n"
            f"- Русский язык. Без названий бренда («{branding.project_name()}», "
            f"«{branding.bot_name()}», «RAMP»).\n"
            "- ПРОСТОЙ ЯЗЫК — глоссарий замен:\n"
            "  CVaR → 'средний убыток в редкий плохой день (примерно раз в месяц)'\n"
            "  волатильность → 'нестабильность' или 'насколько прыгает портфель'\n"
            "  Бета → 'чувствительность к рынку (Beta 2 = двигается вдвое сильнее рынка)'\n"
            "  IR → 'стабильность обгона рынка'\n"
            "  TRC → 'доля в общем риске портфеля'\n"
            "  Recovery-режим → 'рынок восстанавливается после спада'\n"
            "- БАНКОВСКАЯ АНАЛИТИКА: даже без RAG-данных — используй знания о позициях "
            "Goldman Sachs, Barclays, JPMorgan, Morgan Stanley по текущему режиму и секторам.\n"
            "- Каждое число — [Quant Engine]/[SEC EDGAR]/[Regime]/[RAG] или [GS]/[Barclays]/[JPM].\n"
            f"- Риск-профиль: {user_profile}. Режим: {regime_label}.\n"
            "- Stock picks: РЕАЛЬНЫЕ АКЦИИ, не только ETF; имена выводи из данных "
            "отчёта (см. правило DATA-DRIVEN ниже). "
            "why — конкретные цифры (ROE, маржа, Бета) с тегом [Источник].\n"
            f"{plain_rule}"
            f"{style_rule}"
            f"{numbers_rule}"
            f"{currency_rule}"
            f"{benchmark_rule}"
            f"{mandate_rule}"
            f"{regime_confidence_rule}"
            f"{contradiction_rule}"
            f"{signal_sync_rule}"
            f"{held_rule}"
            f"{leverage_rule}"
            f"{ideas_rule}"
            f"{rag_rule}\n"
            f"=== ДАННЫЕ ПОРТФЕЛЯ ===\n"
            f"{_wrap_untrusted('broker_portfolio_json', json.dumps(summary, ensure_ascii=False)[:7000])}"
            f"{rag_block}"
        )

    # ── Deep tier: full prompt with per-section comments, 4 scenarios ──
    picks_spec = f"""
"stock_picks": {{
  "boost_alpha": {{
    "label": "Повышение доходности — активный риск", "desc": "≤100 знаков. Режим: {regime_label}.",
    "picks": [  // 2 реальные акции роста НЕ из портфеля
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: ROE/маржа/beta/momentum + [источник]", "type": "Stock"}}
    ]
  }},
  "rebalance": {{
    "label": "Качественная ребалансировка", "desc": "≤100 знаков.",
    "picks": [  // 2 quality-акции (4-Pillar F+V > 0)
      {{"ticker": "...", "name": "...", "why": "≤200 знаков с SEC-цифрами [SEC EDGAR]", "type": "Stock"}}
    ]
  }},
  "protect_capital": {{
    "label": "Защита капитала", "desc": "≤100 знаков. CVaR/tail-risk hedge.",
    "picks": [  // 1-2 защитных инструмента
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: Beta/корреляция/дивиденд [источник]", "type": "Stock|ETF|Bond"}}
    ]
  }},
  "smart_money": {{
    "label": "Smart Money", "desc": "≤100 знаков. Следование за умными деньгами (институционалы + инсайдеры).",
    "picks": [  // 1-2 идеи с ПРИЗНАКОМ умных денег
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: конкретный сигнал умных денег — покупки инсайдеров (SEC Form 4), накопление институционалов/13F, необычный объём — и согласование с режимом {regime_label} [Smart Money]", "type": "Stock|ETF"}}
    ]
  }}
}}"""

    return (
        "Проанализируй портфель как Senior Financial Analyst. Верни СТРОГО JSON (без текста вне JSON, без markdown).\n"
        "ВСЕ ТЕКСТЫ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.\n"
        "Используй 3-step reasoning: (1) взаимосвязи между метриками, (2) скрытые риски, (3) стратегические рекомендации с числами.\n\n"
        "ЦЕПОЧКА АНАЛИЗА — каждый раздел ссылается на предыдущий:\n"
        "  Риск (CVaR/Vol) → Состав портфеля (hotspots, TRC) → Секторы (перекосы) →\n"
        "  Факторы (β-концентрации) → Режим (Recovery/Stagflation/...) →\n"
        "  Банки (Goldman/Barclays/JPM) → 4-Pillar (F+V+T+C) → Action Plan.\n"
        f"  СКВОЗНАЯ ОСЬ ЦЕПОЧКИ — БЕНЧМАРК ({bench_label}) и МАНДАТ клиента: (1) факторы и «наклон Δ» "
        "трактуй как активный перекос ОТНОСИТЕЛЬНО этого бенчмарка. 🔴 Наклон УЖЕ ПОСЧИТАН "
        "движком и лежит в summary.factor_tilt_text ГОТОВОЙ строкой — цитируй её, НЕ вычитай "
        "беты сам и НЕ подставляй число бенчмарка вместо числа портфеля: у портфеля и у "
        "бенчмарка беты по одной оси РАЗНЫЕ, и перепутать их значит сказать обратное; (2) секторную/классовую концентрацию всегда сверяй с ЛИМИТАМИ мандата — "
        "превышение лимита класса это НАРУШЕНИЕ мандата и самостоятельный риск; (3) режим и идеи "
        "оценивай через призму мандатного риск-профиля (консерватору — хвост, агрессивному — рост); "
        "(4) Action Plan/ai_action_comment ОБЯЗАН вести портфель В рамки мандата (целевая vol, лимиты "
        "классов), а не просто «снизить риск» — назови, какой лимит/цель восстанавливается.\n"
        "Каждый комментарий должен явно указывать, откуда взяты данные И как это связано с соседними разделами.\n\n"
        '{\n'
        '  "verdict": "ОДНО короткое предложение ≤130 знаков — главный риск простыми словами + что делать. БЕЗ нагромождения терминов и чисел",\n'
        '  "plain_summary": "МАКСИМУМ 2 коротких предложения ≤200 знаков — позиция + главный риск + что делать, простым языком",\n'
        '  "bullets": ["5–7 пунктов ≤200 знаков каждый с [Источник] — цепочка: риск → состав → секторы → факторы → действие"],\n'
        '  "ai_cvar_note": "≤120 знаков — простыми словами: сколько в среднем теряется в редкий '
        'плохой день (примерно раз в месяц), нормально ли это для профиля риска [Quant Engine].",\n'
        '  "ai_sharpe_note": "≤120 знаков — простыми словами: окупается ли риск доходностью, '
        'лучше или хуже рынка [Quant Engine].",\n'
        '  "ai_mdd_note": "≤120 знаков — простыми словами: насколько глубоко портфель падал '
        'исторически и что это значит для владельца [Quant Engine].",\n'
        '  "ai_vol_note": "≤120 знаков — простыми словами: насколько сильно скачет стоимость '
        'портфеля за год и укладывается ли это в целевую волатильность мандата [Quant Engine].",\n'
        '  "ai_risk_comment": "260–380 знаков, 2–3 ПОЛНЫХ предложения — (1) средний убыток в редкий '
        'плохой день (CVaR) и нестабильность (Vol) с цифрами и $ потерь, (2) какая позиция вносит '
        'наибольший вклад [см. ai_holdings_comment], (3) как это соотносится с целевой волатильностью '
        'мандата и что сделать в первую очередь [Quant Engine]",\n'
        f'  "ai_benchmark_comment": "≤200 знаков — стабильность обгона (IR) vs бенчмарк {bench_label}: '
        'причины отставания/опережения (бенчмарк называй по имени). Свяжи с нестабильностью портфеля из ai_risk_comment '
        'и секторными перекосами из ai_sector_comment [Quant Engine]",\n'
        '  "ai_performance_comment": "≤180 знаков — доходность 1М/3М/12М/YTD: тренды и причины. '
        'Укажи как макро-режим [см. ai_regime_comment] повлиял на динамику [Quant Engine]",\n'
        '  "ai_regime_comment": "≤300 знаков — текущий режим рынка простыми словами '
        '(рынок восстанавливается / перегрет / в стагфляции), с опорой на ВСЕ драйверы '
        'summary.macro (включая инфляционные ожидания breakeven и их темп). '
        'Если даны RAG-выдержки — рекомендации банков бери ИЗ НИХ с [RAG: файл]; '
        'консенсус по памяти [GS]/[Barclays]/[JPM] — только как дополнение. '
        'Как это влияет на факторы [Quant Engine]",\n'
        '  "regime_confirmation": {\n'
        '    "stance": "confirms | partial | diverges",\n'
        '    "summary": "≤220 знаков — простыми словами: подтверждается ли вывод движка о режиме, '
        'на основании каких независимых сигналов",\n'
        '    "signals": ["5–7 строк ≤90 знаков — каждая начинается с ✓/⚠/✗ + сигнал. '
        'ПОКРЫТЬ обязательно ВСЕ СЕМЬ: (1) МАКРО-ДИНАМИКА — направление темпов из summary.macro[*].trend: '
        'безработица РАСТЁТ/ПАДАЕТ, ВВП УСКОРЯЕТСЯ/ЗАМЕДЛЯЕТСЯ (приоритет ТЕМПА над уровнем); '
        '(2) кривая доходности 10Y−2Y, (3) HY OAS (кредитный спред), (4) VIX (страх рынка), '
        '(5) ИНФЛЯЦИОННЫЕ ОЖИДАНИЯ — breakeven 10Y: уровень vs таргет 2% И темп '
        '(ре-анкоринг ВВЕРХ = ужесточение ФРС против роста ⚠; заякорены/снижаются = поддержка ✓ '
        '— этот сигнал входит в growth-ось движка), '
        '(6) факторные беты портфеля vs ожидаемые для режима ([Barclays]), '
        '(7) банковский консенсус [GS]/[Barclays]/[JPM]"]\n'
        '  },\n'
        '  "ai_holdings_comment": "≤200 знаков — какие позиции занимают наибольшую долю в риске '
        '(TRC — доля в общем риске). Назови конкретные тикеры-hotspots и объясни почему они опасны. '
        'Свяжи с факторами [см. ai_factor_comment] [Quant Engine]",\n'
        '  "ai_sector_comment": "≤170 знаков — какие секторы перевешены и недовешены. '
        'Назови субсектора (напр. внутри Tech: софт vs полупроводники). '
        'Укажи риск ротации при режиме [см. ai_regime_comment]",\n'
        '  "ai_factor_comment": "≤560 знаков — ЕДИНЫЙ вывод, связывающий ОБЕ иллюстрации секции. '
        'Наклон против бенчмарка бери ГОТОВОЙ строкой summary.factor_tilt_text и не пересчитывай: '
        f'(A) радар β (betas — наклоны портфеля; активный Наклон Δ считается против бенчмарка {bench_label}) И (B) «Откуда берётся риск» '
        '(var_shares — доли дисперсии, twins). СТРОГО по summary.factor_decomposition, числа verbatim. '
        '4 шага через «;», ЗАКОНЧИ мысль (НЕ обрывай на полуслове, уложись в лимит): '
        '(1) ИСТОЧНИК РИСКА [из B]: источник с макс. долей var_shares — «X% дисперсии — <источник> '
        '(драйверы: тикеры из top_drivers, с их β из betas)»; если idio_pct > 15% — упомяни специфику бумаг; '
        '(2) СКРЫТАЯ КОНЦЕНТРАЦИЯ [из B, twins]: если twins непуст — пара, corr и суммарный вес: одна и та же '
        'факторная ставка куплена ДВАЖДЫ — это концентрация, а не диверсификация; '
        '(3) НАКЛОН vs РЕЖИМ [из A]: выразительная бета из betas (напр. Value<0 = против дешёвых акций) '
        f'сверена с режимом {regime_label} и рекомендацией Barclays/GS для него; '
        '(4) ЧЕГО НЕ ХВАТАЕТ: фактор с β≈0, который диверсифицировал бы (Value/Size/Commodities/Low Vol), '
        'или «истинный диверсификатор» из most_unique — и КОНКРЕТНОЕ действие с тикером '
        '[Quant Engine][Barclays]",\n'
        '  "ai_4pillar_comment": "≤240 знаков — 4-Pillar (F=фундаментал, V=оценка, T=техника, '
        'C=кредит). ВАЖНО про V простыми словами: оценка сравнивается с НОРМОЙ СЕКТОРА, а не '
        'со всем рынком — у технологий P/E высокий ОТ ПРИРОДЫ (платим за рост), поэтому V≈0 у '
        'теха значит «оценён справедливо ДЛЯ своего сектора», а НЕ «дорого». Отметь позиции с '
        'расходящимися сигналами (F высокий, T низкий = хороший бизнес на спаде) [SEC EDGAR]",\n'
        '  "ai_stress_comment": "≤220 знаков — в худшем сценарии: насколько падает портфель и '
        'какие позиции пострадают сильнее всего (свяжи с TRC из ai_holdings_comment и '
        'факторными бетами из ai_factor_comment). Конкретные цифры просадки [Quant Engine]",\n'
        '  "ai_action_comment": "≤200 знаков — что делать в первую очередь: Trim/Sell hotspots '
        '(из ai_holdings_comment) → затем Buy (из ai_factor_comment и ai_4pillar_comment). '
        'Ожидаемый эффект на долю в риске (TRC) позиций [Quant Engine]",\n'
        '  "ai_effect_comment": "≤220 знаков — чего ожидать после ребалансировки: '
        'средний убыток в редкий плохой день (CVaR) до→после, нестабильность (Vol), Sharpe. '
        '🔴 ЕСЛИ sharpe_delta_suppressed=true — про Sharpe и «отдачу на риск» НЕ ПИШИ ВООБЩЕ: '
        'при отрицательной премии коэффициент не упорядочивает портфели, движок дельту погасил, '
        'и карточка прямо говорит «дельта не показана». Заявить улучшение — прямое противоречие '
        'соседней карточке. Опирайся на волатильность, CVaR и просадку. '
        'ОБЯЗАТЕЛЬНО согласуй с rebalance_verdict из данных: если kind=tradeoff/degradation — '
        'ЯВНО назови ухудшившуюся метрику (напр. рост концентрации Max TRC при урезании топ-позиции) '
        'и НЕ заявляй об одностороннем снижении риска. КУДА уходит высвобожденный вес — '
        'бери ДОСЛОВНО из поля reinvest_destination (свод по rebalance_actions: уже '
        'посчитанные движком тикеры, '
        'имена и +пп). Если reinvest_destination пустое — НЕ называй адресат вообще. '
        'ЗАПРЕЩЕНО называть адресат, которого НЕТ в reinvest_destination — в том '
        'числе «на снижение маржинального долга»: подмена адресата — фактическая '
        'ошибка. 🔴 «Кэш» — ПОЛНОПРАВНЫЙ адресат, а не отговорка: когда движку '
        'некуда честно вложить высвобожденный вес, он оставляет его в кэше и пишет '
        'это в reinvest_destination строкой «Кэш (+N пп)». Есть такая строка — '
        'назови кэш ПРЯМО и не заменяй его «качественными диверсификаторами»: '
        'покупки, которой не было, в отчёте быть не должно [Quant Engine]",\n'
        f'{picks_spec},\n'
        '  "action_plan_text": "≤800 знаков — приоритетные действия: Trim/Sell сначала, '
        'конкретные уровни, cumulative |Δw| ≤ 25% NAV"\n'
        '}\n\n'
        "ПРАВИЛА:\n"
        f"- ВСЕ тексты на РУССКОМ. Без названий бренда («{branding.project_name()}», "
        f"«{branding.bot_name()}», «RAMP»).\n"
        "- ПРОСТОЙ ЯЗЫК — глоссарий замен:\n"
        "  CVaR → 'средний убыток в редкий плохой день (примерно раз в месяц)'\n"
        "  волатильность → 'нестабильность' или 'насколько прыгает портфель'\n"
        "  Бета → 'чувствительность к рынку (Beta 2 = двигается вдвое сильнее рынка)'\n"
        "  IR → 'стабильность обгона рынка'\n"
        "  TRC → 'доля в общем риске портфеля'\n"
        "  Recovery-режим → 'рынок восстанавливается после спада'\n"
        "- Если используешь финансовый термин — сразу поясни его в скобках простыми словами.\n"
        "- НИКОГДА не упоминай в тексте внутренние имена полей (ai_holdings_comment, "
        "ai_sector_comment, ai_risk_comment и т.п.) и не пиши «(см. …)» с именем поля — "
        "это технические ключи, а не ссылки для читателя; пиши законченную самодостаточную прозу.\n"
        "- БАНКОВСКАЯ АНАЛИТИКА: даже без RAG-данных — используй знания о позициях "
        "Goldman Sachs, Barclays, JPMorgan, Morgan Stanley по текущему режиму и секторам. "
        "Теги: [GS], [Barclays], [JPM], [MS].\n"
        "- ПОДТВЕРЖДЕНИЕ РЕЖИМА — заполни поле regime_confirmation:\n"
        f"    движок выдал {regime_label}; УВЕРЕННОСТЬ бери ВЕРБАТИМ из summary.regime.confidence_pct "
        "    (целое %, НЕ округляй сам).\n"
        "    ОБЯЗАТЕЛЬНЫЙ ЧЕКПОИНТ «✓ ИИ подтверждает режим по ДИНАМИКЕ МАКРО»: первый "
        "    сигнал в signals[] подтверждает/опровергает режим по ТЕМПАМ изменения "
        "    (rate-of-change), а НЕ по статичным уровням. Бери summary.macro[*].trend "
        "    (темп за ≥3 наблюдения): для МЕСЯЧНЫХ индикаторов СТРОГО приоритет темпа над "
        "    уровнем — безработица 4.1% РАСТУЩАЯ = охлаждение (против Recovery/Expansion), та же "
        "    4.1% ПАДАЮЩАЯ = ускорение; ВВП — ускорение/замедление важнее абсолютного %. Если "
        "    темп противоречит уровню — верь ТЕМПУ и пометь ⚠. Если trend отсутствует — "
        "    скажи об этом явно и опирайся на уровень как запасной вариант.\n"
        "    Затем проверь режим на остальных НЕЗАВИСИМЫХ сигналах: yield curve 10Y−2Y "
        "    (положительная = рост, инверсия = рецессия), HY OAS (<350 bp = риск-он, >550 bp = "
        "    стресс), VIX (<20 = спокойствие), breakeven (инфляционные ожидания), факторные беты "
        "    портфеля и банковский консенсус. "
        "    Stance: 'confirms' если ≥80% сигналов (включая макро-динамику) согласны, 'partial' "
        "    если 2-3 расходятся, 'diverges' при фундаментальном противоречии (особенно если "
        "    макро-ТЕМПЫ направлены против режима). Каждый сигнал в signals[] — "
        "    одна строка с ✓/⚠/✗ + что именно подтверждает или противоречит.\n"
        "- 3-step reasoning: взаимосвязи → риски → рекомендации с числами.\n"
        "- Каждое число — [Quant Engine], [SEC EDGAR], [Regime] или [RAG: файл].\n"
        f"- Риск-профиль: {user_profile}. Режим: {regime_label}.\n"
        "- РЕАЛЬНЫЕ АКЦИИ, не только ETF; имена выводи из данных отчёта "
        "(см. правило DATA-DRIVEN ниже).\n"
        "- why: конкретные цифры (ROE, маржа, Beta, P/E, momentum) с тегом [источник].\n"
        f"{plain_rule}"
        f"{style_rule}"
        f"{numbers_rule}"
        f"{currency_rule}"
        f"{benchmark_rule}"
        f"{mandate_rule}"
        f"{regime_confidence_rule}"
        f"{contradiction_rule}"
        f"{signal_sync_rule}"
        f"{held_rule}"
        f"{leverage_rule}"
        f"{ideas_rule}"
        f"{rag_rule}"
        f"\n=== ДАННЫЕ ПОРТФЕЛЯ ===\n"
        f"{_wrap_untrusted('broker_portfolio_json', json.dumps(summary, ensure_ascii=False)[:9000])}"
        f"{rag_block}"
    )


# ── Fallback narrative (no API key / failure path) — RUSSIAN ─────────────────

def _fallback_factor_comment(results: dict) -> str:
    """
    Deterministic factor commentary from factor_decomposition — the same
    4-step recipe the LLM follows (источник риска → двойники → идио), so the
    «Источники риска» panel keeps an explanation even on the no-API path.
    Empty string when the engine skipped the decomposition (template hides).
    """
    fd = (results.get("portfolio_metrics") or {}).get("factor_decomposition") or {}
    groups = fd.get("group_shares") or []
    if not groups:
        return ""
    top = max(groups, key=lambda g: abs(g.get("share_pct") or 0))
    driven = fd.get("driven_by") or {}
    drivers: dict[str, float] = {}
    for f in (top.get("factors") or []):
        for d in (driven.get(f) or []):
            t = _safe_ticker(d.get("ticker"))
            if t:
                drivers[t] = drivers.get(t, 0.0) + abs(float(d.get("contribution") or 0))
    top_drv = ", ".join(t for t, _ in sorted(drivers.items(), key=lambda kv: -kv[1])[:3])
    parts = [f"{top.get('share_pct')}% дисперсии портфеля — {top.get('source')}"
             + (f" (драйверы: {top_drv})" if top_drv else "")]
    twins = fd.get("twins") or []
    if twins:
        t0 = twins[0]
        pair = "+".join(_safe_ticker(p) for p in (t0.get("pair") or []))
        parts.append(f"{pair} двигаются как одна ставка (corr {t0.get('systematic_corr')}, "
                     f"вес {t0.get('combined_weight_pct')}%) — концентрация, а не диверсификация")
    idio = fd.get("idio_pct")
    if idio is not None and idio > 15:
        parts.append(f"{idio}% риска — специфика отдельных бумаг, факторами не объясняется")
    # B1 (2026-07-17): назвать бенчмарк по имени — «Наклон Δ» в таблице секции
    # считается против него (fallback-путь должен совпадать с LLM-путём).
    bfp = results.get("benchmark_factor_profile") or {}
    if isinstance(bfp, dict) and bfp.get("name"):
        parts.append(f"активный Наклон Δ в таблице — относительно бенчмарка "
                     f"{bfp.get('name')}")
    return "; ".join(parts) + " [Quant Engine]."


def _fallback_narrative(results: dict, tier: str,
                       user_mandate: dict | None = None) -> dict:
    metrics = results.get("portfolio_metrics") or {}
    regime  = results.get("regime") or {}
    perf    = results.get("performance_table")

    composite  = metrics.get("Composite_Risk_Score") or 0
    risk_label = ("консервативный" if composite < 33 else
                  "умеренный"      if composite < 66 else "агрессивный")
    # Sprint-5.3 (замечания 1+2): short, plain-Russian fallback verdict — one
    # sentence, no jargon (Sharpe/композит/Expansion).  «Уровень риска —
    # {risk_label}» keeps grammatical agreement (уровень — м.р.).
    verdict    = (f"Уровень риска портфеля — {risk_label}. Проверьте "
                  f"концентрацию и при необходимости снизьте крупнейшие позиции.")

    bullets: list[str] = []
    if perf is not None and not perf.empty and "Score_Hotspot" in perf.columns:
        hot = perf[perf["Score_Hotspot"] == True]   # noqa: E712
        if not hot.empty:
            t   = str(hot.iloc[0].get("Ticker"))
            trc = float(hot.iloc[0].get("Euler_Risk_Contribution_Pct") or 0)
            bullets.append(f"🔥 Позиция {t} генерирует {trc:.1f}% общего риска портфеля — "
                           f"рекомендуется частичное сокращение [Quant Engine]")
    cvar = metrics.get("CVaR_95_Daily")
    total_val = results.get("total_value") or 0
    if cvar is not None:
        cvar_dollar = abs(cvar) * total_val if total_val else 0
        bullets.append(f"Ожидаемый убыток в худшие 5% дней: {cvar*100:.1f}% "
                       f"(≈${cvar_dollar:,.0f}) [Quant Engine]")
    mdd = metrics.get("Max_Drawdown")
    if mdd is not None:
        mdd_dollar = abs(mdd) * total_val if total_val else 0
        bullets.append(f"Историческая макс. просадка: {mdd*100:.1f}% "
                       f"(≈${mdd_dollar:,.0f}) [Quant Engine]")
    if regime.get("regime"):
        bullets.append(f"Макро-режим: {regime['regime']} "
                       f"(уверенность {int(round(regime.get('confidence', 0)*100))}%) "
                       f"учтён в скоринге [Regime]")
    if tier == "deep":
        bullets.append("Используйте Action Plan: уровни Buy/Sell/Stop рассчитаны "
                       "из ATR (Wilder RMA) и SMA200 [Quant Engine]")

    # Sprint-5.3: max 2 short plain sentences, no jargon.
    plain_summary = (
        f"Уровень риска портфеля — {risk_label}. "
        + ("Снизьте крупнейшие позиции, чтобы риск не зависел от одной акции."
           if tier == "deep" else
           "Главное — не держать слишком много в одной акции.")
    )

    regime_label = regime.get("regime", "")
    # R-22: офлайн-каталог тоже обязан уважать мандат — иначе клиент
    # без класса `Crypto`/`GlobalETFs` получал из него запрещённые идеи.
    stock_picks  = _fallback_stock_picks(regime_label, tier,
                                         user_mandate=user_mandate)
    stock_picks  = _remove_mandate_banned_picks(stock_picks, user_mandate)

    # Sprint-5 — deterministic Margin-Call warning when the book is levered,
    # so even the no-API fallback path surfaces the leverage risk.
    lm = results.get("leverage_metrics") or {}
    ai_leverage_warning = ""
    if lm.get("is_leveraged"):
        gross = (lm.get("gross_exposure") or 0) * 100
        lev_x = lm.get("leverage_ratio") or 0
        ai_leverage_warning = (
            f"⚠️ Внимание: портфель использует заёмные средства (плечо ≈{lev_x:.2f}x, "
            f"валовая экспозиция ≈{gross:.0f}% капитала). Плечо умножает и прибыль, и "
            "убыток; при сильной просадке возможен Margin Call — принудительное "
            "закрытие позиций брокером по невыгодным ценам [Quant Engine]."
        )
        # R-4: место предупреждения зависит от РАЗМЕРА долга.  Фолбэк ставил
        # его первым всегда — та же ошибка, что в промпте, и «починка обязана
        # накрывать фолбэк» (`§−91` B-1).  На книге с маржой 1.9 % плечо не
        # является главным фактом и первую строку не занимает.
        _debt_pct = abs(min(0.0, lm.get("cash_weight") or 0)) * 100
        if _leverage_salience(_debt_pct) == "minor":
            bullets.append(ai_leverage_warning)
        else:
            bullets.insert(0, ai_leverage_warning)

    action_plan_text = ""
    if tier == "deep":
        action_plan_text = (
            "Приоритет — закрытие концентрационных hotspots (TRC > 20%); "
            "затем точечные покупки по Buy-зонам в активах с Total Score > +1. "
            "Все сделки в пределах 25% оборота NAV [Quant Engine]."
        )

    return {
        "verdict":                  verdict,
        "plain_summary":            plain_summary[:300],
        "bullets":                  bullets[:7 if tier == "deep" else 4],
        "action_plan_text":         action_plan_text,
        # `§−121`: поле выведено из работы — см. комментарий в успешной ветке.
        "ai_action_impact":         "",
        "stock_picks":              stock_picks,
        "used_rag":                 False,
        "model_used":               "fallback",
        "ai_risk_comment":          "",
        "ai_benchmark_comment":     "",
        "ai_performance_comment":   "",
        "ai_regime_comment":        "",
        "ai_holdings_comment":      "",
        "ai_sector_comment":        "",
        # Deterministic factor summary (источник риска → двойники → идио) —
        # keeps the DEEP «Источники риска» panel explained on the no-API path.
        "ai_factor_comment":        _fallback_factor_comment(results) if tier == "deep" else "",
        "ai_4pillar_comment":       "",
        "ai_stress_comment":        "",
        "ai_action_comment":        "",
        "ai_effect_comment":        "",
        "ai_leverage_warning":      ai_leverage_warning,
        "regime_confirmation":      {"stance": "", "summary": "", "signals": []},
        "rag_context":              "",
    }


def _fallback_stock_picks(regime_label: str, tier: str,
                          user_mandate: dict | None = None) -> dict:
    """Rule-based stock picks when Claude is unavailable — includes real stocks.

    R-22 (2026-08-02): выбор кандидата теперь УВАЖАЕТ мандат. Ротация по месяцу
    сохранена (детерминизм внутри месяца), но если выпавший кандидат лежит в
    классе с лимитом 0–0, берётся следующий разрешённый по кругу. Иначе
    «Защита капитала» у клиента без `GlobalETFs` выпадала целиком.
    """
    expansion = regime_label in ("Recovery", "Expansion", "")

    # Sprint-5.1 — the fallback used to be a FROZEN 1-candidate catalogue
    # (always PLTR/JNJ/KO/MTUM in expansion), so no-API users saw byte-
    # identical "ideas" forever.  Each slot now carries 3 candidates and the
    # selection rotates DETERMINISTICALLY by calendar month (stable within a
    # month — reproducible reports; different across months — no more
    # year-long loops).  All candidates are conservative, liquid US names.
    from datetime import date
    _rot = date.today().month

    def _pick(candidates: list[dict], offset: int = 0) -> dict:
        """Ротация по месяцу, но НЕ в запрещённый мандатом класс (R-22)."""
        n = len(candidates)
        start = (_rot + offset) % n
        for i in range(n):
            cand = candidates[(start + i) % n]
            if _mandate_allows(cand.get("ticker", ""), user_mandate):
                return cand
        # Все кандидаты слота запрещены мандатом — отдаём исходный; финальный
        # гард `_remove_mandate_banned_picks` всё равно его вычистит, и слот
        # честно останется пустым, а не покажет запрещённую бумагу.
        return candidates[start]

    _BOOST_EXPANSION = [
        {"ticker": "PLTR", "name": "Palantir Technologies",
         "why": "Рост выручки >30% г/г, высокий momentum 12м, бета ~1.8 — "
                "ставка на режим Expansion [SEC EDGAR] [Regime]", "type": "Stock"},
        {"ticker": "NOW", "name": "ServiceNow",
         "why": "Подписочная выручка +22% г/г, Op. маржа растёт, лидер "
                "enterprise-софта — циклический рост [SEC EDGAR] [Regime]", "type": "Stock"},
        {"ticker": "ANET", "name": "Arista Networks",
         "why": "Сетевая инфраструктура ИИ-ЦОД, выручка +20% г/г, без долга, "
                "ROE >30% — бенефициар capex-цикла [SEC EDGAR] [Regime]", "type": "Stock"},
    ]
    _BOOST_DEFENSIVE = [
        {"ticker": "DBC", "name": "Invesco DB Commodity Index ETF",
         "why": "Товарная экспозиция защищает при стагфляции и Slowdown; низкая "
                "корреляция с техно-позициями [Quant Engine] [Regime]", "type": "ETF"},
        {"ticker": "XLE", "name": "Energy Select Sector SPDR",
         "why": "Энергетика — исторический лидер в позднем цикле/Slowdown; "
                "дивидендная доходность сектора >3% [Regime]", "type": "ETF"},
        {"ticker": "XLU", "name": "Utilities Select Sector SPDR",
         "why": "Коммунальный сектор: низкая бета ~0.5, устойчивые денежные "
                "потоки при замедлении экономики [Regime]", "type": "ETF"},
    ]
    _BALANCE = [
        {"ticker": "JNJ", "name": "Johnson & Johnson",
         "why": "ROE ~25%, Op. маржа ~25%, Debt/Assets <0.4, дивиденд-аристократ "
                "62 года — режимо-устойчивый [SEC EDGAR]", "type": "Stock"},
        {"ticker": "PG", "name": "Procter & Gamble",
         "why": "Op. маржа ~24%, стабильный FCF, 68 лет роста дивиденда — "
                "качество вне цикла [SEC EDGAR]", "type": "Stock"},
        {"ticker": "AVGO", "name": "Broadcom",
         "why": "FCF-маржа >45%, диверсификация чипы+софт, растущий дивиденд — "
                "качество с ростом [SEC EDGAR]", "type": "Stock"},
    ]
    _PROTECT = [
        {"ticker": "KO", "name": "The Coca-Cola Company",
         "why": "Дивиденд-аристократ 61+ лет, Beta ~0.6, маржа >28% — защитный "
                "актив при любом режиме [SEC EDGAR] [Quant Engine]", "type": "Stock"},
        {"ticker": "PEP", "name": "PepsiCo",
         "why": "Beta ~0.55, 52 года роста дивиденда, диверсификация снеки+напитки "
                "— защита с ростом [SEC EDGAR]", "type": "Stock"},
        {"ticker": "AGG", "name": "iShares Core US Aggregate Bond ETF",
         "why": "Инвест-грейд облигации: отрицательная корреляция с акциями в "
                "risk-off, дюрация ~6 лет [Quant Engine]", "type": "ETF"},
    ]
    _BALANCE_DEEP_EXTRA = [
        {"ticker": "COST", "name": "Costco Wholesale",
         "why": "ROE ~30%, стабильная подписочная модель, Revenue Growth ~7% г/г, "
                "низкий долг [SEC EDGAR]", "type": "Stock"},
        {"ticker": "V", "name": "Visa",
         "why": "Op. маржа ~67%, asset-light модель, рост объёмов платежей — "
                "качество-компаундер [SEC EDGAR]", "type": "Stock"},
        {"ticker": "UNH", "name": "UnitedHealth Group",
         "why": "ROE ~25%, защитный health-сектор, стабильный двузначный рост "
                "EPS [SEC EDGAR]", "type": "Stock"},
    ]
    _BOOST_DEEP_EXP_EXTRA = [
        {"ticker": "CRWD", "name": "CrowdStrike Holdings",
         "why": "Лидер кибербезопасности, выручка +33% г/г, переход на "
                "прибыльность, momentum 12м [SEC EDGAR] [Quant Engine]", "type": "Stock"},
        {"ticker": "PANW", "name": "Palo Alto Networks",
         "why": "Платформенная консолидация кибербеза, FCF-маржа ~38%, "
                "рост ARR >15% [SEC EDGAR]", "type": "Stock"},
        {"ticker": "AMAT", "name": "Applied Materials",
         "why": "Полупроводниковое оборудование — рычаг на capex-цикл чипов, "
                "ROE ~45% [SEC EDGAR] [Regime]", "type": "Stock"},
    ]
    _BOOST_DEEP_DEF_EXTRA = [
        {"ticker": "GLD", "name": "SPDR Gold Shares",
         "why": "Золото как хедж в Slowdown/Recession, ~нулевая корреляция "
                "с S&P 500 [Quant Engine] [Regime]", "type": "ETF"},
        {"ticker": "IAU", "name": "iShares Gold Trust",
         "why": "Золото с низкой комиссией (0.25%) — хедж хвостового риска "
                "[Quant Engine] [Regime]", "type": "ETF"},
        {"ticker": "TLT", "name": "iShares 20+ Year Treasury",
         "why": "Длинные трежерис — классический risk-off хедж при рецессии "
                "[Quant Engine] [Regime]", "type": "ETF"},
    ]
    _REGIME_EXPANSION = [
        {"ticker": "MTUM", "name": "iShares MSCI USA Momentum ETF",
         "why": "Momentum-фактор лидирует в Expansion-режиме [Regime] [Quant Engine]",
         "type": "ETF"},
        {"ticker": "IWM", "name": "iShares Russell 2000",
         "why": "Малые капитализации — ранне-цикличная ставка при росте "
                "аппетита к риску [Regime]", "type": "ETF"},
        {"ticker": "XLF", "name": "Financial Select Sector SPDR",
         "why": "Финансы выигрывают от крутой кривой доходности в "
                "Recovery/Expansion [Regime]", "type": "ETF"},
    ]
    _REGIME_DEFENSIVE = [
        {"ticker": "DBC", "name": "Invesco DB Commodity Index ETF",
         "why": "Товарная экспозиция защищает при Slowdown-режиме [Regime]", "type": "ETF"},
        {"ticker": "XLV", "name": "Health Care Select Sector SPDR",
         "why": "Healthcare — защитный сектор позднего цикла со стабильным "
                "спросом [Regime]", "type": "ETF"},
        {"ticker": "USMV", "name": "iShares MSCI USA Min Vol ETF",
         "why": "Min-vol фактор исторически опережает в risk-off фазах "
                "[Regime] [Quant Engine]", "type": "ETF"},
    ]

    picks_boost   = [_pick(_BOOST_EXPANSION if expansion else _BOOST_DEFENSIVE)]
    picks_balance = [_pick(_BALANCE)]
    picks_protect = [_pick(_PROTECT)]

    if tier == "deep":
        picks_boost.append(_pick(
            _BOOST_DEEP_EXP_EXTRA if expansion else _BOOST_DEEP_DEF_EXTRA,
            offset=1))
        picks_balance.append(_pick(_BALANCE_DEEP_EXTRA, offset=1))

    # smart_money: institutional / insider-conviction proxies (B2.4).  The
    # offline catalogue can't read live Form 4, so it uses widely-followed
    # "smart money" proxies; rotates monthly like the rest.
    _SMART_MONEY = [
        {"ticker": "BRK.B", "name": "Berkshire Hathaway",
         "why": "Прокси на аллокацию Баффета — сам по себе индикатор умных денег; "
                "диверсифицированный кэш-генератор [Smart Money]", "type": "Stock"},
        {"ticker": "BLK", "name": "BlackRock",
         "why": "Крупнейший управляющий активами, бенефициар притоков; "
                "высокая институциональная доля [Smart Money]", "type": "Stock"},
        {"ticker": "BX", "name": "Blackstone",
         "why": "Лидер альтернативных активов; индикатор аппетита институционалов "
                "к риску [Smart Money]", "type": "Stock"},
    ]
    picks_smart = [_pick(_SMART_MONEY)]
    return {
        "boost_alpha":     {"label": "Повышение доходности — повышенный риск",
                            "desc":  "Увеличение потенциальной доходности за счёт акций роста и momentum.",
                            "picks": picks_boost},
        "rebalance":       {"label": "Качественная ребалансировка — сохранение профиля",
                            "desc":  "Улучшение качества портфеля без изменения общего риска.",
                            "picks": picks_balance},
        "protect_capital": {"label": "Защита капитала — снижение риска",
                            "desc":  "Защитное позиционирование при макроэкономической неопределённости.",
                            "picks": picks_protect},
        "smart_money":     {"label": "Smart Money — институционалы и инсайдеры",
                            "desc":  "Следование за умными деньгами (накопление фондов / покупки инсайдеров).",
                            "picks": picks_smart},
    }


# ── Public API ───────────────────────────────────────────────────────────────

def generate_narrative(results: dict, tier: str = "base",
                       market_context: str = "",
                       user_risk_profile: str = "Moderate",
                       regime_rag_confirm: list[str] | None = None,
                       user_mandate: dict | None = None) -> dict:
    """
    Returns {verdict, plain_summary, bullets, stock_picks,
             action_plan_text, used_rag}  (`ai_action_impact` — всегда "").

    Base tier  → Claude Haiku  (1200 tokens, fast)
    Deep tier  → Claude Sonnet (6000 tokens, richer prose + RAG synthesis)
    Falls back deterministically when the API is unavailable or fails.
    All output is in Russian.
    """
    api_key  = os.getenv("ANTHROPIC_API_KEY")
    # RAG is consulted for BOTH tiers (_fetch_rag_context runs tier-agnostic
    # in tg_bot), and both Haiku (BASE) and Sonnet (DEEP) receive the bank
    # context in their prompt.  Gating used_rag to deep-only made the BASE
    # report's QC panel claim "RAG: не использован" even when bank excerpts
    # were retrieved and cited — flag the ACTUAL availability instead.
    used_rag = bool(market_context)

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY отсутствует — используется fallback-нарратив.")
        out = _fallback_narrative(results, tier, user_mandate=user_mandate)
        out["used_rag"] = False
        out["model_used"] = "fallback"
        return out

    summary = _summarise_for_prompt(results)
    model   = MODEL_DEEP if tier == "deep" else MODEL_BASE
    max_tok = MAX_TOKENS_DEEP if tier == "deep" else MAX_TOKENS_BASE

    logger.info("AI narrative: запрос к %s (tier=%s, max_tokens=%d, rag=%s)",
                model, tier, max_tok, used_rag)
    try:
        import anthropic
        # `§−122`: у SDK по умолчанию 600 с на чтение × 3 попытки — зависший
        # ответ держал бы поток executor'а и слот пользователя до 30 минут.
        # DEEP укладывается в 1–2 минуты; потолок — env, при истечении отчёт
        # уходит с детерминированным фолбэком, а не ждёт.
        client   = anthropic.Anthropic(api_key=api_key,
                                       timeout=ANTHROPIC_TIMEOUT_S,
                                       max_retries=ANTHROPIC_MAX_RETRIES)
        # Sprint-5: temperature raised from 0.1 (near-deterministic → stale,
        # repeating ideas).  Sprint-5.1: omitted entirely on Opus 4.7/4.8 —
        # those models reject the param with HTTP 400 (§4.2 migration prep).
        _sampling_kwargs: dict = {}
        if _model_supports_temperature(model):
            _sampling_kwargs["temperature"] = NARRATIVE_TEMPERATURE
        response = client.messages.create(
            model       = model,
            max_tokens  = max_tok,
            **_sampling_kwargs,
            # Prompt Caching — the ~25 KB system prompt is byte-identical on
            # every request.  Caching it means repeated calls (base+deep, and
            # many users within the 5-min TTL) read the prefix at ~0.1× instead
            # of full input price.  (Cache is model-scoped: Haiku BASE and
            # Sonnet DEEP keep separate caches — both still benefit at volume.)
            system      = [{
                "type": "text",
                "text": build_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }],
            messages    = [{
                "role": "user",
                "content": _user_prompt(summary, tier=tier,
                                        market_context=market_context,
                                        user_profile=user_risk_profile,
                                        regime_rag_confirm=regime_rag_confirm,
                                        user_mandate=user_mandate),
            }],
            # Structured Outputs — force the report through a typed tool call so
            # the SDK returns a GUARANTEED dict.  No brace-finding, no
            # _repair_truncated_json: JSON parsing is now strictly deterministic.
            tools       = [REPORT_TOOL],
            tool_choice = {"type": "tool", "name": "emit_report"},
        )
        usage      = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        # Deterministic extraction — the forced tool_use block's `input` IS the
        # parsed object.  No text parsing path at all.
        tool_blocks = [b for b in response.content
                       if getattr(b, "type", None) == "tool_use"
                       and getattr(b, "name", "") == "emit_report"]
        if not tool_blocks:
            raise ValueError("Модель не вернула структурированный tool_use (emit_report).")
        parsed = dict(tool_blocks[0].input or {})
        logger.info("AI narrative: structured tool_use OK model=%s (input=%d tok, "
                    "output=%d tok, cache_read=%d tok, keys=%d)",
                    model, usage.input_tokens, usage.output_tokens, cache_read, len(parsed))

        verdict       = str(parsed.get("verdict", "")).strip()
        plain_summary = str(parsed.get("plain_summary", "")).strip()
        bullets       = [str(b).strip() for b in (parsed.get("bullets") or []) if str(b).strip()]
        plan_txt      = str(parsed.get("action_plan_text", "")).strip()
        stock_picks   = parsed.get("stock_picks") or {}

        if not verdict or not bullets:
            raise ValueError(
                f"verdict или bullets пусты (verdict={verdict!r}, bullets_count={len(bullets)})"
            )

        # CoVe: strip RAG citations whose source file is not in market_context.
        _held_now     = _held_symbols(results)
        _is_levered   = bool((results.get("leverage_metrics") or {}).get("is_leveraged"))

        def _clean_top(txt: str) -> str:
            return strip_account_leverage(_expand_bank_short_forms(
                _strip_unverified_rag_citations(txt, market_context), _held_now),
                _is_levered)

        bullets       = [b for b in (_clean_top(b) for b in bullets) if b]
        plan_txt      = _clean_top(plan_txt)
        verdict       = _clean_top(verdict)
        plain_summary = _clean_top(plain_summary)

        # Normalise stock_picks structure.
        stock_picks = _normalise_stock_picks(stock_picks, tier, market_context)
        # Remove picks that already exist in the portfolio (they are not "new ideas").
        stock_picks = _remove_held_picks(stock_picks, results)
        # Post-generation contradiction check: remove Sell/Trim tickers from buy picks.
        stock_picks = _check_pick_contradictions(stock_picks, results)
        # Backfill any scenario the AI left empty (or that the held/contradiction
        # filters drained) from the rule-based fallback, so the user always
        # sees the full 4-scenario strategic menu instead of a silently
        # shrinking idea list.  Held tickers are filtered again so a backfilled
        # pick is never one the user already owns.
        _regime_label = ((results.get("regime") or {}).get("regime")
                         if isinstance(results.get("regime"), dict) else None) or "unknown"
        # B1 (2026-07-17): мандатный гард — детерминированно вычищаем идеи из
        # классов активов с лимитом 0–0 (напр. Crypto у консервативного).
        #
        # R-22 (2026-08-02): гард ПЕРЕЕХАЛ ВЫШЕ backfill. Раньше он шёл после —
        # и опустошал корзину, которую backfill только что гарантированно
        # наполнил, без второго шанса. Живой отчёт 02.08: три карточки при
        # подписи «4 идеи» (у клиента `GlobalETFs`/`Stocks_KZ` = 0–0, а
        # «Защита капитала» — защитные сектор/фактор-ETF). Прежняя мотивация
        # «чтобы и фолбэк прошёл через фильтр» СОХРАНЕНА: теперь фолбэк сам
        # фильтруется мандатом внутри `_backfill_empty_scenarios`.
        stock_picks = _remove_mandate_banned_picks(stock_picks, user_mandate)
        stock_picks = _backfill_empty_scenarios(stock_picks, _regime_label, tier,
                                                 market_context, results,
                                                 user_mandate=user_mandate)
        total_picks = sum(len(s.get("picks", [])) for s in stock_picks.values())

        logger.info("AI narrative: SUCCESS model=%s verdict=%d chars bullets=%d picks=%d",
                    model, len(verdict), len(bullets), total_picks)

        # Extract per-section AI comments.  Strip unverified RAG citations so
        # a comment never references a bank report that wasn't retrieved.
        # `_soft_trim` lands the cut on a sentence boundary so the report
        # never shows half-sentences like "[GS][Bar".
        _held = _held_symbols(results)

        def _clean(txt: str) -> str:
            """Единственная точка пост-обработки текста нарратива.

            Здесь снимаются неподтверждённые [RAG]-ссылки и разворачиваются
            короткие имена банков. Обе правки обязаны стоять ДО аудита ссылок,
            иначе панель провенанса опишет не тот текст, который читает
            пользователь (`§−97`).
            """
            return strip_account_leverage(_expand_bank_short_forms(
                _strip_unverified_rag_citations(txt, market_context), _held),
                _is_levered)

        def _comment(key: str, limit: int = 250) -> str:
            txt = str(parsed.get(key, "")).strip()
            return _soft_trim(_clean(txt), limit)

        # Structured regime-confirmation cell — DEEP tier only.  Validates
        # that the AI cross-checked the engine's regime label against macro
        # drivers, factor betas and bank views.
        def _regime_confirmation() -> dict:
            if tier != "deep":
                return {"stance": "", "summary": "", "signals": []}
            raw = parsed.get("regime_confirmation") or {}
            stance  = str(raw.get("stance", "")).strip().lower()
            if stance not in {"confirms", "partial", "diverges"}:
                stance = ""
            summary = _soft_trim(_clean(str(raw.get("summary", "")).strip()), 260)
            signals_in = raw.get("signals") or []
            # 2026-07-05: cap 6→7 — the checklist gained a mandatory INFLATION
            # (breakeven level⊕темп) checkpoint, mirroring the overlay's third
            # nudge; at 6 the model had to drop a signal to fit.
            signals = [
                _soft_trim(_clean(str(s).strip()), 120)
                for s in signals_in if str(s).strip()
            ][:7]
            return {"stance": stance, "summary": summary, "signals": signals}

        # Build the per-section comments ONCE so the citation audit below counts
        # the SAME strings the report renders (post strip + soft-trim).
        _regime_cmt = _comment("ai_regime_comment", 400 if tier == "deep" else 250)
        _factor_cmt = _comment("ai_factor_comment", 900 if tier == "deep" else 250)
        # Audit 2026-07-04: the effect comment summarises the 8-card «Ожидаемый
        # эффект» grid AND its trade-off caveat — 250 chars amputated the caveat
        # («…это компромисс, а…») on every DEEP run.  Same headroom as regime.
        _effect_cmt = _comment("ai_effect_comment", 400 if tier == "deep" else 250)
        _rc = _regime_confirmation()

        # RAG audit trail — count report-backed [RAG:file] citations vs. bank
        # consensus the model surfaced from memory, across everything the user
        # actually reads.  Feeds the CoVe "ИИ-цитирование банков" checker.
        _cite = _count_rag_citations(
            [verdict, plain_summary, plan_txt, *bullets,
             _regime_cmt, _factor_cmt, _effect_cmt, _rc.get("summary", "")]
            + [_comment(k) for k in (
                "ai_risk_comment", "ai_holdings_comment", "ai_4pillar_comment",
                "ai_stress_comment", "ai_action_comment")])

        return {
            # Sprint-5.3 (замечание 1): hard length cut so the cover verdict
            # stays short even if the model overruns — was 300/400.  These two
            # cover fields keep the STRICT budget (allow_grace=False): the
            # headline boxes have a fixed layout ceiling, unlike the body
            # section comments where a slightly-over complete thought is better.
            "verdict":                  _soft_trim(verdict, 150, allow_grace=False),
            "plain_summary":            _soft_trim(plain_summary, 230, allow_grace=False),
            "bullets":                  bullets[:7 if tier == "deep" else 4],
            "action_plan_text":         _soft_trim(plan_txt, 1000) if tier == "deep" else "",
            # `§−121`: поле ВЫВЕДЕНО ИЗ РАБОТЫ. Промпт просил «количественный
            # прогноз CVaR/Vol/TE/Sharpe после плана» — то есть заставлял модель
            # СЧИТАТЬ то, что движок уже посчитал в панели «Эффект» (`§−44`
            # R-5, `§−97` E-5), а фолбэк печатал выдуманные «−1-2 п.п.». Поле
            # не рендерилось нигде, но оплачивалось каждым DEEP-вызовом и
            # засчитывалось в аудит цитат «того, что читает пользователь».
            # Ключ контракта оставлен пустым: удаление — решение владельца.
            "ai_action_impact":         "",
            "stock_picks":              stock_picks,
            "used_rag":                 used_rag,
            "model_used":               model,
            "ai_cvar_note":             _comment("ai_cvar_note", 160),
            "ai_sharpe_note":           _comment("ai_sharpe_note", 160),
            "ai_mdd_note":              _comment("ai_mdd_note", 160),
            "ai_vol_note":              _comment("ai_vol_note", 160),
            "ai_risk_comment":          _comment("ai_risk_comment", 420 if tier == "deep" else 250),
            "ai_benchmark_comment":     _comment("ai_benchmark_comment"),
            "ai_performance_comment":   _comment("ai_performance_comment"),
            # Audit 2026-07-04: the DEEP regime/factor comments were truncated
            # with a trailing «…» on EVERY run because the model overshot the
            # prompt's ≤220/≤560 guidance and the soft-trim ceiling (250/640)
            # sat just above it.  Ceilings raised (400/900) so the closing
            # sentence completes; the boxes are auto-height so no layout risk.
            "ai_regime_comment":        _regime_cmt,
            "ai_holdings_comment":      _comment("ai_holdings_comment", 400 if tier == "deep" else 250),
            "ai_sector_comment":        _comment("ai_sector_comment", 200),
            # DEEP factor comment ties BOTH illustrations together (β-radar +
            # variance decomposition) via the 4-step institutional recipe —
            # needs headroom so the closing thought isn't clipped mid-sentence
            # (soft-trim still guards the hard ceiling at a WORD boundary).
            "ai_factor_comment":        _factor_cmt,
            "ai_4pillar_comment":       _comment("ai_4pillar_comment"),
            "ai_stress_comment":        validate_stress_comment(_comment("ai_stress_comment")),
            "ai_action_comment":        _comment("ai_action_comment", 400 if tier == "deep" else 250),
            "ai_effect_comment":        _effect_cmt,
            # Sprint-5 margin/leverage trigger output — only the AI fills this.
            # `§−121`: «пусто на книге без маржи» держалось на послушании
            # модели, а Jinja-баннер проверял лишь непустоту текста. Теперь
            # поле гасит ДВИЖОК: нет отрицательного кэша — нет предупреждения.
            "ai_leverage_warning":      (_comment("ai_leverage_warning", 260)
                                         if _is_levered else ""),
            "regime_confirmation":      _rc,
            # RAG audit trail (2026-07-04): how the narrative sourced bank views
            # — report-backed [RAG:file] citations vs. bank consensus from model
            # memory — so the CoVe panel proves whether reports were truly read.
            "rag_file_citations":       _cite["file_cites"],
            "rag_bank_citations":       _cite["bank_cites"],
            "rag_cited_banks":          _cite["banks"],
            # Surface the raw RAG context so integrity_checks can show the
            # actual snippet count.  Without this the integrity pill was
            # hard-wired to "~0 отрывков" even when ChromaDB returned real
            # bank-report excerpts — the bank views WERE reaching the AI,
            # but the UI claimed nothing was retrieved.
            "rag_context":              market_context,
        }
    except Exception as exc:
        logger.warning("AI narrative FAILED (%s) — используется fallback. Модель: %s",
                       exc, model, exc_info=True)
        out = _fallback_narrative(results, tier, user_mandate=user_mandate)
        out["used_rag"]   = False
        out["model_used"] = "fallback"
        return out


def _normalise_stock_picks(raw: dict, tier: str, market_context: str) -> dict:
    """
    Ensure stock_picks has the four scenario keys.
    Strips unverified RAG citations from pick rationale.
    """
    result: dict = {}
    for key in ("boost_alpha", "rebalance", "protect_capital", "smart_money"):
        scenario = raw.get(key) or {}
        picks    = scenario.get("picks") or []
        clean_picks = []
        for p in picks:
            why = _strip_unverified_rag_citations(str(p.get("why", "")), market_context)
            clean_picks.append({
                "ticker": str(p.get("ticker", "")).upper()[:10],
                "name":   str(p.get("name", ""))[:80],
                "why":    why[:220],
                "type":   str(p.get("type", "Stock"))[:10],
            })
        result[key] = {
            "label": str(scenario.get("label", key))[:80],
            "desc":  str(scenario.get("desc", ""))[:200],
            "picks": clean_picks,
        }
    return result


def _held_symbols(results: dict) -> set[str]:
    """Базовые символы позиций портфеля (без суффикса площадки).

    Нужен двоим: фильтру идей и разворачиванию коротких имён банков (`MS` в
    прозе это Morgan Stanley — но НЕ у того, кто держит бумагу `MS`).
    Приватное имя между потребителями не ходит, поэтому функция публичная в
    пределах модуля и одна на обоих (`tests/test_layering.py`).
    """
    perf = results.get("performance_table")
    if perf is None or getattr(perf, "empty", True):
        return set()
    col = perf.get("Ticker", perf.index if getattr(perf.index, "name", None) == "Ticker" else [])
    return {str(x).upper().split(".")[0] for x in col}


def _remove_held_picks(stock_picks: dict, results: dict) -> dict:
    """
    Remove tickers already in the live portfolio from AI idea picks.

    Recommending GOOGL (16.7% held) as a "new idea" is a logical contradiction
    that undermines trust in the report.  boost_alpha / rebalance / protect_capital
    scenarios are all cleaned — the AI should only suggest additions, not echo
    current holdings.
    """
    held: set[str] = _held_symbols(results)
    if not held:
        return stock_picks
    if not held:
        return stock_picks
    for scenario in stock_picks.values():
        if not isinstance(scenario, dict):
            continue
        original = scenario.get("picks", [])
        filtered = [p for p in original
                    if p.get("ticker", "").upper().split(".")[0] not in held]
        if len(filtered) < len(original):
            removed = [p["ticker"] for p in original if p not in filtered]
            logger.info("Held-pick filter: removed %s (already in portfolio)", removed)
        scenario["picks"] = filtered
    return stock_picks


def _check_pick_contradictions(stock_picks: dict, results: dict) -> dict:
    """
    Post-generation sanity check: remove picks that contradict the action_plan.
    If action_plan says Sell/Trim for a ticker, it should NOT appear in
    boost_alpha or rebalance picks (but protect_capital is OK for hedging).
    """
    sell_tickers: set[str] = set()
    # From action_plan
    for ap in (results.get("action_plan") or []):
        act = str(ap.get("action", "")).lower()
        if act in ("sell", "trim", "сократить", "продать"):
            sell_tickers.add(str(ap.get("ticker", "")).upper().split(".")[0])
    # From performance table
    perf = results.get("performance_table")
    if perf is not None and not perf.empty and "Score_Action" in perf.columns:
        for _, row in perf.iterrows():
            act = str(row.get("Score_Action", "")).lower()
            if act in ("sell", "trim"):
                sell_tickers.add(str(row.get("Ticker", "")).upper().split(".")[0])

    if not sell_tickers:
        return stock_picks

    for scenario_key in ("boost_alpha", "rebalance"):
        scenario = stock_picks.get(scenario_key)
        if not scenario:
            continue
        original = scenario.get("picks", [])
        filtered = [p for p in original
                    if p.get("ticker", "").upper().split(".")[0] not in sell_tickers]
        if len(filtered) < len(original):
            removed = [p["ticker"] for p in original if p not in filtered]
            logger.info("Contradiction check: removed %s from %s (Sell/Trim in action_plan)",
                        removed, scenario_key)
        scenario["picks"] = filtered

    return stock_picks


def _remove_mandate_banned_picks(stock_picks: dict,
                                 user_mandate: dict | None) -> dict:
    """
    B1 (2026-07-17): deterministic mandate guard for AI ideas.

    An asset class with a 0–0 limit is EXCLUDED from the client's mandate
    (e.g. Crypto for a conservative profile) — a pick from that class would
    contradict the mandate panel one page above.  The prompt already carries
    the rule; this post-filter GUARANTEES it even when the model (or the
    rule-based backfill) slips.  Classification reuses the same
    `classify_to_asset_key` the Mandate-Compliance panel and Gatekeeper use,
    so «запрещённый класс» means the same thing everywhere.  Pure/graceful:
    no mandate / no banned classes / import failure → picks unchanged.
    """
    if not user_mandate or not stock_picks:
        return stock_picks
    limits = user_mandate.get("limits_dict") or {}
    banned: set[str] = set()
    for key, bounds in limits.items():
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        if lo == 0 and hi == 0:
            banned.add(str(key))
    if not banned:
        return stock_picks
    try:
        from agent.gatekeeper import classify_to_asset_key
    except Exception:                                  # pragma: no cover
        return stock_picks
    for scenario_key, scenario in stock_picks.items():
        if not isinstance(scenario, dict):
            continue
        original = scenario.get("picks", []) or []
        filtered, removed = [], []
        for p in original:
            tkr = str((p or {}).get("ticker", "")).strip()
            if tkr and classify_to_asset_key(tkr) in banned:
                removed.append(tkr)
                continue
            filtered.append(p)
        if removed:
            logger.info("Mandate guard: removed %s from %s (класс запрещён "
                        "лимитом 0–0 мандата)", removed, scenario_key)
        scenario["picks"] = filtered
    return stock_picks


def _mandate_allows(ticker: str, user_mandate: dict | None) -> bool:
    """True, если класс тикера НЕ закрыт лимитом 0–0 мандата (R-22).

    Тот же классификатор, что у мандатной панели и у гарда идей, — «запрещённый
    класс» обязан значить одно и то же во всех трёх местах.
    """
    if not user_mandate:
        return True
    limits = user_mandate.get("limits_dict") or {}
    banned = set()
    for key, bounds in limits.items():
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        if lo == 0 and hi == 0:
            banned.add(str(key))
    if not banned:
        return True
    try:
        from agent.gatekeeper import classify_to_asset_key
    except Exception:                                  # pragma: no cover
        return True
    return classify_to_asset_key(str(ticker or "").strip()) not in banned


def _backfill_empty_scenarios(stock_picks: dict, regime_label: str, tier: str,
                              market_context: str, results: dict,
                              user_mandate: dict | None = None) -> dict:
    """
    Guarantee all four strategic scenarios carry at least one pick.

    The deep/base templates skip any scenario whose `picks` list is empty,
    so a category the AI left blank (or that the held/contradiction filters
    drained) silently shrinks the idea menu from 4 cards to 3.  We backfill
    such gaps from the deterministic rule-based catalogue, then re-run the
    held-ticker filter so a backfilled idea is never one the user owns.

    R-22 (2026-08-02) — МАНДАТ УЧИТЫВАЕТСЯ ЗДЕСЬ, а не только после.
    Живой отчёт 02.08 пришёл с ТРЕМЯ карточками при подписи «4 идеи»: у клиента
    классы `GlobalETFs` и `Stocks_KZ` закрыты лимитом 0–0 (мандатная панель их
    поэтому и не показывает — `_build_mandate_compliance` скрывает строки 0–0),
    а «Защита капитала» — это ровно защитные сектор/фактор-ETF. Гард
    `_remove_mandate_banned_picks` работал ПОСЛЕ backfill и опустошал корзину
    обратно, второго шанса не было. Теперь: (а) фолбэк берёт первого
    РАЗРЕШЁННОГО мандатом кандидата вместо фиксированного по месяцу, (б) сам
    гард вызывается до этой функции, и она добивает то, что он вычистил.
    """
    fallback = _fallback_stock_picks(regime_label, tier,
                                     user_mandate=user_mandate)
    fallback = _normalise_stock_picks(fallback, tier, market_context)
    fallback = _remove_held_picks(fallback, results)
    fallback = _remove_mandate_banned_picks(fallback, user_mandate)

    for key in ("boost_alpha", "rebalance", "protect_capital", "smart_money"):
        scenario = stock_picks.get(key) or {}
        if scenario.get("picks"):
            continue
        fb = fallback.get(key) or {}
        if fb.get("picks"):
            # Preserve the AI's label/desc if present, else take the fallback's.
            # Sprint-5.2: a label equal to the RAW key (normaliser default when
            # the model omitted it) is NOT a real label — prefer the fallback's
            # human-readable one (the 2026-06-12 BASE report showed card titles
            # "boost_alpha"/"rebalance").
            ai_label = str(scenario.get("label") or "").strip()
            if not ai_label or ai_label == key:
                ai_label = fb.get("label", key)
            desc = scenario.get("desc") or fb.get("desc", "")
            # Sprint-5.2 honesty marker: catalogue picks must not masquerade
            # as the AI's own analysis.
            if "резервный каталог" not in desc:
                desc = (desc + " · резервный каталог").strip(" ·")
            stock_picks[key] = {
                "label": ai_label,
                "desc":  desc,
                "picks": fb["picks"],
            }
            logger.info("Idea backfill: scenario '%s' was empty → filled from "
                        "rule-based catalogue (%d picks)", key, len(fb["picks"]))
    return stock_picks


__all__ = ["generate_narrative"]

# Арх-5: имена, которые импортирует `batch_reports`, — это фактический
# контракт модуля, а не внутренность; приватность была неправдой.
# `_user_prompt`/`_summarise_for_prompt` НЕ переименованы: на них завязаны
# 8 тест-файлов, и массовая правка тестов ради косметики имени — плохой
# размен.  Им заведены ПУБЛИЧНЫЕ синонимы, потребитель импортирует их.
user_prompt = _user_prompt
summarise_for_prompt = _summarise_for_prompt
_REPORT_TOOL = REPORT_TOOL
_build_system_prompt = build_system_prompt
