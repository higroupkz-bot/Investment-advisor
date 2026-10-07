"""Конвейер отчёта как публичный фасад: анализ → гейткипер → payload (Ф2 веб-миграции).

Раньше эти шаги жили ПРИВАТНЫМИ функциями `tg_bot`, и второй канал доставки
(веб OMBRI) импортировал `tg_bot._build_pdf_payload` — то есть тянул `aiogram`,
проверку токена Telegram и весь слой доставки ради сборки payload. Теперь оба
канала зовут ЭТИ функции, и отчёт у них один по построению, а не по копии.

Модуль — слой отчёта (L3), корзина «бот отчётов» манифеста
`tests/test_service_boundaries.py`: он импортирует `ai_narrative`/`pdf_payload`,
а ядро не вправе тянуть наверх. От слоя доставки он свободен — `tg_bot` и
`aiogram` сюда не попадают (`tests/test_report_pipeline.py`).

Код перенесён из `tg_bot` без изменения поведения; приватные имена в `tg_bot`
остались алиасами ради существующих тестов.
"""

from __future__ import annotations

import logging
import math
import os
import re

from agent import gatekeeper
from agent.rag_engine import BANK_ORDER, bank_alias_regex, bank_tail_regex
from ai_narrative import generate_narrative
from finance import investment_logic
from finance.scoring import classify_asset_class
from html_renderer import MOCK_DATA
from pdf_payload import TIER_DEEP, TIER_SCENARIO, build_payload
from profile_manager import PROFILE_BENCH_TICKER
from report_charts import (
    build_equity_curve_svg,
    build_factor_betas_table,
    build_factor_radar_svg,
    build_kpi_sparklines,
    compute_factor_betas,
)

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_portfolio", "run_gatekeeper_for", "build_report_payload",
    "resolve_bench_ticker", "fetch_rag_context", "clean_rag_excerpt", "kb_banks",
]


# ── Анализ ───────────────────────────────────────────────────────────────────

def analyze_portfolio(df, *, price_source: str, bench_ticker: str | None = None,
                      risk_mandate: str | None = None, tradernet_client=None) -> dict:
    """Blocking: MAC3-анализ уже загруженного портфеля → `results{}` (`finance/contracts.py`).

    🔴 `§−122`: `price_source` обязателен и ОБЯЗАН совпадать с источником,
    по которому портфель получен: дефолт `freedom` здесь означал бы поход в
    Tradernet за ценами ручного или демо-портфеля (I-12). Источник доезжает
    КАК ЕСТЬ — неизвестный отвергает провайдер цен, а не этот слой.

    `tradernet_client` — клиент на ключах самого пользователя; без него движок
    строит сервисный из env. Для `manual`/`demo` клиент не используется.
    Менеджер берётся через модуль в момент вызова — тесты подменяют его там.
    """
    kwargs = {"price_source": price_source}
    if tradernet_client is not None:
        kwargs["tradernet_client"] = tradernet_client
    return investment_logic.UniversalPortfolioManager(**kwargs).analyze_all(
        df, profile_benchmark=bench_ticker, risk_mandate=risk_mandate)


def run_gatekeeper_for(results: dict, user_profile: dict | None = None) -> dict:
    """Проверка лимитов мандата (рекомендательная, отчёт не блокирует).

    Без профиля — лимиты по умолчанию. С профилем — потолок волатильности
    `target_volatility × 1.2` и лимиты классов профиля, как в боте с Sprint-5.
    """
    if not user_profile:
        return gatekeeper.run_gatekeeper(results)
    gate_limits = {"max_portfolio_volatility": user_profile["target_volatility"] * 1.2}
    return gatekeeper.run_gatekeeper(results, user_limits=gate_limits, user_profile=user_profile)


# ── Бенчмарк ─────────────────────────────────────────────────────────────────

def resolve_bench_ticker(profile: dict | None) -> str | None:
    """Sprint-5 Task 8 — honour the user's SAVED benchmark choice.

    `get_profile` returns the full row (SELECT *), so the explicit
    `benchmark_ticker` picked during onboarding is available.  Previously the
    report always used PROFILE_BENCH_TICKER[profile_name], so the 12-option
    benchmark picker was a dead control — a Conservative user always got AGG,
    an Aggressive one always QQQ, whatever they chose.  Prefer the saved
    selection; fall back to the profile default only when none was stored.
    """
    if not profile:
        return None
    saved = profile.get("benchmark_ticker")
    if saved:
        return saved
    return PROFILE_BENCH_TICKER.get(profile.get("profile_name"))


# ── RAG-контекст для нарратива ──────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


# F-17 (2026-07-10): leading remnants of the issuing bank's own letterhead —
# a chunk often starts mid-header («…J.P. Morgan Putting these pieces
# together…» → body opens with «Morgan Putting…»).  The attribution already
# comes from the retrieval metadata, so a dangling name fragment at the start
# of the EXCERPT is pure noise — strip it when followed by a capitalised word.
# The optional name tails are POSSESSIVE (`?+`, py3.11+): without it the engine
# backtracks on «Morgan Stanley expects…» (lowercase verb → full name fails →
# retry with bare «Morgan» + capital «S» of «Stanley») and strips HALF the
# bank name; possessive tails make the match all-or-nothing.
# §−95: перечень имён — из общего реестра (`agent.rag_engine`), а не пятой
# копией здесь.  Копия молча разошлась: «Merrill» нарратив знал, а этот
# чистильщик — нет, и шапка «Merrill Equities remain…» уезжала в выдержку.
# Полные имена И «хвосты» двусловных: разрыв шапки PDF оставляет в начале
# выдержки как «Morgan Stanley», так и одинокое «Stanley».
#
# Две тонкости старого regex, которые обязаны пережить переезд на реестр:
#  • АТОМАРНАЯ группа `(?>…)`. Совпало длинное имя — откатываться к короткому
#    нельзя. Иначе «Morgan Stanley expects…» (банк — ПОДЛЕЖАЩЕЕ, глагол с
#    маленькой) откатится к обрубку «Morgan» и срежет полфамилии, оставив
#    «Stanley expects…». Раньше это делали посессивные `?+`.
#  • Взгляд вперёд СТРОГО регистрозависимый — `(?-i:…)`. Имена ищем без учёта
#    регистра, но «дальше идёт заглавная» — это и есть признак шапки письма, а
#    не подлежащего; под общим IGNORECASE он совпадал с чем угодно.
_RAG_BANK_REMNANT_RE = re.compile(
    r"^(?>"
    + "|".join(p for b in BANK_ORDER
               for p in (bank_alias_regex(b), bank_tail_regex(b)) if p)
    + r")[\s:—–-]+(?-i:(?=[A-ZА-Я]))", re.IGNORECASE)
# Runs of ≥3 numeric/percent tokens are chart-axis labels scraped from the
# PDF («12% 10% 8% 6% 4% 2%»), never prose — cut the run and what follows it.
_RAG_AXIS_RUN_RE = re.compile(
    r"\s*(?:[-+]?\d+(?:[.,]\d+)?%?\s+){2,}[-+]?\d+(?:[.,]\d+)?%?(?:\s|$)")


def clean_rag_excerpt(text: str, max_len: int = 190) -> str:
    """Turn a raw RAG chunk into a clean, readable excerpt (замечание R2#3 —
    выдержки обрывались на середине слова: «ury/…», «…mod-»; F-17 — «Morgan
    Putting…», «mod- estly», осевые подписи «12% 10% 8%…» из PDF-графиков).

    - collapse whitespace + strip markdown;
    - re-join PDF line-break hyphenation («mod- estly» → «modestly»);
    - drop a leading mid-word fragment OR a dangling bank-letterhead remnant;
    - cut chart-axis number runs (and the label soup that trails them);
    - reject excerpts that are mostly non-prose (letters < 55% of tokens);
    - truncate on a WORD boundary with an ellipsis, never mid-word.
    """
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", str(text or ""))          # de-bold
    t = re.sub(r"\s+", " ", t).strip().lstrip("#*•-—·> ").strip()
    # PDF hyphenation artifact: a line break inside a word survives extraction
    # as «xxx- yyy» (lowercase on both sides) — re-join without the hyphen.
    t = re.sub(r"([a-zа-яё])- ([a-zа-яё])", r"\1\2", t)
    if not t:
        return ""
    if t[:1].islower():                       # opened mid-word → find a clean start
        m = re.search(r"[.!?]\s+(\S)", t)     # right after a sentence end
        if m:
            t = t[m.start(1):]
        else:
            m2 = re.search(r"[A-ZА-Я0-9]", t)  # else first capital / digit (if near)
            if m2 and m2.start() < 45:
                t = t[m2.start():]
    t = _RAG_BANK_REMNANT_RE.sub("", t.strip()).strip()
    # Chart-axis runs: keep the prose BEFORE the first run, drop the run and
    # everything after it (what follows an axis is chart legend soup).
    m_axis = _RAG_AXIS_RUN_RE.search(t)
    if m_axis:
        t = t[:m_axis.start()].rstrip(",;:—–- ")
    t = t.strip()
    if not t:
        return ""
    # Prose gate: if most tokens are numbers/symbols the chunk is a table or
    # a chart, not a quotable sentence — reject so the caller picks the next
    # candidate block (Pass-2 in fetch_rag_context).
    tokens = t.split()
    lettered = sum(1 for w in tokens if re.search(r"[A-Za-zА-Яа-яё]{2,}", w))
    if tokens and lettered / len(tokens) < 0.55:
        return ""
    if len(t) > max_len:                      # truncate on a word boundary + «…»
        t = t[:max_len].rsplit(" ", 1)[0].rstrip(",;:—- ") + "…"
    return t.strip()



def kb_banks(docs: list[dict]) -> list[str]:
    """Issuers actually present in the RAG store, most-covered first.

    §−93: the provenance label was the frozen literal «GS / MS / JPM», so a KB
    holding Citi / Jefferies / Barclays notes still advertised three banks — the
    reader could not tell whether a newly ingested issuer had arrived.  Derived
    here from the SAME `list_documents()` inventory that feeds отчёты/чанки, so
    label and counts can never disagree.  «Unknown» is dropped: it is the
    absence of an issuer, not an issuer.
    """
    tally: dict[str, int] = {}
    for d in docs or []:
        bank = str((d or {}).get("bank") or "").strip()
        if not bank or bank == "Unknown":
            continue
        tally[bank] = tally.get(bank, 0) + int((d or {}).get("chunks", 0) or 0)
    # Chunks desc, then name — ties must not reorder between runs.
    return [b for b, _ in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))]


def fetch_rag_context(results: dict) -> tuple[str, list[str], str, dict]:
    """
    Pull macro + micro RAG excerpts for the AI narrative (deep tier only).
    Also queries for regime confirmation from bank reports.

    Returns (market_context_str, regime_rag_confirm_list, rag_status, kb_stats).

    rag_status is a 3-state flag so the UI can tell the user the truth instead
    of a misleading binary "не использован":
      • "unavailable" — ChromaDB empty or unreachable (RAG never queried)
      • "no_match"    — queried, but no bank report cleared the similarity gate
      • "used"        — relevant bank context was retrieved and fed to the model

    kb_stats = {"docs": <distinct source PDFs in ChromaDB>, "chunks": <embeddings>,
                "banks": [<issuers present, most-covered first>]}
    surfaces how much bank research the knowledge base actually holds, so the
    CoVe panel can prove "RAG sees N reports / M chunks" instead of a bare flag.
    `banks` (§−93) feeds the provenance label, which used to be the FROZEN
    literal «GS / MS / JPM» — after ingesting Citi/Jefferies/Barclays notes the
    report kept naming three banks and hid the other seven.
    """
    _empty_stats = {"docs": 0, "chunks": 0, "banks": []}
    try:
        from agent.rag_engine import FinancialRAG
        rag = FinancialRAG(db_path=os.environ.get("CHROMA_LOCAL_PATH",
                                                   "/app/data/chroma_db"))
        if int(rag.collection.count()) == 0:
            logger.info("RAG database empty — narrative will run without bank context.")
            return "", [], "unavailable", _empty_stats
        # «Отчётов / чанков» must count REAL reports only — list_documents()
        # excludes tmp-named ingestion artifacts (замечание R2#6: показывало 48
        # вместо 29).  Derive BOTH counts from it so they stay consistent.
        try:
            docs = rag.list_documents()   # real sources only
            n_docs   = len(docs)
            n_chunks = sum(int(d.get("chunks", 0) or 0) for d in docs)
            banks    = kb_banks(docs)
        except Exception:
            n_docs, n_chunks = 0, int(rag.collection.count())
            banks = []
        kb_stats = {"docs": n_docs, "chunks": n_chunks, "banks": banks}

        perf = results.get("performance_table")
        tickers: list[str] = []
        if perf is not None and not perf.empty and "Ticker" in perf.columns:
            tickers = [str(t) for t in perf["Ticker"].tolist()
                       if str(t).upper() not in {"USD", "EUR", "RUB", "KZT", "CASH"}][:12]

        macro_query = ("Rating upgrades or downgrades, sector outlook, "
                       "fund flows, recession or expansion calls, currency views")
        macro_ctx = rag.get_market_sentiment(query=macro_query, n_results=3)

        micro_ctx = ""
        if tickers:
            micro_query = "Outlook, target price, recommendation for: " + ", ".join(tickers)
            micro_ctx   = rag.get_market_sentiment(query=micro_query, n_results=3)

        # Regime confirmation — look for reports that discuss the current macro
        # regime.  Each excerpt is tagged with its ISSUING BANK (recovered from
        # the retrieval header) so the DEEP report can attribute the bank view
        # instead of showing an anonymous excerpt (замечание 2026-07-09 #5).
        regime_rag_confirm: list[dict] = []
        regime = results.get("regime") or {}
        regime_label = regime.get("regime", "")
        if regime_label:
            regime_query = (
                f"Market regime {regime_label} GDP growth recession expansion "
                "economic cycle leading indicators PMI yield curve"
            )
            # Pull MORE than we show so we can pick DISTINCT banks (замечание
            # R2#3: раньше все 3 выдержки были из одного JPMorgan).
            regime_raw = rag.get_market_sentiment(query=regime_query, n_results=6)
            if regime_raw and "NO PDF DATA" not in regime_raw:
                # The raw context is Markdown with retrieval-header lines
                # («--- [дата] файл — БАНК · секция (актуальность: …) ---»).  Split
                # it into (bank, body) BLOCKS — the header starts a block, the
                # lines after it (until the next header) are the chunk body — then
                # build ONE clean excerpt per block (whole-body, sentence-bounded)
                # instead of an arbitrary mid-sentence line.
                blocks: list[tuple[str, str]] = []
                cur_bank, cur_body = "", []
                for line in regime_raw.split("\n"):
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("---"):
                        if cur_body:
                            blocks.append((cur_bank, " ".join(cur_body)))
                            cur_body = []
                        m = re.search(r"—\s*([^·()]+?)\s*(?:·|\(|$)", s)
                        cur_bank = (m.group(1).strip() if m else "")
                        if cur_bank in ("—", "Unknown"):
                            cur_bank = ""
                        continue                     # retrieval header, not text
                    cur_body.append(s)
                if cur_body:
                    blocks.append((cur_bank, " ".join(cur_body)))

                seen_banks: set[str] = set()
                seen_text:  set[str] = set()

                def _add_excerpt(bank: str, body: str) -> bool:
                    exc = clean_rag_excerpt(body)
                    if len(exc) < 45 or exc in seen_text:
                        return False
                    regime_rag_confirm.append({"text": exc, "bank": bank})
                    seen_text.add(exc)
                    return True

                # Pass 1 — one clean excerpt per DISTINCT bank.
                for bank, body in blocks:
                    bkey = (bank or "").lower()
                    if bkey and bkey in seen_banks:
                        continue
                    if _add_excerpt(bank, body):
                        if bkey:
                            seen_banks.add(bkey)
                        if len(regime_rag_confirm) >= 3:
                            break
                # Pass 2 — if fewer than 3 banks were available, fill the rest.
                if len(regime_rag_confirm) < 3:
                    for bank, body in blocks:
                        if _add_excerpt(bank, body) and len(regime_rag_confirm) >= 3:
                            break

        sections = []
        if macro_ctx and "NO PDF DATA" not in macro_ctx:
            sections.append("=== MACRO TRENDS ===\n" + macro_ctx)
        if micro_ctx and "NO PDF DATA" not in micro_ctx:
            sections.append("=== MICRO INSIGHTS ===\n" + micro_ctx)
        ctx = "\n\n".join(sections)
        # Queried successfully — "used" iff something cleared the gate.
        return ctx, regime_rag_confirm, ("used" if ctx else "no_match"), kb_stats
    except Exception as exc:
        logger.info("RAG context fetch skipped: %s", exc)
        return "", [], "unavailable", _empty_stats


# ── Payload отчёта ───────────────────────────────────────────────────────────

def build_report_payload(results: dict, tier: str,
                       user_bench_ticker: str | None = None,
                       prev_snapshot: dict | None = None,
                       user_risk_profile: str = "Moderate",
                       user_profile: dict | None = None) -> dict:
    """
    Build the PDF payload from analyze_all() output.

    v2 (default): delegates to pdf_payload.build_payload — produces the rich
    schema consumed by report_basic.html (2pp) and report_deep.html (4pp),
    enriched with SVG charts and an AI narrative.

    v1 (legacy, REPORT_VERSION=v1 env): retains the old shape so the legacy
    report.html keeps rendering without code changes elsewhere.
    """
    # Scenario tier — детерминированный, БЕЗ ИИ-вызовов и RAG (0 LLM-API →
    # 1 токен).  Собственная схема `data.scenario`; рендерится своим Jinja-
    # шаблоном (html_renderer маршрутизирует scenario минуя Premium).
    if (tier or "").lower() == TIER_SCENARIO:
        from finance.scenario_report import build_scenario_payload
        return build_scenario_payload(results)

    if os.getenv("REPORT_VERSION", "v2").lower() != "v1":
        # Both tiers pull RAG context (macro + micro + regime confirmation).
        # Deep tier gets full 6000-char context; base tier gets 2000-char
        # version (truncated in ai_narrative) to keep latency reasonable.
        market_context, regime_rag_confirm, rag_status, rag_kb = fetch_rag_context(results)

        ai_summary = generate_narrative(
            results,
            tier=tier,
            market_context=market_context,
            user_risk_profile=user_risk_profile,
            # Фаза 4 (блок 1): режим-специфичные банковские выдержки теперь
            # видны МОДЕЛИ (не только чипам отчёта) — ai_regime_comment и
            # regime_confirmation обязаны опираться на них в первую очередь.
            regime_rag_confirm=regime_rag_confirm,
            # B1 (2026-07-17): мандат клиента (limits_dict/профиль) — идеи ИИ
            # не должны предлагать классы активов с лимитом 0–0.
            user_mandate=user_profile,
        )
        # Propagate the 3-state RAG status into the summary so the integrity
        # panel shows the truth (used / queried-no-match / unavailable) for
        # BOTH tiers — not just deep, and not the misleading binary.
        ai_summary["rag_status"] = rag_status
        # KB inventory (distinct PDFs + chunk embeddings) so the CoVe/quality
        # panels can prove HOW MUCH bank research RAG actually holds and read.
        ai_summary["rag_kb_docs"]   = int((rag_kb or {}).get("docs", 0) or 0)
        ai_summary["rag_kb_chunks"] = int((rag_kb or {}).get("chunks", 0) or 0)
        # §−93: issuers present in the KB — the provenance label names THEM
        # instead of the frozen «GS / MS / JPM».
        ai_summary["rag_kb_banks"]  = [str(b) for b in (rag_kb or {}).get("banks") or []]
        payload = build_payload(
            results, tier,
            ai_summary=ai_summary,
            user_bench_ticker=user_bench_ticker,
            prev_snapshot=prev_snapshot,
            regime_rag_confirm=regime_rag_confirm,
            user_profile=user_profile,
        )
        if tier == TIER_DEEP:
            payload["equity_curve_svg"]    = build_equity_curve_svg(results)
            payload["factor_radar_svg"]    = build_factor_radar_svg(results)
            payload["factor_betas"]        = build_factor_betas_table(results)
            payload["factor_coverage_pct"] = compute_factor_betas(results)[2]
            payload["used_rag"]            = bool(market_context)
        # KPI sparklines — wired for BOTH tiers (both templates surface them
        # in the cover KPI strip).  None when history < 90 daily obs;
        # template gating then hides the chart cells.
        payload["kpi_sparklines"] = build_kpi_sparklines(results)
        return payload

    # ── v1 legacy fallback ──────────────────────────────────────────────────
    metrics    = results.get("portfolio_metrics") or {}
    perf_df    = results.get("performance_table")
    total_val  = _safe_float(results.get("total_value"), 1.0) or 1.0

    cvar_raw   = _safe_float(metrics.get("CVaR_95_Daily"),        0.0)
    sharpe_raw = _safe_float(metrics.get("Sharpe_Ratio"),         float("nan"))
    var_raw    = _safe_float(metrics.get("VaR_95_Daily"),         0.0)
    mdd_raw    = _safe_float(metrics.get("Max_Drawdown"),         0.0)
    vol_raw    = _safe_float(metrics.get("Total_Volatility_Ann"), 0.0)

    cvar_str         = f"{cvar_raw * 100:.1f}%"
    sharpe_str       = f"{sharpe_raw:.2f}" if not math.isnan(sharpe_raw) else "—"
    var_95_daily_str = f"{var_raw * 100:.1f}%"
    max_drawdown_str = f"{mdd_raw * 100:.1f}%"
    risk_pct         = min(100, max(0, int(vol_raw / 0.40 * 100)))

    # ── Aggregate P/L since position entry ─────────────────────────────────
    total_pnl  = 0.0
    total_cost = 0.0
    if perf_df is not None and not perf_df.empty:
        if "PnL" in perf_df.columns:
            total_pnl = float(perf_df["PnL"].fillna(0).sum())
        if "Total_Cost" in perf_df.columns:
            total_cost = float(perf_df["Total_Cost"].fillna(0).sum())
    total_return_pct = (total_pnl / total_cost) if total_cost > 0 else 0.0

    assets: list[dict] = []
    if perf_df is not None and not perf_df.empty:
        for _, row in perf_df.iterrows():
            ticker     = str(row.get("Ticker", "—"))
            cur_val    = _safe_float(row.get("Current_Value"), 0.0)
            weight_pct = cur_val / total_val * 100
            euler      = _safe_float(row.get("Euler_Risk_Contribution_Pct"), 0.0)
            pnl_abs    = _safe_float(row.get("PnL"), 0.0)
            ret_pct    = _safe_float(row.get("Return_Pct"), 0.0)
            assets.append({
                "ticker":      ticker,
                "weight":      f"{weight_pct:.1f}%",
                "asset_class": classify_asset_class(ticker),
                "euler_risk":  f"{euler:.1f}%",
                "pnl_pct":     f"{ret_pct * 100:+.1f}%",   # P/L since entry, %
                "pnl_abs":     f"{pnl_abs:+,.0f}",         # P/L since entry, $
                "pnl_color":   "pos" if ret_pct >= 0 else "neg",
            })

    payload: dict = {
        # KPI strip
        "cvar":            cvar_str,
        "sharpe":          sharpe_str,
        "var_95_daily":    var_95_daily_str,    # canonical key
        "max_drawdown":    max_drawdown_str,    # now real MaxDD
        "risk_pct":        risk_pct,
        # Aggregate P/L since position entry
        "pnl_total_abs":   f"{total_pnl:+,.0f}",
        "pnl_total_pct":   f"{total_return_pct * 100:+.1f}%",
        "pnl_total_color": "pos" if total_return_pct >= 0 else "neg",
        # Holdings
        "assets":          assets or MOCK_DATA["assets"],
    }

    if tier == "deep":
        bm_data   = results.get("benchmark_comparison") or {}
        scenarios = []
        for bm_name, bm in bm_data.items():
            # Prefer the annualised excess (consistent scale with IR / TE).
            # Fall back to period total only if annualised is unavailable
            # (legacy engine output / first-run before refresh).
            excess_ann = bm.get("Excess_Return_Ann")
            if excess_ann is None:
                excess_ann = _safe_float(bm.get("Excess_Return"), 0.0)
            else:
                excess_ann = _safe_float(excess_ann, 0.0)
            pnl_str = f"+{excess_ann*100:.1f}%" if excess_ann >= 0 else f"{excess_ann*100:.1f}%"
            ir      = _safe_float(bm.get("Information_Ratio"), 0.0)
            beat    = "✅ Обыгрывает" if bm.get("Beating_Benchmark") else "❌ Отстаёт"
            scenarios.append({
                "name":        bm_name,
                "probability": f"IR: {ir:.2f}" if ir else "—",
                "pnl":         pnl_str,
                "driver":      f"{beat} бенчмарк",
            })
        payload["scenarios"] = scenarios

    return payload
