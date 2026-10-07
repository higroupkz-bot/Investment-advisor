"""Оркестратор отчёта: `UniversalPortfolioManager` и семь стадий конвейера.

Арх-3.10: выделен из `finance/investment_logic.py` (остался фасадом).

Форма `analyze_all` закреплена ИСПОЛНЯЕМО — `tests/test_engine_orchestrator.py`
пинит размер тела, ПОРЯДОК стадий, полноту проводки и единственность выхода.
Порядок не косметика: стадии общаются через состояние движка (`_last_*`),
читаемое из `getattr`, поэтому перестановка меняет числа МОЛЧА.

Конвейер:

    1 _stage_normalize      источник и канон книги
    2 _stage_price_and_fx   цены, валюта строк, стоимость, веса
    3 _stage_data_quality   чекеры качества (блоки B и C)
    4 _stage_risk_model     факторная модель, ATR, форвардная E[r]
    5 _stage_benchmarks     бенчмарки, периоды, стресс
    6 _stage_scoring        секторы, внешние сигналы, 4-Pillar
    7 _stage_overlay        Black-Litterman, план, эффект
      _build_results        ЕДИНСТВЕННОЕ место сборки контракта

Каналы состояния движка после разреза принадлежат ровно одной стадии:
`_last_psd_repair` / `_last_regression_nobs` — стадии 4, `_last_ortho_betas` —
стадии 5, `_last_sparse_dropped` — стадии 7, `_last_port_log_returns` — сборке.
"""

import os
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any

from finance.broker_api import RealPortfolioRequired
from finance.contracts import AnalyzeResults
from finance.period_returns import (
    MIN_OVERLAP_TDAYS as _MIN_OVERLAP_TDAYS,
    compute_period_returns_table as _compute_period_returns_table,
    build_portfolio_log_returns as _build_portfolio_log_returns,
    compute_benchmark_stats as _compute_benchmark_stats,
)
from finance.stress import run_stress_scenarios as _run_stress_scenarios
from finance.stress import residualize_shocks as _residualize_factor_vector
from finance.simulate import simulate_after_plan as _simulate_after_plan
from services.macro_data import MacroFeed as _MacroFeed

# Относительные импорты внутри пакета: `test_layering` их намеренно не считает
# нарушением, поэтому приватные хелперы движка доступны без переименования.
from .risk_engine import (
    MAC3RiskEngine,
    _MANDATE_BL_CONSTRAINTS,
    _is_leveraged_etp,
    _safe_num,
    apply_leveraged_forward,
)
from .market_preview import MarketDataPreview

logger = logging.getLogger("Antigravity_RiskEngine")


def factor_premia_daily(all_data: pd.DataFrame,
                        factor_tickers: dict[str, str],
                        ortho_betas: dict[str, dict[str, float]]) -> dict[str, float]:
    """Дневная ожидаемая лог-доходность каждого фактора — в пространстве БЕТ.

    Среднее дневной лог-доходности фонда-фактора за всю историю (F-2: в
    дневном лог-пространстве, аннуализация — одна, на итоговом прогнозе
    бумаги).  H-1 (`§−134`): беты бумаги к стилю/EM живут в ОЧИЩЕННОМ от
    родителя пространстве (BLOCK 3.5), поэтому премия ребёнка —
    рыночно-нейтральная: μ_ребёнка − Σ β̂·μ_родителя.  Та же линейная карта
    F-1, что переводит шоки стресса (`stress.residualize_shocks`).  Прежде
    сырое μ(MTUM) несло в себе премию рынка, уже вошедшую через бету рынка —
    прогноз бумаги завышался на Σ β_стиль · β̂(стиль→рынок) · μ_рынка
    (живой DEEP 02.10: ожидаемая 12.4% против BL 5.5%).  Без ортогонализации
    (`ortho_betas` пуст) — сырые средние, как и беты.
    """
    keys = [k for k, v in factor_tickers.items() if v in all_data.columns]
    if not keys:
        return {}
    etfs = [factor_tickers[k] for k in keys]
    f_data = all_data[etfs]
    daily_log = np.log(f_data / f_data.shift(1)).dropna()
    means = daily_log.mean()
    raw = {k: float(means.get(v, 0.0)) for k, v in zip(keys, etfs)}
    return _residualize_factor_vector(raw, ortho_betas or {})


@dataclass
class _AnalysisCtx:
    """Состояние ОДНОГО прогона `analyze_all` (Арх-3.1).

    Зачем контекст, а не передача аргументами
    -----------------------------------------
    Замер потока данных показал, что он СТРОГО НАКАПЛИВАЮЩИЙ: множество живых
    переменных на границах растёт от 3 до 34, минимума в середине нет — почти
    всё, что посчитано, доживает до финальной сборки (она читает 41 имя).
    Значит «узких» швов, где стадии передавали бы друг другу два-три значения,
    не существует, и цепочка функций дала бы сигнатуры на 20–34 параметра —
    читать их тяжелее, чем нынешнюю простыню, то есть цель фазы была бы
    провалена. Стадии вместо этого ДОПИСЫВАЮТ поля сюда.

    Что это НЕ является
    -------------------
    Не хранилищем состояния движка. Шесть каналов (`_last_psd_repair`,
    `_last_regression_nobs`, `_last_ortho_betas`, `_last_sparse_dropped`,
    `_last_port_log_returns`, `_last_fx_records`) по-прежнему живут на
    `self.engine` и читаются через `getattr`. Именно они задают ПОРЯДОК
    стадий, поэтому переставлять стадии нельзя — молча изменятся числа.

    Поля добавляются по мере выноса стадий (`ARCHITECTURE_FOR_AGENTS.md §4`).
    Имена совпадают с прежними локальными переменными намеренно: так диф
    каждой подзадачи остаётся читаемым, а golden-фикстура сверяется побайтово.
    """

    # ── стадия 1: нормализация книги ─────────────────────────────────────────
    df: Any
    _merged_rows: Any
    _dropped_rows: Any
    _portfolio_source: Any
    # ── стадия 2: цены, валюта, веса ─────────────────────────────────────────
    total_portfolio_value: float
    weights_dict: Any
    priced_at_cost: Any
    _fx_rows: Any
    _rep_ccy: Any
    history_result: Any
    # ── стадия 3: чекеры качества ────────────────────────────────────────────
    _dq_report: Any
    # ── стадия 4: риск-модель ────────────────────────────────────────────────
    cov_matrix: Any
    port_metrics: Any
    perf: Any
    _model_uncovered: Any
    # ── стадия 5: бенчмарки, периоды, стресс ────────────────────────────────
    profile_benchmark: Any
    benchmark_factor_profile: Any
    benchmark_results: Any
    period_returns_table: Any
    return_coverage: Any
    stress_scenarios: Any
    # ── стадия 6: секторы, скоринг, режим ────────────────────────────────────
    sector_exposure: Any
    factor_scores: Any
    macro_drivers: Any
    regime_reading: Any
    technicals_map: Any
    asset_scores: Any
    cds_summary: Any
    # ── стадия 7: наложения ──────────────────────────────────────────────────
    bl_records: Any
    action_plan_rows: Any
    expected_effect: Any
    smart_money: Any


class UniversalPortfolioManager:
    COLUMN_ALIASES = {
        'Ticker': ['Symbol', 'Asset', 'Тикер', 'Name'],
        'Quantity': ['Qty', 'Amount', 'Shares', 'Кол-во'],
        'Purchase_Price': ['Buy_Price', 'Price_Buy', 'Entry_Price', 'Цена_Покупки', 'Avg_Price']
    }

    def __init__(self, price_source: str = "freedom", tradernet_client=None):
        # price_source прокидывается в движок и решает, откуда берутся ЦЕНЫ.
        # Дефолт 'freedom' сохраняет поведение всех существующих вызовов.
        # tradernet_client — клиент на ключах самого пользователя (веб); без него
        # движок строит сервисный из env. Раньше веб подменял `engine._tradernet_client`.
        self.engine = MAC3RiskEngine(price_source=price_source,
                                     tradernet_client=tradernet_client)

    def prefetch_market_data(self, candidate_tickers) -> "MarketDataPreview":
        """Public facade for the bot's Step-1 progress panel (H-3).

        Encapsulates every engine-internal access the Telegram layer used to
        reach for directly, returning a ready-to-render summary.  The bot must
        call THIS instead of poking `manager.engine.*`.
        """
        eng = self.engine
        risky_tickers = [t for t in (candidate_tickers or [])
                         if str(t).upper() not in eng.NON_RISK_ASSETS]
        data, history_result = eng.get_market_data(risky_tickers)

        loaded_count = (
            len([c for c in data.columns if not data[c].isna().all()])
            if not data.empty else 0
        )
        internal_tickers = set(list(eng.factor_tickers.values()) + eng.BENCHMARK_EXTRA)
        resolved_portfolio = list(dict.fromkeys(eng.resolve_tickers(risky_tickers)))
        portfolio_loaded = len([
            t for t in resolved_portfolio
            if t in data.columns and not data[t].isna().all()
        ])
        # Подмена неликвидной бумаги на ликвидный прокси (original → proxy).
        # Читаем SSOT `proxy_for`, а не сравниваем строки: прежняя эвристика
        # («резолв ≠ TICKER.US и ≠ TICKER») считала прокси ЛЮБУЮ нормализацию
        # формата, поэтому `KSPI`→`KSPI.KZ` и `BITCOIN`→`BTC-USD` попадали в
        # список подмен — пользователю показали бы «ваш KSPI заменён», хотя это
        # та же бумага.  H-1: раскрывать надо ровно реальные подмены.
        proxy_map: dict = {}
        for t in risky_tickers:
            proxy = eng.proxy_for(t)
            if proxy is not None:
                proxy_map[t] = proxy

        return MarketDataPreview(
            data               = data,
            history_result     = history_result,
            risky_tickers      = risky_tickers,
            resolved_portfolio = resolved_portfolio,
            internal_tickers   = internal_tickers,
            loaded_count       = loaded_count,
            portfolio_loaded   = portfolio_loaded,
            portfolio_total    = len(resolved_portfolio),
            proxy_map          = proxy_map,
        )

    # Колонки, которые склейка дубликатов складывает как ВЕЛИЧИНЫ.
    _MERGE_SUM_COLS = ("Quantity",)

    def _normalize_positions(self, df):
        """Приводит фрейм позиций к контракту движка ДО всей математики.

        Делает ровно две вещи, обе — снятие дефектов сквозного аудита
        (`AUDIT_360_2026-07-30.md §2.1/§2.2`), и обе требуются Фазой 4
        (`PHASE_04 §3.2`: «дубликат тикера → суммирование, средневзвешенная цена»):

        **A-1. Склейка дубликатов.** Одна бумага может прийти дважды: два лота,
        два счёта, один ISIN на двух биржах. Раньше дубликат доезжал до
        `a_data.columns = valid_originals`, давал две колонки с одним именем, и
        Ridge падал с `DuplicateError` из недр sklearn — отчёт не строился
        ВООБЩЕ. Количество складывается, цена покупки — средневзвешенная по
        количеству (Σqty·price / Σqty): это и есть корректная база стоимости.

        **A-2. Нечисловое количество.** `Quantity = NaN` не роняло отчёт — оно
        протекало через `Current_Value` в веса и делало `NaN` ВСЕ головные
        метрики (Vol/Return/Sharpe), при внешне нормальной стоимости портфеля.
        Это хуже падения: читатель не узнаёт, что цифр нет. Такие строки
        отбрасываются и РАСКРЫВАЮТСЯ (`results["dropped_rows"]`), как это давно
        делает sparse-guard.

        Возвращает `(df, merged, dropped)`, где `merged` = [(тикер, сколько
        строк)], `dropped` = [(тикер, причина)].
        """
        merged: list[tuple[str, int]] = []
        dropped: list[tuple[str, str]] = []
        if df is None or df.empty or 'Ticker' not in df.columns:
            return df, merged, dropped

        df = df.copy()
        df['Ticker'] = df['Ticker'].astype(str).str.strip()

        # ── A-2: количество и цена обязаны быть числами ────────────────────
        # `errors="coerce"` превращает мусор ('', 'н/д', None, текст) в NaN, а
        # NaN здесь — единственный честный сигнал «строка непригодна».
        for col in ('Quantity', 'Purchase_Price'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Quantity' in df.columns:
            _bad = ~np.isfinite(df['Quantity'].to_numpy(dtype=float, na_value=np.nan))
            if _bad.any():
                for t in df.loc[_bad, 'Ticker']:
                    dropped.append((str(t), "нечисловое количество"))
                logger.warning(
                    "MAC3 ENGINE ▸ отброшено %d строк(и) с нечисловым количеством: %s "
                    "(иначе NaN обнулил бы Vol/Return/Sharpe всего портфеля)",
                    int(_bad.sum()), ", ".join(t for t, _ in dropped))
                df = df.loc[~_bad].copy()

        if df.empty:
            return df, merged, dropped

        # ── A-1: склейка дубликатов ────────────────────────────────────────
        dup_mask = df['Ticker'].duplicated(keep=False)
        if not dup_mask.any():
            return df, merged, dropped

        has_price = 'Purchase_Price' in df.columns
        has_qty   = 'Quantity' in df.columns
        rows = []
        for ticker, grp in df.groupby('Ticker', sort=False):
            if len(grp) == 1:
                rows.append(grp.iloc[0])
                continue
            merged.append((str(ticker), int(len(grp))))
            row = grp.iloc[0].copy()
            if has_qty:
                qty_sum = float(grp['Quantity'].sum())
                row['Quantity'] = qty_sum
                if has_price:
                    # Средневзвешенная цена покупки = Σ(qty·price)/Σqty.
                    # Вырожденный случай Σqty ≈ 0 (лонг и шорт гасят друг друга)
                    # оставляет простое среднее — делить на ноль нельзя, а
                    # выбрасывать позицию значит потерять её экспозицию.
                    if abs(qty_sum) > 1e-12:
                        row['Purchase_Price'] = float(
                            (grp['Quantity'] * grp['Purchase_Price']).sum()) / qty_sum
                    else:
                        row['Purchase_Price'] = float(grp['Purchase_Price'].mean())
            elif has_price:
                row['Purchase_Price'] = float(grp['Purchase_Price'].mean())
            # Цена брокера у одной и той же бумаги одинакова в любой строке —
            # берём первую непустую, чтобы склейка не теряла котировку.
            if 'Broker_Current_Price' in grp.columns:
                _bp = grp['Broker_Current_Price'].dropna()
                if not _bp.empty:
                    row['Broker_Current_Price'] = _bp.iloc[0]
            # −37: валюта инструмента одна на все его лоты — сохраняем первую
            # непустую, иначе склейка обнулит признак и цена покупки останется
            # неконвертированной.  H-2 (`PHASE_04 §5d`) требует конверсии ДО
            # склейки; здесь это соблюдено: склейка идёт по УЖЕ размеченным
            # строкам, а конверсия — позже, единым курсом на бумагу.
            if 'Currency' in grp.columns:
                _cc = grp['Currency'].astype(str).str.strip()
                _cc = _cc[_cc.ne("") & _cc.ne("nan")]
                if not _cc.empty:
                    row['Currency'] = _cc.iloc[0]
            rows.append(row)

        logger.info(
            "MAC3 ENGINE ▸ склеены дубликаты позиций: %s "
            "(количество суммировано, цена покупки — средневзвешенная)",
            ", ".join(f"{t}×{n}" for t, n in merged))
        out = pd.DataFrame(rows).reset_index(drop=True)
        out.attrs = dict(df.attrs)
        return out, merged, dropped

    def _standardize_columns(self, df):
        for standard, aliases in self.COLUMN_ALIASES.items():
            actual_col = next((c for c in df.columns if c in aliases or c == standard), None)
            if actual_col: df = df.rename(columns={actual_col: standard})
        return df

    def _stage_normalize(self, source) -> "tuple[pd.DataFrame, list, list]":
        """Стадия 1: проверить ИСТОЧНИК книги и привести её к канону (Арх-3.2).

        Здесь портфель ещё не считается — только доказывается, что считать
        ЕСТЬ ЧТО и что строки пригодны:

        * три гейта источника (`_ramp_is_fallback` / `_ramp_is_mock` / пустой
          фрейм). Гейт fallback-мока — не формальность: без него отчёт
          строился бы по синтетике, когда брокер молчит, и пользователь принял
          бы его за свой (инцидент `AUDIT §−30`);
        * `_standardize_columns` — имена колонок;
        * `_normalize_positions` — склейка дублей позиции (A-1) и отбрасывание
          строк с нечисловым количеством (A-2). Обе операции обязаны быть
          НАЗВАНЫ пользователю, поэтому возвращаются реестры, а не молча
          применяются.

        Возвращает `(df, merged_rows, dropped_rows, portfolio_source)` — вход
        остальных стадий. Источник читается здесь и передаётся дальше значением:
        `df.attrs` ниже по конвейеру не переживает `join`/`reset_index`.
        Самая изолированная стадия конвейера (вход 0 переменных), поэтому
        вынесена первой: на ней обкатан приём.
        """
        if isinstance(source, pd.DataFrame): raw_df = source
        else: raise ValueError("Неверный источник данных.")

        # ── MAC3 GATE: portfolio source verification ──────────────────────────
        # Detect the three possible DataFrame origins stamped by FreedomConnector:
        #   _ramp_is_mock=True, _ramp_is_fallback=True  → broker API failed, fallback mock
        #   _ramp_is_mock=True, _ramp_source='demo'     → user explicitly chose template/demo
        #   (no attrs)                                  → live broker data
        _is_fallback = raw_df.attrs.get('_ramp_is_fallback', False)
        _is_mock     = raw_df.attrs.get('_ramp_is_mock',     False)
        # Источник СОСТАВА книги. Читается ЗДЕСЬ, на входе: `df.attrs` не
        # переживает `join`/`reset_index` ниже по конвейеру, поэтому позже
        # спрашивать уже некого.
        _portfolio_source = str(raw_df.attrs.get('_ramp_source')
                                or ('demo' if _is_mock else 'freedom'))

        # ── EMPTY-PORTFOLIO GUARD ────────────────────────────────────────────
        # 0-row frame → `total = df['Current_Value'].sum()` = 0 →
        # `df['Current_Value']/total` = all-NaN → Ridge fit + bootstrap CVaR
        # explode with cryptic errors.  Bail out with a user-facing message
        # the existing handler (`RealPortfolioRequired` is caught in
        # tg_bot._run_analysis_background) already understands.
        if raw_df is None or raw_df.empty:
            logger.warning("MAC3 ENGINE GATE ▸ empty portfolio (0 rows) — halting.")
            raise RealPortfolioRequired(
                "У вас нет открытых позиций для анализа.\n\n"
                "Откройте хотя бы одну позицию у брокера или выберите "
                "«Демо-режим» в настройках подключения."
            )

        if _is_fallback:
            logger.error(
                "MAC3 ENGINE GATE ▸ FALLBACK mock portfolio detected "
                "(broker API returned no live data). "
                "Halting immediately — skipping market data fetch and all MAC3 calculations."
            )
            raise RealPortfolioRequired(
                "Не удалось получить данные портфеля от Freedom Broker. "
                "Проверьте ваши API-ключи и попробуйте снова.\n\n"
                "Если хотите запустить анализ на шаблонном портфеле, "
                "выберите «Демо-режим» в настройках подключения."
            )

        if _is_mock:
            logger.info(
                "MAC3 ENGINE ▸ Portfolio source: DEMO (template). "
                "Running analysis on synthetic mock positions."
            )
        else:
            logger.info(
                "MAC3 ENGINE ▸ Portfolio source: REAL (live Freedom Broker data). "
                "Initiating full MAC3 Risk Engine pipeline."
            )
        # ─────────────────────────────────────────────────────────────────────

        df = self._standardize_columns(raw_df)
        df, _merged_rows, _dropped_rows = self._normalize_positions(df)
        df = df.set_index('Ticker')
        return df, _merged_rows, _dropped_rows, _portfolio_source

    def _stage_data_quality(self, df, weights_dict, all_data, _fx_unconvertible):
        """Стадия 3: чекеры качества входных данных, блоки B и C (Арх-3.2→3.3).

        Отвечает на вопрос «можно ли вообще строить отчёт по этим данным» и
        при уровне BLOCK поднимает `DataQualityBlocked` — отказ БЕЗ списания
        токена (списание живёт после доставки отчёта, поэтому исключение
        здесь гарантирует, что пользователь не заплатил).

        Имена параметров совпадают с прежними локальными переменными
        намеренно: тело стадии перенесено ДОСЛОВНО, и это главная опора
        доказательства, что разрез ничего не изменил.

        Возвращает отчёт чекеров (уровень DEGRADE) либо `None`, если модуль
        `data_checks` недоступен.
        """
        # ── Ф-3 (2026-08-02): ЧЕКЕРЫ КАЧЕСТВА ДАННЫХ, блоки B и C ───────────
        # `data_checks.py` (526 строк, 56 тестов) полгода existed как библиотека
        # без потребителя.  Точка вызова — ЗДЕСЬ, а не в слое бота, как
        # предполагал `PHASE_03 §3a.1`: бот между Шагом 1 и Шагом 2 знает
        # количество и цену покупки, но НЕ знает текущую стоимость и веса —
        # без них C-8 выродился бы в подсчёт тикеров (ровно ловушка Т-6,
        # ради которой блок C и написан).  Здесь оба есть, а дорогая часть
        # (факторная модель, SEC, скоринг) ещё не запускалась.
        #
        # 🔴 МОСТ ПРОСТРАНСТВ ИМЁН — без него подключение блокировало БЫ ВСЕХ.
        # Замерено на демо-портфеле: `values` идут ключами ПОРТФЕЛЯ (`AAPL`,
        # `TLT`), а `data.columns` — РАЗРЕШЁННЫМИ (`AAPL.US`, `TLT.US`), и
        # C-8 доложил «не загружены цены по 7 из 9 бумаг (70% стоимости)» на
        # книге, которая строит нормальный отчёт.  Поэтому и веса, и стоимости
        # переводятся в namespace матрицы через `resolve_tickers` — тот же
        # SSOT, что решает, что скачивать и что класть в фактор-модель.
        def _dq_requested_days() -> int:
            """То же окно, что запросил `get_market_data` — C-4 отличает
            УСЕЧЕНИЕ выгрузки от молодого листинга только по нему."""
            try:
                return max(90, min(3650, int(os.getenv("HISTORY_LOOKBACK_DAYS", "1825"))))
            except (TypeError, ValueError):
                return 1825

        _dq_report = None
        try:
            from finance import data_checks as _dc
            _profile = _dc.profile_for_source(self.engine.price_source)

            def _res_key(_t):
                _r = self.engine.resolve_tickers([_t])
                return _r[0] if _r else str(_t)

            # 🔴 При КОЛЛИЗИИ ключей веса и стоимости СУММИРУЮТСЯ, а не
            # затираются: две неликвидные бумаги законно делят один прокси
            # (§−38 намеренно сохраняет дубликаты в `valid_resolved`).
            # Наивный dict-comprehension схлопывал их в одну запись, Σ весов
            # переставала равняться 1.0, и C-10 объявлял «внутреннюю ошибку
            # расчёта долей» на совершенно корректной книге — поймано
            # регрессией (`test_two_illiquid_names_sharing_one_proxy`).
            _dq_weights: dict[str, float] = {}
            _dq_values: dict[str, float] = {}
            for _t in df.index:
                _k = _res_key(_t)
                _dq_weights[_k] = _dq_weights.get(_k, 0.0) + \
                    float(weights_dict.get(_t, 0.0) or 0.0)
                _dq_values[_k] = _dq_values.get(_k, 0.0) + \
                    float(df.at[_t, 'Current_Value'] or 0.0)
            _dq_report = _dc.merge_reports(
                _dc.check_price_matrix(all_data, profile=_profile),
                _dc.check_portfolio_sufficiency(
                    data=all_data, weights=_dq_weights, values=_dq_values,
                    profile=_profile,
                    requested_days=_dq_requested_days(),
                    unconvertible_currencies=_fx_unconvertible,
                ),
            )
            for _f in _dq_report.findings:
                logger.info("Ф-3 %s [%s] %s", _f.id, _f.level.value, _f.message)
            if _dq_report.blocked:
                # Отказ БЕЗ списания токена: та же ветка, что `RealPortfolioRequired`
                # (списание живёт после CHECKPOINT 3, поэтому исключение здесь
                # гарантирует, что пользователь не заплатил).
                raise _dc.DataQualityBlocked(_dq_report)
        except ImportError:                            # pragma: no cover
            logger.warning("Ф-3: data_checks недоступен — проверки пропущены")
        return _dq_report

    def _stage_price_and_fx(self, df):
        """Стадия 2: цены позиций, валюта строк, стоимость и веса (Арх-3.4).

        Самая насыщенная стадия конвейера — здесь книга превращается из списка
        бумаг в оценённый портфель:

        * загрузка истории и разрешение тикеров, включая ПРОКСИ неликвидных
          бумаг (H-1). Прокси влияет ТОЛЬКО на риск: цена и P&L всегда от
          реального тикера, иначе нота, купленная по 100, оценивалась бы по
          последней цене прокси;
        * Base Currency Approach (−37): строка в чужой валюте переводится в
          валюту отчёта, и в реестр `fx_converted_rows` попадают только
          ФАКТИЧЕСКИ сконвертированные — заявить пересчёт, которого не было,
          нельзя (D-2);
        * три гарда капитала: `NaN`, отрицательный (заём превысил активы) и
          нулевой. Каждый говорит СВОЁ — при отрицательном капитале совет
          «проверьте подключение» был бы прямой ложью (A-6);
        * `weights_dict` — вход всей последующей матричной математики.

        Возвращает 10 значений. Это не признак плохого разреза, а ЗАМЕР:
        стадия действительно производит столько состояния, и все десять
        читаются дальше по конвейеру. Их конечный дом — `_AnalysisCtx`.
        """
        # Extract broker-provided current prices before any further processing.
        # Used as fallback for instruments with no Tradernet history
        # (KZ bonds with ISIN tickers, cash balances from Freedom API "acc").
        broker_current_prices: dict = {}
        if "Broker_Current_Price" in df.columns:
            broker_current_prices = df["Broker_Current_Price"].dropna().to_dict()
            df = df.drop(columns=["Broker_Current_Price"])

        # Скачиваем цены на Рисковые активы
        risky_tickers = [t for t in df.index if t not in self.engine.NON_RISK_ASSETS]

        all_data, history_result = self.engine.get_market_data(risky_tickers)
        # Фундаментальные метрики берутся ТОЛЬКО из SEC EDGAR (см. ниже,
        # batch_fundamental_scan). SEC EDGAR покрывает ROE/Debt/Margins/Growth
        # точнее и без зависимости от стороннего API (10-K/10-Q филинги).

        # Установка Current Price.
        # Приоритет:
        #   (1) NON_RISK_ASSETS (кэш) → 1.0
        #   (2) своя ценовая история провайдера (НЕ прокси — см. ниже)
        #   (3) Цена брокера (Freedom API mkt_price) — fallback для КЗ облигаций без данных
        #   (4) Purchase_Price — удержание по цене покупки, когда рыночной цены нет
        #
        # 🔴 ПРОКСИ НЕ ДАЁТ ЦЕНУ (2026-07-29, H-1).  Раньше ветка (2) брала
        # `all_data[res]` для ЛЮБОГО резолвинга, включая подмену: AIX-нота
        # получала последнюю цену прокси-ETF.  Нота, купленная по 100 и стоящая
        # у брокера 103.5, оценивалась в 65 (цена VWOB) — стоимость портфеля
        # падала на треть, а P&L показывал −35% убытка, которого не было
        # (с прежним LQD.US ≈ 110 знак был обратный: фабриковалась +10% прибыль).
        # Прокси существует ТОЛЬКО для ковариации; цена — всегда за реальным
        # тикером, поэтому проксированные бумаги идут сразу в ветки (3)/(4).
        #
        # При этом проксированная бумага ОСТАЁТСЯ в фактор-модели (у неё есть
        # ряд прокси), поэтому в `broker_priced_only` она НЕ попадает — иначе
        # смысл слоя прокси исчез бы.
        #
        # ВАЖНО: resolve_tickers() пропускает NON_RISK_ASSETS → список короче df.index.
        # Используем per-ticker резолюцию вместо zip, чтобы избежать сдвига индексов.
        current_prices = {}
        broker_priced_only: set = set()  # тикеры, оцениваемые только через брокера (без фактор-модели)
        priced_at_cost: set = set()      # проксированные бумаги без рыночной цены → по цене покупки
        priced_from_matrix: set = set()  # −37: цена взята из УЖЕ сконвертированной матрицы
        for orig in df.index:
            orig_str = str(orig).upper().strip()
            if orig_str in self.engine.NON_RISK_ASSETS:
                current_prices[orig] = 1.0
                continue
            resolved_list = self.engine.resolve_tickers([orig])
            res = resolved_list[0] if resolved_list else orig_str
            is_proxied = self.engine.proxy_for(orig_str) is not None
            if not is_proxied and res in all_data.columns:
                current_prices[orig] = all_data[res].iloc[-1]
                # −37: эта цена пришла из МАТРИЦЫ, а она уже сконвертирована
                # `_apply_fx_conversion`.  Повторно её умножать нельзя.
                priced_from_matrix.add(orig)
            elif orig in broker_current_prices:
                # KZ облигации / AIX-ноты: используем mkt_price от Freedom API
                current_prices[orig] = broker_current_prices[orig]
                if not is_proxied:
                    broker_priced_only.add(orig)
                logger.info(
                    "Используется цена брокера для %s = %.4f (нет своей истории)",
                    orig, broker_current_prices[orig],
                )
            elif is_proxied:
                # Неликвидная бумага без брокерской цены — удержание по цене
                # покупки (P&L = 0 честнее сфабрикованного по чужому ETF).
                p_price = df.loc[orig, 'Purchase_Price']
                current_prices[orig] = p_price if pd.notna(p_price) else 100.0
                priced_at_cost.add(orig)
                logger.info(
                    "%s оценена по цене покупки: нет ни своей истории, ни цены "
                    "брокера (прокси %s используется только для риска)", orig, res)

        df['Current_Price'] = pd.Series(current_prices)
        df = df.dropna(subset=['Current_Price'])

        # ── −37: ВСЁ В БАЗОВОЙ ВАЛЮТЕ (2026-08-02) ───────────────────────────
        # Матрица цен уже сконвертирована (`_apply_fx_conversion`), но две
        # величины приходили СЫРЫМИ и портили результат:
        #
        #   1. `Purchase_Price` — цена покупки в валюте сделки.  Для KASE-бумаги
        #      (100 шт по 12 000 ₸ при базовой USD) `Total_Cost` = $1 200 000
        #      против `Current_Value` ≈ $2 500 → доходность −99.8%.
        #   2. КЭШ в неба́зовой валюте — цена пиннится в 1.0 безусловно, поэтому
        #      2 500 000 ₸ (≈$4 800) входили в портфель как **$2 500 000**
        #      (замерено): портфель раздувался в сотни раз вместе с весами,
        #      риском и мандатной панелью.
        #
        # Решение владельца: ЕДИНЫЙ ТЕКУЩИЙ КУРС — тот же множитель, которым
        # сконвертирована последняя строка матрицы цен (`fx_rate_to_base`), без
        # учёта движения курса с момента покупки.
        #
        # Математическое следствие, которое ОБЯЗАНО быть раскрыто в отчёте:
        # при едином курсе он сокращается тождественно
        #   Return = (P_now·r − P_buy·r)/(P_buy·r) = (P_now − P_buy)/P_buy,
        # то есть доходность позиции — в ВАЛЮТЕ ИНСТРУМЕНТА, а не долларового
        # инвестора.  Это конвенция, а не приближение: стоимость, веса, мандат
        # и вся риск-математика при этом корректны (курс в числителе и
        # знаменателе один).  Раскрытие — `results["fx_converted_rows"]`.
        _rep_ccy = self.engine.reporting_currency.value
        _fx_rows: list[dict] = []
        # Ф-3: валюты, для которых курса НЕ нашлось. В −37 факт только
        # логировался; теперь он доезжает до чекера C-11 и до CoVe — строка,
        # оставшаяся в своей валюте, обязана быть названа, а не утонуть в логе.
        _fx_unconvertible: list[str] = []
        if 'Currency' in df.columns:
            for _t in df.index:
                _ccy = str(df.at[_t, 'Currency'] or "").upper().strip()
                if not _ccy or _ccy == _rep_ccy:
                    continue
                _rate = self.engine.fx_rate_to_base(_ccy)
                if _rate is None or not np.isfinite(_rate) or _rate <= 0:
                    logger.warning(
                        "−37: нет курса %s→%s для %s — строка остаётся в своей "
                        "валюте (раскрывается в отчёте)", _ccy, _rep_ccy, _t)
                    if _ccy not in _fx_unconvertible:
                        _fx_unconvertible.append(_ccy)
                    continue
                # Цена ПОКУПКИ всегда в валюте сделки → конвертируем всегда.
                _pp = _safe_num(df.at[_t, 'Purchase_Price'])
                if _pp is not None:
                    df.at[_t, 'Purchase_Price'] = _pp * _rate
                # ТЕКУЩАЯ цена — только если она НЕ из матрицы: матрица уже
                # прошла `_apply_fx_conversion`, второй множитель испортил бы
                # её.  Конвертировать надо ровно те источники, что дают цену в
                # валюте инструмента: цена брокера, цена покупки (priced_at_cost)
                # и пиннящаяся в 1.0 строка кэша.
                if _t not in priced_from_matrix:
                    _cp = _safe_num(df.at[_t, 'Current_Price'])
                    if _cp is not None:
                        df.at[_t, 'Current_Price'] = _cp * _rate
                _fx_rows.append({"ticker": str(_t), "currency": _ccy,
                                 "rate": round(float(_rate), 8)})
            if _fx_rows:
                logger.info("−37: в базовую валюту (%s) переведено %d строк: %s",
                            _rep_ccy, len(_fx_rows),
                            ", ".join(f"{r['ticker']}({r['currency']})" for r in _fx_rows))

        # Стоимости и Веса
        df['Total_Cost'] = df['Quantity'] * df['Purchase_Price']
        df['Current_Value'] = df['Quantity'] * df['Current_Price']
        df['PnL'] = df['Current_Value'] - df['Total_Cost']
        # BLK-2: guard against Purchase_Price == 0 (gifted shares, corporate
        # actions, broker data gaps).  A bare PnL/0 yields ±inf — and inf is
        # NOT caught by a downstream .fillna(0) — so one zero-cost row would
        # corrupt the whole portfolio return.  Replace the 0 denominator with
        # NaN first, then fill the resulting NaN with 0.0 (flat return).
        df['Return_Pct'] = df['PnL'].div(df['Total_Cost'].replace(0, np.nan)).fillna(0.0)

        total_portfolio_value = float(df['Current_Value'].sum())
        # Second guard: even with rows, if every Current_Price is 0 or all
        # rows are NON_RISK_ASSETS with zero balance, total is 0.  Don't
        # divide — propagate the same user-friendly error.
        # A-6 (2026-08-02): ТРИ РАЗНЫХ состояния под одним сообщением.
        # Раньше все они печатали «Стоимость портфеля = 0. Проверьте подключение
        # к брокеру» — и при ОТРИЦАТЕЛЬНОМ капитале это прямая ложь: связь с
        # брокером в порядке, данные доехали, просто маржинальный заём превысил
        # активы. Пользователь шёл проверять подключение вместо того, чтобы
        # закрывать плечо. Диагноз теперь называет то, что произошло.
        if not np.isfinite(total_portfolio_value):
            logger.warning("MAC3 ENGINE ▸ total portfolio value is NaN — halting.")
            raise RealPortfolioRequired(
                "Не удалось посчитать стоимость портфеля — в данных брокера "
                "встретились некорректные числа. Повторите анализ позже; "
                "если ошибка повторяется, напишите в /support."
            )
        if total_portfolio_value < 0:
            _neg_assets = float(df.loc[df['Current_Value'] > 0, 'Current_Value'].sum())
            _neg_debt = float(-df.loc[df['Current_Value'] < 0, 'Current_Value'].sum())
            logger.warning(
                "MAC3 ENGINE ▸ NEGATIVE equity: assets=%.2f debt=%.2f net=%.2f — halting.",
                _neg_assets, _neg_debt, total_portfolio_value)
            raise RealPortfolioRequired(
                "Заёмные средства превышают стоимость активов — собственный "
                "капитал счёта отрицательный, и доли позиций посчитать нельзя "
                "(они делятся на капитал). Это НЕ сбой подключения: данные "
                "получены. Сократите плечо или пополните счёт и повторите анализ."
            )
        if total_portfolio_value == 0:
            logger.warning("MAC3 ENGINE ▸ total portfolio value is 0 — halting.")
            raise RealPortfolioRequired(
                "Стоимость портфеля = 0. Проверьте подключение к брокеру "
                "или откройте позиции — анализ невозможен на пустом счёте."
            )
        weights_dict = (df['Current_Value'] / total_portfolio_value).to_dict()
        return (df, all_data, history_result, broker_priced_only, priced_at_cost, _rep_ccy, _fx_rows, _fx_unconvertible, total_portfolio_value, weights_dict)

    def _stage_risk_model(self, df, all_data, weights_dict, broker_priced_only,
                          history_result, _dq_report):
        """Стадия 4: факторная риск-модель, ATR, форвардная E[r] (Арх-3.7).

        Сердце движка: здесь книга получает ковариацию, беты, вклады в риск и
        обе доходности — реализованную и форвардную. Вынесена ШЕСТОЙ по счёту:
        контекст и приём к этому моменту обкатаны пятью стадиями, а последней
        остаётся `_stage_overlay` (вход 20 — самая связная).

        Что внутри:

        * `calculate_structural_risk` — факторная модель Barra-стиля. Из неё
          выходят `cov_matrix`, факторный фрейм и `port_metrics`, и она же
          ЗАПОЛНЯЕТ состояние движка (`_last_*`), которым живут стадии ниже;
        * C-6: вырожденную ковариацию движок чинит сам (`_nearest_psd`) и
          правильно делает — отчёт лучше построить; но чинит МОЛЧА, поэтому
          чекер доливает находку в тот же отчёт качества. В STRICT — отказ;
        * ATR (OHLC, иначе Close-only) и фундаменталка SEC EDGAR — обе под
          `try`, отчёт обязан строиться и без них;
        * σ_ETP плечевых бумаг (P-1/P-7) — считается ОДИН раз и питает и
          forward-drag здесь, и path-dependent стресс в стадии 5;
        * форвардная E[r] = β·μ БЕЗ Ridge-интерсепта (F-14) плюс гейт окна:
          при регрессионном окне короче торгового года портфельная панель
          гаснет в «—». Именно этот гейт закрыл живой дефект обложки
          «ожид. дох. 218.4% · Sharpe 11.29».

        🔴 Чем доказан перенос. Эталон здесь защищает СЛАБЕЕ, чем на других
        стадиях, и это ЗАМЕРЕНО, а не предположено: из шести каналов состояния
        движка два мертвы в обоих сценариях фикстуры — `_last_psd_repair`
        (`None`, то есть ветка C-6 внутри этой стадии не исполняется вовсе) и
        `_last_fx_records` (`[]`, известное ограничение: `get_market_data`
        мокается целиком). Плюс покрытие стадии всем набором — 72.8%.

        Поэтому доказательством служит опора №1 — ДОСЛОВНОСТЬ переноса
        (сверка тела с исходным блоком построчно), а не числа снимка: перенос
        без изменения текста не может сломать ветку, которую никто не
        исполняет. Дотягивать фикстуру до вырожденной ковариации на шаге
        разреза НЕ нужно — это отдельная задача со своим тестом (§−61, §0c).

        Возвращает 7 значений в порядке вычисления. `df` и `_dq_report` стоят
        и на входе, и на выходе: стадия их ДОПОЛНЯЕТ (джойны колонок; находка
        C-6 в отчёт качества), а не создаёт заново.
        """
        # MAC3 Structural Risk.
        # Исключены из факторной модели:
        #   • NON_RISK_ASSETS (кэш) — нулевая волатильность по определению
        #   • broker_priced_only — КЗ облигации без ценовой истории в Tradernet;
        #     их веса в weights_dict корректно разбавляют риск акций.
        # Облигации, разрешённые в proxy-ETF (BIL/LQD/HYG), ОСТАЮТСЯ в actual_risky —
        # у них есть ценовая история и они корректно обрабатываются Rates-фактором.
        actual_risky = [
            t for t in df.index
            if t not in self.engine.NON_RISK_ASSETS and t not in broker_priced_only
        ]
        cov_matrix, factor_df, port_metrics = self.engine.calculate_structural_risk(all_data, actual_risky, weights_dict)

        # ── C-6 (2026-08-02): вырожденная факторная ковариация ──────────────
        # Вход появляется ТОЛЬКО здесь — матрица считается внутри вызова выше,
        # уже после того, как стали известны веса.  Поэтому C-6 идёт вторым,
        # коротким проходом и доливается в тот же отчёт: сигнатуры чекеров
        # блока C при этом не тронуты (56 тестов Фазы 3 действуют как есть).
        # Движок вырождение ЧИНИТ сам (`_nearest_psd`) и правильно делает —
        # отчёт лучше построить; но чинит молча, а `PHASE_03 §3a.4` требует об
        # этом СООБЩАТЬ. В STRICT — отказ, там источник один.
        try:
            from finance import data_checks as _dc6
            _c6 = _dc6.check_covariance_health(
                getattr(self.engine, "_last_psd_repair", None),
                profile=_dc6.profile_for_source(self.engine.price_source))
            if _c6.findings:
                for _f in _c6.findings:
                    logger.info("Ф-3 %s [%s] %s", _f.id, _f.level.value, _f.message)
                _dq_report = _dc6.merge_reports(_dq_report, _c6) \
                    if _dq_report is not None else _c6
                if _c6.blocked:
                    raise _dc6.DataQualityBlocked(_dq_report)
        except ImportError:                            # pragma: no cover
            pass
        
        # ATR (Intraday Margin Standard — SEC 34-105226)
        # Uses True ATR (OHLC) when available, falls back to Close-only
        atr_df = self.engine.calculate_atr(
            all_data, actual_risky,
            ohlc_data=getattr(history_result, 'ohlc_data', None),
        )
        if not atr_df.empty:
            df = df.join(atr_df)

        # SEC EDGAR Fundamental Scores (sector-normalized)
        # Build sector_map from MAC3RiskEngine.TICKER_SECTOR for sector-aware thresholds
        sector_map = {t: self.engine.get_ticker_sector(t) for t in actual_risky}
        try:
            from finance.sec_edgar import batch_fundamental_scan
            sec_df = batch_fundamental_scan(actual_risky, sector_map=sector_map)
            if not sec_df.empty:
                df = df.join(sec_df)
        except Exception as e:
            logger.warning(f"SEC EDGAR scan пропущен: {e}")
        # H-4 (`§−132`): Altman Z — классическая формула с РЫНОЧНОЙ
        # капитализацией в X4.  Цена известна только здесь; она в валюте
        # ОТЧЁТА, отчётность SEC — в USD, поэтому капитализация переводится
        # обратно (нет курса → Z пуст, а не балансовый суррогат).
        try:
            from finance.sec_edgar import apply_market_altman
            _rep = self.engine.reporting_currency.value
            if _rep == "USD":
                _usd_per_unit = 1.0
            else:
                _usd_rate = self.engine.fx_rate_to_base("USD")
                _usd_per_unit = (1.0 / _usd_rate
                                 if _usd_rate and np.isfinite(_usd_rate) and _usd_rate > 0
                                 else None)
            df = apply_market_altman(df, usd_per_price_unit=_usd_per_unit)
        except Exception as e:
            logger.warning(f"Altman Z по капитализации пропущен: {e}")
        
        # Джойним факторы к общему DF
        df = df.join(factor_df)
        
        # Для кэша выставляем волатильность 0
        if 'Residual_Vol_Ann' in df.columns:
            df['Residual_Vol_Ann'] = df['Residual_Vol_Ann'].fillna(0)
            df['Euler_Risk_Contribution_Pct'] = df['Euler_Risk_Contribution_Pct'].fillna(0)

        # Ожидаемая доходность: annualised CAPM-style return from factor betas.
        #
        # Correct approach: betas are from DAILY log-return regression, so
        # expected return must be built from the SAME daily timescale and then
        # annualised — NOT from the cumulative period log-return (which mixes
        # timescales and produces nonsensical "expected" values).
        #
        # Algorithm:
        #   1. Compute mean daily log-return for each factor ETF.
        #   2. GEOMETRIC annualisation: E[r_ann] = exp(mean_daily_log * 252) - 1
        #      This avoids the arithmetic overestimate (arithmetic: mean*252).
        #   3. Map ETF tickers → factor key names (Beta_Market ← Market ← SPY.US).
        #   4. Expected return = Σ beta_k * E[r_k_ann] — β·μ ONLY.  The Ridge
        #      intercept (realised idiosyncratic drift) is EXCLUDED from the
        #      forward forecast (F-14); it stays visible via Alpha_Specific.
        #   5. Specific Alpha = actual return - expected return (in %).
        # P-1/P-7: σ_ETP (daily, ddof=1) for each registry leveraged-ETP name,
        # from its OWN price history — shared by the forward-drag adjustment
        # and the path-dependent stress branch.  ≥60 obs floor (sparse-guard
        # parity); thinner names simply stay absent from the map.
        letf_sigma_map: dict[str, float] = {}
        try:
            for _t in df.index:
                if not _is_leveraged_etp(_t):
                    continue
                _res_l = self.engine.resolve_tickers([_t])
                _col = _res_l[0] if _res_l else None
                if _col and _col in all_data.columns:
                    _pr = all_data[_col].dropna()
                    if len(_pr) >= _MIN_OVERLAP_TDAYS:
                        _lr = np.log(_pr / _pr.shift(1)).dropna()
                        _sd = float(_lr.std(ddof=1))
                        if np.isfinite(_sd) and _sd > 0:
                            letf_sigma_map[str(_t)] = _sd
        except Exception as _exc:
            logger.warning("LETF sigma map skipped: %s", _exc)
            letf_sigma_map = {}

        # H-1 (`§−134`): премии факторов — в ТОМ ЖЕ пространстве, что и беты.
        factor_mu_daily_by_key = pd.Series(factor_premia_daily(
            all_data, self.engine.factor_tickers,
            getattr(self.engine, "_last_ortho_betas", {}) or {}), dtype=float)

        beta_cols = [c for c in df.columns if str(c).startswith('Beta_')]
        if beta_cols and not factor_mu_daily_by_key.empty:
            beta_factor_keys = [c[len('Beta_'):] for c in beta_cols]
            mu_vec = factor_mu_daily_by_key.reindex(beta_factor_keys).fillna(0.0).values
            # F-14 (2026-07-10, пост-релизный фикс «218.4%»): the FORWARD
            # expectation is now β·μ ONLY — the Ridge intercept (realised
            # idiosyncratic drift) is EXCLUDED from the forecast.  Institutional
            # convention: alpha is not a persistent premium, and annualising it
            # geometrically explodes on short regression windows — after F-6
            # removed the bfill look-ahead, the common window honestly shrinks
            # to the youngest listing, and a leveraged single-stock ETF's bull
            # streak extrapolated to «ожид. дох. 218.4% · Sharpe 11.29» on the
            # live cover.  β are bounded by Ridge and μ_f is estimated on the
            # FULL factor history (factor ETFs always span the whole lookback),
            # so the β·μ expectation stays economically sane.  The realised
            # drift remains visible через Alpha_Specific below.
            factor_log_daily = df[beta_cols].fillna(0).values @ mu_vec
            expected_log_daily = factor_log_daily
            # F-23/P-1: leveraged daily-reset ETPs — lower the forward by the
            # CONTRACTUAL variance drag −½L(L−1)σ_u² + fee −ER/252 when the
            # multiplier L is in the registry (finance/leveraged.py), else by
            # the empirical min(α̂,0) fallback.  Ordinary names untouched.
            if 'Specific_Alpha_Daily' in df.columns:
                try:
                    # letf_sigma_map построена выше (≥60 obs floor); имена
                    # тоньше порога падают в α̂-фолбэк (α̂=0 → no-op,
                    # бит-в-бит пре-P-1 поведение).
                    expected_log_daily, _lev_details = apply_leveraged_forward(
                        expected_log_daily, list(df.index),
                        df['Specific_Alpha_Daily'], letf_sigma_map)
                    if _lev_details and isinstance(port_metrics, dict):
                        port_metrics["leveraged_adjustments"] = _lev_details
                        # Back-compat key (pre-P-1 consumers/tests).
                        port_metrics["leveraged_drag_tickers"] = \
                            list(_lev_details.keys())
                except Exception as _exc:
                    logger.warning("Leveraged-ETP drag skipped: %s", _exc)
            df['Expected_Return'] = np.exp(
                expected_log_daily * self.engine.trading_days) - 1.0
            # Belt-and-suspenders: clamp the per-asset FORWARD expectation to a
            # publishable band — a runaway β on a noisy short window must not
            # print a four-digit forecast.  Clamped names are counted for QC.
            _ER_LO, _ER_HI = -0.50, 1.00
            _clamped_n = int(((df['Expected_Return'] < _ER_LO)
                              | (df['Expected_Return'] > _ER_HI)).sum())
            if _clamped_n:
                logger.warning(
                    "Expected_Return clamped to [%.0f%%, %.0f%%] for %d asset(s).",
                    _ER_LO * 100, _ER_HI * 100, _clamped_n)
            df['Expected_Return'] = df['Expected_Return'].clip(_ER_LO, _ER_HI)
            if isinstance(port_metrics, dict):
                port_metrics["expected_return_clamped_n"] = _clamped_n
            # Specific Alpha: actual holding-period return minus the
            # FACTOR-IMPLIED return (in %) — i.e. the realised excess the
            # factor model does NOT explain.  This is exactly the textbook
            # «specific alpha» reading; the forecast column above stays clean.
            df['Alpha_Specific'] = (df['Return_Pct'] - df['Expected_Return']) * 100

        # ── BLOCK 5: portfolio-level FORWARD expected annual return ──────────
        # Aggregate the per-asset CAPM expectations into a single portfolio
        # number the UI can show against risk:
        #     E[r_port] = Σ_i w_i · E[r_i]   +   min(0, w_cash) · r_f
        # Cash rows sit in df with E[r] = 0: free broker cash earns nothing,
        # so its drag is priced in.  A MARGIN loan (w_cash < 0) is charged at
        # r_f — a lower bound on the broker's rate, never zero (`§−121`; the
        # old `max(0, 1 − w_risky)` residual made leverage free).  Names with
        # no finite E[r] still sit in the residual at r_f.  This is distinct
        # from `Annualised_Return` (the REALISED trailing CAGR) — it is the
        # forward expectation implied by today's factor betas + factor premia.
        # The result is fed back into the SAME port_metrics dict the structural
        # model returned, so every downstream surface reads one figure.
        if 'Expected_Return' in df.columns and isinstance(port_metrics, dict):
            try:
                # F-14: gate the ANNUALISED forward panel on the regression
                # window.  With the F-6 look-ahead fix the common window is the
                # honest intersection of listings — when it is shorter than one
                # trading year, an annualised forward claim off it is noise
                # (the live «218.4% · Sharpe 11.29» cover bug).  The per-asset
                # Expected_Return column stays (bounded β·μ), only the
                # portfolio-level panel degrades to «—».
                _nobs = int(getattr(self.engine, "_last_regression_nobs", 0) or 0)
                port_metrics["expected_return_window_days"] = _nobs
                if _nobs < self.engine.trading_days:
                    logger.warning(
                        "Forward expected-return panel skipped: regression "
                        "window %d < %d trading days.",
                        _nobs, self.engine.trading_days)
                    raise StopIteration          # → graceful skip below
                _exp = df['Expected_Return']
                _w_risky = 0.0
                _er_weighted = 0.0
                for _t in df.index:
                    _w = float(weights_dict.get(_t, 0.0) or 0.0)
                    _er = _exp.get(_t)
                    if _er is None or not np.isfinite(float(_er)):
                        continue
                    _w_risky += _w
                    _er_weighted += _w * float(_er)
                _rfr = float(port_metrics.get("risk_free_rate_annual") or 0.0)
                _cash_w = max(0.0, 1.0 - _w_risky)
                _margin_w = min(0.0, sum(
                    float(w or 0.0) for t, w in weights_dict.items()
                    if str(t).upper() in self.engine.NON_RISK_ASSETS))
                _port_exp = _er_weighted + _cash_w * _rfr + _margin_w * _rfr
                _vol = float(port_metrics.get("Total_Volatility_Ann") or 0.0)
                port_metrics["Expected_Return_Annual"] = _port_exp
                # Forward (ex-ante) Sharpe — expected excess return per unit of
                # structural vol.  Complements the realised Sharpe_Ratio above.
                port_metrics["Expected_Sharpe"] = (
                    (_port_exp - _rfr) / _vol if _vol > 0 else None)
                port_metrics["Expected_Return_Cash_Weight"] = round(_cash_w, 4)
                port_metrics["Expected_Return_Invested_Weight"] = round(_w_risky, 4)
            except StopIteration:
                pass                             # sub-year window — panel «—»
            except Exception as _exc:
                logger.warning("portfolio expected-return aggregation skipped: %s", _exc)

        return (actual_risky, cov_matrix, port_metrics, _dq_report, df,
                letf_sigma_map, beta_cols)


    def _stage_benchmarks(self, df, all_data, weights_dict, actual_risky,
                          port_metrics, total_portfolio_value, letf_sigma_map,
                          profile_benchmark):
        """Стадия 5: сравнение с бенчмарками, периоды и стресс-сценарии (Арх-3.5).

        Первая стадия, которая опирается на состояние, оставленное риск-моделью:
        `_last_ortho_betas` движка читается через `getattr` и уходит в стресс.
        Поэтому она вынесена РАНЬШЕ самой риск-модели (3.7) — если неявный
        порядок где-то нарушится, это всплывёт здесь, на компактном блоке, а не
        в сердце движка.

        Что внутри:

        * набор бенчмарков: профильный ставится ПЕРВЫМ. Порядок словаря тут
          содержателен, но НЕ по той причине, которую называл комментарий до
          §−64: Tracking Error считается ПО КАЖДОМУ бенчмарку отдельно
          (`_compute_benchmark_stats` на пару, ниже), поэтому перестановка не
          двигает ни одного числа. Порядок несёт ПОДПИСЬ: `pdf_payload` берёт
          `next(iter(period_returns_table))` и по нему решает, чьим именем
          назвать карточку «Рост против рынка». Собьётся порядок — числа
          останутся верными, а подпись станет чужой.
          🔴 Снимок этого НЕ стережёт: `golden_support.normalize` сортирует
          ключи словарей, то есть порядок для него невидим. Стережёт
          `BenchmarkOrderTest` — он и есть здесь опора, а не эталон;
        * портфельная лог-серия строится ОДИН раз и переиспользуется и в TE/IR,
          и в таблице периодов, поэтому окно у них общее (иначе IR считался бы
          в одном окне, а доходность — в другом);
        * `benchmark_factor_profile` (B1): реальные беты бенчмарка тем же
          Ridge-пайплайном, что и активы. `None` допустим — потребители обязаны
          это переживать;
        * стресс-сценарии: беты уже подогнаны, шок мапится в то же residual-
          пространство (F-1), плечевые ETP считаются path-dependent (P-7).

        Возвращает 6 значений; `port_resolved` нужен позже оверлею (3.8), а не
        только здесь — поэтому он в выходе, хотя вычислен «по дороге».
        """
        # ═══════════════ BENCHMARK COMPARISON ═══════════════
        # Профильный бенчмарк ставится ПЕРВЫМ, потому что первый ключ доезжает
        # до подписи карточки «Рост против рынка» (`pdf_payload` читает
        # `next(iter(period_returns_table))`).  Прежняя редакция этой строки
        # объясняла порядок расчётом Tracking Error — это неверно с F-15:
        # TE/IR считаются ПО КАЖДОМУ бенчмарку на своём pair inner-join (ниже),
        # и порядок на них не влияет вовсе.  §−64.
        benchmarks: dict[str, str] = {}
        if profile_benchmark:
            benchmarks["Профильный бенчмарк"] = profile_benchmark

        # Бенчмарки в формате Tradernet (.US ETF-прокси для индексов).
        # EEM.US и EMB.US уже запрошены как EM-факторы → данные гарантированно есть.
        benchmarks.update({
            'S&P 500':      'SPY.US',   # ETF-прокси на ^GSPC
            'Nasdaq 100':   'QQQ.US',   # ETF-прокси на ^NDX
            'Russell 2000': 'IWM.US',   # ETF-прокси на ^RUT
            'MSCI EM':      'EEM.US',   # Emerging Markets equity index — EM/KASE сравнение
            'EM Bonds':     'EMB.US',   # JP Morgan EM Bond index — для KZ bond holders
        })
        # ── Weighted portfolio log-return series — built ONCE, robustly ────
        # Sparse-history constituents (thinly-traded local listings) are
        # dropped from the panel and the surviving weights renormalised, so
        # one bad name can no longer collapse the overlap window for the
        # whole book.  See period_returns.build_portfolio_log_returns.
        port_resolved = self.engine.resolve_tickers(actual_risky)
        col_weights: dict[str, float] = {}
        for orig, res in zip(actual_risky, port_resolved):
            if res in all_data.columns:
                col_weights[res] = col_weights.get(res, 0.0) + float(weights_dict.get(orig, 0.0))
        port_log_series, return_coverage = _build_portfolio_log_returns(
            all_data[list(col_weights)] if col_weights else None, col_weights
        )
        if return_coverage["dropped"]:
            logger.info("Return series: dropped %d sparse-history name(s): %s "
                        "(covered weight %.1f%%)",
                        len(return_coverage["dropped"]),
                        ", ".join(return_coverage["dropped"]),
                        return_coverage["covered_weight"] * 100)
        # H-3 / D-5 (`§−128`): ЧИСЛА сравнения с рынком (TE/IR, таблица
        # периодов) — из ТОГО ЖЕ ряда книги, что и обложка
        # (`_last_port_log_returns`: кэш 0%, маржа по rf, композит F-22).
        # Ряд выше перенормирует веса на 100% инвестированной книги: кэш не
        # разбавлял, плечо не увеличивало — эталон `base` давал 12М +36.4% в
        # таблице против +30.7% по ряду обложки, `leveraged_fx` +33.8% против
        # +36.7%. «Доходность за 12 мес» на обложке и строка 12М таблицы
        # обязаны быть одним числом. `return_coverage` остаётся от ряда выше:
        # это телеметрия выпавших имён, а не база чисел.
        _book_series = getattr(self.engine, "_last_port_log_returns", None)
        if isinstance(_book_series, pd.Series) and len(_book_series.dropna()) >= 2:
            port_log_series = _book_series.dropna()

        # Benchmark log-returns — one Series per benchmark (per-pair join).
        bm_logs: dict[str, pd.Series] = {}
        for bm_name, bm_ticker in benchmarks.items():
            if bm_ticker in all_data.columns:
                bm_prices = all_data[bm_ticker].dropna()
                if len(bm_prices) >= 2:
                    bm_logs[bm_name] = np.log(bm_prices / bm_prices.shift(1)).dropna()

        # ═══════════════ BENCHMARK FACTOR PROFILE (B1 2026-07-17) ═══════════════
        # Реальные факторные беты выбранного бенчмарка (тот же Ridge-пайплайн,
        # что и активы) — столбец «бенчмарк» в секции «Факторное разложение»
        # больше не константа (Market=1, rest=0)=S&P 500.  Для SPY.US беты
        # ≈ (Market≈1, стили≈0) → отчёт обратносовместим по построению.
        # None (нет бенчмарка / нет истории) — потребители обязаны переживать.
        benchmark_factor_profile: dict | None = None
        if profile_benchmark:
            _bm_betas = self.engine.fit_factor_betas(all_data, profile_benchmark)
            if _bm_betas:
                try:
                    from profile_manager import BENCHMARK_LIST as _BM_NAMES
                except Exception:                     # pragma: no cover
                    _BM_NAMES = {}
                benchmark_factor_profile = {
                    "ticker": profile_benchmark,
                    "name":   _BM_NAMES.get(profile_benchmark, profile_benchmark),
                    "betas":  _bm_betas,
                }
            else:
                logger.info(
                    "Benchmark factor profile: беты для %s недоступны — секция "
                    "факторов использует S&P-константу (graceful fallback).",
                    profile_benchmark)

        # ═══════════════ BENCHMARK COMPARISON ═══════════════
        # TE / IR / annualised excess are computed per benchmark on a PAIR
        # inner-join (compute_benchmark_stats) — a benchmark's own short
        # history can no longer null the comparison, and numerator and
        # denominator share one aligned window (no IR scale bug).
        benchmark_results: dict[str, dict] = {}
        port_return = float(df['Return_Pct'].fillna(0).values @ np.array(
            [weights_dict.get(t, 0) for t in df.index]))
        for bm_name, bm_ticker in benchmarks.items():
            if bm_ticker not in all_data.columns:
                continue
            bm_prices = all_data[bm_ticker].dropna()
            if len(bm_prices) <= 1:
                continue
            bm_return = float(bm_prices.iloc[-1] / bm_prices.iloc[0]) - 1.0
            stats = _compute_benchmark_stats(port_log_series, bm_logs.get(bm_name),
                                             trading_days=self.engine.trading_days)
            benchmark_results[bm_name] = {
                "Benchmark_Return":     bm_return,                  # period total (display)
                "Portfolio_Return":     port_return,                # period total (display)
                "Excess_Return":        port_return - bm_return,    # period total (display)
                "Portfolio_Ann_Return": stats["port_ann_return"]   if stats else None,
                "Benchmark_Ann_Return": stats["bm_ann_return"]     if stats else None,
                "Excess_Return_Ann":    stats["excess_ann"]        if stats else None,
                "Tracking_Error":       stats["tracking_error"]    if stats else None,
                "Information_Ratio":    stats["information_ratio"] if stats else None,
                "Beating_Benchmark":    port_return > bm_return,
            }

        # ═══════════════ MULTI-PERIOD RETURNS TABLE ═══════════════
        # Reuses the SAME robust port_log_series + per-benchmark bm_logs, so
        # every period row shares one aligned window with the TE/IR figures.
        period_returns_table: dict[str, dict] = {}
        try:
            if port_log_series is not None and bm_logs:
                period_returns_table = _compute_period_returns_table(
                    port_log_series, bm_logs
                )
        except Exception as exc:  # never block the rest of the pipeline
            logger.warning("period_returns_table build failed: %s", exc)
            period_returns_table = {}

        # ═══════════════ STRESS SCENARIOS ═══════════════
        # Parametric factor-shock scenarios using the betas just fitted.  The
        # default catalog ships 7 scenarios (5 direct, 2 proxy).  Result is a
        # list[dict] — always present, possibly empty when perf_df is too
        # thin to support any scenario.
        try:
            # Recovery rate = portfolio's own annualised return, clamped to a
            # realistic band [8%, 18%].  A growth book recovers faster than
            # the generic 8% market drift, but we cap at 18% so an
            # unsustainable trailing CAGR (e.g. +36%) cannot fabricate an
            # absurdly fast recovery.  Floor 8% = long-run market average.
            _ann_ret = float(port_metrics.get("Annualised_Return") or 0.0)
            _recovery_rate = min(0.18, max(0.08, _ann_ret))
            stress_scenarios = _run_stress_scenarios(
                perf_df      = df.reset_index(),
                total_value  = total_portfolio_value,
                port_metrics = port_metrics,
                ann_return_baseline = _recovery_rate,
                # F-1: the Beta_* columns in perf_df live in the RESIDUAL factor
                # space when orthogonalization ran; hand the fitted child→parent
                # betas to the stress engine so it maps the raw shock catalog
                # into the same space (invariant to FACTOR_ORTHOGONALIZE).
                ortho_betas  = getattr(self.engine, "_last_ortho_betas", {}) or {},
                # P-7: реестровые daily-reset ETP считаются path-dependent —
                # (1+X_u)^L·exp(−½L(L−1)σ_u²·63)−1 вместо линейного β·shock
                # с капом ±35% (кап срезал КОНТРАКТНУЮ выпуклость плеча).
                letf_sigma_daily = letf_sigma_map,
            )
        except Exception as exc:
            logger.warning("stress scenarios build failed: %s", exc)
            stress_scenarios = []

        return (port_resolved, benchmark_factor_profile, benchmark_results,
                period_returns_table, return_coverage, stress_scenarios)

    def _stage_scoring(self, df, weights_dict, port_metrics, beta_cols,
                       all_data, actual_risky):
        """Стадия 6: секторы, аггравторы гейджа, внешние сигналы, 4-Pillar (Арх-3.6).

        Стадия внешних сервисов: макро (FRED), режим, техникалс, CDS, SEC —
        каждый под своим `try`, потому что отчёт обязан строиться и без них.
        Отсюда её главное свойство для разреза: **большая часть веток здесь —
        аварийные**, их не исполняет ни один тест (покрытие 73.9% / 75.0%,
        §0c). Поэтому доказательством служит дословность переноса, а не снимок:
        перенос без изменения текста не может сломать ветку, которую никто не
        исполняет.

        🔴 У стадии есть НЕВИДИМЫЙ выход: она дописывает в `port_metrics`
        ключи `Composite_Risk_Score` и `composite_aggravators` — то есть
        мутирует словарь, пришедший из риск-модели, ПО ССЫЛКЕ. В возвращаемом
        кортеже этого не видно, и убирать мутацию сейчас нельзя: гейдж обязан
        считаться на тех же mandate+аггравторах, что и обложка с `simulate`
        (инвариант `finance/CLAUDE.md`), а любая правка поведения на шаге
        разреза запрещена. Кандидат в отдельную задачу после 3.11.

        Возвращает 9 значений — в порядке их вычисления, а не «по важности»:
        так соответствие блоку читается глазами.
        """
        # Sector exposure analysis
        sector_exposure = self.engine.get_sector_exposure(list(df.index), weights_dict)

        # ── Composite-risk aggravators (2026-07-18) ────────────────────────────
        # Re-score the composite gauge with the STRUCTURAL/TAIL signals the base
        # three (vol/CVaR/single-name-ERC) miss — leverage, single-sector
        # concentration and the realised max drawdown.  Live audit: a 73%-tech,
        # margin-funded book with a −43.5% historical drawdown read «48 ·
        # Умеренный».  The aggravators are bounded and can only RAISE the gauge;
        # a clean, diversified, unlevered book is unchanged.  Computed HERE
        # (sector_exposure + weights available) and passed to BOTH the verdict
        # gauge and the effect simulator so «до/после» stays consistent.
        _agg_lev_ratio = None
        try:
            _lw = sum(max(0.0, float(w)) for w in weights_dict.values())
            _nw = sum(float(w) for w in weights_dict.values())
            _agg_lev_ratio = round(_lw / _nw, 4) if _nw > 1e-9 else 1.0
        except Exception:
            _agg_lev_ratio = None
        _agg_sector_top_pct = None
        try:
            # Use the SAME super-group concentration the report headlines
            # («Tech-комплекс 73%»), not the biggest SINGLE sector (59%) — the
            # correlated Technology+Semiconductors block is the real
            # concentration the gauge must reflect (SSOT: asset_taxonomy).
            from finance.asset_taxonomy import top_sector_concentration_pct
            _top = top_sector_concentration_pct(sector_exposure or {})
            _agg_sector_top_pct = _top if _top > 0 else None
        except Exception:
            _agg_sector_top_pct = None
        try:
            _base_vol = float(port_metrics.get("Total_Volatility_Ann") or 0.0)
            _base_cvar = float(port_metrics.get("CVaR_95_Daily") or 0.0)
            _base_erc = float(port_metrics.get("Max_Euler_Risk_Pct") or 0.0)
            _base_mdd = port_metrics.get("Max_Drawdown")
            _mandate_str = str(getattr(self.engine, "risk_mandate", "MODERATE"))
            port_metrics["Composite_Risk_Score"] = self.engine._composite_risk_score(
                _base_vol, _base_cvar, _base_erc, _mandate_str,
                max_drawdown=(float(_base_mdd) if _base_mdd is not None else None),
                leverage_ratio=_agg_lev_ratio,
                sector_top_pct=_agg_sector_top_pct)
            # Surface the aggravators for the QC/AI layer (why the gauge is
            # higher than the raw vol/CVaR would suggest).
            port_metrics["composite_aggravators"] = {
                "max_drawdown":   (float(_base_mdd) if _base_mdd is not None else None),
                "leverage_ratio": _agg_lev_ratio,
                "sector_top_pct": _agg_sector_top_pct,
            }
        except Exception as _exc:
            logger.warning("composite aggravators skipped: %s", _exc)

        # ═══════════════ FACTOR SCORES GROUPING ═══════════════
        # Group A (Style Factors): MAC3-derived betas, alpha, volatility
        # Group B (Fundamental Factors): SEC EDGAR metrics
        factor_scores = {}
        perf = df.reset_index()
        for _, row in perf.iterrows():
            ticker = row.get('Ticker', '?')
            group_a = {}
            group_b = {}
            # Style factors (Group A)
            for col in beta_cols:
                if col in row.index:
                    group_a[col] = row[col]
            for col in ['Residual_Vol_Ann', 'Euler_Risk_Contribution_Pct',
                        'Expected_Return', 'Alpha_Specific', 'Specific_Alpha_Daily']:
                if col in row.index:
                    group_a[col] = row[col]
            # Fundamental factors (Group B)
            for col in ['Fundamental_Score', 'Fundamental_Sector',
                        'SEC_Op_Margin', 'SEC_Debt_to_Assets',
                        'SEC_ROE', 'SEC_Revenue_Growth_YoY', 'SEC_Filing_Date']:
                if col in row.index:
                    group_b[col] = row[col]
            factor_scores[ticker] = {
                'group_a_style': group_a,
                'group_b_fundamental': group_b,
            }

        # ═══════════════ MACRO DRIVERS (FRED) ═══════════════
        # 6-series pack (yield curve / HY OAS / VIX / breakeven / unemployment /
        # real-GDP growth) used by the DEEP P5 regime page.  Disk-cached for
        # 12h; gracefully degrades to status="missing" when FRED_API_KEY is
        # unset and to status="stale" on transient network failures.  Fetched
        # BEFORE classification so the regime can (optionally) consume it.
        macro_drivers: dict = {}
        try:
            macro_drivers = _MacroFeed().get_regime_drivers()
        except Exception as exc:
            logger.warning("Macro drivers fetch skipped: %s", exc)
            macro_drivers = {}

        # ═══════════════ MACRO REGIME CLASSIFICATION ═══════════════
        # Reuses the factor ETF prices already in `all_data` — no extra calls.
        # BLOCK 3.4: the FRED macro pack is passed in; it is only consulted when
        # REGIME_MACRO_OVERLAY=1 (otherwise the classifier is unchanged).
        try:
            from finance.regime import RegimeClassifier
            regime_reading = RegimeClassifier().classify(all_data, macro=macro_drivers)
        except Exception as exc:
            logger.warning("Regime classification skipped: %s", exc)
            regime_reading = None

        # ═══════════════ TECHNICALS (pillar C) ═══════════════
        # Computes RSI/MACD/SMA/Bollinger/52w-Hi/Mom-12m1m per-asset.
        # Uses the same close-price frame already loaded; no new fetches.
        technicals_map: dict = {}
        try:
            from finance.technicals import compute_technicals
            tech_sector_map = {
                t: self.engine.get_ticker_sector(t) for t in actual_risky
            }
            technicals_map = compute_technicals(
                close_prices = all_data,
                tickers      = actual_risky,
                volume_frame = None,            # OHLC volumes wired in a later phase
                sector_map   = tech_sector_map,
            )
        except Exception as exc:
            logger.warning("Technicals computation skipped: %s", exc)
            technicals_map = {}

        # ═══════════════ CDS FEED (free layer) ═══════════════
        # Lazy-init.  When CDS_DISABLED=1 (e.g. unit tests) we skip the feed
        # entirely.  Failures are caught and the scoring orchestrator falls
        # back to SEC-only Credit signals.
        cds_lookup = None
        if os.getenv("CDS_DISABLED") != "1":
            try:
                from finance.cds_feed import CDSFeed, make_lookup
                cds_lookup = make_lookup(CDSFeed())
            except Exception as exc:
                logger.info("CDS feed unavailable, continuing without it: %s", exc)
                cds_lookup = None

        # Tally CDS coverage so the CoVe lineage row reflects reality
        # (instead of the silent "no per-ticker CDS attached" placeholder).
        # Single extra pass over the cache — microseconds for ≤20 tickers.
        # C-1 (`§−129`): `loaded` — только чтения, ВОШЕДШИЕ в C-пиллар
        # эмитента; рыночный индекс HY считается отдельно (`market_only`).
        # Прежде любой непустой ответ шёл в `loaded`, и строка CoVe писала
        # «ok N/N», когда у всех бумаг было одно рыночное число.
        cds_summary: dict = {"enabled": cds_lookup is not None,
                              "checked": 0, "loaded": 0, "market_only": 0,
                              "gated_out": 0}
        if cds_lookup is not None:
            try:
                from finance.cds_feed import enters_issuer_score
                checked = list(actual_risky)
                readings = [cds_lookup(t) for t in checked]
                n_loaded = sum(1 for r in readings if enters_issuer_score(r))
                n_market = sum(1 for r in readings
                               if r and r.get("bps") is not None
                               and not enters_issuer_score(r))
                cds_summary.update({
                    "checked":     len(checked),
                    "loaded":      n_loaded,
                    "market_only": n_market,
                    "gated_out":   len(checked) - n_loaded - n_market,
                })
            except Exception as exc:
                logger.info("CDS coverage tally skipped: %s", exc)

        # ═══════════════ 4-PILLAR SCORING (F/V/T/C) ═══════════════
        # Produces AssetScore per ticker, including CDS signals when available.
        asset_scores: dict = {}
        try:
            from finance.scoring_orchestrator import score_portfolio
            asset_scores = score_portfolio(
                perf_table = perf,
                technicals = technicals_map,
                regime     = regime_reading,
                cds_lookup = cds_lookup,
                # Sprint-5.1 (S2): in production, small-sector F-pillar
                # z-scores use the LIVE SEC cohort of sector leaders before
                # falling back to the static 2020-25 constants.
                dynamic_benchmarks = os.getenv("SECTOR_COHORT_DISABLED") != "1",
            )
            # Materialise the score columns into the perf table for downstream
            # PDF rendering — this is purely additive, no existing column is
            # overwritten.
            if asset_scores:
                score_rows = []
                for t in perf["Ticker"].tolist():
                    sc = asset_scores.get(str(t))
                    if sc is None:
                        score_rows.append({
                            "Ticker": t,
                            "Score_Fundamentals": None,
                            "Score_Valuations":   None,
                            "Score_Technicals":   None,
                            "Score_Credit":       None,
                            "Score_Total":        None,
                            "Score_Action":       None,
                            "Score_Hotspot":      False,
                        })
                    else:
                        score_rows.append({
                            "Ticker": t,
                            "Score_Fundamentals": sc.fundamentals,
                            "Score_Valuations":   sc.valuations,
                            "Score_Technicals":   sc.technicals,
                            "Score_Credit":       sc.credit,
                            "Score_Total":        sc.total,
                            "Score_Action":       sc.action,
                            "Score_Hotspot":      sc.hotspot,
                        })
                score_df = pd.DataFrame(score_rows).set_index("Ticker")
                # Join back onto perf without losing existing columns.
                perf = perf.set_index("Ticker").join(score_df, how="left").reset_index()
        except Exception as exc:
            logger.warning("Scoring orchestrator skipped: %s", exc)

        return (sector_exposure, _agg_lev_ratio, factor_scores, perf,
                macro_drivers, regime_reading, technicals_map, cds_summary,
                asset_scores)

    def _stage_overlay(self, df, all_data, weights_dict, actual_risky,
                       port_resolved, broker_priced_only, cov_matrix,
                       port_metrics, perf, asset_scores, technicals_map,
                       total_portfolio_value, _agg_lev_ratio, _fx_rows):
        """Стадия 7: наложения поверх посчитанного портфеля (Арх-3.8).

        Последняя стадия конвейера и самая связная — она читает почти всё, что
        произвели предыдущие шесть. План оценивал интерфейс в 20/8; замер дал
        **13 входов и 5 выходов**: часть «живых» переменных, как и на 3.5/3.7,
        оказалась целиком внутренней.

        Порядок ТРЁХ из четырёх блоков содержателен, и это не стилистика —
        каждый следующий питается предыдущим (Smart Money стоит особняком:
        он ничего не читает у соседей и ничего им не даёт):

            Black-Litterman ─→ Action Plan ─→ Expected Effect
                    │                │              ↑
                    └────────────────┴──────────────┘
              bl_records кормит и план, и симуляцию; action_plan_rows — симуляцию

        * **Black-Litterman**: reverse-optimisation prior + views из оценок.
          Мандат КОНСТРАНИТ оптимизатор (δ, turnover, cap на имя), поэтому
          консервативная книга получает меньшие отклонения, а не «один размер
          на всех»;
        * **Action Plan**: зоны покупки/продажи от ATR+SMA+RSI. Считается ДО
          симуляции (BLOCK 2.3), чтобы панель эффекта двигали именно
          high-priority идеи, а не полный BL-таргет — иначе «Идеи → План →
          Эффект» перестают быть одной историей;
        * **Smart Money**: слой инсайдеров заполняется ВСЕГДА, даже когда
          провайдер выключен, — иначе секция просто исчезнет из отчёта вместо
          честного «слой готов, источник не подключён»;
        * **Expected Effect**: пересчёт обложечных метрик под план. Реинвест
          ДЕКОНЦЕНТРИРУЕТ (иначе освободившийся вес возвращался в тот же
          перегретый сектор), а блок-лист реинвеста закрывает имена, которые
          модель не может симулировать честно (L-13).

        Каждый блок под своим `try`: отчёт обязан строиться, даже если
        оптимизатор или внешний слой отказали. Все пять выходов инициализируются
        ДО своего `try`, поэтому возврат безопасен при любом отказе.

        Неявный канал ровно один — `_last_sparse_dropped` движка, и читается он
        ДВАЖДЫ: в Action Plan (F-20, чтобы строка объяснила, почему у неё нет
        беты) и в блок-листе реинвеста (L-13, где к нему добавляется
        `broker_priced_only`). Два почти одинаковых вычисления одного множества
        — кандидат на объединение, но НЕ на шаге разреза: фаза поведенчески
        пустая, а у слияния свой замер и свой тест (`§−66`).

        Возвращает 5 значений в порядке вычисления.
        """
        # ═══════════════ BLACK-LITTERMAN TARGETS ═══════════════
        # Reverse-optimisation prior + score-derived views → posterior
        # target weights.  Skipped when there's no covariance matrix
        # (degenerate portfolio) or when all scores are zero.
        bl_records: list[dict] | None = None
        try:
            from finance.black_litterman import black_litterman, views_from_scores
            if not cov_matrix.empty and asset_scores:
                bl_tickers = [t for t in cov_matrix.index]
                if bl_tickers:
                    P, Q, conf = views_from_scores(
                        {t: {"total": getattr(s, "total", None)}
                         for t, s in asset_scores.items()},
                        bl_tickers,
                    )
                    # Sprint-5 Task 4 — the investor mandate now CONSTRAINS the
                    # optimiser (was fully mandate-agnostic): a Conservative
                    # book gets a higher risk-aversion δ (smaller tilts), a
                    # tighter turnover cap and a tighter single-name cap than an
                    # Aggressive one, so BL targets respect the approved mandate
                    # instead of being a one-size-fits-all output.
                    _mandate = str(getattr(self.engine, "risk_mandate", "MODERATE")).upper()
                    _bl_cfg = _MANDATE_BL_CONSTRAINTS.get(
                        _mandate, _MANDATE_BL_CONSTRAINTS["MODERATE"])
                    bl_res = black_litterman(
                        cov              = cov_matrix,
                        tickers          = bl_tickers,
                        current_weights  = {t: weights_dict.get(t, 0) for t in bl_tickers},
                        views_P          = P if P.size else None,
                        views_Q          = Q if Q.size else None,
                        view_confidence  = conf if conf.size else None,
                        risk_aversion     = _bl_cfg["risk_aversion"],
                        max_active_share  = _bl_cfg["max_active_share"],
                        max_single_weight = _bl_cfg["max_single_weight"],
                    )
                    bl_records = bl_res.as_records()
        except Exception as exc:
            logger.warning("Black-Litterman skipped: %s", exc)
            bl_records = None

        # ═══════════════ ACTION PLAN ═══════════════
        # Buy zone / Sell target / Stop loss anchored to ATR + SMA + RSI.
        # BLOCK 2.3: computed BEFORE the Expected-Effect simulation so the
        # panel can be driven by the HIGH-PRIORITY (non-deferred) action items
        # rather than the full BL target — keeping Идеи → Action Plan →
        # Ожидаемый эффект one consistent story.
        action_plan_rows: list[dict] = []
        # Доля книги вне факторной модели — дефолт на случай сбоя блока ниже.
        _model_uncovered = {"weight_pct": 0.0, "names": []}
        try:
            from finance.action_plan import build_action_plan
            ap_scores = {
                t: {"action":  s.action,
                    "total":   s.total,
                    "hotspot": s.hotspot}
                for t, s in asset_scores.items()
            }
            # F-20: originals of the names the sparse-guard excluded from the
            # structural model (resolved → original via base-symbol match), so
            # the plan can say WHY a row has no beta/target/quantity.
            _sparse_res = set(getattr(self.engine, "_last_sparse_dropped", []) or [])
            _uncovered = {
                orig for orig, res in zip(actual_risky, port_resolved)
                if res in _sparse_res
                or str(res).split(".")[0] in {str(s).split(".")[0]
                                              for s in _sparse_res}
            } if _sparse_res else set()
            rows = build_action_plan(
                perf_table      = perf,
                asset_scores    = ap_scores,
                technicals_map  = technicals_map,
                bl_records      = bl_records,
                portfolio_value = total_portfolio_value,
                # Sprint-5.1 (A3): stop/take distances scale with the mandate
                # (Conservative tighter, Aggressive wider) — levels are no
                # longer one-size-fits-all.
                risk_mandate    = self.engine.risk_mandate,
                uncovered       = _uncovered,
                # A-2: курсы ФАКТИЧЕСКИ выполненных конверсий — по ним план
                # повторит уровни в валюте торговли. Реестр, а не свежий запрос:
                # уровни обязаны сходиться с ценой в той же карточке.
                fx_rates        = {r["currency"]: r["rate"] for r in (_fx_rows or [])},
            )
            action_plan_rows = [r.as_dict() for r in rows]
            # 2026-08-03: доля книги ВНЕ факторной модели. Отчёт печатал
            # «покрытие 100%» — это доля загруженных фактор-СЕРИЙ, а позиция
            # весом 5.3% бет не имела вовсе; читатель понимал строку как
            # «портфель покрыт полностью». Считаем здесь, где известны и
            # исключения, и веса.
            _unc_w = sum(float(weights_dict.get(t, 0.0) or 0.0) for t in _uncovered)
            _model_uncovered = {
                "weight_pct": round(_unc_w * 100.0, 2),
                "names": sorted(str(t) for t in _uncovered),
            }
        except Exception as exc:
            logger.warning("Action plan skipped: %s", exc)
            action_plan_rows = []

        # ═══════════════ SMART MONEY (insider SEC Form-4) ═══════════════
        # BLOCK 3.5 / B2.4 — always populate results["smart_money"] so the DEEP
        # report can RENDER the insider layer.  Its status is visible even when
        # gated OFF: the panel then shows "слой готов — источник Form-4 не
        # подключён" instead of the section being absent.  No live EDGAR crawl
        # here (gated provider); default per-ticker state is "disabled".
        smart_money: dict = {}
        try:
            from finance.smart_money import build_insider_signals
            sm_tickers = [str(t) for t in df.index][:25]
            smart_money = build_insider_signals(sm_tickers)
        except Exception as exc:
            logger.warning("Smart Money skipped: %s", exc)
            smart_money = {}

        # ═══════════════ EXPECTED EFFECT (after-plan simulation) ═══════════
        # Re-evaluates cover-page metrics under the HIGH-PRIORITY action items
        # (BLOCK 2.3).  Falls back to the BL target — and then to current
        # weights — when no actionable rows exist; the "after" then equals
        # "before" and every delta is 0, which the report can detect.
        expected_effect: dict | None = None
        try:
            # BLOCK 2.3: the panel simulates ONLY the high-priority moves that
            # survived the turnover cap (Buy/Sell/Trim with |Δw|>0).  Deferred
            # and Hold rows leave their weight unchanged; falls back to the BL
            # target when there are no actionable rows.
            from finance.simulate import (high_priority_target_weights,
                                          external_diversifier_candidates)
            sector_map_for_sim = {t: self.engine.get_ticker_sector(t)
                                   for t in df.index}
            # L-13 (2026-07-19): reinvest-eligibility blocklist — the names the
            # structural model can't честно simulate (sparse-history exclusions
            # + broker-priced-only proxies, e.g. the FFSPC AIX note that the
            # live report «bought» +12пп into becoming the new Max-TRC 36%).
            # Leveraged ETPs are filtered inside simulate via the registry.
            _reinvest_block: set = set(broker_priced_only)
            _sparse_res = set(getattr(self.engine, "_last_sparse_dropped", []) or [])
            if _sparse_res:
                _sparse_bases = {str(s).split(".")[0] for s in _sparse_res}
                _reinvest_block |= {
                    orig for orig, res in zip(actual_risky, port_resolved)
                    if res in _sparse_res
                    or str(res).split(".")[0] in _sparse_bases}
            # 2026-07-18: pass the sector map so the reinvest de-CONCENTRATES —
            # freed weight from selling the over-weight tech must NOT be poured
            # back into more tech (the old reinvest bought the highest-BL held
            # names, which ARE the concentrated mega-caps → IT share and risk
            # went UP, contradicting the mandate).  Diversifiers first, else cash.
            # L-18 (2026-07-19): external global-ETF sleeve — кандидаты только
            # из ФАКТОРНОЙ панели (история уже скачана этим же прогоном),
            # с реальным покрытием ≥60 торговых дней и НЕ из держимых имён.
            # Порядок кандидатов задаёт мандат (см. simulate.EXTERNAL_DIVERSIFIERS).
            _held_names = {str(t) for t in df.index}
            _ext_cands = [
                c for c in external_diversifier_candidates(
                    str(getattr(self.engine, "risk_mandate", "MODERATE")))
                if c.get("panel") in all_data.columns
                and int(all_data[c["panel"]].notna().sum()) >= 60
                and str(c.get("ticker")) not in _held_names
            ]
            target_weights, hp_tickers, hp_actions = high_priority_target_weights(
                weights_dict, action_plan_rows, bl_records,
                sector_by_ticker=sector_map_for_sim,
                reinvest_blocklist=_reinvest_block,
                external_candidates=_ext_cands)

            # Build daily log-returns matrix for assets the cov matrix knows
            # about.  Reuses all_data (already loaded) so no extra fetch.
            sim_daily_log: pd.DataFrame | None = None
            if not cov_matrix.empty:
                cov_tickers = list(cov_matrix.index)
                resolved    = self.engine.resolve_tickers(cov_tickers)
                avail_cols  = [r for r in resolved if r in all_data.columns]
                if avail_cols:
                    sub = all_data[avail_cols].dropna()
                    if len(sub) >= 2:
                        log_df = np.log(sub / sub.shift(1)).dropna()
                        # Map columns back to ORIGINAL (display) tickers.
                        col_map = {res: orig for orig, res in zip(cov_tickers, resolved)
                                   if res in avail_cols}
                        log_df = log_df.rename(columns=col_map)
                        sim_daily_log = log_df

            # L-18: колонки внешних ETF-покупок — в sim-панель (под display-
            # тикером), чтобы simulate_after_plan расширил ковариацию и
            # sample-метрики реальными данными, а не считал покупку кэшем.
            _ext_bought = [a for a in hp_actions
                           if a.get("is_external") and a.get("ticker")]
            if _ext_bought:
                _panel_by_tkr = {str(c["ticker"]): c["panel"] for c in _ext_cands}
                _ext_cols = {}
                for a in _ext_bought:
                    _t = str(a["ticker"])
                    _pcol = _panel_by_tkr.get(_t)
                    if _pcol and _pcol in all_data.columns:
                        _ser = all_data[_pcol].dropna()
                        if len(_ser) >= 2:
                            _ext_cols[_t] = np.log(_ser / _ser.shift(1)).dropna()
                if _ext_cols:
                    _ext_df = pd.DataFrame(_ext_cols)
                    sim_daily_log = (_ext_df if sim_daily_log is None
                                     else sim_daily_log.join(_ext_df, how="left"))

            expected_effect = _simulate_after_plan(
                perf_df           = df.reset_index(),
                risk_matrix       = cov_matrix,
                daily_log_returns = sim_daily_log,
                bl_records        = bl_records,
                current_metrics   = port_metrics,
                risk_free_rate    = self.engine.current_rfr_annual,
                target_weights    = target_weights,
                sector_by_ticker  = sector_map_for_sim,
                # 2026-07-18: same mandate + leverage the verdict gauge uses so
                # the effect «до/после» index mirrors the cover gauge (both now
                # carry the structural/tail aggravators).
                mandate           = str(getattr(self.engine, "risk_mandate", "MODERATE")),
                leverage_ratio    = _agg_lev_ratio,
            )
            # Tag which tickers drove the panel so the UI can show the
            # before/after delta is scoped to the high-priority ideas.
            if isinstance(expected_effect, dict):
                expected_effect["high_priority_tickers"] = hp_tickers
                expected_effect["high_priority_actions"] = hp_actions
                expected_effect["driver"] = ("high_priority_action_plan"
                                             if hp_tickers else "bl_target_fallback")
        except Exception as exc:
            logger.warning("Expected-effect simulator skipped: %s", exc)
            expected_effect = None

        return (bl_records, action_plan_rows, _model_uncovered, smart_money,
                expected_effect)


    def analyze_all(self, source, scenario_shocks=None, profile_benchmark: str | None = None,
                    risk_mandate: str | None = None) -> AnalyzeResults:
        """
        profile_benchmark: Tradernet ETF ticker for the user's mandate benchmark
        (e.g. 'AGG.US' for Conservative, 'SPY.US' for Moderate, 'QQQ.US' for Aggressive).
        When provided it is computed first and labelled 'Профильный бенчмарк' in results.

        risk_mandate: CONSERVATIVE / MODERATE / AGGRESSIVE (or an RU/EN profile
        name / numeric score) — drives the composite-risk-score calibration
        (H4).  When None, the engine keeps its current mandate (MODERATE
        default).
        """
        if risk_mandate is not None:
            from finance.scoring import normalize_risk_mandate as _nrm
            self.engine.risk_mandate = _nrm(risk_mandate)
        (df, _merged_rows, _dropped_rows,
         _portfolio_source) = self._stage_normalize(source)

        (df, all_data, history_result, broker_priced_only, priced_at_cost,
         _rep_ccy, _fx_rows, _fx_unconvertible, total_portfolio_value,
         weights_dict) = self._stage_price_and_fx(df)

        _dq_report = self._stage_data_quality(
            df, weights_dict, all_data, _fx_unconvertible)

        (actual_risky, cov_matrix, port_metrics, _dq_report, df,
         letf_sigma_map, beta_cols) = self._stage_risk_model(
            df, all_data, weights_dict, broker_priced_only,
            history_result, _dq_report)

        (port_resolved, benchmark_factor_profile, benchmark_results,
         period_returns_table, return_coverage, stress_scenarios) = (
            self._stage_benchmarks(df, all_data, weights_dict, actual_risky,
                                   port_metrics, total_portfolio_value,
                                   letf_sigma_map, profile_benchmark))

        (sector_exposure, _agg_lev_ratio, factor_scores, perf,
         macro_drivers, regime_reading, technicals_map, cds_summary,
         asset_scores) = self._stage_scoring(df, weights_dict, port_metrics,
                                            beta_cols, all_data, actual_risky)

        (bl_records, action_plan_rows, _model_uncovered, smart_money,
         expected_effect) = self._stage_overlay(
            df, all_data, weights_dict, actual_risky, port_resolved,
            broker_priced_only, cov_matrix, port_metrics, perf, asset_scores,
            technicals_map, total_portfolio_value, _agg_lev_ratio, _fx_rows)


        # ── Арх-3.1: состояние прогона собрано в один контекст ─────────────
        # Введён на 3.1 ПАРАЛЛЕЛЬНО прежним локальным переменным — так
        # достаточность контекста доказывалась ДО того, как резать функцию.
        # После 3.8 это уже не дубль: локальные выше — ровно выходы семи
        # стадий, и контекст стал единственным, что переходит от слоя стадий
        # к сборке (`ARCHITECTURE_FOR_AGENTS.md §4 Арх-3`).
        ctx = _AnalysisCtx(
            _dq_report=_dq_report,
            _dropped_rows=_dropped_rows,
            _fx_rows=_fx_rows,
            _merged_rows=_merged_rows,
            _portfolio_source=_portfolio_source,
            _model_uncovered=_model_uncovered,
            _rep_ccy=_rep_ccy,
            action_plan_rows=action_plan_rows,
            asset_scores=asset_scores,
            benchmark_factor_profile=benchmark_factor_profile,
            benchmark_results=benchmark_results,
            bl_records=bl_records,
            cds_summary=cds_summary,
            cov_matrix=cov_matrix,
            df=df,
            expected_effect=expected_effect,
            factor_scores=factor_scores,
            history_result=history_result,
            macro_drivers=macro_drivers,
            perf=perf,
            period_returns_table=period_returns_table,
            port_metrics=port_metrics,
            priced_at_cost=priced_at_cost,
            profile_benchmark=profile_benchmark,
            regime_reading=regime_reading,
            return_coverage=return_coverage,
            sector_exposure=sector_exposure,
            smart_money=smart_money,
            stress_scenarios=stress_scenarios,
            technicals_map=technicals_map,
            total_portfolio_value=total_portfolio_value,
            weights_dict=weights_dict,
        )
        return self._build_results(ctx)

    def _build_results(self, ctx: "_AnalysisCtx") -> AnalyzeResults:
        """ЕДИНСТВЕННОЕ место сборки контракта `results{}` (Арх-3.1).

        Сюда же переехал расчёт плеча/экспозиции — он и есть стадия 8
        «сборка результата»: читает готовые веса, ничего не считает заново.

        Форма возврата описана в `finance/contracts.py`; `tests/test_contracts_results.py`
        сверяет ключи ЭТОГО литерала с контрактом по AST, а
        `tests/test_contracts_golden.py` — сами числа с эталоном.
        """
        # ── Leverage / gross exposure (read-only, AFTER risk math) ─────────
        # Margin debt manifests as a NEGATIVE weight on the cash leg (e.g.
        # USD wt = -17.7% → leverage 117.7%).  We compute these metrics
        # AFTER all matrix maths so the linear-algebra pipeline still
        # consumes the raw weights_dict unchanged.
        _NON_RISK = self.engine.NON_RISK_ASSETS
        long_weight = sum(max(0.0, float(w)) for w in ctx.weights_dict.values())
        # R-7 (2026-08-02): валовая экспозиция — Σ|w| по ПОЗИЦИЯМ, без кэша.
        # Маржинальный остаток — это ФИНАНСИРОВАНИЕ, а не рыночная позиция:
        # прежний Σ|w| по всем строкам считал |−20.2%| маржи как экспозицию и
        # печатал «валовая ≈140.4%» рядом с собственным «плечо ≈1.2x» в одном
        # предложении вердикта (экономически 120.2% = leverage_ratio × NAV).
        # Для книги с шортами формула остаётся верной: лонги + |шорты|.
        gross_expo  = sum(abs(float(w)) for t, w in ctx.weights_dict.items()
                          if str(t).upper() not in _NON_RISK)
        net_expo    = sum(float(w)           for w in ctx.weights_dict.values())
        cash_weight = sum(float(ctx.weights_dict.get(t, 0.0)) for t in ctx.weights_dict
                          if str(t).upper() in _NON_RISK)
        # Strict threshold guards float noise on a fully-funded book.
        is_leveraged = cash_weight < -0.001
        leverage_metrics = {
            "gross_exposure":   round(gross_expo, 6),
            "net_exposure":     round(net_expo,   6),
            "long_weight":      round(long_weight, 6),
            "cash_weight":      round(cash_weight, 6),
            "leverage_ratio":   round(long_weight / net_expo, 4) if net_expo > 1e-9 else 1.0,
            "is_leveraged":     bool(is_leveraged),
        }

        return {
            "performance_table": ctx.perf,
            "risk_matrix": ctx.cov_matrix,
            "total_portfolio_pnl": ctx.df['PnL'].sum(),
            "total_value": ctx.total_portfolio_value,
            "portfolio_metrics": ctx.port_metrics,
            "leverage_metrics":  leverage_metrics,
            # H3 (Phase-3): date-indexed portfolio log-return series for the
            # Telegram equity-curve SVG — the bot no longer recomputes it.
            "port_log_returns":  getattr(self.engine, "_last_port_log_returns", None),
            # H4: the mandate actually used for the composite-risk calibration.
            "risk_mandate":      getattr(self.engine, "risk_mandate", "MODERATE"),
            "risk_free_rate": self.engine.current_rfr_annual,
            "benchmark_comparison": ctx.benchmark_results,
            # The user's chosen benchmark TICKER (e.g. QQQ.US), so downstream
            # visuals (equity-curve line) can honour the actual choice instead
            # of defaulting to the first concrete benchmark.  None → no profile
            # benchmark selected.
            "profile_benchmark_ticker": ctx.profile_benchmark,
            # B1 (2026-07-17): факторные беты мандатного бенчмарка для секции
            # «Факторное разложение» — {ticker, name, betas} или None.
            "benchmark_factor_profile": ctx.benchmark_factor_profile,
            "period_returns_table": ctx.period_returns_table,
            "return_series_coverage": ctx.return_coverage,
            # H-1 (2026-07-29): раскрытие подмены неликвидных бумаг.
            # `proxy_substitutions` — {оригинал: прокси}: риск и беты этих строк
            # посчитаны по ликвидному индексу, а не по самой бумаге.
            # `priced_at_cost` — у этих бумаг нет ни своей истории, ни цены
            # брокера, поэтому они стоят по цене покупки (P&L = 0).
            # Отчёт ОБЯЗАН показать оба списка: иначе пользователь не поймёт,
            # почему у его структурной ноты «чужая» бета.
            "proxy_substitutions": {
                str(t): p for t in ctx.df.index
                if (p := self.engine.proxy_for(t)) is not None
            },
            "priced_at_cost": sorted(str(t) for t in ctx.priced_at_cost),
            # A-1 / A-2 (2026-08-01): что движок сделал с входным фреймом ДО
            # математики.  `PHASE_04 §3.2` требует показывать результат склейки
            # пользователю, а отброшенная строка обязана быть названа — молча
            # терять позицию нельзя.
            # −37: какие строки переведены в базовую валюту и по какому курсу.
            # Отчёт ОБЯЗАН это показать: доходность таких позиций — в валюте
            # инструмента (единый курс сокращается), а не долларовая.
            "fx_converted_rows": ctx._fx_rows,
            # Доля портфеля вне факторной модели (короткая история котировок).
            "model_uncovered": ctx._model_uncovered,
            # Ф-3: DEGRADE-находки чекеров. BLOCK сюда не попадает — он
            # выбросил исключение выше и отчёта не будет вовсе.
            "data_quality": {
                "profile": ctx._dq_report.profile.value if ctx._dq_report else None,
                "findings": [
                    {"id": f.id, "level": f.level.value, "message": f.message,
                     "ticker": f.ticker}
                    for f in (ctx._dq_report.findings if ctx._dq_report else [])
                ],
            },
            "reporting_currency": ctx._rep_ccy,
            "portfolio_source": ctx._portfolio_source,
            "merged_positions": [{"ticker": t, "rows": n} for t, n in ctx._merged_rows],
            "dropped_rows":     [{"ticker": t, "reason": r} for t, r in ctx._dropped_rows],
            "stress_scenarios": ctx.stress_scenarios,
            "expected_effect": ctx.expected_effect,
            "cds_summary":     ctx.cds_summary,
            "history_result": ctx.history_result,
            "sector_exposure": ctx.sector_exposure,
            "factor_scores": ctx.factor_scores,
            # Phase 2 additions
            "regime":            ctx.regime_reading.as_dict() if ctx.regime_reading else None,
            "macro_drivers":     ctx.macro_drivers,
            "technicals":        {t: {"score": r.score, "raw": r.raw,
                                       "components": r.components}
                                  for t, r in ctx.technicals_map.items()},
            "asset_scores":      {t: {
                                      "fundamentals": s.fundamentals,
                                      "valuations":   s.valuations,
                                      "technicals":   s.technicals,
                                      "credit":       s.credit,
                                      "total":        s.total,
                                      "action":       s.action,
                                      "hotspot":      s.hotspot,
                                      "credit_applicable": getattr(s, "credit_applicable", True),
                                      "fundamentals_applicable": getattr(s, "fundamentals_applicable", True),
                                  } for t, s in ctx.asset_scores.items()},
            # Phase 3 additions
            "black_litterman":   ctx.bl_records,
            "action_plan":       ctx.action_plan_rows,
            # BLOCK 3.5 / B2.4 — insider (Form-4) signals for the DEEP layer.
            "smart_money":       ctx.smart_money,
        }