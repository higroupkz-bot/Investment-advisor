"""Гибрид PR-3: агрегированный отчёт и меню источников в боте.

Задание `pico_prompt_hybrid_architecture_fallback_v2` PR-3, решения D-1, D-5,
D-7, D-9, D-10; таблица §I.5 — S-2, S-3, S-4, S-5, S-10.

Главное — **I-15** на уровне бота (`test_aggregated_requires_live_broker_fetch`):
при fallback-моке, при отозванных ключах и при пустом vault у не-админа
`UniversalPortfolioManager(price_source="aggregated")` не создаётся, то есть
клиент Tradernet для ручных тикеров не появляется вовсе.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_broker_fallback import (  # noqa: E402
    FallbackTestBase,
    _BotRecorder,
    _buttons,
    _fallback_mock,
)
from test_manual_fsm_flow import _FakeCallback, _FakeMessage, _StubManager  # noqa: E402

BROKER_ROWS = [
    {"Ticker": "AAPL", "Quantity": 5.0, "Purchase_Price": 140.0,
     "Broker_Current_Price": 190.0, "Asset_Type": "Акция",
     "Raw_Ticker": "AAPL.US", "Currency": "USD"},
    {"Ticker": "USD", "Quantity": 100.0, "Purchase_Price": 1.0,
     "Broker_Current_Price": 1.0, "Asset_Type": "Кэш",
     "Raw_Ticker": "USD", "Currency": "USD"},
]
MANUAL_NO_OVERLAP = "TLT.US 20 95 USD\nCASH:KZT 1000000"
MANUAL_OVERLAP = "AAPL.US 10 150 USD\nTLT.US 20 95 USD"


def _broker(*_a) -> pd.DataFrame:
    return pd.DataFrame(BROKER_ROWS)


async def make_profile(db, user_id: int) -> None:
    """Пройденная анкета: без мандата меню отправляет на /start (§−124)."""
    await db.save_profile(
        telegram_id=user_id, score=10, profile_name="Умеренный",
        target_volatility=0.10, target_te=0.04, selected_assets=["US_EQUITY"],
        limits_dict={"US_EQUITY": [30, 60]}, benchmark_ticker="SPY.US")


class HybridTestBase(FallbackTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._orig_hybrid = os.environ.get("HYBRID_PORTFOLIO_ENABLED")
        os.environ["HYBRID_PORTFOLIO_ENABLED"] = "on"
        self.built_sources: list[str] = []
        outer = self

        class _Recorder(_StubManager):
            def __init__(self, price_source: str = "freedom") -> None:
                outer.built_sources.append(price_source)
                super().__init__(price_source)

        self._hybrid = [
            patch.object(self.tg, "UniversalPortfolioManager", _Recorder),
            patch("finance.investment_logic.UniversalPortfolioManager", _Recorder),
            patch.object(self.tg, "_has_vault_keys_sync", lambda _uid: True),
        ]
        for p in self._hybrid:
            p.start()

    async def asyncTearDown(self) -> None:
        for p in self._hybrid:
            p.stop()
        if self._orig_hybrid is None:
            os.environ.pop("HYBRID_PORTFOLIO_ENABLED", None)
        else:
            os.environ["HYBRID_PORTFOLIO_ENABLED"] = self._orig_hybrid
        await super().asyncTearDown()

    async def _go(self, data: str, fetch=_broker) -> _FakeCallback:
        cb = _FakeCallback(data, user_id=self.USER_ID)
        handler = (self.tg.cb_aggregated_overlap if data.startswith("agg:")
                   else self.tg.cb_report_source if data.startswith("src:")
                   else self.tg.cb_report_tier)
        with patch.object(self.tg, "_fetch_portfolio_sync", fetch):
            await handler(cb, self.state)
            await asyncio.sleep(0)
        return cb

    @property
    def aggregated_built(self) -> bool:
        return "aggregated" in self.built_sources


# ── I-15 ─────────────────────────────────────────────────────────────────────

class AggregatedRequiresLiveBrokerFetchTest(HybridTestBase):
    """S-10 — `test_aggregated_requires_live_broker_fetch` на уровне бота."""

    async def test_aggregated_requires_live_broker_fetch(self) -> None:
        from finance.broker_api import BrokerAuthError

        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)

        def _auth(*_a):
            raise BrokerAuthError("revoked")

        cases = {
            "fallback": dict(fetch=lambda *_a: _fallback_mock("api_error")),
            "auth": dict(fetch=_auth),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                self.built_sources.clear()
                cb = await self._go("rptgo:aggregated:base", **kw)
                self.assertFalse(self.aggregated_built, cb.message.all_text)
                self.assertFalse(self.started)
                self.assertIn("не списан", cb.message.all_text)
                self.assertTrue(await self._slot_is_free())

        with self.subTest(case="empty_vault_non_admin"):
            self.built_sources.clear()
            with patch.object(self.tg, "_get_keys_sync", lambda _uid: None), \
                 patch.object(self.tg, "_is_admin", lambda _uid: False):
                cb = await self._go("rptgo:aggregated:base",
                                    fetch=lambda *_a: self.fail("брокер не спрашивается"))
            self.assertFalse(self.aggregated_built)
            self.assertFalse(self.started)
            self.assertIn("Брокер не подключён", cb.message.all_text)

    async def test_background_refuses_without_proof(self) -> None:
        """Вторая линия: фон с `aggregated` без доказательства не строит менеджер."""
        bot = _BotRecorder()
        await self.real_background(
            bot=bot, chat_id=self.USER_ID, user_id=self.USER_ID, tier="base", cost=1,
            df=pd.DataFrame(BROKER_ROWS), bench_tick=None, source="aggregated",
            aggregated_composition={"freedom_positions": 2, "manual_positions": 0,
                                    "overlaps": []},
            freedom_proof=None)
        self.assertFalse(self.aggregated_built)
        self.assertTrue(any("Агрегированный отчёт невозможен" in t for t, _ in bot.sent))
        self.assertTrue(await self._slot_is_free())


# ── успешный путь ─────────────────────────────────────────────────────────────

class AggregatedHappyPathTest(HybridTestBase):

    async def test_merged_portfolio_goes_to_background(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)
        cb = await self._go("rptgo:aggregated:deep")
        self.assertEqual(len(self.started), 1, cb.message.all_text)
        run = self.started[0]
        self.assertEqual(run["source"], "aggregated")
        self.assertEqual(run["cost"], self.tg.TIER_COST["deep"], "D-10: тариф тот же")
        self.assertEqual(run["df"].attrs, {"_ramp_source": "aggregated"})
        self.assertEqual(sorted(run["df"]["Ticker"]), ["AAPL", "KZT", "TLT", "USD"])
        self.assertEqual(run["aggregated_composition"],
                         {"freedom_positions": 2, "manual_positions": 2, "overlaps": []})
        self.assertTrue(run["freedom_proof"].live)
        self.assertEqual(run["freedom_proof"].key_origin, "vault")
        text = cb.message.all_text
        self.assertIn("Freedom Broker (2 поз.) + ручной ввод (2 поз.)", text)
        self.assertIn("Tradernet для всех позиций", text)

    async def test_background_puts_composition_into_results(self) -> None:
        """§I.6: состав доезжает до CoVe; менеджер — `aggregated` за гейтом."""
        from portfolio_aggregation import FreedomSource

        proof = FreedomSource("k", connector_factory=lambda *_a: SimpleNamespace(
            fetch_portfolio=lambda: pd.DataFrame(BROKER_ROWS))).load()
        seen: list[dict] = []

        def _prefetch(_self, candidates):
            return SimpleNamespace(
                data=pd.DataFrame({"AAPL": [1.0, 1.1]}), risky_tickers=["AAPL"],
                history_result=SimpleNamespace(failed={}, retried=[]),
                loaded_count=1, internal_tickers=set(), resolved_portfolio=["AAPL"],
                portfolio_loaded=1, portfolio_total=1, proxy_map={})

        def _payload(results, tier, **_kw):
            seen.append(dict(results))
            raise RuntimeError("стоп после payload")

        comp = {"freedom_positions": 2, "manual_positions": 1, "overlaps": ["AAPL"]}
        with patch.object(_StubManager, "prefetch_market_data", _prefetch, create=True), \
             patch.object(self.tg, "_analyze_existing_portfolio_sync",
                          lambda *_a, **_k: {"portfolio_source": "aggregated"}), \
             patch.object(self.tg, "run_gatekeeper_for",
                          lambda *_a, **_k: {"critical": [], "warnings": []}), \
             patch.object(self.tg, "_build_pdf_payload", _payload):
            await self.real_background(
                bot=_BotRecorder(), chat_id=self.USER_ID, user_id=self.USER_ID,
                tier="base", cost=1, df=pd.DataFrame(BROKER_ROWS), bench_tick=None,
                source="aggregated", aggregated_composition=comp, freedom_proof=proof)
        self.assertEqual(self.built_sources, ["aggregated"])
        self.assertEqual(seen[0]["aggregated_composition"], comp)

        from finance.data_lineage import _aggregated_source_status
        row = _aggregated_source_status(seen[0])
        self.assertIn("Freedom Broker (2 поз.) + ручной ввод (1 поз.)", row["source"])

    async def test_effective_cost(self) -> None:
        for tier, cost in self.tg.TIER_COST.items():
            self.assertEqual(self.tg._effective_cost(tier, "aggregated"), cost)


# ── D-5: пересечения ─────────────────────────────────────────────────────────

class OverlapScreenTest(HybridTestBase):

    async def test_overlap_is_shown_before_the_report(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_OVERLAP)
        cb = await self._go("rptgo:aggregated:base")
        self.assertFalse(self.started)
        self.assertIn("и на счёте Freedom, и в ручном портфеле", cb.message.all_text)
        self.assertIn("AAPL", cb.message.all_text)
        datas = _buttons(cb.message)
        agg = [d for d in datas if d.startswith("agg:sum:base:")]
        self.assertEqual(len(agg), 1)
        self.assertIn("mp:show", datas)
        self.assertIn("cancel", datas)
        self.assertTrue(await self._slot_is_free())

        cb2 = await self._go(agg[0])
        self.assertEqual(len(self.started), 1, cb2.message.all_text)
        self.assertEqual(self.started[0]["aggregated_composition"]["overlaps"], ["AAPL"])

    async def test_confirmation_of_another_set_shows_screen_again(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_OVERLAP)
        cb = await self._go("agg:sum:base:00000000")
        self.assertFalse(self.started)
        self.assertIn("и на счёте Freedom, и в ручном портфеле", cb.message.all_text)


# ── D-7 и прочие отказы ──────────────────────────────────────────────────────

class AggregatedRefusalsTest(HybridTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)

    async def test_broker_down_offers_manual_report(self) -> None:
        """D-7: «агрегированный сейчас невозможен» + причина + ручной отчёт."""
        cb = await self._go("rptgo:aggregated:base",
                            fetch=lambda *_a: _fallback_mock("waf_block"))
        text = cb.message.all_text
        self.assertIn("Агрегированный отчёт сейчас невозможен", text)
        self.assertIn("не пройдёт сам", text, "текст — по причине waf_block (§−94)")
        self.assertIn("fb:manual:base:waf_block", _buttons(cb.message))
        self.assertFalse(self.started)

    async def test_broker_timeout(self) -> None:
        import time

        def _slow(*_a):
            time.sleep(0.5)
            return _broker()

        with patch.object(self.tg, "broker_fetch_budget_s", lambda: 0.05):
            cb = await self._go("rptgo:aggregated:base", fetch=_slow)
        self.assertIn("не ответил", cb.message.all_text)
        self.assertIn("fb:manual:base:timeout", _buttons(cb.message))
        self.assertTrue(await self._slot_is_free())

    async def test_broker_error_body_never_shown(self) -> None:
        """S-2: только код, никогда `str(exc)`."""
        from freedom_portfolio.client import BrokerAPIError

        def _raise(*_a):
            raise BrokerAPIError("HTTP 502 BODY-SECRET")

        cb = await self._go("rptgo:aggregated:base", fetch=_raise)
        self.assertNotIn("BODY-SECRET", cb.message.all_text)
        self.assertIn("Код для поддержки", cb.message.all_text)

    async def test_empty_broker_account(self) -> None:
        from finance.broker_api import BrokerEmptyPortfolioError

        def _raise(*_a):
            raise BrokerEmptyPortfolioError("empty")

        cb = await self._go("rptgo:aggregated:base", fetch=_raise)
        self.assertIn("На счёте Freedom нет позиций", cb.message.all_text)
        self.assertIn("rptgo:manual:base", _buttons(cb.message))

    async def test_currency_conflict(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 1 80000 KZT")
        cb = await self._go("rptgo:aggregated:base")
        self.assertIn("в разных валютах", cb.message.all_text)
        self.assertFalse(self.started)

    async def test_empty_manual_portfolio(self) -> None:
        """§−124: отказ — ДО похода к брокеру и с кнопкой туда, где его устранить."""
        await self.db.delete_manual_portfolio(self.USER_ID)
        cb = await self._go("rptgo:aggregated:base",
                            fetch=lambda *_a: self.fail("брокер не спрашивается"))
        self.assertIn("заполните ручной портфель", cb.message.all_text)
        self.assertIn("mp:show", _buttons(cb.message))
        self.assertFalse(self.started)

    async def test_logs_carry_no_keys_or_positions(self) -> None:
        """S-3: в логах — user_id, причина, error_id, счётчики."""
        with self.assertLogs(level="INFO") as logs:
            await self._go("rptgo:aggregated:base")
        joined = "\n".join(logs.output)
        for secret in ("user-key", "user-secret", "TLT", "1000000"):
            self.assertNotIn(secret, joined)


# ── меню и недоверенные callback_data ─────────────────────────────────────────

class SourceMenuTest(HybridTestBase):

    async def _menu(self) -> _FakeMessage:
        msg = _FakeMessage("", user_id=self.USER_ID)
        await self.tg._show_analysis_menu(msg, "", user_id=self.USER_ID)
        return msg

    async def test_all_sources_when_available(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)
        msg = await self._menu()
        # §−124: под списком — строка навигации (портфель · меню).
        self.assertEqual(_buttons(msg), ["src:freedom", "src:manual", "src:aggregated",
                                         "src:demo", "home:portfolio", "home:menu"])

    async def test_unavailable_sources_are_explained(self) -> None:
        with patch.object(self.tg, "_has_vault_keys_sync", lambda _uid: False), \
             patch.object(self.tg, "_is_admin", lambda _uid: False):
            msg = await self._menu()
        datas = _buttons(msg)
        self.assertNotIn("src:freedom", datas)
        self.assertNotIn("src:aggregated", datas)
        self.assertIn("Недоступно", msg.all_text)
        self.assertIn("нужен подключённый Freedom", msg.all_text)

    async def test_step2_tiers_carry_source(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)
        cb = await self._go("src:aggregated")
        datas = _buttons(cb.message)
        # §−124: тиры по одному в строке (парой они обрезались на телефоне) + навигация.
        self.assertEqual(datas, ["rpt:aggregated:base", "rpt:aggregated:scenario",
                                 "rpt:aggregated:deep", "home:report", "home:menu"])
        self.assertTrue(all(len(d.encode()) <= 64 for d in datas))
        cb2 = await self._go("rpt:aggregated:deep")
        self.assertIn("rptgo:aggregated:deep", _buttons(cb2.message))
        self.assertIn("2 токен", cb2.message.all_text)
        self.assertFalse(self.started, "экран цены ничего не запускает")

    async def test_empty_manual_leads_to_input(self) -> None:
        await self._go("src:manual")
        self.assertEqual(await self.state.get_state(), self.tg.ManualPortfolio.Input)

    async def test_forged_callbacks(self) -> None:
        """S-5: `rpt:stooq:base`, `rpt:aggregated:vip` и компания."""
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)
        for data in ("rpt:stooq:base", "rpt:aggregated:vip", "rptgo:aggregated:vip",
                     "rptgo:tradernet:base", "src:stooq", "src:", "agg:sum:vip:00000000",
                     "agg:sum:base:XYZ", "rptx:aggregated:base"):
            with self.subTest(data=data):
                cb = await self._go(data, fetch=lambda *_a: self.fail("брокер"))
                self.assertFalse(self.started)
                self.assertEqual(cb.message.sent, [])
                self.assertTrue(await self._slot_is_free())

    async def test_old_buttons_with_hybrid_flag_off(self) -> None:
        """S-4: кнопка из прошлой ревизии при выключенном флаге."""
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)
        os.environ["HYBRID_PORTFOLIO_ENABLED"] = "off"
        for data in ("src:aggregated", "rpt:aggregated:base", "rptgo:aggregated:base",
                     "agg:sum:base:00000000"):
            with self.subTest(data=data):
                cb = await self._go(data, fetch=lambda *_a: self.fail("брокер"))
                self.assertIn("недоступен", cb.message.all_text)
                self.assertFalse(self.started)
        self.assertFalse(self.aggregated_built)

    async def test_hybrid_requires_manual_flag(self) -> None:
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        self.assertFalse(self.tg.hybrid_portfolio_enabled())

    async def test_flag_off_keeps_the_old_menu(self) -> None:
        """I-9: без флага гибрида — прежнее меню тиров."""
        os.environ["HYBRID_PORTFOLIO_ENABLED"] = "off"
        msg = await self._menu()
        self.assertEqual(_buttons(msg), ["analysis:base", "analysis:scenario",
                                         "analysis:deep", "home:portfolio", "home:menu"])

    async def test_handlers_registered(self) -> None:
        # `build_dispatcher()` здесь не вызывается: роутеры модульные и
        # прикрепляются к диспетчеру один раз на процесс.
        import inspect
        names = set(re.findall(r"register\((\w+)", inspect.getsource(
            self.tg.build_dispatcher)))
        for name in ("cb_report_source", "cb_report_tier", "cb_aggregated_overlap"):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()


# ── аудит §−123 ──────────────────────────────────────────────────────────────

class AuditBotFindingsTest(HybridTestBase):

    def _stage_patches(self, seen: list, results: dict):
        def _prefetch(_self, candidates):
            return SimpleNamespace(
                data=pd.DataFrame({"AAPL": [1.0, 1.1]}), risky_tickers=["AAPL"],
                history_result=SimpleNamespace(failed={}, retried=[]),
                loaded_count=1, internal_tickers=set(), resolved_portfolio=["AAPL"],
                portfolio_loaded=1, portfolio_total=1, proxy_map={})

        snapshots: list = []

        async def _get_snap(*_a, **_k):
            snapshots.append("read")
            return {"risk_score": 50}

        async def _save_snap(**_k):
            snapshots.append("write")

        def _payload(res, tier, **kw):
            seen.append({"results": dict(res), "prev": kw.get("prev_snapshot")})
            return {"risk_pct": 10}

        self.snapshots = snapshots
        return [
            patch.object(_StubManager, "prefetch_market_data", _prefetch, create=True),
            patch.object(self.tg, "_analyze_existing_portfolio_sync",
                         lambda *_a, **_k: dict(results)),
            patch.object(self.tg, "run_gatekeeper_for",
                         lambda *_a, **_k: {"critical": [], "warnings": []}),
            patch.object(self.tg, "_build_pdf_payload", _payload),
            patch.object(self.tg, "render_report_html", lambda *_a, **_k: "<html/>"),
            patch.object(self.tg, "write_report_html", lambda *_a, **_k: "/tmp/x.html"),
            patch.object(self.tg, "upload_report",
                         lambda *_a, **_k: "https://example.invalid/r.html"),
            patch.object(self.tg, "get_last_report_snapshot", _get_snap),
            patch.object(self.tg, "save_report_snapshot", _save_snap),
        ]

    async def _bg(self, **kw):
        bot = _BotRecorder()
        seen: list = []
        results = kw.pop("results", {"performance_table": pd.DataFrame({"Ticker": ["AAPL"]})})
        patches = self._stage_patches(seen, results)
        for p in patches:
            p.start()
        try:
            await self.real_background(bot=bot, chat_id=self.USER_ID, user_id=self.USER_ID,
                                       tier="base", cost=0, bench_tick=None, **kw)
        finally:
            for p in patches:
                p.stop()
        return bot, seen

    async def test_unpriced_position_is_announced(self) -> None:
        df = pd.DataFrame({"Ticker": ["AAPL", "ZZZQ", "USD"], "Quantity": [1, 1, 1],
                           "Asset_Type": ["Акция", "Акция", "Кэш"]})
        bot, seen = await self._bg(df=df, source="manual")
        self.assertTrue(any("Не вошли в расчёт" in t and "ZZZQ" in t for t, _ in bot.sent))
        self.assertEqual(seen[0]["results"]["unpriced_positions"], ["ZZZQ"])

    async def test_month_over_month_skipped_for_other_portfolios(self) -> None:
        """Дельта «против прошлого месяца» не сравнивает разные составы книги."""
        from portfolio_aggregation import FreedomSource

        proof = FreedomSource("k", connector_factory=lambda *_a: SimpleNamespace(
            fetch_portfolio=lambda: pd.DataFrame(BROKER_ROWS))).load()
        df = pd.DataFrame(BROKER_ROWS)
        _bot, seen = await self._bg(df=df, source="aggregated", freedom_proof=proof,
                                    aggregated_composition={"freedom_positions": 2,
                                                            "manual_positions": 0,
                                                            "overlaps": []})
        self.assertIsNone(seen[0]["prev"])
        self.assertEqual(self.snapshots, [])
        _bot, seen = await self._bg(df=df, source="manual",
                                    broker_fallback="брокер не ответил вовремя")
        self.assertIsNone(seen[0]["prev"])
        self.assertEqual(self.snapshots, [])
        _bot, seen = await self._bg(df=df, source="freedom")
        self.assertEqual(seen[0]["prev"], {"risk_score": 50}, "брокерская история прежняя")
        self.assertEqual(self.snapshots, ["read", "write"])

    async def test_fallback_report_keeps_unfinished_draft(self) -> None:
        """Отчёт по СОХРАНЁННОМУ портфелю не стирает незаконченный черновик."""
        await self.db.save_manual_portfolio(self.USER_ID, MANUAL_NO_OVERLAP)
        await self.db.save_manual_draft(self.USER_ID, "MSFT 1 400")
        cb = _FakeCallback("fb:manual:base:timeout", user_id=self.USER_ID)
        await self.tg.cb_fallback_manual(cb, self.state)
        await asyncio.sleep(0)
        self.assertEqual(len(self.started), 1, cb.message.all_text)
        self.assertIs(self.started[0].get("delete_draft"), False)
        _bot, _seen = await self._bg(df=pd.DataFrame(BROKER_ROWS), source="manual",
                                     delete_draft=False)
        self.assertIsNotNone(await self.db.get_manual_draft(self.USER_ID))

    async def test_storage_failure_is_never_silent(self) -> None:
        """Сбой хранилища на экране/кнопках — код поддержки, а не тишина."""
        async def _boom(_uid):
            raise RuntimeError("db down")

        await make_profile(self.db, self.USER_ID)
        with patch.object(self.tg, "get_manual_portfolio", _boom):
            # §−124: /portfolio — экран «Мой портфель»; сбой хранилища там
            # назван «недоступен», а не «пуст» (это разные факты).
            msg = _FakeMessage("/portfolio", user_id=self.USER_ID)
            await self.tg.cmd_portfolio(msg, self.state)
            self.assertIn("Ручной портфель — недоступен", msg.all_text)
            cb = _FakeCallback("mp:show", user_id=self.USER_ID)
            await self.tg.cb_manual_portfolio(cb, self.state)
            self.assertIn("Код ошибки для поддержки", cb.message.all_text)
            cb = _FakeCallback("mp:rmlist", user_id=self.USER_ID)
            await self.tg.cb_manual_portfolio(cb, self.state)
            self.assertIn("Код ошибки для поддержки", cb.message.all_text)
            menu = _FakeMessage("", user_id=self.USER_ID)
            await self.tg._show_analysis_menu(menu, "", user_id=self.USER_ID)
            self.assertIn("src:demo", _buttons(menu), "меню источников не пропало")

    async def test_screen_shows_exact_quantities(self) -> None:
        """0.005 BTC печаталось как «0.01» — экран врал о сохранённом."""
        await self.db.save_manual_portfolio(self.USER_ID, "BTC-USD 0.005 65000.5 USD")
        cb = _FakeCallback("mp:show", user_id=self.USER_ID)
        await self.tg.cb_manual_portfolio(cb, self.state)
        self.assertIn("0.005", cb.message.all_text)
        self.assertIn("65 000.5", cb.message.all_text)


class OverLimitPromiseTest(HybridTestBase):
    """Аудит `§−123`: сообщение о непринятом длинном вводе не обещает ложного."""

    async def test_hybrid_with_saved_portfolio_says_old_one_is_used(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 1 100 USD")
        await self._send_text("\n".join(f"T{i:03d} 1 10 USD" for i in range(51)))
        cb = await self._tap("confirm")
        self.assertIn("не сохранён", cb.message.all_text)
        self.assertIn("прежнему", cb.message.all_text)
        self.assertEqual(await self._stored(), "AAPL.US 1 100 USD")
