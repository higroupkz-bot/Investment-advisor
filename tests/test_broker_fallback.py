"""Гибрид PR-2: ручной отчёт вместо брокерского, когда Freedom недоступен.

Задание `pico_prompt_hybrid_architecture_fallback_v2` PR-2, решения D-7/D-8,
таблица §I.5: S-1, S-2, S-4, S-5, S-11.

Что охраняется
--------------
* Текст отказа — ПО ПРИЧИНЕ (`_broker_outage_advice`, `§−94`): «серверы
  недоступны» честно только для `api_error` и таймаута; для отозванных ключей
  — «ключи неверны».
* Бюджет ожидания брокера (D-8): по таймауту — отказ без списания, слот
  освобождён ровно один раз.
* Кнопка `fb:manual:<tier>:<причина>` проходит ВЕСЬ путь подтверждения заново
  и строит обычный `manual`-отчёт: Tradernet в нём нет ни байта (I-12, S-11).
* При выключенном флаге ручного ввода сообщения об отказе — ровно прежние (I-9).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_manual_fsm_flow import _FakeCallback  # noqa: E402
from test_manual_portfolio_store import StoreTestBase  # noqa: E402

STORED = "AAPL.US 10 150 USD\nCASH:USD 1000"


def _buttons(message) -> list[str]:
    out = []
    for _text, kwargs in message.sent + message.edited:
        kb = kwargs.get("reply_markup")
        if kb is not None:
            out += [b.callback_data for row in kb.inline_keyboard for b in row]
    return out


def _fallback_mock(reason: str) -> pd.DataFrame:
    from finance.demo_portfolio import build_demo_portfolio

    df = build_demo_portfolio()
    df.attrs.update({"_ramp_is_mock": True, "_ramp_is_fallback": True,
                     "_ramp_fallback_reason": reason})
    return df


class _BotRecorder:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    async def send_message(self, _chat_id, text: str, **kwargs):
        self.sent.append((text, kwargs))
        return SimpleNamespace(edit_text=self._edit)

    async def _edit(self, *_a, **_k):
        return None


class FallbackTestBase(StoreTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self.db.init_user(self.USER_ID)
        await self.db.credit_tokens(self.USER_ID, 10, reason="test")
        self.started: list[dict] = []
        # Настоящая фоновая задача — для тестов, которые гоняют именно её.
        self.real_background = self.tg._run_analysis_background

        async def _fake_background(**kwargs):
            self.started.append(kwargs)

        async def _freedom_source(_uid):
            return "freedom", "freedom"

        self._extra = [
            patch.object(self.tg, "_run_analysis_background", _fake_background),
            patch.object(self.tg, "_resolve_portfolio_source", _freedom_source),
            patch.object(self.tg, "_get_keys_sync",
                         lambda _uid: ("login", "user-key", "user-secret")),
        ]
        for p in self._extra:
            p.start()

    async def asyncTearDown(self) -> None:
        for p in self._extra:
            p.stop()
        await super().asyncTearDown()

    async def _confirm(self, fetch) -> _FakeCallback:
        cb = _FakeCallback("confirm:base:menu", user_id=self.USER_ID)
        with patch.object(self.tg, "_fetch_portfolio_sync", fetch):
            await self.tg.cb_confirm(cb, self.state)
            await asyncio.sleep(0)
        return cb

    async def _fb(self, data: str) -> _FakeCallback:
        cb = _FakeCallback(data, user_id=self.USER_ID)
        with patch.object(self.tg, "_fetch_portfolio_sync",
                          side_effect=AssertionError("брокер не спрашивается")):
            await self.tg.cb_fallback_manual(cb, self.state)
            await asyncio.sleep(0)
        return cb


class OutageOffersManualReportTest(FallbackTestBase):

    async def test_fallback_mock_offers_manual_report(self) -> None:
        """S-1: мок не показан; отказ по причине + кнопка ручного отчёта."""
        await self.db.save_manual_portfolio(self.USER_ID, STORED)
        cb = await self._confirm(lambda *_a: _fallback_mock("waf_block"))
        text = cb.message.all_text
        self.assertIn("Freedom Broker сейчас недоступен", text)
        self.assertNotIn("успешно получен", text)
        self.assertNotIn("BTC-USD", text, "мок не доезжает до превью")
        self.assertIn("fb:manual:base:waf_block", _buttons(cb.message))
        self.assertFalse(self.started)
        self.assertTrue(await self._slot_is_free())

    async def test_empty_manual_portfolio_offers_input(self) -> None:
        cb = await self._confirm(lambda *_a: _fallback_mock("api_error"))
        self.assertIn("Ручной портфель пуст, отчёт сейчас невозможен", cb.message.all_text)
        self.assertEqual(_buttons(cb.message), ["mp:add"])

    async def test_flag_off_keeps_the_old_message(self) -> None:
        """I-9: без флага — один прежний отказ, без кнопок ручного пути."""
        await self.db.save_manual_portfolio(self.USER_ID, STORED)
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        cb = await self._confirm(lambda *_a: _fallback_mock("api_error"))
        self.assertEqual(len(cb.message.sent), 1)
        self.assertEqual(_buttons(cb.message), [])
        self.assertNotIn("ручн", cb.message.sent[0][0].lower())

    async def test_auth_error_says_keys_not_servers(self) -> None:
        from finance.broker_api import BrokerAuthError

        await self.db.save_manual_portfolio(self.USER_ID, STORED)

        def _raise(*_a):
            raise BrokerAuthError("revoked")

        cb = await self._confirm(_raise)
        text = cb.message.all_text
        self.assertIn("ключи неверны", text)
        self.assertNotIn("Серверы Freedom Broker", text)
        self.assertIn("fb:manual:base:auth", _buttons(cb.message))
        self.assertTrue(await self._slot_is_free())

    async def test_unexpected_error_shows_only_error_id(self) -> None:
        """S-2: тело ответа брокера (`client._decode`) пользователю не уходит."""
        from freedom_portfolio.client import BrokerAPIError

        await self.db.save_manual_portfolio(self.USER_ID, STORED)

        def _raise(*_a):
            raise BrokerAPIError("HTTP 500: <html>BODY-SECRET</html>")

        cb = await self._confirm(_raise)
        self.assertNotIn("BODY-SECRET", cb.message.all_text)
        self.assertIn("Код ошибки для поддержки", cb.message.all_text)
        self.assertIn("fb:manual:base:error", _buttons(cb.message))

    async def test_budget_timeout(self) -> None:
        """D-8: брокер молчит дольше бюджета — отказ, слот свободен."""
        await self.db.save_manual_portfolio(self.USER_ID, STORED)

        def _slow(*_a):
            time.sleep(0.5)
            return pd.DataFrame()

        with patch.object(self.tg, "broker_fetch_budget_s", lambda: 0.05):
            cb = await self._confirm(_slow)
        text = cb.message.all_text
        self.assertIn("Серверы Freedom Broker сейчас недоступны", text)
        self.assertIn("не списан", text)
        self.assertIn("fb:manual:base:timeout", _buttons(cb.message))
        self.assertFalse(self.started)
        self.assertTrue(await self._slot_is_free())

    async def test_budget_default_and_clamp(self) -> None:
        from portfolio_aggregation import broker_fetch_budget_s

        with patch.dict(os.environ, {"BROKER_FETCH_BUDGET_S": ""}):
            self.assertEqual(broker_fetch_budget_s(), 60)
        with patch.dict(os.environ, {"BROKER_FETCH_BUDGET_S": "1"}):
            self.assertEqual(broker_fetch_budget_s(), 10)
        with patch.dict(os.environ, {"BROKER_FETCH_BUDGET_S": "9999"}):
            self.assertEqual(broker_fetch_budget_s(), 180)

    async def test_empty_broker_account_is_not_a_fallback(self) -> None:
        """Пустой счёт — не сбой: ручной отчёт для freedom не предлагается."""
        from finance.broker_api import BrokerEmptyPortfolioError

        await self.db.save_manual_portfolio(self.USER_ID, STORED)

        def _raise(*_a):
            raise BrokerEmptyPortfolioError("empty")

        cb = await self._confirm(_raise)
        self.assertIn("Портфель пуст", cb.message.all_text)
        self.assertNotIn("fb:", " ".join(_buttons(cb.message)))


class FallbackButtonTest(FallbackTestBase):

    async def test_button_builds_a_manual_report(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, STORED)
        await self.db.save_connection_mode(self.USER_ID, "freedom")
        cb = await self._fb("fb:manual:base:timeout")
        self.assertEqual(len(self.started), 1, cb.message.all_text)
        run = self.started[0]
        self.assertEqual(run["source"], "manual")
        self.assertEqual(run["cost"], self.tg.TIER_COST["base"], "тариф тот же (D-10)")
        self.assertEqual(run["broker_fallback"], "брокер не ответил вовремя")
        self.assertEqual(sorted(run["df"]["Ticker"]), ["AAPL", "USD"])
        self.assertIn("Отчёт по ручным активам; брокер был недоступен", cb.message.all_text)
        self.assertEqual(await self.db.get_connection_mode_explicit(self.USER_ID),
                         "freedom", "кнопка не меняет режим по умолчанию")

    async def test_button_without_portfolio_refuses(self) -> None:
        cb = await self._fb("fb:manual:deep:api_error")
        self.assertFalse(self.started)
        self.assertIn("не списан", cb.message.all_text)
        self.assertTrue(await self._slot_is_free())

    async def test_forged_callbacks_are_ignored(self) -> None:
        """S-5: источник, тир и причина — по allowlist."""
        await self.db.save_manual_portfolio(self.USER_ID, STORED)
        for data in ("fb:manual:vip:timeout", "fb:manual:base:hack",
                     "fb:stooq:base:timeout", "fb:manual:base", "fb:manual:base:timeout:x"):
            with self.subTest(data=data):
                await self._fb(data)
                self.assertFalse(self.started)
                self.assertTrue(await self._slot_is_free())

    async def test_old_button_with_flag_off(self) -> None:
        """S-4: кнопка из прошлой ревизии при выключенном флаге."""
        await self.db.save_manual_portfolio(self.USER_ID, STORED)
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        cb = await self._fb("fb:manual:base:timeout")
        self.assertFalse(self.started)
        self.assertIn("недоступен", cb.message.all_text)

    async def test_busy_slot(self) -> None:
        """S-6: двойное нажатие — второй запрос не стартует."""
        await self.db.save_manual_portfolio(self.USER_ID, STORED)
        self.assertTrue(await self.tg._try_acquire_user_slot(self.USER_ID))
        try:
            cb = await self._fb("fb:manual:base:timeout")
            self.assertIn("уже выполняется", cb.message.all_text)
            self.assertFalse(self.started)
        finally:
            await self.tg._release_user_slot(self.USER_ID)

    async def test_registered_in_dispatcher(self) -> None:
        # `build_dispatcher()` здесь не вызывается: роутеры модульные и
        # прикрепляются к диспетчеру один раз на процесс.
        import inspect
        names = set(re.findall(r"register\((\w+)", inspect.getsource(
            self.tg.build_dispatcher)))
        self.assertIn("cb_fallback_manual", names)


class ManualFallbackIsolationTest(FallbackTestBase):
    """S-11: `test_manual_fallback_never_builds_tradernet_client`."""

    async def test_manual_fallback_never_builds_tradernet_client(self) -> None:
        from finance.investment_logic import MAC3RiskEngine

        built: list[str] = []
        for p in self._extra:
            p.stop()
        self._extra = []

        class _Provider:
            name = "stooq"

            def fetch(self, tickers, *, days):
                raise RuntimeError("стоп после выбора провайдера")

        with patch.object(MAC3RiskEngine, "_get_tradernet_client",
                          lambda _self: built.append("tradernet")), \
             patch.dict("finance.price_providers._REGISTRY",
                        {"stooq": lambda: _Provider()}):
            bot = _BotRecorder()
            df = pd.DataFrame({"Ticker": ["AAPL", "USD"], "Quantity": [1, 100],
                               "Purchase_Price": [150, 1]})
            df.attrs["_ramp_source"] = "manual"
            await self.tg._run_analysis_background(
                bot=bot, chat_id=self.USER_ID, user_id=self.USER_ID, tier="base",
                cost=1, df=df, bench_tick=None, source="manual",
                broker_fallback="брокер не ответил вовремя")
        self.assertEqual(built, [], "ручной fallback создал клиента Tradernet")
        self.assertTrue(any("Шаг 1 не удался" in t for t, _ in bot.sent),
                        "расчёт дошёл до провайдера цен — ветка проверена")
        offers = [kw for _t, kw in bot.sent if kw.get("reply_markup") is not None]
        self.assertFalse(offers, "ручному отчёту не предлагается ручной же fallback")
        self.assertTrue(await self._slot_is_free())


class FallbackReachesCoveTest(FallbackTestBase):
    """Причина отказа брокера доезжает до `results` → строка CoVe (§I.6)."""

    async def test_reason_is_put_into_results(self) -> None:
        for p in self._extra:
            p.stop()
        self._extra = []
        seen: list[dict] = []

        class _Manager:
            def __init__(self, price_source="freedom") -> None:
                self.engine = SimpleNamespace(price_source=price_source)

            def prefetch_market_data(self, candidates):
                return SimpleNamespace(
                    data=pd.DataFrame({"AAPL": [1.0, 1.1]}), risky_tickers=["AAPL"],
                    history_result=SimpleNamespace(failed={}, retried=[]),
                    loaded_count=1, internal_tickers=set(),
                    resolved_portfolio=["AAPL"], portfolio_loaded=1,
                    portfolio_total=1, proxy_map={})

            def analyze_all(self, df, profile_benchmark=None, risk_mandate=None):
                return {"portfolio_source": "manual"}

        def _payload(results, tier, **_kw):
            seen.append(dict(results))
            raise RuntimeError("стоп после сборки payload")

        with patch("finance.investment_logic.UniversalPortfolioManager", _Manager), \
             patch.object(self.tg, "UniversalPortfolioManager", _Manager), \
             patch.object(self.tg, "run_gatekeeper_for",
                          lambda *_a, **_k: {"critical": [], "warnings": []}), \
             patch.object(self.tg, "_build_pdf_payload", _payload):
            await self.tg._run_analysis_background(
                bot=_BotRecorder(), chat_id=self.USER_ID, user_id=self.USER_ID,
                tier="base", cost=1, df=pd.DataFrame({"Ticker": ["AAPL"]}),
                bench_tick=None, source="manual",
                broker_fallback="брокер не ответил вовремя")
        self.assertEqual(seen[0]["broker_fallback_reason"], "брокер не ответил вовремя")

        from finance.data_lineage import _manual_source_status
        self.assertIn("Freedom Broker был недоступен",
                      _manual_source_status(seen[0])["note"])


class HistoryFailureOfferTest(FallbackTestBase):
    """Шаг 1 брокерского отчёта упал — ручной отчёт ОТДЕЛЬНОЙ кнопкой."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        for p in self._extra:
            p.stop()
        self._extra = []

    async def _run_with_prefetch(self, prefetch) -> _BotRecorder:
        class _Manager:
            def __init__(self, price_source="freedom") -> None:
                self.engine = SimpleNamespace(price_source=price_source)

            prefetch_market_data = prefetch

        bot = _BotRecorder()
        df = pd.DataFrame({"Ticker": ["AAPL"], "Quantity": [1]})
        with patch("finance.investment_logic.UniversalPortfolioManager", _Manager):
            await self.tg._run_analysis_background(
                bot=bot, chat_id=self.USER_ID, user_id=self.USER_ID, tier="deep",
                cost=2, df=df, bench_tick=None, source="freedom")
        return bot

    async def test_stage1_exception_offers_manual(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, STORED)

        def _boom(_self, _c):
            raise RuntimeError("history down")

        bot = await self._run_with_prefetch(_boom)
        datas = [b.callback_data for _t, kw in bot.sent if kw.get("reply_markup")
                 for row in kw["reply_markup"].inline_keyboard for b in row]
        self.assertIn("fb:manual:deep:history", datas)
        self.assertTrue(await self._slot_is_free())

    async def test_no_series_offers_manual(self) -> None:
        await self.db.save_manual_portfolio(self.USER_ID, STORED)

        def _empty(_self, _c):
            return SimpleNamespace(
                data=pd.DataFrame(), risky_tickers=[], history_result=SimpleNamespace(
                    failed={}, retried=[]), loaded_count=0, internal_tickers=set(),
                resolved_portfolio=[], portfolio_loaded=0, portfolio_total=0,
                proxy_map={})

        bot = await self._run_with_prefetch(_empty)
        self.assertTrue(any("ни одной серии" in t for t, _ in bot.sent))
        datas = [b.callback_data for _t, kw in bot.sent if kw.get("reply_markup")
                 for row in kw["reply_markup"].inline_keyboard for b in row]
        self.assertIn("fb:manual:deep:history", datas)


if __name__ == "__main__":
    unittest.main()
