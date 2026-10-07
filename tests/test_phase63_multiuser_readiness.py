"""`§−122` — готовность к 10+ пользователям: один отчёт не должен держать всех.

Единственный пользователь этих дефектов не видит — они проявляются, когда
отчёты идут ПАРАЛЛЕЛЬНО:

1. **Сборка payload шла в event loop.** `_build_pdf_payload` — это RAG-запрос
   (ChromaDB + ONNX-эмбеддинг, 4 с даже на пустой базе) и HTTP-вызов Anthropic
   (30–120 с для DEEP). В DEEP/BASE-ветке функция вызывалась БЕЗ executor'а —
   при том, что сценарная ветка ту же функцию в executor оборачивает. Пока
   модель писала нарратив одному пользователю, бот не читал апдейты Telegram
   ни для кого: не отвечал на /start, не двигал статус-строки других отчётов.
2. **Загрузка отчёта в GCS шла в event loop** (`upload_report`: PUT + подпись
   URL — сеть) — та же заморозка на секунды на КАЖДЫЙ отчёт.
3. **Stage 2 строила менеджер без источника цен.** `_analyze_existing_portfolio_sync`
   создавал `UniversalPortfolioManager()` с дефолтом `freedom`, хотя Stage 1
   создала его с `price_source=source`. Для `manual`/`demo` это значит: цены
   Stage 2 — из Tradernet (нарушение I-12) и второй сетевой поход за теми же
   рядами. Тест I-12 (`test_price_source_is_never_silently_freedom`) этого не
   видел: он падает на ПЕРВОМ конструкторе и до Stage 2 не доходит — а Stage 2
   берёт имя из модульного импорта `tg_bot`, а не из локального.

Тесты гоняют НАСТОЯЩИЙ `_run_analysis_background` со стабами на границах
(движок, LLM, GCS) и записывают, в каком потоке и с каким источником звали.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.environ.setdefault("OMBRI_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")


class _FakeMessage:
    def __init__(self, text: str = "") -> None:
        self.text = text

    async def edit_text(self, text: str, **_kw):
        self.text = text
        return self


class _FakeBot:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, _chat_id, text: str, **_kw):
        self.sent.append(text)
        return _FakeMessage(text)


def _fake_preview(tickers: list[str]) -> SimpleNamespace:
    data = pd.DataFrame({t: [1.0, 1.01, 1.02] for t in tickers})
    return SimpleNamespace(
        data=data, risky_tickers=list(tickers), resolved_portfolio=list(tickers),
        internal_tickers=set(), loaded_count=len(tickers),
        portfolio_loaded=len(tickers), portfolio_total=len(tickers),
        proxy_map={}, history_result=SimpleNamespace(retried=[], failed={}),
    )


class BackgroundReportIsMultiuserSafeTest(unittest.IsolatedAsyncioTestCase):
    USER_ID = 63001

    async def asyncSetUp(self) -> None:
        import db_tokenomics as db
        import tg_bot
        self.db, self.tg = db, tg_bot
        self._orig_path = db.DB_PATH
        db.DB_PATH = Path(tempfile.mkdtemp(prefix="ramp-mu-")) / "tokenomics.db"
        await db.init_db()
        tg_bot._IN_FLIGHT_USERS.clear()
        self.loop_thread = threading.current_thread()
        self.seen_sources: list[str] = []
        self.payload_thread: list[threading.Thread] = []
        self.upload_thread: list[threading.Thread] = []
        self.render_thread: list[threading.Thread] = []

        outer = self

        class _Recorder:
            """`UniversalPortfolioManager` в той части, которую читает флоу."""

            def __init__(self, price_source: str = "freedom") -> None:
                outer.seen_sources.append(price_source)
                self.engine = SimpleNamespace(price_source=price_source)

            def prefetch_market_data(self, candidates):
                return _fake_preview([c for c in candidates if c != "USD"])

            def analyze_all(self, df, profile_benchmark=None, risk_mandate=None):
                return {"performance_table": pd.DataFrame([{"Ticker": "AAPL"}]),
                        "portfolio_metrics": {"Sharpe_Ratio": 0.5},
                        "total_value": 1000.0}

        def _payload(results, tier, **_kw):
            outer.payload_thread.append(threading.current_thread())
            return {"risk_pct": 10}

        def _render(payload, **_kw):
            outer.render_thread.append(threading.current_thread())
            return "<html/>"

        def _upload(path, **_kw):
            outer.upload_thread.append(threading.current_thread())
            return "https://example.invalid/report.html"

        self._patches = [
            patch.object(tg_bot, "UniversalPortfolioManager", _Recorder),
            patch("finance.investment_logic.UniversalPortfolioManager", _Recorder),
            patch.object(tg_bot, "run_gatekeeper_for",
                         lambda *_a, **_k: {"critical": [], "warnings": []}),
            patch.object(tg_bot, "_build_pdf_payload", _payload),
            patch.object(tg_bot, "render_report_html", _render),
            patch.object(tg_bot, "write_report_html", lambda html, **_k: "/tmp/x.html"),
            patch.object(tg_bot, "upload_report", _upload),
        ]
        for p in self._patches:
            p.start()

    async def asyncTearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self.tg._IN_FLIGHT_USERS.clear()
        self.db.DB_PATH = self._orig_path

    async def _run(self, source: str, tier: str = "deep") -> _FakeBot:
        bot = _FakeBot()
        df = pd.DataFrame({"Ticker": ["AAPL", "USD"], "Quantity": [1, 100]})
        await self.tg._run_analysis_background(
            bot=bot, chat_id=self.USER_ID, user_id=self.USER_ID, tier=tier,
            cost=0, df=df, bench_tick=None, source=source)
        return bot

    async def test_every_stage_uses_the_chosen_price_source(self) -> None:
        for source in ("manual", "demo", "freedom"):
            with self.subTest(source=source):
                self.seen_sources.clear()
                await self._run(source)
                self.assertEqual(len(self.seen_sources), 2,
                                 "ожидались ДВА конструктора: Stage 1 и Stage 2")
                self.assertEqual(self.seen_sources, [source, source])

    async def test_payload_build_leaves_the_event_loop(self) -> None:
        """LLM + RAG — не в потоке loop'а: иначе один DEEP замораживает бота."""
        await self._run("freedom")
        self.assertEqual(len(self.payload_thread), 1)
        self.assertIsNot(self.payload_thread[0], self.loop_thread)

    async def test_render_and_upload_leave_the_event_loop(self) -> None:
        await self._run("freedom")
        self.assertEqual(len(self.upload_thread), 1)
        self.assertIsNot(self.upload_thread[0], self.loop_thread)
        self.assertEqual(len(self.render_thread), 1)
        self.assertIsNot(self.render_thread[0], self.loop_thread)

    async def test_report_is_delivered_and_slot_released(self) -> None:
        """Стабы не сломали доставку: ссылка ушла, слот свободен."""
        bot = await self._run("freedom")
        self.assertTrue(any("Открыть отчёт" in t for t in bot.sent))
        self.assertNotIn(self.USER_ID, self.tg._IN_FLIGHT_USERS)
        self.assertTrue(await self.tg._try_acquire_user_slot(self.USER_ID))
        await self.tg._release_user_slot(self.USER_ID)


# ═══════════════ Общий потолок одновременных расчётов ═══════════════

class ReportGateTest(unittest.IsolatedAsyncioTestCase):
    """Слот — по пользователю; гейт — на всех. Лишние ждут и знают, сколько
    перед ними. Семафор подменяется на свежий: asyncio привязывает его к
    loop'у при первом ожидании, а у каждого теста loop свой."""

    async def asyncSetUp(self) -> None:
        import tg_bot
        self.tg = tg_bot
        self._orig = (tg_bot._REPORT_GATE, tg_bot._REPORTS_RUNNING,
                      tg_bot._REPORTS_WAITING)
        tg_bot._REPORT_GATE = asyncio.Semaphore(1)
        tg_bot._REPORTS_RUNNING = 0
        tg_bot._REPORTS_WAITING = 0

    async def asyncTearDown(self) -> None:
        (self.tg._REPORT_GATE, self.tg._REPORTS_RUNNING,
         self.tg._REPORTS_WAITING) = self._orig

    async def test_second_report_waits_and_is_told_its_place(self) -> None:
        tg = self.tg
        b1, b2, b3 = _FakeBot(), _FakeBot(), _FakeBot()
        await tg._enter_report_gate(b1, 1)           # единственное место
        self.assertEqual(b1.sent, [])                # свободно — без сообщений
        t2 = asyncio.create_task(tg._enter_report_gate(b2, 2))
        t3 = asyncio.create_task(tg._enter_report_gate(b3, 3))
        await asyncio.sleep(0.05)
        self.assertFalse(t2.done() or t3.done())
        self.assertTrue(any("в очереди" in m for m in b2.sent), b2.sent)
        self.assertTrue(any("перед вами ещё 1" in m for m in b3.sent), b3.sent)
        self.assertEqual(tg._REPORTS_WAITING, 2)
        self.assertEqual(tg._REPORTS_RUNNING, 1)
        tg._leave_report_gate()
        await asyncio.wait_for(t2, 1.0)
        await asyncio.sleep(0.01)
        self.assertFalse(t3.done())                  # FIFO: третий всё ещё ждёт
        tg._leave_report_gate()
        await asyncio.wait_for(t3, 1.0)
        tg._leave_report_gate()
        self.assertEqual((tg._REPORTS_WAITING, tg._REPORTS_RUNNING), (0, 0))
        self.assertFalse(tg._REPORT_GATE.locked())

    async def test_stuck_queue_is_loud_in_the_log(self) -> None:
        """Гейт, который никто не отпустил, — тихий отказ. Ожидание дольше
        порога обязано писать WARNING, не теряя места в очереди."""
        tg = self.tg
        orig = tg._QUEUE_WARN_S
        tg._QUEUE_WARN_S = 0.05
        try:
            await tg._enter_report_gate(_FakeBot(), 1)       # занято
            with self.assertLogs("ombri.bot", level="WARNING") as cm:
                waiter = asyncio.create_task(tg._enter_report_gate(_FakeBot(), 2))
                await asyncio.sleep(0.18)
                self.assertFalse(waiter.done())
                tg._leave_report_gate()
                await asyncio.wait_for(waiter, 1.0)
            self.assertTrue(any("гейт не отпущен" in m for m in cm.output), cm.output)
            tg._leave_report_gate()
            self.assertEqual((tg._REPORTS_WAITING, tg._REPORTS_RUNNING), (0, 0))
        finally:
            tg._QUEUE_WARN_S = orig

    async def test_gate_is_released_when_stage_one_fails(self) -> None:
        """Упавший расчёт обязан вернуть место — иначе очередь встаёт навсегда."""
        import db_tokenomics as db
        tg = self.tg
        orig_path = db.DB_PATH
        db.DB_PATH = Path(tempfile.mkdtemp(prefix="ramp-gate-")) / "tokenomics.db"
        await db.init_db()
        tg._IN_FLIGHT_USERS.clear()

        class _Broken:
            def __init__(self, price_source: str = "freedom") -> None:
                pass

            def prefetch_market_data(self, _c):
                raise RuntimeError("брокер лёг")

        try:
            with patch.object(tg, "UniversalPortfolioManager", _Broken), \
                 patch("finance.investment_logic.UniversalPortfolioManager", _Broken):
                bot = _FakeBot()
                await tg._run_analysis_background(
                    bot=bot, chat_id=7, user_id=7, tier="base", cost=1,
                    df=pd.DataFrame({"Ticker": ["AAPL"]}), bench_tick=None,
                    source="freedom")
            self.assertTrue(any("Шаг 1 не удался" in m for m in bot.sent))
            self.assertFalse(tg._REPORT_GATE.locked())
            self.assertEqual(tg._REPORTS_RUNNING, 0)
        finally:
            db.DB_PATH = orig_path
            tg._IN_FLIGHT_USERS.clear()


# ═══════════════ Аренды слотов: graceful shutdown снимает СВОИ ═══════════════

class OwnerLeaseReleaseTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        import db_tokenomics as db
        self.db = db
        self._orig = db.DB_PATH
        db.DB_PATH = Path(tempfile.mkdtemp(prefix="ramp-lease-")) / "tokenomics.db"
        await db.init_db()

    async def asyncTearDown(self) -> None:
        self.db.DB_PATH = self._orig

    async def test_only_the_owners_leases_are_removed(self) -> None:
        db = self.db
        self.assertTrue(await db.acquire_report_lock(1, "inst-A"))
        self.assertTrue(await db.acquire_report_lock(2, "inst-A"))
        self.assertTrue(await db.acquire_report_lock(3, "inst-B"))
        self.assertEqual(await db.release_report_locks_for_owner("inst-A"), 2)
        # Пользователи A свободны сразу, а не через TTL…
        self.assertTrue(await db.acquire_report_lock(1, "inst-C"))
        self.assertTrue(await db.acquire_report_lock(2, "inst-C"))
        # …а держатель B не задет.
        self.assertFalse(await db.acquire_report_lock(3, "inst-C"))
        self.assertEqual(await db.release_report_locks_for_owner("inst-A"), 0)

    async def test_shutdown_watcher_calls_the_owner_release(self) -> None:
        """Связь с ботом — по коду: снятие стоит в `_watch_shutdown` после
        закрытия сессии и адресовано `_INSTANCE_ID`."""
        import inspect
        import tg_bot
        src = inspect.getsource(tg_bot.main)
        after_close = src.split("bot.session.close()", 1)[1]
        self.assertIn("release_report_locks_for_owner(_INSTANCE_ID)", after_close)


# ═══════════════ Потолок ожидания Anthropic ═══════════════

class AnthropicTimeoutTest(unittest.TestCase):

    def test_client_gets_a_bounded_timeout(self) -> None:
        import ai_narrative
        seen: dict = {}

        class _Client:
            def __init__(self, **kw):
                seen.update(kw)
                self.messages = SimpleNamespace(
                    create=lambda **_k: (_ for _ in ()).throw(RuntimeError("stub")))

        results = {"portfolio_metrics": {}, "total_value": 1.0,
                   "performance_table": pd.DataFrame([{"Ticker": "AAPL"}])}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}), \
             patch("anthropic.Anthropic", _Client):
            out = ai_narrative.generate_narrative(results, tier="base")
        self.assertEqual(seen.get("timeout"), ai_narrative.ANTHROPIC_TIMEOUT_S)
        self.assertLessEqual(ai_narrative.ANTHROPIC_TIMEOUT_S, 1800.0)
        self.assertEqual(out.get("model_used"), "fallback")   # отчёт всё равно есть


# ═══════════════ Деплой пинит лимиты явно ═══════════════

class DeployPinsConcurrencyTest(unittest.TestCase):
    """`gcloud run deploy --set-env-vars` ЗАМЕНЯЕТ весь набор переменных
    (урок `PREMIUM_REPORT_ENABLED`, 2026-06-27): значение, не записанное в
    `cloudbuild.yaml`, живёт ровно до следующего деплоя."""

    def test_limits_are_pinned_in_cloudbuild(self) -> None:
        path = Path(__file__).resolve().parent.parent / "cloudbuild.yaml"
        if not path.exists():
            self.skipTest("cloudbuild.yaml не входит в деплой-образ (гейт репозитория)")
        text = path.read_text(encoding="utf-8")
        self.assertIn("MAX_CONCURRENT_REPORTS=", text)
        self.assertIn("ANTHROPIC_TIMEOUT_S=", text)


if __name__ == "__main__":
    unittest.main()
