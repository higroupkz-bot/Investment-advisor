"""Фасад конвейера отчёта `report_pipeline` (Ф2 веб-миграции OMBRI).

Второй канал доставки (веб) строил отчёт через `tg_bot._build_pdf_payload` и
подменял `engine._tradernet_client` — то есть импортировал слой доставки с
`aiogram` и лез в приватное поле движка. Здесь пинится то, на что веб теперь
опирается: фасад импортируется без бота, источник цен доезжает как есть,
клиент брокера — публичный параметр и не утекает в ручной/демо-портфель (I-12),
а бот зовёт ТЕ ЖЕ функции.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"


class ImportsWithoutDeliveryTest(unittest.TestCase):

    def test_facade_import_does_not_load_bot_or_aiogram(self) -> None:
        """Отдельный процесс: в общем процессе тестов `aiogram` уже загружен другими."""
        code = ("import sys, report_pipeline; "
                "print(sorted(m for m in ('aiogram', 'tg_bot') if m in sys.modules))")
        env = {**os.environ, "PYTHONPATH": str(_SRC)}
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=_SRC,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")

    def test_bot_takes_pipeline_from_facade(self) -> None:
        """Бот не держит вторую копию конвейера: имена приходят из фасада."""
        tree = ast.parse((_SRC / "tg_bot.py").read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module == "report_pipeline"
                 for a in n.names}
        self.assertTrue({"analyze_portfolio", "build_report_payload", "run_gatekeeper_for",
                         "resolve_bench_ticker"} <= names, names)
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertFalse(defined & {"_build_pdf_payload", "_fetch_rag_context",
                                    "_resolve_bench_ticker", "_clean_rag_excerpt", "_kb_banks"})


class AnalyzePortfolioTest(unittest.TestCase):

    def _run(self, **kw):
        import report_pipeline
        built: list[dict] = []

        class _Manager:
            def __init__(self, **ctor):
                built.append(ctor)

            def analyze_all(self, df, profile_benchmark=None, risk_mandate=None):
                return {"bench": profile_benchmark, "mandate": risk_mandate}

        with patch("finance.investment_logic.UniversalPortfolioManager", _Manager):
            res = report_pipeline.analyze_portfolio("df", **kw)
        return built[0], res

    def test_source_and_mandate_pass_through(self) -> None:
        ctor, res = self._run(price_source="manual", bench_ticker="SPY.US",
                              risk_mandate="AGGRESSIVE")
        self.assertEqual(ctor, {"price_source": "manual"})
        self.assertEqual(res, {"bench": "SPY.US", "mandate": "AGGRESSIVE"})

    def test_price_source_is_required(self) -> None:
        import report_pipeline
        with self.assertRaises(TypeError):
            report_pipeline.analyze_portfolio("df")   # дефолт freedom = I-12 для чужого источника

    def test_client_passed_only_when_given(self) -> None:
        client = object()
        ctor, _ = self._run(price_source="freedom", tradernet_client=client)
        self.assertIs(ctor["tradernet_client"], client)


class TradernetClientParameterTest(unittest.TestCase):

    def _client_seen_by_provider(self, source: str, client) -> object:
        from finance.investment_logic import UniversalPortfolioManager
        engine = UniversalPortfolioManager(price_source=source, tradernet_client=client).engine
        seen = {}

        def _factory(src, *, client=None):
            seen["client"] = client
            return object()

        with patch("finance.price_providers.provider_for_source", _factory):
            engine._get_price_provider()
        return seen["client"]

    def test_broker_source_uses_given_client(self) -> None:
        client = object()
        self.assertIs(self._client_seen_by_provider("freedom", client), client)

    def test_manual_and_demo_never_get_the_client(self) -> None:
        """I-12: ключи брокера пользователя не превращают ручной портфель в брокерский."""
        for source in ("manual", "demo"):
            with self.subTest(source=source):
                self.assertIsNone(self._client_seen_by_provider(source, object()))


class GatekeeperForTest(unittest.TestCase):

    def test_without_profile_runs_defaults(self) -> None:
        import report_pipeline
        with patch("agent.gatekeeper.run_gatekeeper", lambda *a, **k: (a, k)):
            self.assertEqual(report_pipeline.run_gatekeeper_for({"r": 1}, None), (({"r": 1},), {}))

    def test_profile_caps_volatility(self) -> None:
        import report_pipeline
        profile = {"target_volatility": 0.10}
        with patch("agent.gatekeeper.run_gatekeeper", lambda *a, **k: k):
            kw = report_pipeline.run_gatekeeper_for({}, profile)
        self.assertAlmostEqual(kw["user_limits"]["max_portfolio_volatility"], 0.12)
        self.assertIs(kw["user_profile"], profile)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
