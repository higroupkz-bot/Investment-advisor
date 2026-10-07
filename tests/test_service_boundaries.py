"""Манифест сервисов монорепозитория: ядро · бот отчётов · бот данных (`§−121`).

Два Telegram-бота — это два Cloud Run сервиса из ОДНОГО образа: разные
процессы, разные event loop, разные токены, разные файлы состояния. Сбой
тяжёлого рендера в боте отчётов не трогает загрузчик на рантайме — это
гарантирует инфраструктура. А вот на уровне КОДА гарантию даёт только этот
файл: он делит каждый модуль `src/` на четыре корзины и запрещает стрелки
между сервисами, включая ЛЕНИВЫЕ импорты внутри функций.

Замер `§−121` до правил: 0 нарушений. Разделение уже держалось — но на
соглашении, а соглашение без гейта отрастает обратно (`§−54`, `§−91`).

Корзины
-------
* ``DATA_BOT``   — загрузчик котировок (`ingest_entrypoint` и всё, что только он);
* ``REPORT_BOT`` — бот отчётов (`entrypoint` и всё, что только он);
* ``TOOLING``    — ручные CLI и мёртвые модули: ни один сервис их не грузит;
* ``CORE``       — всё остальное: общий движок, данные, утилиты. Корзина по
  УМОЛЧАНИЮ — поэтому новый модуль, импортирующий слой отчёта, упадёт здесь и
  потребует осознанно положить его в корзину сервиса.

Физический переезд в `apps/report_bot` + `apps/data_bot` — отдельные PR по плану
`docs/ARCHITECTURE_FOR_AGENTS.md §8`; этот манифест — его фаза С-0.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

DATA_BOT = frozenset({
    "ingest_entrypoint", "ingest_bot", "ingest_access",
    "services.quote_ingest", "services.quote_publisher",
})

REPORT_BOT = frozenset({
    "entrypoint", "tg_bot", "db_tokenomics", "services.report_storage",
    "pdf_payload", "premium_payload", "premium_renderer", "html_renderer",
    "ai_narrative", "pdf_charts", "report_charts", "report_mocks",
    "agent.gatekeeper", "agent.rag_engine",
    # Фасад конвейера отчёта (Ф2 веб-миграции): тянет ai_narrative/pdf_payload,
    # поэтому он в боте отчётов, а не в ядре. Веб зовёт его без tg_bot/aiogram.
    "report_pipeline",
    "finance.data_lineage", "finance.scenario_report", "finance.scenario_engine",
    "finance.portfolio_series", "finance.manual_portfolio", "finance.security",
    # Гибрид (I-15): пакет собирает состав из брокера и ручного ввода, а ручной
    # ввод (`finance.manual_portfolio`) живёт в боте отчётов — значит и пакет.
    "portfolio_aggregation", "portfolio_aggregation.aggregator",
    "portfolio_aggregation.edits", "portfolio_aggregation.flags",
    "portfolio_aggregation.gate", "portfolio_aggregation.sources",
})

TOOLING = frozenset({
    "batch_reports", "test_live_api", "finance.setup_vault",
    "finance.tool_plugins", "agent.advisor_bot",
    "freedom_portfolio.__main__", "freedom_portfolio.display",
    "freedom_portfolio.websocket",
})


def _modules() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        parts = list(p.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            out[".".join(parts)] = p
    return out


def _imports(name: str, mods: dict[str, Path]) -> set[str]:
    """Все импорты модуля — и верхнего уровня, и внутри функций."""
    path = mods[name]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - node.level + 1] if node.level > 1 else pkg
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            candidates = [mod] + [f"{mod}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            candidates = [a.name for a in node.names]
        else:
            continue
        for cand in candidates:
            bits = cand.split(".")
            for i in range(len(bits), 0, -1):
                key = ".".join(bits[:i])
                if key in mods:
                    found.add(key)
                    break
    found.discard(name)
    return found


def _closure(root: str, mods: dict[str, Path]) -> set[str]:
    seen: set[str] = set()
    stack = [root]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(_imports(m, mods) - seen)
    return seen


class ServiceManifestTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mods = _modules()
        cls.graph = {m: _imports(m, cls.mods) for m in cls.mods}
        cls.core = set(cls.mods) - DATA_BOT - REPORT_BOT - TOOLING

    def test_manifest_names_real_modules(self) -> None:
        """Опечатка в манифесте молча выключила бы правило для модуля."""
        for bucket in (DATA_BOT, REPORT_BOT, TOOLING):
            self.assertFalse(bucket - set(self.mods), sorted(bucket - set(self.mods)))
        self.assertFalse(DATA_BOT & REPORT_BOT)
        self.assertFalse((DATA_BOT | REPORT_BOT) & TOOLING)

    def test_core_never_reaches_up_into_a_bot(self) -> None:
        bad = sorted((m, n) for m in self.core for n in self.graph[m]
                     if n in DATA_BOT or n in REPORT_BOT)
        self.assertFalse(bad, f"ядро импортирует модуль сервиса: {bad}")

    def test_bots_never_import_each_other(self) -> None:
        bad = sorted((m, n) for m in DATA_BOT for n in self.graph[m] if n in REPORT_BOT)
        bad += sorted((m, n) for m in REPORT_BOT for n in self.graph[m] if n in DATA_BOT)
        self.assertFalse(bad, f"сервис импортирует другой сервис: {bad}")

    def test_service_modules_are_reachable_from_their_entrypoint(self) -> None:
        """Модуль сервиса, недостижимый из его точки входа, — мёртвый или
        положен не в ту корзину."""
        for entry, bucket in (("entrypoint", REPORT_BOT),
                              ("ingest_entrypoint", DATA_BOT)):
            reach = _closure(entry, self.mods)
            with self.subTest(service=entry):
                self.assertFalse(bucket - reach, sorted(bucket - reach))

    def test_tooling_stays_out_of_both_services(self) -> None:
        """Ручной инструмент, ставший импортом сервиса, — уже не инструмент."""
        for entry in ("entrypoint", "ingest_entrypoint"):
            leak = _closure(entry, self.mods) & TOOLING
            with self.subTest(service=entry):
                self.assertFalse(leak, sorted(leak))

    def test_the_scanner_sees_lazy_imports(self) -> None:
        """Инструмент проверен: ленивый импорт внутри функции виден, иначе
        все правила выше пусты для самого частого способа их нарушить."""
        lazy = self.graph["services.quote_ingest"]
        self.assertIn("finance.stooq_provider", lazy)   # импорт внутри функции


if __name__ == "__main__":
    unittest.main()
