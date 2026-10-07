"""Границы модулей исполняемы: приватное — внутреннее, импорты смотрят вниз.

Раунд: архитектурный трек `docs/ARCHITECTURE_FOR_AGENTS.md §4`, фаза Арх-5.

Что охраняется и почему
-----------------------
**1. Ни один модуль `src/` не импортирует `_приватное` имя из другого модуля.**

Пока такие импорты есть, «внутреннего» в проекте не существует: переименование
служебной функции ломает чужой модуль, хотя поведение не изменилось. Замер до
Арх-5 — **27 таких импортов** (построчный grep показывал 8: он не видел
многоимённые `from x import (_a, _b, …)`). Стало 0.

Лечение было не косметическим: имя, которое нужно двум модулям, — это контракт,
поэтому оно стало публичным, а приватный синоним остался алиасом, чтобы не
трогать 37 тест-файлов, которые на него ссылаются.

**2. Слой отчёта не импортирует слой доставки.**

`html_renderer` (L3) импортировал из `tg_bot` (L4) четыре имени — то есть
рендер отчёта зависел от модуля с Telegram-клавиатурами и ради одного
спарклайна затаскивал весь `aiogram`. Билдеры переехали в `report_charts`
(L3), и теперь `tg_bot` разрешено импортировать только `entrypoint` — точке
входа это положено, она поднимает бота.

Почему это тест, а не запись в доке: конвенция без исполняемого гейта в этом
репозитории уже дважды деградировала (`AUDIT §−50`, `§−54`).
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

#: Верхние пакеты/модули проекта — импорт из них считается ВНУТРЕННИМ.
_PROJECT_ROOTS = {
    "finance", "services", "agent", "freedom_portfolio", "pdf_payload",
    "premium_payload", "premium_renderer", "tg_bot", "ai_narrative",
    "html_renderer", "report_mocks", "report_charts", "pdf_charts",
    "batch_reports", "db_tokenomics", "entrypoint", "profile_manager",
    "env_config", "branding", "report_pipeline",
    # Бот-загрузчик котировок (IB-3). Без записи здесь его импорты считались бы
    # ВНЕШНИМИ, и проверка приватных кросс-импортов прошла бы мимо новых
    # модулей, оставшись зелёной.
    "ingest_bot", "ingest_entrypoint", "ingest_access",
    # Гибридный портфель (I-15) — свой пакет, его импорты тоже под правилом.
    "portfolio_aggregation",
}

#: `tg_bot` — слой доставки (L4). Импортировать его вправе только точка входа.
_MAY_IMPORT_TG_BOT = {"entrypoint.py", "tg_bot.py"}


def _iter_src_files():
    for path in sorted(_SRC.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


class NoPrivateCrossModuleImportsTest(unittest.TestCase):
    """`_имя` принадлежит своему модулю. Нужно двоим — сделай публичным."""

    def test_no_private_imports_between_modules(self) -> None:
        offenders: list[str] = []
        for path, tree in _iter_src_files():
            rel = path.relative_to(_SRC.parent)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level or not node.module:
                    continue        # относительный импорт — внутри пакета, это норма
                if node.module.split(".")[0] not in _PROJECT_ROOTS:
                    continue        # stdlib / сторонняя библиотека — не наше дело
                for alias in node.names:
                    name = alias.name
                    if name.startswith("_") and not name.startswith("__"):
                        offenders.append(f"{rel}:{node.lineno} from {node.module} import {name}")
        self.assertFalse(
            offenders,
            "приватные имена не ходят между модулями — нужное двоим делай публичным "
            "(приватный алиас можно оставить ради существующих тестов):\n  "
            + "\n  ".join(offenders),
        )


class ReportLayerDoesNotImportDeliveryTest(unittest.TestCase):
    """L3 (отчёт) не тянет из L4 (бот). Иначе рендер зависит от aiogram."""

    def test_only_entrypoint_imports_tg_bot(self) -> None:
        offenders: list[str] = []
        for path, tree in _iter_src_files():
            if path.name in _MAY_IMPORT_TG_BOT:
                continue
            for node in ast.walk(tree):
                mods: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                for mod in mods:
                    if mod.split(".")[0] == "tg_bot":
                        rel = path.relative_to(_SRC.parent)
                        offenders.append(f"{rel}:{node.lineno} → {mod}")
        self.assertFalse(
            offenders,
            "импорт `tg_bot` за пределами точки входа — это инверсия слоёв "
            "(нижний слой тянет из верхнего):\n  " + "\n  ".join(offenders),
        )

    def test_report_charts_is_free_of_delivery_deps(self) -> None:
        """`report_charts` обязан оставаться лёгким: без aiogram и без бота.

        Смысл переноса именно в этом — иначе смоук-рендер отчёта снова начнёт
        требовать Telegram-стек.

        Проверяется по ИМПОРТАМ, а не поиском подстроки: докстринг модуля
        объясняет, откуда и почему код переехал, и упоминает оба имени. Тест
        на подстроку падал бы на собственной документации — тот же промах,
        что разобран в `AUDIT §−52` (регулярка на «маржинальн» ловила
        операционную маржу вместо плеча).
        """
        tree = ast.parse((_SRC / "report_charts.py").read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
        offenders = [m for m in imported if m.split(".")[0] in {"aiogram", "tg_bot"}]
        self.assertFalse(
            offenders,
            f"`report_charts` не должен зависеть от слоя доставки; найдено: {offenders}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
