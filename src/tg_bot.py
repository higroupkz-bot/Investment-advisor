"""
OMBRI Telegram Bot — aiogram 3.x async entry point.

Deep-link format: t.me/<bot>?start=<source_slug>  (хэндл — `branding.bot_username()`,
  переопределяется env `BOT_USERNAME`). `start=scn_<n>` → «Применить идею» → Scenario tier.
Analysis tiers:
  - base  : 1 token  → MAC3 CVaR + allocation table
  - deep  : 2 tokens → base + scenario analysis + fundamental signals

Onboarding FSM (new users only):
  Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Universe → Benchmark → MandateReview → Connection → Analysis

PortfolioConnection FSM:
  connect:template → save mode → Analysis menu
  connect:freedom  → Login → ApiKey → save encrypted → Analysis menu
"""

import asyncio
import logging
import os
import re
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import (
    TelegramConflictError,
    TelegramNetworkError,
    TelegramServerError,
)
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

import branding
from env_config import env_int
from db_tokenomics import (
    acquire_report_lock,
    approve_mandate,
    assert_persistent_state,
    credit_tokens,
    deduct_tokens,
    delete_manual_draft,
    delete_manual_portfolio,
    InsufficientFundsError,
    get_balance,
    get_benchmark_ticker,
    get_connection_mode_explicit,
    get_manual_draft,
    get_manual_portfolio,
    get_profile,
    get_last_report_snapshot,
    init_db,
    init_user,
    MANUAL_DRAFT_MAX_BYTES,
    ManualDraftTooLarge,
    release_report_lock,
    release_report_locks_for_owner,
    save_benchmark_ticker,
    save_connection_mode,
    save_manual_draft,
    save_manual_portfolio,
    save_profile,
    save_report_snapshot,
)
from finance.broker_api import (
    BrokerAuthError,
    BrokerEmptyPortfolioError,
    FreedomConnector,
    RealPortfolioRequired,
)
from finance.data_checks import DataQualityBlocked
from finance.investment_logic import UniversalPortfolioManager
from finance.security import SecureVault, MasterKeyRotatedError
# Гибрид (I-15): брокер + ручной ввод. Пакет — L1, импорт вниз.
from portfolio_aggregation import (
    AGGREGATED_SOURCE,
    AggregatedNotPermitted,
    AggregationRefused,
    FreedomSource,
    KEY_ORIGIN_ADMIN_SERVICE,
    KEY_ORIGIN_VAULT,
    ManualSource,
    PortfolioAggregator,
    aggregated_manager,
    count_positions,
    unpriced_positions,
    load_with_budget,
    apply_edit as _mp_apply_edit,
    broker_fetch_budget_s,
    canonical_text as _mp_canonical_text,
    entries_of as _mp_entries_of,
    FLAG_ADMINS,
    FLAG_ON,
    HYBRID_PORTFOLIO_ENV,
    manual_max_positions,
    rollout_mode,
    remove_at as _mp_remove_at,
    version_tag as _mp_version_tag,
)
from html_renderer import render_report_html, write_report_html
from services.report_storage import upload_report
from pdf_payload import TIER_BASE, TIER_DEEP, TIER_SCENARIO
# Ф2 веб-миграции: анализ → гейткипер → payload живут в фасаде `report_pipeline`,
# его же зовёт веб. Приватные имена — алиасы ради существующих тестов.
from report_pipeline import (
    analyze_portfolio,
    build_report_payload as _build_pdf_payload,
    clean_rag_excerpt as _clean_rag_excerpt,
    fetch_rag_context as _fetch_rag_context,
    kb_banks as _kb_banks,
    resolve_bench_ticker as _resolve_bench_ticker,
    run_gatekeeper_for,
)
from profile_manager import (
    ASSET_DISPLAY, ASSET_KEYS, BENCHMARK_LIST,
    PROFILE_BENCH_TICKER, RiskProfileManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger("ombri.bot")

#: Токен ЭТОГО бота. Проект переехал с имени RAMP на OMBRI (`§−101`), поэтому
#: имя переменной объявлено, а не вписано в три места: сообщение об ошибке,
#: страж общего токена у загрузчика и тесты читают его отсюда.
TOKEN_ENV = "OMBRI_BOT_TOKEN"
#: Прежнее имя той же переменной. Читается ВТОРЫМ и с предупреждением: пока
#: секрет в Secret Manager не переименован, бот обязан подниматься. Молча
#: принимать прежнее имя нельзя — тогда переезд не закончится никогда.
LEGACY_TOKEN_ENV = "RAMP_BOT_TOKEN"


def read_bot_token() -> str:
    """Токен бота: новое имя, при его отсутствии — прежнее, с предупреждением.

    🔴 Читается на уровне МОДУЛЯ, и это fail-closed по замыслу: бот без токена
    не «работает частично», он не работает. Прежнее чтение было
    `os.environ["RAMP_BOT_TOKEN"]` — отсутствие переменной роняло ИМПОРТ с
    голым `KeyError`, по симптому неотличимым от гонки импортов (`§−98`).
    Здесь причина названа прямо.
    """
    value = str(os.getenv(TOKEN_ENV, "") or "").strip()
    if value:
        return value
    value = str(os.getenv(LEGACY_TOKEN_ENV, "") or "").strip()
    if value:
        logger.warning(
            "%s пуст — взял устаревший %s. Переименуйте секрет в Secret Manager "
            "и переключите подстановку `_BOT_TOKEN_SECRET` в cloudbuild.yaml.",
            TOKEN_ENV, LEGACY_TOKEN_ENV)
        return value
    raise RuntimeError(
        f"{TOKEN_ENV} не задан (прежнее имя {LEGACY_TOKEN_ENV} тоже пусто). "
        "Боту нужен токен от @BotFather: в проде он приезжает из Secret "
        "Manager через `--set-secrets`, локально — из .env.")


def md_safe(text: str) -> str:
    """Экранирует служебные символы legacy-Markdown в ПОДСТАВЛЯЕМОМ значении.

    🔴 Telegram в режиме `Markdown` разбирает `_..._` где угодно, включая
    середину слова. Хэндл вида `@OMBRI_support_bot` несёт ДВА подчёркивания —
    они схлопываются в курсив, и пользователь видит `@OMBRIsupportbot`, то
    есть КОПИРУЕТ несуществующий контакт. Непарное подчёркивание хуже:
    Telegram отвечает 400 «can\'t parse entities», и сообщение не уходит
    вовсе. Дефолтные имена бренда (`OMBRIOS`, `OMBRI`) безопасны, но значение
    приходит из окружения и может быть любым — поэтому экранируем.
    """
    return re.sub(r"([_*`\[])", r"\\\1", text)


BOT_TOKEN: str = read_bot_token()
if ":" not in BOT_TOKEN:
    # Never echo any part of the secret — only its length on the error path.
    raise ValueError(
        f"Invalid {TOKEN_ENV} format — expected 'id:secret' (len={len(BOT_TOKEN)})"
    )
# The numeric id BEFORE ':' is public (it's the bot's user id); the secret
# AFTER ':' must never be logged.  Log only the public id segment.
logger.info("Starting bot id=%s", BOT_TOKEN.split(":", 1)[0])
# B1: vault DB path is overridable via env so the encrypted broker
# credentials live on the persistent gcsfuse volume (/mnt/state) in prod.
# Without this every Cloud Run redeploy wipes the file and forces every
# user through re-onboarding (which itself re-leaks their keys into chat).
# Local dev keeps the historical repo-relative default.
VAULT_DB: str = os.getenv(
    "VAULT_DB_PATH",
    str(Path(__file__).parent.parent / "data" / "users_vault.db"),
)

# ── FSM ───────────────────────────────────────────────────────────────────────

class Onboarding(StatesGroup):
    Q1            = State()
    Q2            = State()
    Q3            = State()
    Q4            = State()
    Q5            = State()
    Q6            = State()
    Universe      = State()
    Benchmark     = State()
    MandateReview = State()


class PortfolioConnection(StatesGroup):
    Login     = State()
    ApiKey    = State()
    SecretKey = State()


class AnalysisFlow(StatesGroup):
    awaiting_approval = State()


class ManualPortfolio(StatesGroup):
    """Ручной ввод портфеля (Manual Portfolio, Фаза 5).

    Два состояния, потому что решений у пользователя ровно два: что ввести и
    согласен ли он с тем, как мы это поняли. Экран подтверждения между ними
    обязателен по §3.4 задания: опечатка в количестве (лишний ноль) не видна в
    числах отчёта, но полностью искажает веса, TRC и CVaR — доля позиции на
    экране делает её заметной глазом.
    """

    Input   = State()   # ждём текст (файл — Фаза 7)
    Confirm = State()   # показан экран подтверждения
    Edit    = State()   # гибрид PR-1: точечные правки сохранённого портфеля (+/−)


class MandateEdit(StatesGroup):
    """B1 (2026-07-17) — точечное редактирование мандата из /mandate.

    Benchmark/Universe переиспользуют состояния Onboarding.* с флагом
    ``edit_mode=True`` в FSM-данных (обработчики ob:* ветвятся по флагу);
    отдельное состояние нужно только выбору риск-профиля вручную.
    """
    Profile = State()


# ── Onboarding question definitions ──────────────────────────────────────────
# Each option tuple: (button_label, score_points, callback_data)

NUM_QUESTIONS = 6

QUESTIONS: list[dict] = [
    {
        "state": Onboarding.Q1,
        "text":  "🕐 *Вопрос 1 из 6*\nКаков ваш инвестиционный горизонт?",
        "options": [
            ("Менее 1 года",                         1, "ob:q1:1"),
            ("От 1 до 3 лет",                        2, "ob:q1:2"),
            ("Более 3 лет",                          3, "ob:q1:3"),
        ],
    },
    {
        "state": Onboarding.Q2,
        "text":  "🎯 *Вопрос 2 из 6*\nВаша главная инвестиционная цель?",
        "options": [
            ("Сохранение капитала",                  1, "ob:q2:1"),
            ("Умеренный рост",                       2, "ob:q2:2"),
            ("Максимальный рост",                    3, "ob:q2:3"),
        ],
    },
    {
        "state": Onboarding.Q3,
        "text":  "📉 *Вопрос 3 из 6*\nЕсли ваш портфель упадёт на 20%, вы:",
        "options": [
            ("Продам всё, чтобы остановить потери",  1, "ob:q3:1"),
            ("Подожду восстановления",               2, "ob:q3:2"),
            ("Докуплю на просадке",                  3, "ob:q3:3"),
        ],
    },
    {
        "state": Onboarding.Q4,
        "text":  "📚 *Вопрос 4 из 6*\nВаш опыт в инвестировании?",
        "options": [
            ("Нет опыта",                            1, "ob:q4:1"),
            ("До 3 лет",                             2, "ob:q4:2"),
            ("Более 3 лет / профессионал",           3, "ob:q4:3"),
        ],
    },
    {
        "state": Onboarding.Q5,
        "text":  "🛡 *Вопрос 5 из 6*\nНасколько комфортно вы себя чувствуете финансово?",
        "options": [
            ("Живу от зарплаты до зарплаты",         1, "ob:q5:1"),
            ("Есть накопления на несколько месяцев", 2, "ob:q5:2"),
            ("Финансово защищён(а) на год и более",  3, "ob:q5:3"),
        ],
    },
    {
        "state": Onboarding.Q6,
        "text":  "💼 *Вопрос 6 из 6*\nНасколько стабилен ваш доход?",
        "options": [
            ("Нестабильный / фриланс",               1, "ob:q6:1"),
            ("Стабильная зарплата",                  2, "ob:q6:2"),
            ("Несколько источников дохода",           3, "ob:q6:3"),
        ],
    },
]

# ── Constants ─────────────────────────────────────────────────────────────────

# Токен-тариф (2026-07-07): base 1 · scenario 1 · deep 2.
TIER_COST  = {"base": 1, "scenario": 1, "deep": 2}
TIER_LABEL = {"base": "Базовый отчёт",
              "scenario": "Сценарный анализ",
              "deep": "Глубокий анализ"}
TIER_SCENARIO_LABEL = TIER_LABEL[TIER_SCENARIO]

# Цена токена (ВАЖНОЕ изменение 2026-07-17): 1 токен = 2 500 ₸ (KZT);
# пакет — 10 токенов за 25 000 ₸.  Единственный источник правды для копирайта
# /balance, /topup и /help (менять ЗДЕСЬ, не в текстах).
TOKEN_PRICE_KZT      = 2_500
TOKEN_PACK_TOKENS    = 10
TOKEN_PACK_PRICE_KZT = TOKEN_PRICE_KZT * TOKEN_PACK_TOKENS   # 25 000 ₸

SOURCE_GREETING: dict[str, str] = {
    "news_apple": "новостей Apple",
    "news_kaspi":  "новостей Kaspi",
}

# ── Manual Portfolio (Фаза 5) ────────────────────────────────────────────────
MANUAL_PORTFOLIO_ENV = "MANUAL_PORTFOLIO_ENABLED"


def _rollout_enabled(env_name: str, user_id: int | None) -> bool:
    """Флаг раскатки для КОНКРЕТНОГО пользователя (`§−124`).

    `on` — всем; `admins` — только `ADMIN_USER_IDS`; иначе никому. Без
    `user_id` (клавиатура без контекста, список команд) видно только `on`:
    неизвестный пользователь — не администратор.
    """
    mode = rollout_mode(env_name)
    if mode == FLAG_ON:
        return True
    if mode == FLAG_ADMINS:
        return user_id is not None and _is_admin(int(user_id))
    return False


def manual_portfolio_enabled(user_id: int | None = None) -> bool:
    """Флаг отката ручного ввода. По умолчанию — ВЫКЛЮЧЕН.

    Значения: `on` (всем), `admins` (только `ADMIN_USER_IDS` — ступень раскатки,
    `§−124`), остальное — выключено. Проверка пофамильная, поэтому каждый
    вызов, у которого есть пользователь, обязан его передать.

    Значение читается ФУНКЦИЕЙ, а не константой модуля: константа защёлкнулась
    бы на импорте, и ни один тест не смог бы проверить оба состояния бота
    (правило §6.4 плана, образец — `factor_orthogonalize_enabled`).

    Почему дефолт «off», хотя фаза сдана. 🔴 Прежняя редакция докстринга
    ссылалась на отсутствие `StooqProvider` — это обоснование ПРОТУХЛО: Фаза 9
    доставлена (`§−84`), провайдер существует, и `provider_for_source("manual")`
    возвращает именно его. Решение держать флаг выключенным осталось, но
    держится оно на другом: включение упирается не в код, а в ДАННЫЕ и ПРАВО —
    чек-лист `OPERATOR_STOOQ §13.1` (глубина и покрытие залитой базы, замер
    конвенции, условия использования, живой прогон отчёта глазами).

    Разница существенна для того, кто будет флаг включать: раньше выходило, что
    ждать надо разработчика, теперь — что оператора. Устаревшее обоснование
    опаснее отсутствующего, оно убеждает не проверять (`§−97` D-5).
    """
    return _rollout_enabled(MANUAL_PORTFOLIO_ENV, user_id)


def hybrid_portfolio_enabled(user_id: int | None = None) -> bool:
    """Флаг меню источников и агрегированного отчёта (`HYBRID_PORTFOLIO_ENABLED`).

    Дефолт — ВЫКЛЮЧЕН (I-9). Требует включённого ручного ввода: агрегированный
    отчёт без ручного портфеля бессмыслен, а ручной портфель без флага ручного
    ввода недоступен. Читается функцией — по той же причине, что и соседний флаг.
    """
    return (manual_portfolio_enabled(user_id)
            and _rollout_enabled(HYBRID_PORTFOLIO_ENV, user_id))


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _source_label(slug: str) -> str:
    return SOURCE_GREETING.get(slug, f"канала «{slug}»")


def _get_keys_sync(user_id: int):
    """Blocking helper — must be run in a thread executor."""
    vault = SecureVault(db_name=VAULT_DB)
    return vault.get_user_keys(str(user_id))


def _save_keys_sync(user_id: int, login: str, api_key: str, secret_key: str) -> None:
    """Blocking helper — must be run in a thread executor."""
    vault = SecureVault(db_name=VAULT_DB)
    vault.save_user_keys(str(user_id), login, api_key, secret_key)


def _has_vault_keys_sync(user_id: int) -> bool:
    """Blocking helper — existence check only, never decrypts (Fix B)."""
    vault = SecureVault(db_name=VAULT_DB)
    return vault.has_user(str(user_id))


async def _resolve_portfolio_source(user_id: int) -> tuple[str, str | None]:
    """Resolve the portfolio source for a report request (Fixes B + C).

    Returns ``(source, stored_mode)`` where source is one of:
      * ``'freedom'``      — live broker portfolio;
      * ``'manual'``       — the user typed the portfolio in by hand (Фаза 5);
      * ``'demo'``         — the user EXPLICITLY chose the template portfolio;
      * ``'undetermined'`` — no explicit choice and no keys → the caller must
        refuse to build a (paid) report and ask the user to pick a source.

    Vault keys ALWAYS win (product decision 2026-07-16): stored keys are proof
    the user linked their broker, so even a lost/'template' stored mode
    resolves to 'freedom' — and the mode is self-healed back to 'freedom' so
    the recovery stops firing (incident 2026-07-14, user 88202680).
    """
    stored = await get_connection_mode_explicit(user_id)
    if stored == "freedom":
        # Vault check is redundant here — the freedom branch resolves keys
        # itself (re-link message / admin service keys when the vault is
        # empty), so we skip a vault open on the common path.
        return "freedom", stored
    # F-4 (Фаза 5 §2.1): ветка manual стоит ДО проверки vault, и порядок здесь
    # содержательный.  Само-лечение ниже трактует ключи в vault как
    # доказательство привязки брокера — но пользователь, у которого ключи
    # когда-то БЫЛИ, а сегодня он вводит портфель руками, получил бы молчаливый
    # возврат в freedom и не смог бы построить ручной отчёт вообще.  Явный,
    # только что сделанный выбор свежее, чем факт наличия ключей.
    #
    # Для 'template' и None само-лечение сохраняется в полной силе: его смысл
    # («ключи = доказательство привязки») не нарушается, и регресс инцидента
    # 2026-07-14 закрыт теми же тестами, что и раньше.
    if stored == "manual":
        return "manual", stored
    loop = asyncio.get_running_loop()
    if await loop.run_in_executor(None, _has_vault_keys_sync, user_id):
        logger.warning(
            "CONN MODE RECOVERED user=%s: режим был %r, но в vault есть "
            "ключи → freedom (само-лечение).", user_id, stored,
        )
        await save_connection_mode(user_id, "freedom")
        return "freedom", stored
    if stored == "template":
        return "demo", stored
    return "undetermined", stored


def _effective_cost(tier: str, source: str) -> int:
    """Token price of a report given its portfolio source.

    Демо-отчёты бесплатны (product decision 2026-07-16) — платится только
    однозначно определённый ЖИВОЙ источник.
    """
    return 0 if source == "demo" else TIER_COST[tier]


def _fetch_portfolio_sync(api_key: str, secret_key: str = "", login: str = ""):
    """Blocking: только подключение к брокеру и загрузка позиций (без анализа)."""
    return FreedomConnector(api_key, secret_key, login).fetch_portfolio()


def _analyze_existing_portfolio_sync(df, bench_ticker: str | None = None,
                                     risk_mandate: str | None = None,
                                     price_source: str = "freedom") -> dict:
    """Blocking: только MAC3 анализ уже загруженного DataFrame (фасад `report_pipeline`).

    🔴 `§−122`: `price_source` ОБЯЗАН совпадать со Stage 1. Прежде здесь
    строился `UniversalPortfolioManager()` с дефолтом `freedom`, и Stage 2
    ходила за ценами в Tradernet даже для `manual`/`demo` (I-12) — второй
    сетевой поход за теми же рядами. Тест I-12 этого не видел: он падал на
    первом конструкторе и до этой строки не доходил.
    """
    return analyze_portfolio(df, price_source=price_source, bench_ticker=bench_ticker,
                             risk_mandate=risk_mandate)


def _format_portfolio_preview(df) -> str:
    """
    Сжатая Markdown-таблица портфеля для Telegram-сообщения.

    Колонки: Тикер | Тип | Кол-во | Сумма | Валюта.

    Тип инструмента берётся из ``df["Asset_Type"]`` если он есть (поставлен
    ``broker_api.classify_instrument``); иначе вычисляется на лету.

    🔴 **Итог считается ПО ВАЛЮТАМ, а не одной строкой** (`AUDIT §−86`).
    Прежняя редакция складывала `Quantity × Purchase_Price` по всем позициям и
    подписывала результат «Сумма $». Портфель с тенговой бумагой давал в превью
    900 000 «долларов» вместо примерно 1 800 — и показывалось это ровно перед
    тем, как пользователь соглашался на расчёт. Дефект был и на брокерском пути
    (`KSPI.KZ` приходит от брокера в тенге), но ручной ввод — его главный
    источник: тенговая книга и есть целевая аудитория.

    Курс здесь НЕ спрашивается сознательно: перевод — это математика, а её
    место в `finance/*`, не в слое доставки. Экран подтверждения ручного ввода
    уже показывает корректный пересчёт (`build_confirmation`), а превью
    отвечает на другой вопрос — «правильно ли мы поняли ваши позиции».
    """
    if df is None or df.empty:
        return "_(портфель пуст)_"

    df2 = df.copy()
    if "Quantity" in df2.columns and "Purchase_Price" in df2.columns:
        df2["_value"] = df2["Quantity"] * df2["Purchase_Price"]
    else:
        df2["_value"] = df2.get("Quantity", 0)

    df2 = df2.sort_values("_value", ascending=False)

    # Source of ticker: column "Ticker" (RangeIndex case) or index when named.
    if "Ticker" in df2.columns:
        tickers = df2["Ticker"].astype(str).tolist()
    else:
        tickers = [str(i) for i in df2.index.tolist()]

    # Source of type: column "Asset_Type" if present, else heuristic on ticker.
    if "Asset_Type" in df2.columns:
        types = df2["Asset_Type"].astype(str).tolist()
    else:
        try:
            from finance.broker_api import classify_instrument
            types = [classify_instrument(t) for t in tickers]
        except Exception:
            types = ["—"] * len(tickers)

    if "Currency" in df2.columns:
        currencies = [str(c or "").upper() or "—" for c in df2["Currency"]]
    else:
        # Колонки нет — у брокерского фрейма она появляется не всегда.
        # Молчаливый доллар был бы тем же враньём, только тише, поэтому прочерк.
        currencies = ["—"] * len(df2)

    rows = ["```",
            f"{'Тикер':<12} {'Тип':<10} {'Кол-во':>10} {'Сумма':>14}  Валюта",
            "─" * 56]

    totals: dict[str, float] = {}
    for i, (_, row) in enumerate(df2.iterrows()):
        ticker = tickers[i] if i < len(tickers) else "?"
        kind   = types[i]   if i < len(types)   else "—"
        ccy    = currencies[i] if i < len(currencies) else "—"
        qty    = row.get("Quantity", 0) or 0
        value  = row.get("_value", 0) or 0
        totals[ccy] = totals.get(ccy, 0.0) + float(value)
        rows.append(f"{ticker[:12]:<12} {kind[:10]:<10} {qty:>10.2f} "
                    f"{value:>14.2f}  {ccy}")

    rows.append("─" * 56)
    rows.append(f"{'Позиций':<12} {'':<10} {len(df2):>10}")
    for ccy, total in sorted(totals.items(), key=lambda kv: -kv[1]):
        rows.append(f"{'Итого':<12} {'':<10} {'':>10} {total:>14.2f}  {ccy}")
    rows.append("```")
    return "\n".join(rows)


# ── Визуальные блоки отчёта (алиасы) ──────────────────────────────────────────

# Арх-5: визуальные блоки отчёта переехали в `report_charts` (слой L3).
# С Ф2 их вызывает `report_pipeline.build_report_payload`, а здесь остались
# прежние ПРИВАТНЫЕ имена — на них ссылаются существующие тесты.  Публичные имена
# существуют ради `html_renderer`: раньше он импортировал их ОТСЮДА, то есть
# слой отчёта тянул наверх, в слой доставки, и ради спарклайна затаскивал
# весь `aiogram`.  `ARCHITECTURE_FOR_AGENTS.md` §3.1.
from report_charts import (
    RADAR_FACTOR_AXES as _RADAR_FACTOR_AXES,
    BENCH_FACTOR_BETAS as _BENCH_FACTOR_BETAS,
    sparkline_svg as _sparkline_svg,
    build_kpi_sparklines as _build_kpi_sparklines,
    build_equity_curve_svg as _build_equity_curve_svg,
    compute_factor_betas as _compute_factor_betas,
    build_factor_radar_svg as _build_factor_radar_svg,
    build_factor_betas_table as _build_factor_betas_table,
)


def _broker_outage_advice(reason: str, error_id: str) -> str:
    """Совет пользователю ПО ПРИЧИНЕ отказа брокера, а не один на все случаи.

    §−94: все три ветки `broker_api.fetch_portfolio` печатали «обычно это
    проходит за 5–15 минут». Для `waf_block` это неправда — Cloudflare отбивает
    ПО IP, и само оно не рассосётся (рестарт лишь перекатывает IP из общего
    пула, отсюда же «работало и вдруг перестало»). Для `parse_error` это тоже
    неправда: ответ пришёл, разобрать не смогли — ждать можно вечно.
    Неизвестная причина трактуется как временная — это прежнее поведение и
    самый мягкий для пользователя вариант, но код ошибки печатается всегда,
    иначе поддержке не за что зацепиться.
    """
    if reason == "waf_block":
        # 🔴 R-9: у блока по IP есть КОНКРЕТНАЯ устранимая причина — сервис
        # ходит наружу через общий пул адресов Cloud Run, который WAF брокера
        # и отбивает.  Пока статический egress не включён, каждый рестарт
        # перекатывает IP, и «работало и вдруг перестало» будет повторяться.
        # Состояние приезжает переменной `STATIC_EGRESS` (см. `cloudbuild.yaml`,
        # шаг `configure-egress`); неизвестное значение трактуем как «не
        # настроен» — иначе умолчание молча выглядело бы как «всё в порядке».
        _tail = ""
        if os.getenv("STATIC_EGRESS", "off").strip().lower() != "on":
            _tail = ("\n\n_Диагностика: сервис работает без статического "
                     "исходящего IP, поэтому адрес меняется при каждом "
                     "перезапуске._")
        return ("Брокер отклонил запрос на уровне защиты от ботов — это блок по "
                "IP-адресу, он *не пройдёт сам*. Мы уже видим проблему."
                + _tail + f"\n\nКод для поддержки: `{error_id}`")
    if reason == "parse_error":
        return ("Брокер ответил, но мы не смогли разобрать ответ — это ошибка на "
                "*нашей* стороне, и повтор её не вылечит. Мы уже видим проблему.\n\n"
                f"Код для поддержки: `{error_id}`")
    return ("Брокерский API не вернул ваш портфель — обрыв соединения или сбой "
            "на стороне брокера. Обычно это проходит за 5–15 минут.\n\n"
            f"Попробуйте ещё раз чуть позже. Код для поддержки: `{error_id}`")


# ── Keyboard builders ─────────────────────────────────────────────────────────

def kb_question(options: list[tuple[str, int, str]], q_num: int = 1) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=label, callback_data=cb)]
        for label, _, cb in options
    ]
    if q_num > 1:
        rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_universe(selected: set[str]) -> InlineKeyboardMarkup:
    rows = []
    for key in ASSET_KEYS:
        label = ("✅ " if key in selected else "") + ASSET_DISPLAY[key]
        rows.append([InlineKeyboardButton(text=label, callback_data=f"ob:uni:{key}")])
    rows.append([
        InlineKeyboardButton(text="Подтвердить выбор ➡️", callback_data="ob:uni:confirm")
    ])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_benchmark_compact(recommended: str | None) -> InlineKeyboardMarkup:
    """Sprint-5 Task 2 — Progressive Disclosure benchmark menu.

    Instead of dumping all 12 ETFs at once, show ONE confirm-with-recommended
    button plus an "Изменить" button that expands the full catalogue on demand.
    """
    rec_name = BENCHMARK_LIST.get(recommended or "", recommended or "—")
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"✅ Продолжить с «{rec_name}»",
                              callback_data="ob:bench:confirm")],
        [InlineKeyboardButton(text="✏️ Изменить рекомендуемый бенчмарк",
                              callback_data="ob:bench:expand")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")],
    ])


def kb_benchmark(current: str | None = None) -> InlineKeyboardMarkup:
    """Full benchmark selection keyboard (expanded view).

    Reached only after the user taps "Изменить" on the compact menu — so the
    12-option list is opt-in, not dumped on every user.
    """
    rows = []
    for ticker, display_name in BENCHMARK_LIST.items():
        prefix = "✅ " if ticker == current else ""
        rows.append([
            InlineKeyboardButton(
                text=f"{prefix}{display_name}",
                callback_data=f"ob:bench:{ticker}",
            )
        ])
    rows.append([
        InlineKeyboardButton(text="Продолжить ➡️", callback_data="ob:bench:confirm")
    ])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_mandate_review() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Утвердить мандат", callback_data="ob:mandate:approve"),
        InlineKeyboardButton(text="✏️ Изменить",        callback_data="ob:mandate:edit"),
    ]])


def kb_connect_choice(user_id: int | None = None) -> InlineKeyboardMarkup:
    """Откуда брать портфель. Брокер — первым: это основной путь (`§−124`)."""
    rows = [[InlineKeyboardButton(text="🔗 Подключить Freedom Broker",
                                  callback_data="connect:freedom")]]
    # I-9: при выключенном флаге бот обязан вести себя РОВНО как до Фазы 5 —
    # ни кнопки, ни обработчика (гард дублируется в `cb_connect_choice`, чтобы
    # старое сообщение с кнопкой, отправленное при включённом флаге, не стало
    # обходным путём после выключения).
    if manual_portfolio_enabled(user_id):
        rows.append([InlineKeyboardButton(text="✍️ Ввести портфель вручную",
                                          callback_data="connect:manual")])
    rows.append([InlineKeyboardButton(text="📋 Демо-портфель (бесплатно)",
                                      callback_data="connect:template")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_analysis_choice(source: str | None = None) -> InlineKeyboardMarkup:
    """Тиры — по одному в строке: парой «Базовый (1 ток…» обрезался на телефоне.

    `source` — портфель отчёта: цена на кнопке та же, что спишется
    (`_effective_cost`, демо — бесплатно). Без источника — тариф `TIER_COST`.
    """
    def _label(tier: str) -> str:
        cost = _effective_cost(tier, source) if source else TIER_COST[tier]
        price = "бесплатно" if cost == 0 else _tokens(cost)
        return f"{_TIER_ICON[tier]} {_TIER_SHORT[tier]} · {price}"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=_label("base"), callback_data="analysis:base")],
        [InlineKeyboardButton(text=_label("scenario"), callback_data="analysis:scenario")],
        [InlineKeyboardButton(text=_label("deep"), callback_data="analysis:deep")],
        [InlineKeyboardButton(text="💼 Портфель", callback_data="home:portfolio"),
         InlineKeyboardButton(text="🏠 Меню", callback_data="home:menu")],
    ])


def kb_confirm(tier: str, context_slug: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="✅ Запустить",
            callback_data=f"confirm:{tier}:{context_slug}",
        ),
        InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"),
    ]])


# ── /mandate menu (B1 2026-07-17: progressive disclosure вместо ре-анкеты) ───

def kb_mandate_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎯 Сменить бенчмарк",
                              callback_data="mandate:edit:bench")],
        [InlineKeyboardButton(text="🧬 Изменить классы активов",
                              callback_data="mandate:edit:universe")],
        [InlineKeyboardButton(text="⚖️ Изменить риск-профиль",
                              callback_data="mandate:edit:profile")],
        [InlineKeyboardButton(text="🔄 Пройти анкету заново",
                              callback_data="mandate:edit:requiz")],
        # §−124: «Мой мандат и настройки» не вело к портфелю вовсе — источник
        # нельзя было сменить после онбординга. Портфель — соседним пунктом.
        [InlineKeyboardButton(text="💼 Мой портфель", callback_data="home:portfolio"),
         InlineKeyboardButton(text="🏠 Меню", callback_data="home:menu")],
    ])


# Опорный балл для ручного выбора профиля («экспертный» режим): середина
# диапазона каждой полосы _PROFILE_MAP (6–8 / 9–12 / 13–15 / 16–18).
_PROFILE_PIVOT_SCORE = {
    "Консервативный":       7,
    "Умеренный":            10,
    "Умеренно-агрессивный": 14,
    "Агрессивный":          17,
}


def kb_mandate_profile(current: str | None) -> InlineKeyboardMarkup:
    rows = []
    for name, score in _PROFILE_PIVOT_SCORE.items():
        prefix = "✅ " if name == current else ""
        rows.append([InlineKeyboardButton(
            text=f"{prefix}{name}",
            callback_data=f"mandate:profile:{score}")])
    rows.append([InlineKeyboardButton(text="⬅️ Назад",
                                      callback_data="mandate:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_mandate_changed() -> InlineKeyboardMarkup:
    """CTA после сохранённого изменения мандата."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Новый отчёт",
                              callback_data="mandate:report")],
        [InlineKeyboardButton(text="🎛 Мандат", callback_data="mandate:back"),
         InlineKeyboardButton(text="🏠 Меню", callback_data="home:menu")],
    ])


def _mandate_overview_text(profile: dict) -> str:
    """Текущий мандат из строки БД (get_profile) — шапка меню /mandate."""
    name     = profile.get("profile_name") or "—"
    vol_pct  = int(round(float(profile.get("target_volatility") or 0) * 100))
    te_pct   = int(round(float(profile.get("target_te") or 0) * 100))
    bench_tk = _resolve_bench_ticker(profile)
    bench    = BENCHMARK_LIST.get(bench_tk or "", bench_tk or "—")
    limits   = profile.get("limits_dict") or {}
    lines = [
        "🎛 *Ваш инвестиционный мандат*\n",
        f"Профиль: *{name}*",
        f"Бенчмарк: *{bench}*",
        f"Целевая волатильность: *{vol_pct}%* · Tracking Error: *{te_pct}%*",
        "\n📊 *Классы активов:*",
    ]
    for key in ASSET_KEYS:
        if key not in limits:
            continue
        try:
            lo, hi = limits[key]
        except (TypeError, ValueError):
            continue
        display = ASSET_DISPLAY.get(key, key)
        if not lo and not hi:
            lines.append(f"  •  {display}: ❌ не включено")
        else:
            lines.append(f"  •  {display}: {lo}–{hi}%")
    lines.append(
        "\nЧто изменить?\n"
        "_Изменения бесплатны и вступят в силу в следующем отчёте._"
    )
    return "\n".join(lines)


async def _reset_state_keep_message(state: FSMContext) -> None:
    """Сброс FSM-состояния/edit-режима с сохранением id активного сообщения,
    чтобы следующий экран редактировался на месте, а не плодил сообщения."""
    data      = await state.get_data()
    ob_msg_id = data.get("ob_message_id")
    await state.clear()
    if ob_msg_id:
        await state.update_data(ob_message_id=ob_msg_id)


async def _show_mandate_menu(target: Message | CallbackQuery,
                             state: FSMContext, user_id: int) -> None:
    """Показать/обновить экран /mandate (переиспользует _edit_or_answer)."""
    profile = await get_profile(user_id)
    if profile is None:
        msg = target.message if _is_callback(target) else target
        await msg.answer(
            "⚠️ У вас ещё нет профиля. Используйте /start для регистрации.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    await _reset_state_keep_message(state)
    await _edit_or_answer(target, state,
                          _mandate_overview_text(profile), kb_mandate_menu())


# ── Shared UI helpers ─────────────────────────────────────────────────────────

async def _edit_or_answer(
    target: Message | CallbackQuery,
    state: FSMContext,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """Edit the active onboarding message when possible; otherwise send new."""
    data      = await state.get_data()
    ob_msg_id = data.get("ob_message_id")

    if _is_callback(target) and ob_msg_id:
        try:
            await target.message.edit_text(
                text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )
            return
        except Exception:
            pass

    if _is_callback(target):
        sent = await target.message.answer(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    else:
        sent = await target.answer(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    await state.update_data(ob_message_id=getattr(sent, "message_id", None))


async def send_question(
    target: Message | CallbackQuery,
    state: FSMContext,
    q_idx: int,
) -> None:
    """Advance to Q q_idx (0-based) or to Universe/Benchmark steps after Q6."""
    if q_idx < NUM_QUESTIONS:
        q = QUESTIONS[q_idx]
        await state.set_state(q["state"])
        await _edit_or_answer(target, state, q["text"], kb_question(q["options"], q_num=q_idx + 1))
    else:
        data     = await state.get_data()
        selected = set(data.get("universe", []))
        await state.set_state(Onboarding.Universe)
        if "universe" not in data:
            await state.update_data(universe=[])
        await _edit_or_answer(
            target, state,
            "🌍 *Выбор классов активов*\n\n"
            "Отметьте классы активов, которые вы хотите включить в портфель:",
            kb_universe(selected),
        )


# ══════════════════════════════════════════════════════════════════════════════
# НАВИГАЦИЯ (§−124): главное меню, «Мой портфель», один вид экранов
# ══════════════════════════════════════════════════════════════════════════════
# Почему так. До `§−124` у вернувшегося пользователя не было ни главного меню,
# ни пути к смене источника портфеля: экран подключения появлялся только в
# онбординге и при ошибке, а «Мой мандат и настройки» вёл к мандату и больше
# никуда. Демо-пользователь не мог подключить брокера, а ручной портфель и
# агрегированный отчёт не были видны ниоткуда.
#
# Правила экранов:
#   * у каждого экрана есть следующий шаг и путь назад (🏠 Меню);
#   * навигация по меню ПРАВИТ нажатое сообщение (`_screen`), а не плодит новые;
#     информационные сообщения (готовый отчёт, списание) не правятся никогда —
#     их кнопка `home:open` присылает меню НОВЫМ сообщением;
#   * цена на кнопке — та, что спишется (`_effective_cost`): демо бесплатно.

_TIER_ICON = {"base": "📊", "scenario": "🎯", "deep": "🔬"}
_TIER_SHORT = {"base": "Базовый", "scenario": "Сценарный", "deep": "Глубокий"}
_TIER_LINES = (
    "📊 *Базовый* — риск, доходность, состав, идеи\n"
    "🎯 *Сценарный* — вклад позиций в риск, 3 макро-сценария\n"
    "🔬 *Глубокий* — всё из базового + факторы и стресс-тесты"
)
#: Портфель отчёта — в тексте (строчными) и на кнопке выбора.
_PORTFOLIO_LABEL = {
    "freedom": "Freedom Broker",
    "manual": "ручной портфель",
    AGGREGATED_SOURCE: "Freedom + ручной",
    "demo": "демо-портфель",
}
_HOME_ACTIONS = frozenset({"menu", "open", "report", "portfolio", "mandate",
                           "balance", "topup", "help"})
_PF_RE = re.compile(r"^pf:(freedom|demo)$")


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русское склонение по числу: 1 токен · 2 токена · 5 токенов · 11 токенов."""
    k = abs(int(n))
    if k % 10 == 1 and k % 100 != 11:
        return one
    if k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14):
        return few
    return many


def _tokens(n: int) -> str:
    return f"{n} {_plural(n, 'токен', 'токена', 'токенов')}"


def _positions(n: int) -> str:
    return f"{n} {_plural(n, 'позиция', 'позиции', 'позиций')}"


def _portfolio_label(source: str | None) -> str:
    return _PORTFOLIO_LABEL.get(str(source or ""), "портфель")


def _is_callback(target) -> bool:
    """Нажатие кнопки, а не сообщение: у нажатия есть `data` и `message`.

    Утиная проверка, а не только `isinstance`: экран не должен молча уходить
    во всплывающее `callback.answer()` из-за обёртки над апдейтом.
    """
    return isinstance(target, CallbackQuery) or (
        hasattr(target, "data") and getattr(target, "message", None) is not None)


def _btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def kb_home() -> InlineKeyboardMarkup:
    """Главное меню: одно главное действие и четыре раздела."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [_btn("📊 Новый отчёт", "home:report")],
        [_btn("💼 Мой портфель", "home:portfolio"), _btn("🎛 Мандат", "home:mandate")],
        [_btn("💳 Баланс", "home:balance"), _btn("❓ Помощь", "home:help")],
    ])


def kb_nav(*extra: InlineKeyboardButton, new_message: bool = False) -> InlineKeyboardMarkup:
    """Строка навигации: дополнительные кнопки + «🏠 Меню».

    `new_message=True` — для сообщений, которые нельзя затирать (готовый
    отчёт, строка списания): меню придёт отдельным сообщением.
    """
    home = _btn("🏠 Меню", "home:open" if new_message else "home:menu")
    return InlineKeyboardMarkup(inline_keyboard=[[*extra, home]])


async def _screen(target: Message | CallbackQuery, text: str,
                  reply_markup: InlineKeyboardMarkup | None = None, *,
                  edit: bool = True) -> None:
    """Показать экран меню: ПРАВКА нажатого сообщения, иначе новое сообщение.

    Правка — только для нажатия кнопки (`CallbackQuery`) и только при
    `edit=True`. «Сообщение не изменилось» (повторное нажатие) — не повод
    слать дубль; любая другая ошибка правки (старое сообщение, удалено) —
    повод прислать экран заново: молчание хуже дубля (`§−104`).
    """
    if _is_callback(target) and edit:
        try:
            await target.message.edit_text(text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=reply_markup)
            return
        except Exception as exc:                       # noqa: BLE001
            if "not modified" in str(exc).lower():
                return
    msg = target.message if _is_callback(target) else target
    await msg.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)


async def _portfolio_overview(user_id: int) -> dict:
    """Что подключено у пользователя — для главного меню и «Мой портфель».

    Ни одно чтение не имеет права уронить экран (`§−104`): сбой хранилища
    показывается как «недоступен», а не как «пуст» — это разные факты.
    """
    loop = asyncio.get_running_loop()
    try:
        has_keys = _is_admin(user_id) or bool(
            await loop.run_in_executor(None, _has_vault_keys_sync, user_id))
    except Exception as exc:                           # noqa: BLE001
        logger.warning("NAV: vault недоступен user=%s: %s", user_id, type(exc).__name__)
        has_keys = False
    manual_on = manual_portfolio_enabled(user_id)
    # Два счётчика, и это не дубль: `manual_n` — БУМАГИ (то же правило, что у
    # лимита, `count_positions`), `manual_lines` — строки вместе с кэшем.
    # «Пуст» — только когда строк нет: портфель из одного кэша не пустой.
    # None — не прочитан (сбой хранилища), это «недоступен», а не «пуст».
    manual_n: int | None = None
    manual_lines: int | None = None
    if manual_on:
        try:
            text, unreadable = await _load_manual_portfolio_text(user_id)
            if not unreadable:
                entries = (await loop.run_in_executor(None, _mp_entries_sync, text)
                           if text.strip() else [])
                manual_n, manual_lines = count_positions(entries), len(entries)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("NAV: ручной портфель не прочитан user=%s: %s",
                           user_id, type(exc).__name__)
    try:
        default, _stored = await _resolve_portfolio_source(user_id)
    except Exception as exc:                           # noqa: BLE001
        logger.warning("NAV: источник не определён user=%s: %s", user_id,
                       type(exc).__name__)
        default = "undetermined"
    return {"has_keys": has_keys, "manual_on": manual_on, "manual_n": manual_n,
            "manual_lines": manual_lines, "default": default,
            "hybrid": hybrid_portfolio_enabled(user_id)}


def _manual_status(ov: dict) -> str:
    """«3 позиции» · «только кэш» · «пуст» · «недоступен» — одно правило везде."""
    if ov["manual_lines"] is None:
        return "недоступен"
    if not ov["manual_lines"]:
        return "пуст"
    return _positions(ov["manual_n"]) if ov["manual_n"] else "только кэш"


def _overview_line(ov: dict) -> str:
    """Одна строка о портфелях для главного меню: что подключено."""
    parts = []
    if ov["has_keys"]:
        parts.append("Freedom Broker ✅")
    if ov["manual_on"] and (ov["manual_lines"] != 0 or ov["default"] == "manual"):
        parts.append(f"ручной — {_manual_status(ov)}")
    if parts:
        return ("💼 Портфели: " if len(parts) > 1 else "💼 Портфель: ") + " · ".join(parts)
    if ov["default"] == "demo":
        return "💼 Портфель: демо — отчёты бесплатны"
    return "💼 Портфель не подключён — начните с «Мой портфель»"


def _report_sources(ov: dict) -> list[str]:
    """Портфели, по которым есть смысл заказать отчёт, — в порядке показа.

    Демо — всегда последним. По этому списку «📊 Новый отчёт» решает, нужен ли
    вопрос «По какому портфелю?»: при одном настоящем портфеле — сразу к тирам.
    """
    out = []
    if ov["has_keys"]:
        out.append("freedom")
    if ov["manual_on"] and (ov["manual_lines"] or ov["default"] == "manual"):
        out.append("manual")
    if ov["hybrid"] and ov["has_keys"] and ov["manual_lines"]:
        out.append(AGGREGATED_SOURCE)
    out.append("demo")
    return out


_NEEDS_FREEDOM = "подключите Freedom Broker"
_NEEDS_FREEDOM_FOR_AGG = "нужен подключённый Freedom Broker"
_NEEDS_MANUAL = "заполните ручной портфель"


def _source_refusal(source: str, ov: dict) -> str | None:
    """Почему отчёт по этому портфелю сейчас НЕЛЬЗЯ; `None` — можно.

    Проверяется на КАЖДОМ нажатии (S-4): кнопка живёт в чате дольше условий.
    Каждый портфель проверяется своим условием — ключи, флаг ручного ввода,
    флаг гибрида; общий выключатель меню не нужен.
    """
    if source == "freedom" and not ov["has_keys"]:
        return _NEEDS_FREEDOM
    if source == "manual" and not ov["manual_on"]:
        return "ручной ввод сейчас выключен"
    if source == AGGREGATED_SOURCE:
        if not ov["hybrid"]:
            return "отчёт «Freedom + ручной» сейчас выключен"
        if not ov["has_keys"]:
            return _NEEDS_FREEDOM_FOR_AGG
        if not ov["manual_lines"]:
            return _NEEDS_MANUAL
    return None


def _kb_refusal(source: str, why: str) -> InlineKeyboardMarkup:
    """Отказ ведёт туда, где причину можно устранить."""
    if why in (_NEEDS_FREEDOM, _NEEDS_FREEDOM_FOR_AGG):
        fix = _btn("🔗 Freedom Broker", "pf:freedom")
    elif why == _NEEDS_MANUAL:
        fix = _btn("✏️ Ручной портфель", "mp:show")
    else:
        fix = _btn("💼 Мой портфель", "home:portfolio")
    return kb_nav(fix)


async def _require_profile(target: Message | CallbackQuery, user_id: int) -> bool:
    """Меню без мандата бессмысленно: анкета — первый шаг (`/start`)."""
    if await get_profile(user_id) is not None:
        return True
    msg = target.message if _is_callback(target) else target
    await msg.answer("👋 Сначала короткая анкета (6 вопросов) — нажмите /start.")
    return False


async def _show_home(target: Message | CallbackQuery, user_id: int, *,
                     edit: bool = True) -> None:
    ov = await _portfolio_overview(user_id)
    balance = await get_balance(user_id)
    await _screen(target,
                  "🏠 *Главное меню*\n\n"
                  f"{_overview_line(ov)}\n"
                  f"💳 Баланс: *{_tokens(balance)}*",
                  kb_home(), edit=edit)


async def _open_report(target: Message | CallbackQuery, state: FSMContext,
                       user_id: int, *, edit: bool = True) -> None:
    """«📊 Новый отчёт».

    Один настоящий портфель — сразу тиры по нему; несколько (или гибрид) —
    вопрос «По какому портфелю?». Портфель ЯВНО едет в callback_data
    (`src:`/`rpt:`/`rptgo:`), а не берётся из скрытого «режима по умолчанию»:
    тот сам себя перелечивает в `freedom` при ключах в vault, и кнопка «демо»
    молча строила бы платный брокерский отчёт.
    """
    await state.clear()
    ov = await _portfolio_overview(user_id)
    real = [s for s in _report_sources(ov) if s != "demo"]
    if ov["hybrid"] or len(real) >= 2:
        await _show_source_menu(target, user_id, edit=edit, ov=ov)
    elif real:
        await _show_tiers_for(target, state, user_id, real[0], ov=ov, edit=edit)
    elif ov["default"] == "demo":
        await _show_tiers_for(target, state, user_id, "demo", ov=ov, edit=edit)
    else:
        await _screen(target, "📡 *Сначала выберите портфель.*\n\n"
                              "Откуда брать позиции для отчёта?",
                      kb_connect_choice(user_id), edit=edit)


async def _show_tiers_for(target: Message | CallbackQuery, state: FSMContext,
                          user_id: int, source: str, *, ov: dict | None = None,
                          edit: bool = True) -> None:
    """Тиры по КОНКРЕТНОМУ портфелю. Пустой ручной портфель — сразу к вводу."""
    ov = ov or await _portfolio_overview(user_id)
    why = _source_refusal(source, ov)
    if why is not None:
        await _screen(target, f"ℹ️ Портфель недоступен: {why}.",
                      _kb_refusal(source, why), edit=edit)
        return
    if source == "manual" and not ov["manual_lines"]:
        draft = await get_manual_draft(user_id)
        if not str((draft or {}).get("text") or "").strip():
            msg = target.message if _is_callback(target) else target
            await _manual_ask_for_input(msg, state)
            return
    multi = ov["hybrid"] or len([s for s in _report_sources(ov) if s != "demo"]) >= 2
    await _screen(target,
                  f"📊 *Выберите тип анализа* · {_portfolio_label(source)}\n\n"
                  f"{_TIER_LINES}\n\n"
                  + ("📋 Отчёты по демо-портфелю бесплатны." if source == "demo"
                     else "💳 Токен списывается только за готовый отчёт."),
                  kb_report_tiers(source, multi=multi), edit=edit)


async def _show_analysis_menu(target: Message | CallbackQuery, slug: str,
                              user_id: int | None = None, *,
                              edit: bool = False) -> None:
    """Выбор тира. При гибриде — сначала выбор портфеля (D-9).

    Портфель отчёта назван в шапке, цены на кнопках — по нему: прежняя шапка
    молчала об источнике, а демо-пользователю обещала списать токен.
    """
    msg = target.message if _is_callback(target) else target
    uid = user_id if user_id is not None else getattr(
        getattr(msg, "chat", None), "id", None)
    # Гибрид D-9: меню двухшаговое (источник → тир). Только при включённом
    # флаге (I-9): без него меню ровно прежнее. Чат с ботом личный, поэтому
    # id чата — это id пользователя, когда вызывающий его не передал.
    if uid is not None and hybrid_portfolio_enabled(int(uid)):
        await _show_source_menu(target, int(uid), edit=edit)
        return
    source = None
    if uid is not None:
        try:
            source, _stored = await _resolve_portfolio_source(int(uid))
        except Exception as exc:                       # noqa: BLE001
            logger.warning("NAV: источник не определён user=%s: %s", uid,
                           type(exc).__name__)
    priced = source if source in ("freedom", "manual", "demo") else None
    if slug:
        text = (f"👋 Вы пришли из нашего канала _{_md_safe(_source_label(slug))}_.\n\n"
                "Посмотрим, как эта новость влияет на ваш портфель. "
                "Выберите тип анализа:")
    else:
        head = "📊 *Выберите тип анализа*"
        if priced:
            head += f" · {_portfolio_label(priced)}"
        text = (f"{head}\n\n{_TIER_LINES}\n\n"
                + ("📋 Отчёты по демо-портфелю бесплатны." if priced == "demo"
                   else "💳 Токен списывается только за готовый отчёт."))
    await _screen(target, text, kb_analysis_choice(priced), edit=edit)


def _price_screen(tier: str, source: str | None, cost: int, balance: int,
                  go_data: str) -> tuple[str, InlineKeyboardMarkup]:
    """Экран цены перед запуском — одинаковый для обоих меню."""
    head = f"{_TIER_ICON[tier]} *{TIER_LABEL[tier]}*"
    if source:
        head += f" · {_portfolio_label(source)}"
    if cost > balance:
        return (f"{head}\n\n❌ Не хватает токенов: нужно *{_tokens(cost)}*, "
                f"на балансе *{_tokens(balance)}*.",
                InlineKeyboardMarkup(inline_keyboard=[[
                    _btn("💰 Пополнить", "home:topup"), _btn("🏠 Меню", "home:menu")]]))
    price = ("📋 Бесплатно — демо-портфель." if cost == 0 else
             f"💳 Стоимость: *{_tokens(cost)}* — спишется только после готового отчёта.")
    return (f"{head}\n\n{price}\nБаланс: *{_tokens(balance)}*.",
            InlineKeyboardMarkup(inline_keyboard=[[
                _btn("✅ Запустить", go_data), _btn("❌ Отмена", "cancel")]]))


async def _show_portfolio_hub(target: Message | CallbackQuery, user_id: int, *,
                              edit: bool = True) -> None:
    """«💼 Мой портфель»: что подключено и куда нажать, чтобы поменять."""
    ov = await _portfolio_overview(user_id)
    lines = ["💼 *Мой портфель*", "",
             "🔗 Freedom Broker — " + ("подключён ✅" if ov["has_keys"] else "не подключён")]
    if ov["manual_on"]:
        lines.append(f"✏️ Ручной портфель — {_manual_status(ov)}")
    lines.append("📋 Демо — шаблонный портфель, бесплатно")
    lines.append("")
    lines.append("Нажмите на портфель, чтобы управлять им или заказать отчёт.")
    rows = [[_btn("🔗 Freedom Broker", "pf:freedom")]]
    if ov["manual_on"]:
        rows.append([_btn("✏️ Ручной портфель", "mp:show")])
    rows.append([_btn("📋 Демо-портфель", "pf:demo")])
    rows.append([_btn("🏠 Меню", "home:menu")])
    await _screen(target, "\n".join(lines),
                  InlineKeyboardMarkup(inline_keyboard=rows), edit=edit)


async def _start_key_entry(target: Message | CallbackQuery, state: FSMContext,
                           slug: str = "") -> None:
    """Подключение Freedom Broker: три шага, отмена — в один тап."""
    await state.update_data(slug=slug)
    await state.set_state(PortfolioConnection.Login)
    await _screen(target,
                  "🔗 *Подключение Freedom Broker* · шаг 1 из 3\n\n"
                  f"{branding.project_name()} получает доступ только на ЧТЕНИЕ — "
                  "сделки бот совершать не может. Ключи хранятся зашифрованными; "
                  "сообщения с ними потом удалите из чата.\n\n"
                  "Введите *логин* Freedom Broker:",
                  InlineKeyboardMarkup(inline_keyboard=[[_btn("❌ Отмена", "cancel")]]))


async def _show_balance(target: Message | CallbackQuery, user_id: int, *,
                        edit: bool = True) -> None:
    balance = await get_balance(user_id)
    price_str = f"{TOKEN_PRICE_KZT:,}".replace(",", " ")
    pack_str = f"{TOKEN_PACK_PRICE_KZT:,}".replace(",", " ")
    await _screen(target,
                  f"💳 *Баланс: {_tokens(balance)}*\n\n"
                  f"1 токен = {price_str} ₸ · пакет {_tokens(TOKEN_PACK_TOKENS)} = "
                  f"{pack_str} ₸\n"
                  "Токен списывается только за готовый отчёт; демо — бесплатно.",
                  kb_nav(_btn("💰 Пополнить", "home:topup")), edit=edit)


async def _show_topup(target: Message | CallbackQuery, *, edit: bool = True) -> None:
    price_str = f"{TOKEN_PRICE_KZT:,}".replace(",", " ")
    pack_str = f"{TOKEN_PACK_PRICE_KZT:,}".replace(",", " ")
    await _screen(target,
                  "💰 *Пополнение*\n\n"
                  f"Пакет {_tokens(TOKEN_PACK_TOKENS)} — *{pack_str} ₸* "
                  f"(1 токен = {price_str} ₸).\n"
                  f"Оплата пока через поддержку: {md_safe(branding.support_contact())}.",
                  kb_nav(), edit=edit)


async def _show_help(target: Message | CallbackQuery, user_id: int, *,
                     edit: bool = True) -> None:
    """Короткая карта бота: три шага, затем справка по ручному вводу/токенам."""
    base_c, scn_c, deep_c = (TIER_COST["base"], TIER_COST["scenario"],
                             TIER_COST["deep"])
    price_str = f"{TOKEN_PRICE_KZT:,}".replace(",", " ")
    pack_str = f"{TOKEN_PACK_PRICE_KZT:,}".replace(",", " ")
    lines = [
        f"❓ *Как пользоваться {branding.bot_name()}*",
        "",
        "1️⃣ *Мой портфель* — /portfolio: подключите Freedom Broker (только "
        "чтение)"
        + (", введите портфель вручную" if manual_portfolio_enabled(user_id) else "")
        + " или возьмите демо.",
        "2️⃣ *Новый отчёт* — /report:",
        f"   📊 Базовый · {_tokens(base_c)} — риск, доходность, состав, идеи",
        f"   🎯 Сценарный · {_tokens(scn_c)} — вклад позиций в риск, 3 сценария",
        f"   🔬 Глубокий · {_tokens(deep_c)} — + факторы, стресс-тесты, "
        "аналитика банков",
        "3️⃣ *Мандат* — /mandate: риск-профиль, бенчмарк, классы активов. "
        "Бесплатно.",
        "",
    ]
    # I-9: строки ручного портфеля — только при включённом ручном вводе.
    if manual_portfolio_enabled(user_id):
        lines.append("✏️ *Ручной портфель*: правки сообщением `+AAPL 10 150`, "
                     "`-AAPL 5`, `-AAPL`. Удалить всё — /forget\\_portfolio.")
    if hybrid_portfolio_enabled(user_id):
        lines.append("🌐 *Freedom + ручной* — один отчёт по обоим портфелям; "
                     "котировки Tradernet.")
    lines.append(f"💳 *Токены* — /balance: 1 токен = {price_str} ₸, "
                 f"пакет {TOKEN_PACK_TOKENS} = {pack_str} ₸. Списание — только за "
                 "готовый отчёт; демо бесплатно.")
    lines.append("🛟 *Поддержка* — /support")
    await _screen(target, "\n".join(lines), kb_nav(), edit=edit)


async def cb_home(callback: CallbackQuery, state: FSMContext) -> None:
    """`home:<раздел>` — главное меню и его разделы. callback_data — по allowlist."""
    await callback.answer()
    action = str(callback.data or "").split(":", 1)[-1]
    user_id = callback.from_user.id
    if action not in _HOME_ACTIONS:
        logger.warning("NAV: подделанный callback user=%s", user_id)
        return
    if not await _require_profile(callback, user_id):
        return
    if action in ("menu", "open"):
        await state.clear()
        await _show_home(callback, user_id, edit=(action == "menu"))
    elif action == "report":
        await _open_report(callback, state, user_id)
    elif action == "portfolio":
        await state.clear()
        await _show_portfolio_hub(callback, user_id)
    elif action == "mandate":
        # Меню мандата правит «своё» сообщение (`_edit_or_answer`) — отдаём ему
        # нажатое, чтобы переход был на месте, а не новым сообщением.
        await state.update_data(ob_message_id=getattr(callback.message, "message_id", None))
        await _show_mandate_menu(callback, state, user_id)
    elif action == "balance":
        await _show_balance(callback, user_id)
    elif action == "topup":
        await _show_topup(callback)
    elif action == "help":
        await _show_help(callback, user_id)


async def cb_portfolio_card(callback: CallbackQuery, state: FSMContext) -> None:
    """`pf:freedom` / `pf:demo` — карточка портфеля в «💼 Мой портфель»."""
    await callback.answer()
    m = _PF_RE.match(str(callback.data or ""))
    user_id = callback.from_user.id
    if not m:
        logger.warning("NAV: подделанный callback user=%s", user_id)
        return
    if not await _require_profile(callback, user_id):
        return
    back = _btn("⬅️ Мой портфель", "home:portfolio")
    if m.group(1) == "freedom":
        ov = await _portfolio_overview(user_id)
        if not ov["has_keys"]:
            await _start_key_entry(callback, state)
            return
        await _screen(callback,
                      "🔗 *Freedom Broker* — подключён ✅\n"
                      "Доступ только на чтение; ключи хранятся зашифрованными.",
                      InlineKeyboardMarkup(inline_keyboard=[
                          [_btn("📊 Отчёт по Freedom", "src:freedom")],
                          [_btn("🔑 Заменить ключи", "connect:freedom")],
                          [back]]))
        return
    await _screen(callback,
                  "📋 *Демо-портфель*\n"
                  "Шаблонный портфель, чтобы посмотреть, как выглядят отчёты. "
                  "Бесплатно.",
                  InlineKeyboardMarkup(inline_keyboard=[
                      [_btn("📊 Демо-отчёт", "src:demo")], [back]]))


async def cmd_report(message: Message, state: FSMContext) -> None:
    """/report — сразу к заказу отчёта."""
    if await _require_profile(message, message.from_user.id):
        await _open_report(message, state, message.from_user.id, edit=False)


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING ROUTER
# ══════════════════════════════════════════════════════════════════════════════

onboarding_router = Router(name="onboarding")


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    slug    = message.text.split(maxsplit=1)[1].strip() if " " in (message.text or "") else ""
    profile = await get_profile(user_id)

    # «Применить идею» deep-link from a delivered report: t.me/<bot>?start=scn_<n>.
    # The static HTML report cannot charge a token, so it hands off here and the
    # bot runs the Scenario tier + charges the 1 token via the normal confirm
    # flow.  Strip the marker so it is NOT mistaken for a news-source slug.
    scn_idea: str | None = None
    if slug.startswith("scn_"):
        scn_idea = re.sub(r"[^0-9A-Za-z]", "", slug[4:])[:8] or "—"
        slug = ""

    if profile is None:
        # NEW user — launch onboarding; token grant is deferred until mandate approval.
        await state.update_data(slug=slug, scn_idea=scn_idea)
        await state.set_state(Onboarding.Q1)
        q    = QUESTIONS[0]
        sent = await message.answer(
            f"👋 *Добро пожаловать в {branding.project_name()} — Risk & Asset Management Platform!*\n\n"
            "Прежде чем начать, пройдите короткое анкетирование (6 вопросов), "
            "чтобы мы могли составить ваш персональный инвестиционный мандат.\n\n"
            + q["text"],
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_question(q["options"], q_num=1),
        )
        await state.update_data(ob_message_id=sent.message_id)

    else:
        # RETURNING user — original flow unchanged.
        await init_user(user_id)  # returns False — no double-grant
        if scn_idea is not None:
            # Report «Применить идею» → Scenario-tier confirmation.  Reuses the
            # existing scenario flow (kb_confirm → cb_confirm): the token is
            # charged here, bot-side, only after the report is rendered.
            cost    = TIER_COST["scenario"]
            balance = await get_balance(user_id)
            await state.update_data(tier="scenario", context_slug="idea")
            await message.answer(
                f"🎯 *Сценарный анализ по идее №{scn_idea} из вашего отчёта*\n\n"
                "Прогоним ваш портфель через сценарную модель: вклад позиций в "
                "риск (Euler-MCTR), фондирование и walk-forward бэктест.\n\n"
                f"💳 Спишется *{cost}* токен · баланс: *{balance}*.\n\n"
                "Запустить сценарный анализ?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_confirm("scenario", "idea"),
            )
            await state.set_state(AnalysisFlow.awaiting_approval)
        elif slug:
            await state.update_data(context_slug=slug)
            await message.answer(
                f"👋 Привет! Я вижу, вы пришли из нашего канала _{_md_safe(_source_label(slug))}_.\n\n"
                "Я могу проанализировать, как эта новость повлияет на ваш портфель.\n\n"
                "💰 *Базовый отчёт:* 1 токен\n"
                "🎯 *Сценарный анализ:* 1 токен\n"
                "🔬 *Глубокий анализ:* 2 токена\n\n"
                "Начать анализ?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_analysis_choice(),
            )
        else:
            # A profile can exist while the portfolio SOURCE is undetermined
            # (connection lost pre-persistence / legacy 'template' default not
            # migrated).  /start is the natural place to heal that: offer the
            # source choice FIRST, otherwise the user loops between the menu
            # and the «Источник портфеля не выбран» guard in cb_confirm.
            try:
                source, _stored = await _resolve_portfolio_source(user_id)
            except Exception as exc:               # noqa: BLE001 — never block /start
                logger.warning("Source resolution on /start failed for %s: %s",
                               user_id, exc)
                source = None
            if source == "undetermined":
                await message.answer(
                    "📡 *Сначала подключите источник портфеля.*\n\n"
                    "Похоже, подключение не завершено или было сброшено. "
                    "Выберите, как анализировать ваш портфель:",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb_connect_choice(user_id),
                )
                return
            # §−124: вернувшийся пользователь попадает в ГЛАВНОЕ МЕНЮ, а не
            # сразу в тиры: иначе разделы «Мой портфель», «Мандат», «Баланс»
            # были бы видны только тем, кто знает команды.
            await _show_home(message, user_id, edit=False)


# ── Q1–Q6 answer handler ──────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.regexp(r"^ob:q[1-6]:\d$"),
    StateFilter(Onboarding.Q1, Onboarding.Q2, Onboarding.Q3,
                Onboarding.Q4, Onboarding.Q5, Onboarding.Q6),
)
async def cb_question_answer(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, q_part, pts_str = callback.data.split(":")
    q_num = int(q_part[1])
    pts   = int(pts_str)
    await state.update_data(**{f"q{q_num}": pts})
    await send_question(callback, state, q_num)


# ── Back button handler ───────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:back",
    StateFilter(Onboarding.Q2, Onboarding.Q3, Onboarding.Q4,
                Onboarding.Q5, Onboarding.Q6,
                Onboarding.Universe, Onboarding.Benchmark),
)
async def cb_back(callback: CallbackQuery, state: FSMContext) -> None:
    """Roll back to the previous onboarding step."""
    await callback.answer()
    data = await state.get_data()
    # B1: в edit-режиме (/mandate) «Назад» возвращает в меню мандата, а не
    # в предыдущий шаг несуществующей анкеты.
    if data.get("edit_mode"):
        await _show_mandate_menu(callback, state, callback.from_user.id)
        return
    current = await state.get_state()
    # Map current state → previous q_idx (0-based)
    back_map = {
        Onboarding.Q2.state:       0,  # back to Q1
        Onboarding.Q3.state:       1,  # back to Q2
        Onboarding.Q4.state:       2,
        Onboarding.Q5.state:       3,
        Onboarding.Q6.state:       4,
        Onboarding.Universe.state: 5,  # back to Q6
    }
    prev_idx = back_map.get(current)
    if prev_idx is not None:
        await send_question(callback, state, prev_idx)
    elif current == Onboarding.Benchmark.state:
        # Back from Benchmark → Universe
        data = await state.get_data()
        selected = set(data.get("universe", []))
        await state.set_state(Onboarding.Universe)
        await _edit_or_answer(
            callback, state,
            "🌍 *Выбор классов активов*\n\n"
            "Отметьте классы активов, которые вы хотите включить в портфель:",
            kb_universe(selected),
        )


# ── Universe toggle ───────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.startswith("ob:uni:"),
    ~F.data.endswith("confirm"),
    StateFilter(Onboarding.Universe),
)
async def cb_universe_toggle(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    asset_key = callback.data[len("ob:uni:"):]
    data      = await state.get_data()
    universe: list[str] = list(data.get("universe", []))
    if asset_key in universe:
        universe.remove(asset_key)
    else:
        universe.append(asset_key)
    await state.update_data(universe=universe)
    await callback.message.edit_reply_markup(reply_markup=kb_universe(set(universe)))


# ── Universe confirm ──────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:uni:confirm",
    StateFilter(Onboarding.Universe),
)
async def cb_universe_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data     = await state.get_data()
    universe = data.get("universe", [])

    if not universe:
        await callback.message.answer("⚠️ Выберите хотя бы один класс активов.")
        return

    # B1 (edit-режим /mandate): баллы профиля НЕ трогаем — меняется только
    # вселенная/лимиты.  Сохраняем в БД сразу и возвращаемся в мандат-саммари
    # (НЕ идём дальше по онбордингу к выбору бенчмарка/подключению).
    if data.get("edit_mode"):
        user_id = callback.from_user.id
        stored  = await get_profile(user_id)
        if stored is None:
            await callback.message.answer(
                "⚠️ Профиль не найден — пройдите /start.",
                parse_mode=ParseMode.MARKDOWN)
            await state.clear()
            return
        # Опорный балл из БД (клампим в диапазон анкеты на случай легаси).
        score   = max(6, min(18, int(stored.get("score") or 10)))
        profile = RiskProfileManager.score_to_profile(score)
        limits  = RiskProfileManager.apply_universe(profile, universe)
        await save_profile(
            telegram_id       = user_id,
            score             = int(stored.get("score") or score),
            profile_name      = stored.get("profile_name") or profile["name"],
            target_volatility = float(stored.get("target_volatility")
                                      or profile["target_vol"]),
            target_te         = float(stored.get("target_te")
                                      or profile["target_te"]),
            selected_assets   = universe,
            limits_dict       = limits,
            benchmark_ticker  = stored.get("benchmark_ticker"),
        )
        # Пользователь явно подтвердил изменение кнопкой — мандат остаётся
        # утверждённым (save_profile сбрасывает флаг для полной ре-анкеты).
        await approve_mandate(user_id)
        updated = await get_profile(user_id)
        await _edit_or_answer(
            callback, state,
            "✅ *Классы активов обновлены.*\n\n"
            + _mandate_overview_text(updated or {}),
            kb_mandate_changed(),
        )
        await _reset_state_keep_message(state)
        return

    # Score from 6 questions (range 6-18)
    score   = sum(data.get(f"q{i}", 0) for i in range(1, NUM_QUESTIONS + 1))
    profile = RiskProfileManager.score_to_profile(score)

    # Store profile data for later use
    await state.update_data(profile_data={
        "name":       profile["name"],
        "target_vol": profile["target_vol"],
        "target_te":  profile["target_te"],
        "score":      score,
        "limits":     RiskProfileManager.apply_universe(profile, universe),
    })

    # Transition to Benchmark selection
    default_bench = PROFILE_BENCH_TICKER.get(profile["name"], "SPY.US")
    await state.update_data(benchmark_ticker=default_bench)
    await state.set_state(Onboarding.Benchmark)
    await _edit_or_answer(
        callback, state,
        "📊 *Бенчмарк для вашего портфеля*\n\n"
        f"На основе профиля *{profile['name']}* мы рекомендуем "
        f"*{BENCHMARK_LIST.get(default_bench, default_bench)}*.\n\n"
        "Продолжите с рекомендованным или измените его:",
        kb_benchmark_compact(default_bench),
    )


# ── Benchmark expand (Progressive Disclosure) ─────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:bench:expand",
    StateFilter(Onboarding.Benchmark),
)
async def cb_benchmark_expand(callback: CallbackQuery, state: FSMContext) -> None:
    """Sprint-5 Task 2 — reveal the full 12-ETF list only on explicit request."""
    await callback.answer()
    data    = await state.get_data()
    current = data.get("benchmark_ticker")
    await _edit_or_answer(
        callback, state,
        "📊 *Выберите бенчмарк*\n\n"
        "ℹ️ Бенчмарк — эталон, с которым сравнивается ваш портфель "
        "(доходность, Tracking Error, факторное разложение).\n\n"
        "Отметьте предпочитаемый индекс/фактор:",
        kb_benchmark(current=current),
    )


# ── Benchmark toggle ──────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.startswith("ob:bench:"),
    ~F.data.endswith("confirm"),
    ~F.data.endswith("expand"),
    StateFilter(Onboarding.Benchmark),
)
async def cb_benchmark_toggle(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    ticker = callback.data[len("ob:bench:"):]
    await state.update_data(benchmark_ticker=ticker)
    await callback.message.edit_reply_markup(reply_markup=kb_benchmark(current=ticker))


# ── Benchmark confirm ─────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:bench:confirm",
    StateFilter(Onboarding.Benchmark),
)
async def cb_benchmark_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data      = await state.get_data()
    bench_tk  = data.get("benchmark_ticker")

    # B1 (edit-режим /mandate): прямой антидот к багу-первопричине — смена
    # ТОЛЬКО бенчмарка за 2 тапа, без повторной анкеты и без биллинга.
    # save_benchmark_ticker пишет мгновенно и НЕ сбрасывает утверждение мандата.
    if data.get("edit_mode"):
        user_id = callback.from_user.id
        if bench_tk:
            await save_benchmark_ticker(user_id, bench_tk)
        updated = await get_profile(user_id)
        await _edit_or_answer(
            callback, state,
            "✅ *Бенчмарк обновлён.* Следующий отчёт будет сравнивать портфель "
            f"с *{BENCHMARK_LIST.get(bench_tk or '', bench_tk or '—')}* — и в "
            "доходности, и в факторном разложении.\n\n"
            + _mandate_overview_text(updated or {}),
            kb_mandate_changed(),
        )
        await _reset_state_keep_message(state)
        return

    prof      = data["profile_data"]
    universe  = data.get("universe", [])

    summary = RiskProfileManager.build_mandate_summary(
        prof, prof["limits"], benchmark_ticker=bench_tk,
    )
    await state.set_state(Onboarding.MandateReview)
    await _edit_or_answer(callback, state, summary, kb_mandate_review())


# ── Mandate approve ───────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:mandate:approve",
    StateFilter(Onboarding.MandateReview),
)
async def cb_mandate_approve(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    user_id  = callback.from_user.id
    data     = await state.get_data()
    prof     = data["profile_data"]
    slug     = data.get("slug", "")
    universe = data.get("universe", [])
    bench_tk = data.get("benchmark_ticker")

    await save_profile(
        telegram_id       = user_id,
        score             = prof["score"],
        profile_name      = prof["name"],
        target_volatility = prof["target_vol"],
        target_te         = prof["target_te"],
        selected_assets   = universe,
        limits_dict       = prof["limits"],
        benchmark_ticker  = bench_tk,
    )
    await approve_mandate(user_id)
    await init_user(user_id)   # grants tokens — user is new here

    # Preserve slug for the connection sub-flow, then reset everything else.
    await state.clear()
    if slug:
        await state.update_data(slug=slug)

    balance = await get_balance(user_id)
    await callback.message.answer(
        f"🎉 *Мандат утверждён — добро пожаловать в {branding.project_name()}!*\n"
        f"Профиль: *{prof['name']}* · на счёте *{_tokens(balance)}*.\n\n"
        "Последний шаг — откуда брать портфель?",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_connect_choice(user_id),
    )


# ── Mandate edit ──────────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:mandate:edit",
    StateFilter(Onboarding.MandateReview),
)
async def cb_mandate_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data     = await state.get_data()
    universe = set(data.get("universe", []))
    await state.set_state(Onboarding.Universe)
    await _edit_or_answer(
        callback, state,
        "🌍 *Выбор классов активов*\n\nОтметьте классы активов для вашего портфеля:",
        kb_universe(universe),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONNECTION ROUTER
# ══════════════════════════════════════════════════════════════════════════════

portfolio_router = Router(name="portfolio_connection")


@portfolio_router.callback_query(F.data.startswith("connect:"))
async def cb_connect_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, mode  = callback.data.split(":", 1)
    user_id  = callback.from_user.id
    fsm_data = await state.get_data()
    slug     = fsm_data.get("slug", "")

    if mode == "template":
        await save_connection_mode(user_id, "template")
        await callback.message.edit_text(
            "✅ *Демо-портфель выбран* — отчёты по нему бесплатны.\n"
            "Подключить свой портфель можно в любой момент: «💼 Мой портфель».",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        await _show_analysis_menu(callback.message, slug)

    elif mode == "manual":
        # I-9, второй гард: кнопки при выключенном флаге нет, но СТАРОЕ
        # сообщение с кнопкой живёт в чате вечно.  Без проверки здесь выключение
        # флага не выключало бы фичу — а флаг существует ровно ради отката.
        if not manual_portfolio_enabled(user_id):
            logger.info("MANUAL: нажата кнопка при выключенном флаге user=%s", user_id)
            await callback.message.answer(
                "ℹ️ Ручной ввод портфеля пока недоступен. "
                "Выберите демо-режим или подключите брокера.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_connect_choice(user_id),
            )
            return
        await save_connection_mode(user_id, "manual")
        await state.update_data(slug=slug)
        draft = await get_manual_draft(user_id)
        if draft and str(draft.get("text") or "").strip():
            # §9: пользователь мог уйти на середине и вернуться через неделю.
            # Молча подставить недельный черновик нельзя — портфель с тех пор
            # мог измениться; молча выбросить тоже нельзя — это минуты работы.
            await state.set_state(ManualPortfolio.Input)
            await callback.message.edit_text(
                f"📝 *Найден черновик от {_fmt_draft_age(draft)}*\n\n"
                "Продолжить с него или начать заново?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_manual_draft(),
            )
            return
        await _manual_ask_for_input(callback.message, state)

    elif mode == "freedom":
        await _start_key_entry(callback, state, slug)


@portfolio_router.message(StateFilter(PortfolioConnection.Login))
async def msg_login(message: Message, state: FSMContext) -> None:
    await state.update_data(connect_login=message.text.strip())
    await state.set_state(PortfolioConnection.ApiKey)
    await message.answer(
        "🔑 Шаг 2 из 3 — введите *API Key*:",
        parse_mode=ParseMode.MARKDOWN,
    )


@portfolio_router.message(StateFilter(PortfolioConnection.ApiKey))
async def msg_api_key(message: Message, state: FSMContext) -> None:
    await state.update_data(connect_api_key=message.text.strip())
    await state.set_state(PortfolioConnection.SecretKey)
    # CRITICAL: live broker API key was just transmitted as plain-text — purge
    # it from the chat IMMEDIATELY (bot needs Delete-Messages permission in a
    # group; in a 1:1 chat with the user, bots can delete their OWN sent
    # messages but NOT the user's; Telegram restricts that to admins).  If
    # the delete fails (permission, network), we surface a clear instruction
    # to the user as the fallback.  We delete BEFORE asking for the next
    # secret so even an interrupted onboarding doesn't leave the key visible.
    try:
        await message.delete()
    except Exception as exc:                       # noqa: BLE001
        logger.info("Couldn't auto-delete API-key message for %s: %s",
                    message.from_user.id, exc)
        await message.answer(
            "⚠️ *Удалите вручную* предыдущее сообщение с API-ключом из чата — "
            "у бота нет прав на автоматическое удаление в этом диалоге.",
            parse_mode=ParseMode.MARKDOWN,
        )
    await message.answer(
        "🔑 Шаг 3 из 3 — введите *Secret Key*:",
        parse_mode=ParseMode.MARKDOWN,
    )


@portfolio_router.message(StateFilter(PortfolioConnection.SecretKey))
async def msg_secret_key(message: Message, state: FSMContext) -> None:
    secret_key = message.text.strip()
    data       = await state.get_data()
    login      = data.get("connect_login", "")
    api_key    = data.get("connect_api_key", "")
    slug       = data.get("slug", "")
    user_id    = message.from_user.id

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _save_keys_sync, user_id, login, api_key, secret_key)
    await save_connection_mode(user_id, "freedom")
    await state.clear()

    # CRITICAL: secret key just travelled through the chat in plaintext —
    # purge it BEFORE we acknowledge save, so even if the next call fails
    # the secret is gone from Telegram's chat history.
    delete_failed = False
    try:
        await message.delete()
    except Exception as exc:                       # noqa: BLE001
        delete_failed = True
        logger.info("Couldn't auto-delete Secret-key message for %s: %s",
                    user_id, exc)

    # H2 — strict, minimalist security acknowledgement.  The auto-delete above
    # usually succeeds in groups but Telegram forbids bots from deleting a
    # user's message in a 1:1 chat, so the reminder below is shown regardless
    # (it IS the mitigation — ask the user to delete their key message now).
    ack_text = (
        "✅ *Freedom Broker подключён.* Ключи хранятся только в зашифрованном виде.\n\n"
        "⚠️ Удалите из чата сообщения с ключами — у бота нет права сделать это за вас."
    )
    await message.answer(ack_text, parse_mode=ParseMode.MARKDOWN)
    await _show_analysis_menu(message, slug)


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL PORTFOLIO (Фаза 5) — ввод текстом, экран подтверждения, черновик
# ══════════════════════════════════════════════════════════════════════════════

#: Сколько позиций показываем в таблице подтверждения.  Лимит сообщения
#: Telegram — 4096 символов, а строка таблицы ≈ 55; 200 позиций не поместятся
#: физически.  Показываем самые ДОРОГИЕ: опечатка в количестве, ради которой
#: экран и существует, всплывает именно наверху списка (§9).
_MANUAL_PREVIEW_ROWS = 20
#: Сколько отвергнутых строк перечисляем поимённо.
_MANUAL_PREVIEW_ERRORS = 10
#: Сколько пар «как ввели → как поняли» показываем (синонимы, прокси).
_MANUAL_PREVIEW_PAIRS = 10
#: Жёсткий потолок сообщения Telegram. Сообщение длиннее API не примет —
#: `TelegramBadRequest`, и пользователь не увидит СВОЙ ПОРТФЕЛЬ вообще, хотя
#: разобран он корректно. Резерв в 96 символов — на маркер обрезки.
_TELEGRAM_TEXT_LIMIT = 4096

_MANUAL_INPUT_HELP = (
    "✍️ *Введите портфель одним сообщением*\n\n"
    "Одна позиция — одна строка:\n"
    "`ТИКЕР  КОЛИЧЕСТВО  ЦЕНА_ПОКУПКИ  [ВАЛЮТА]`\n\n"
    "```\n"
    "AAPL 10 150.50\n"
    "KSPI 200 45000 KZT\n"
    "CASH:USD 5000\n"
    "CASH:KZT 2 500 000\n"
    "```\n"
    "• валюта — необязательна, обычно она видна по тикеру;\n"
    "• кэш — два поля: `CASH:ВАЛЮТА СУММА`;\n"
    "• десятичная запятая и разряды через пробел допустимы;\n"
    "• строки с `#` игнорируются.\n\n"
    "После ввода покажу, как я вас понял, — и только потом посчитаю."
)


def kb_manual_draft() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="▶️ Продолжить",   callback_data="manual:resume"),
        InlineKeyboardButton(text="🆕 Начать заново", callback_data="manual:fresh"),
    ]])


def kb_manual_confirm() -> InlineKeyboardMarkup:
    """Три кнопки, а не две (B-4): без «Отмены» пользователь, увидевший, что
    распозналось не то, оставался бы заперт в FSM."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Рассчитать", callback_data="manual:confirm")],
        [
            InlineKeyboardButton(text="✏️ Исправить", callback_data="manual:edit"),
            InlineKeyboardButton(text="❌ Отмена",    callback_data="manual:cancel"),
        ],
    ])


def _md_safe(text: str) -> str:
    """Убрать из ПОЛЬЗОВАТЕЛЬСКОГО фрагмента символы разметки Telegram.

    Причины отказа парсера цитируют ввод («тикер «AAPL*» не распознан»), а
    незакрытая `*` или `_` роняет всё сообщение целиком с
    `TelegramBadRequest: can't parse entities` — то есть пользователь не увидит
    НИ ОДНОЙ ошибки вместо одной непонятой строки.
    """
    return re.sub(r"[*_`\[\]]", "", str(text))


def _fmt_amount(value: float | None, digits: int = 2) -> str:
    """Число с пробелом-разделителем разрядов: `2 500 000.00`."""
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ")


def _fmt_pairs(pairs: list[tuple[str, str]]) -> str:
    """«как ввели → как поняли», с ограничением количества.

    Список растёт ЛИНЕЙНО по числу позиций: книга из 200 бумаг, введённых
    синонимами, давала 200 пар и раздувала сообщение до 5 292 символов при
    лимите Telegram 4 096 — API отвергал его целиком, и пользователь не видел
    СВОЙ ПОРТФЕЛЬ вообще, хотя разобран тот был корректно (замерено).
    """
    shown = ", ".join(f"{_md_safe(a)} → {_md_safe(b)}"
                      for a, b in pairs[:_MANUAL_PREVIEW_PAIRS])
    rest = len(pairs) - _MANUAL_PREVIEW_PAIRS
    return shown if rest <= 0 else f"{shown} и ещё {rest}"


def _clip_to_telegram_limit(text: str) -> str:
    """Последний рубеж перед `send_message`: сообщение обязано пройти.

    Каждый блок экрана ограничен по отдельности, но блоков восемь, и их сумма
    остаётся способом превысить лимит — особенно после того, как в экран
    добавят девятый. Обрезка по границе строки уродлива, зато пользователь
    видит портфель и кнопку «Рассчитать»; отвергнутое сообщение не даёт ему
    ничего.
    """
    if len(text) <= _TELEGRAM_TEXT_LIMIT:
        return text
    marker = "\n… сообщение сокращено, чтобы уместиться в лимит Telegram"
    body = text[:_TELEGRAM_TEXT_LIMIT - len(marker)]
    cut = body.rfind("\n")
    if cut > 0:
        body = body[:cut]                    # режем по границе строки, не по букве
    # Обрыв внутри кодового блока оставил бы непарный ``` — Telegram отвергнет
    # такое сообщение по разметке, то есть обрезка сама себя обесценит.
    if body.count("```") % 2:
        body += "\n```"
    return body + marker


def _fmt_fx_note(currency: str, base: str, rate: float) -> str:
    """Курс в том виде, в каком его КОТИРУЮТ, а не в каком применяют.

    Движок работает мультипликатором «цена × rate», и для тенге это
    0.00222 — число арифметически верное и человечески нечитаемое: владелец
    тенгового остатка знает свой курс как «450 ₸ за доллар». Поэтому пара
    разворачивается так, чтобы показанное число было ≥ 1; смысл при этом не
    меняется, обе записи — один и тот же курс.
    """
    if rate <= 0:                                      # pragma: no cover
        return f"1 {currency} = {rate} {base}"
    if rate >= 1:
        return f"1 {currency} = {_fmt_amount(rate, 4)} {base}"
    return f"1 {base} = {_fmt_amount(1.0 / rate, 4)} {currency}"


def _fit(text: str, width: int) -> str:
    """Обрезать до `width` с многоточием — обрезка молча искажает тикер.

    `FFSPC6.1028.AIX`, урезанный до `FFSPC6.1028`, выглядит как ДРУГАЯ бумага
    (и именно такая бумага в отчёте существует), поэтому факт обрезки обязан
    быть виден.
    """
    s = str(text)
    return s if len(s) <= width else s[:width - 1] + "…"


def _fmt_draft_age(draft: dict) -> str:
    """Дата создания черновика как «20.07», либо «неизвестной давности».

    Формат `CURRENT_TIMESTAMP` у SQLite — `YYYY-MM-DD HH:MM:SS`, но полагаться
    на него без проверки нельзя: строку могли записать миграцией или иным
    клиентом, а падение ради подписи к кнопке недопустимо.
    """
    raw = str(draft.get("created_at") or "").strip()
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").strftime("%d.%m")
    except (ValueError, TypeError):
        return "неизвестной давности"


async def _manual_ask_for_input(message: Message, state: FSMContext,
                                prefill: str = "") -> None:
    """Экран ввода. `prefill` — прошлый текст для правки.

    Telegram не умеет подставлять текст в поле ввода, поэтому «предзаполнение»
    (§3, переход `manual:edit`) — это прошлый ввод отдельным сообщением-блоком:
    его можно скопировать одним касанием, поправить строку и отправить обратно.
    """
    await state.set_state(ManualPortfolio.Input)
    await message.answer(_MANUAL_INPUT_HELP, parse_mode=ParseMode.MARKDOWN)
    if prefill.strip():
        await message.answer(
            "Ваш прошлый ввод — скопируйте, поправьте и пришлите снова:\n"
            f"```\n{prefill.strip()[:3500]}\n```",
            parse_mode=ParseMode.MARKDOWN,
        )


def _manual_review_sync(text: str) -> tuple:
    """Разбор + доли + pre-flight ОДНИМ блокирующим вызовом (для executor).

    Почему в executor: `build_confirmation` спрашивает курс, а это сетевой
    поход (FRED), и делать его прямо в обработчике сообщения — значит держать
    long-poll aiogram и показывать пользователю зависший интерфейс (B-2).

    Возвращает `(report, view, coverage)`.
    """
    from finance.manual_portfolio import (
        build_confirmation, parse_portfolio_text, preflight_coverage,
    )
    engine = UniversalPortfolioManager(price_source="manual").engine
    report = parse_portfolio_text(text, engine)
    view = build_confirmation(report, engine)
    days = env_int("HISTORY_LOOKBACK_DAYS", 1825, lo=90, hi=3650)
    try:
        coverage = preflight_coverage(report, engine, days=days)
    except Exception as exc:                           # noqa: BLE001
        # Pre-flight — подсказка, а не условие расчёта: сбой чтения кэша не
        # имеет права отнять у пользователя разобранный портфель.
        logger.warning("MANUAL pre-flight пропущен: %s", exc)
        coverage = None
    return report, view, coverage


class ManualInputUnusable(RuntimeError):
    """Ручной портфель не из чего собрать: черновика нет или он не разобрался.

    Отдельный тип, а не общий `Exception`: путь расчёта ловит его РЯДОМ с
    брокерскими отказами и обязан отличать «нам нечего считать» от «сломалось».
    Первое — вина ввода и лечится подсказкой, второе — наша, и там код ошибки
    для поддержки.
    """


def _manual_frame_sync(text: str):
    """Текст черновика → DataFrame позиций (для executor).

    🔴 Разбор повторяется, а не переиспользуется с экрана подтверждения, и это
    осознанно: между подтверждением и расчётом мог быть рестарт контейнера,
    после которого FSM-состояние пусто, а черновик в SQLite жив (`PHASE_05 §5`).
    Источник правды — черновик, поэтому считаем от него.

    Сеть не трогается: `build_confirmation` (курс через FRED) здесь не нужен —
    экран уже показан, а расчёту нужен только фрейм.
    """
    from finance.manual_portfolio import parse_portfolio_text

    engine = UniversalPortfolioManager(price_source="manual").engine
    report = parse_portfolio_text(text, engine)
    if not report.valid:
        raise ManualInputUnusable(
            "ни одной позиции не распознано" if not report.failed
            else f"не распознано ни одной позиции из {len(report.failed)}")
    return report.to_dataframe()


def _format_manual_confirmation(view, coverage=None) -> str:
    """Экран подтверждения (`PHASE_05 §4`). Только ВИД — числа уже посчитаны.

    Доля позиции здесь обязательна, а не желательна: опечатка в количестве
    (лишний ноль) не видна ни в одном числе отчёта, но полностью искажает веса,
    TRC и CVaR. Именно доля делает её заметной глазом.
    """
    lines = ["📋 *Ваш портфель — проверьте перед расчётом*", ""]
    body: list[str] = []

    shown = view.positions[:_MANUAL_PREVIEW_ROWS]
    if shown:
        body.append(f"{'Тикер':<12}{'Распознан':<16}{'Кол-во':>10}"
                    f"{'Цена':>12}{'Доля':>8}")
        body.append("─" * 58)
    for row in shown:
        share = "—" if row.share is None else f"{row.share * 100:.1f}%"
        body.append(f"{_fit(_md_safe(row.raw_ticker), 11):<12}"
                    f"{_fit(_md_safe(row.resolved), 15):<16}"
                    f"{_fmt_amount(row.quantity, 2):>10}"
                    f"{_fmt_amount(row.price, 2):>12}"
                    f"{share:>8}")
    hidden = view.positions[_MANUAL_PREVIEW_ROWS:]
    if hidden:
        rest = sum(r.value_base or 0.0 for r in hidden)
        body.append(f"… и ещё {len(hidden)} позиций на "
                    f"{_fmt_amount(rest)} {view.base_currency}")

    if view.cash:
        body.append("")
        body.append("💵 Кэш")
        for row in view.cash:
            share = "—" if row.share is None else f"{row.share * 100:.1f}%"
            label = (row.currency if row.currency == view.base_currency
                     else f"{row.currency} → {view.base_currency}")
            body.append(f"{label:<28}{_fmt_amount(row.quantity, 2):>14}{share:>8}")

    body.append("")
    body.append(f"Итого: {_fmt_amount(view.total_base)} {view.base_currency}")
    lines.append("```\n" + "\n".join(body) + "\n```")

    if not view.shares_are_meaningful:
        lines.append("⚠️ Суммарная стоимость не положительна — доли не показаны.")

    for note in view.fx_notes:
        stamp = f" (на {note.as_of})" if note.as_of else ""
        lines.append(f"💱 {note.currency} → {view.base_currency}: "
                     f"{_fmt_fx_note(note.currency, view.base_currency, note.rate)}"
                     f"{stamp}")
    if view.unconvertible:
        lines.append(
            "⚠️ Нет курса для " + ", ".join(_md_safe(c) for c in view.unconvertible)
            + " — позиции в этой валюте *не учтены* в итоге и в расчёт не пойдут.")
    if view.proxied:
        lines.append(
            f"🔁 Для риск-модели заменены: {_fmt_pairs(view.proxied)}. "
            "Это НЕ смена бумаги: стоимость и P&L считаются по вашей позиции, "
            "заменитель нужен только там, где у бумаги нет своей истории цен.")
    if view.auto_resolved:
        lines.append(f"🔎 Распознаны как: {_fmt_pairs(view.auto_resolved)}")

    note = _manual_coverage_note(coverage)
    if note:
        lines.append(note)

    if view.excluded:
        lines.append(f"⚠️ *Исключено строк: {len(view.excluded)}*")
        for line_no, reason in view.excluded[:_MANUAL_PREVIEW_ERRORS]:
            lines.append(f"  • строка {line_no}: {_md_safe(reason)}")
        rest = len(view.excluded) - _MANUAL_PREVIEW_ERRORS
        if rest > 0:
            lines.append(f"  • … и ещё {rest}")

    return _clip_to_telegram_limit("\n".join(lines))


def _manual_coverage_note(coverage) -> str:
    """Строка про ценовую историю — ТОЛЬКО когда источнику есть что сказать.

    🔴 Условие переписано вместе с источником ответа (`AUDIT §−86`). Раньше
    отвечал кэш Фазы 1, а он у ручного провайдера был холодным ВСЕГДА: в
    `unknown` попадал весь портфель, и предупреждение «по 12 бумагам истории
    нет» было бы ложью — отсюда требование «часть книги покрыта, часть нет».

    Теперь отвечает БАЗА КОТИРОВОК, и она знает не «спрашивали ли мы раньше», а
    есть ли бумага вообще. Поэтому «не покрыто НИЧЕГО» — валидный и самый
    важный ответ: пользователь узнаёт до расчёта, что отчёт не построится.
    Единственное, что по-прежнему обязано молчать, — недоступная база
    (`answered=False`): «базы нет» и «вашей бумаги нет» — разные утверждения.
    """
    if coverage is None or not getattr(coverage, "answered", False):
        return ""
    if not coverage.unknown:
        return ""
    names = ", ".join(_md_safe(t) for t in coverage.unknown[:10])
    rest = len(coverage.unknown) - 10
    if rest > 0:
        names += f" и ещё {rest}"
    if not coverage.covered:
        return ("🔴 *Ценовой истории нет ни по одной бумаге:* " + names +
                ". Отчёт по такому списку не построится — проверьте написание "
                "тикеров. Токен не спишется в любом случае.")
    return ("ℹ️ Ценовой истории нет по: " + names +
            ". Эти позиции честно выпадут из расчёта — отчёт об этом скажет.")


#: Сколько тикеров с причиной показывать в отказе — остальное «и ещё N».
_NO_QUOTES_SHOWN = 10


def _public_quote_reason(reason: str) -> str:
    """Причина провайдера → текст для чата. Текст исключения базы (путь к
    файлу) наружу не идёт (F-6): «недоступна» и всё."""
    text = str(reason or "").strip() or "нет котировок"
    if text.startswith("база котировок недоступна"):
        return "база котировок недоступна"
    return text


def _manual_no_quotes_text(tickers, failed: dict, *, base_down: bool) -> str:
    """Отказ ручного отчёта на шаге 1: ни одной котировки (`§−125`).

    `tickers` — бумаги портфеля в виде движка, `failed` — причины провайдера
    по тикеру (`history_result.failed`). Раньше причины уходили только в лог,
    а пользователь получал «движок упал» и «проверьте подключение к брокеру».
    """
    if base_down:
        head = ("⚠️ *Отчёт не построен: база котировок для ручного портфеля "
                "сейчас недоступна.*\n\nЭто сбой на нашей стороне, а не в вашем "
                "портфеле. Повторите позже; если повторяется — /support.")
    else:
        head = ("⚠️ *Отчёт не построен: ни по одной бумаге нет котировок.*\n\n"
                "Ручной портфель оценивается по базе котировок, а не по брокеру. "
                "Что не нашлось:")
    lines = [head]
    shown = list(dict.fromkeys(str(t) for t in tickers))
    for t in shown[:_NO_QUOTES_SHOWN]:
        lines.append(f"• `{_md_safe(t)}` — {_md_safe(_public_quote_reason(failed.get(t)))}")
    if len(shown) > _NO_QUOTES_SHOWN:
        lines.append(f"• … и ещё {len(shown) - _NO_QUOTES_SHOWN}")
    if not base_down:
        lines.append("\nПроверьте написание тикеров (`AAPL`, `MSFT`) или уберите "
                     "эти позиции в «✏️ Ручной портфель».")
    return "\n".join(lines)


@portfolio_router.message(
    StateFilter(ManualPortfolio.Input, ManualPortfolio.Confirm), F.text)
async def msg_manual_input(message: Message, state: FSMContext) -> None:
    """Текст портфеля → экран подтверждения. Разбор НИКОГДА не бросает.

    Состояние `Confirm` обрабатывается ТЕМ ЖЕ хендлером намеренно. Увидев на
    экране подтверждения ошибку, человек чаще всего просто присылает
    исправленный список, не нажимая «Исправить». Без этого фильтра такое
    сообщение не подходило ни под один хендлер (`msg_text_fallback` стоит под
    `StateFilter(None)`) и ПРОПАДАЛО молча: пользователь набрал портфель
    заново, а бот не ответил ничего.
    """
    user_id = message.from_user.id
    text = message.text or ""

    # 🔴 MP-D · I-9, ТРЕТИЙ гард флага.  Четыре соседних входа в ручной флоу
    # флаг спрашивают, а этот — нет: он закрыт только `StateFilter`, то есть
    # состоянием FSM.  Безопасно это было ровно по одной причине: хранилище
    # состояний живёт в ПАМЯТИ, и смена флага = новая ревизия Cloud Run =
    # пустое FSM.  При переезде на Redis состояние переживёт рестарт, и
    # выключение флага перестанет выключать фичу — то есть флаг отката
    # перестанет откатывать.  Гард ставится сейчас, пока цена ошибки нулевая.
    if not manual_portfolio_enabled(user_id):
        logger.info("MANUAL: ввод при выключенном флаге user=%s", user_id)
        await state.clear()
        await message.answer(
            "ℹ️ Ручной ввод портфеля пока недоступен. "
            "Выберите демо-режим или подключите брокера.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_connect_choice(user_id),
        )
        return

    if len(text.encode("utf-8")) > MANUAL_DRAFT_MAX_BYTES:
        await message.answer(
            "⚠️ Слишком длинный ввод. Пришлите портфель до "
            f"{MANUAL_DRAFT_MAX_BYTES // 1024} КБ — "
            "загрузка файлом появится отдельно.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        async with user_slot(user_id):
            loop = asyncio.get_running_loop()
            report, view, coverage = await loop.run_in_executor(
                None, _manual_review_sync, text)

            if not view.has_positions:
                # Полный провал разбора: остаёмся в Input, черновик НЕ пишем —
                # сохранять нечего, а перезапись стёрла бы прошлый рабочий ввод.
                head = ("😕 Не удалось разобрать ни одной позиции.\n\n"
                        if view.excluded else
                        "😕 Не вижу ни одной позиции в этом сообщении.\n\n")
                details = "\n".join(
                    f"  • строка {n}: {_md_safe(r)}"
                    for n, r in view.excluded[:_MANUAL_PREVIEW_ERRORS])
                await message.answer(head + details + "\n\nПопробуйте ещё раз.",
                                     parse_mode=ParseMode.MARKDOWN)
                return

            # §5: ОДНА запись за флоу, ровно на переходе Input → Confirm.
            # SQLite лежит на gcsfuse — писать на каждое сообщение значило бы
            # платить задержкой за то, что пользователю не нужно.
            try:
                await save_manual_draft(user_id, text)
            except ManualDraftTooLarge as exc:         # уже проверено выше
                logger.warning("MANUAL: черновик не сохранён user=%s: %s",
                               user_id, exc)
            except Exception as exc:                   # noqa: BLE001
                # Черновик — удобство, а не условие расчёта: сбой БД не имеет
                # права отнять у пользователя только что разобранный портфель.
                logger.warning("MANUAL: черновик не сохранён user=%s: %s",
                               user_id, exc)

            await state.update_data(manual_text=text)
            await state.set_state(ManualPortfolio.Confirm)
            await message.answer(_format_manual_confirmation(view, coverage),
                                 parse_mode=ParseMode.MARKDOWN,
                                 reply_markup=kb_manual_confirm())
    except SlotBusy:
        # Слот общий с расчётом отчёта, поэтому занять его мог и предыдущий
        # ввод, и идущий фоновый анализ — формулировка покрывает оба случая
        # честно, вместо того чтобы угадывать.
        await message.answer(
            "⏳ *Секунду — у вас уже идёт обработка.*\n\n"
            "Пришлите текст ещё раз, когда предыдущий запрос завершится.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:                           # noqa: BLE001
        error_id = uuid.uuid4().hex[:12]
        logger.exception("MANUAL: разбор упал [%s] user=%s: %s",
                         error_id, user_id, exc)
        await message.answer(
            "😔 Не удалось обработать ввод.\n\n"
            f"Код ошибки для поддержки: `{error_id}`\n"
            "✅ Токен *не списан* — попробуйте ещё раз.",
            parse_mode=ParseMode.MARKDOWN,
        )


@portfolio_router.callback_query(F.data.startswith("manual:"))
async def cb_manual_action(callback: CallbackQuery, state: FSMContext) -> None:
    """Кнопки ручного флоу: черновик, подтверждение, правка, отмена."""
    await callback.answer()
    action = callback.data.split(":", 1)[1]
    user_id = callback.from_user.id
    data = await state.get_data()

    if not manual_portfolio_enabled(user_id):
        # Тот же гард, что и на входе: сообщение с кнопками переживает
        # выключение флага.
        await state.clear()
        await callback.message.answer(
            "ℹ️ Ручной ввод портфеля пока недоступен.",
            reply_markup=kb_connect_choice(user_id),
        )
        return

    if action == "fresh":
        await delete_manual_draft(user_id)
        await _manual_ask_for_input(callback.message, state)
        return

    if action == "resume":
        draft = await get_manual_draft(user_id)
        text = str((draft or {}).get("text") or "")
        if not text.strip():
            await _manual_ask_for_input(callback.message, state)
            return
        # Черновик прогоняется через АКТУАЛЬНЫЙ парсер, а не восстанавливается
        # разобранным: правила разбора между версиями меняются (за одну Фазу 4
        # их поменяли трижды), и сохранённый результат прошлой версии был бы
        # тихо неверным.
        await callback.message.answer("⏳ Восстанавливаю черновик…")
        try:
            async with user_slot(user_id):
                loop = asyncio.get_running_loop()
                _report, view, coverage = await loop.run_in_executor(
                    None, _manual_review_sync, text)
        except SlotBusy:
            await callback.message.answer(
                "⏳ Секунду — идёт другая обработка. Нажмите ещё раз.")
            return
        except Exception as exc:                           # noqa: BLE001
            error_id = uuid.uuid4().hex[:12]
            logger.exception("MANUAL: восстановление черновика упало [%s] "
                             "user=%s: %s", error_id, user_id, exc)
            await callback.message.answer(
                "😔 Не удалось восстановить черновик.\n\n"
                f"Код ошибки для поддержки: `{error_id}`\n"
                "Ваш ввод сохранён — попробуйте ещё раз.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        if not view.has_positions:
            await callback.message.answer(
                "😕 Черновик больше не разбирается — начнём заново.")
            await delete_manual_draft(user_id)
            await _manual_ask_for_input(callback.message, state)
            return
        await state.update_data(manual_text=text)
        await state.set_state(ManualPortfolio.Confirm)
        await callback.message.answer(
            _format_manual_confirmation(view, coverage),
            parse_mode=ParseMode.MARKDOWN, reply_markup=kb_manual_confirm())
        return

    if action == "edit":
        await _manual_ask_for_input(callback.message, state,
                                    prefill=str(data.get("manual_text") or ""))
        return

    if action == "cancel":
        # Отмена — это ЯВНОЕ решение пользователя выбросить ввод, поэтому
        # черновик удаляется здесь и только здесь (сбой отчёта его сохраняет).
        await delete_manual_draft(user_id)
        await state.clear()
        await callback.message.answer(
            "❌ Ручной ввод отменён. Токены не списаны.\n\n"
            "Выберите источник портфеля:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_connect_choice(user_id),
        )
        return

    if action == "confirm":
        slug = str(data.get("slug") or "")
        # Гибрид PR-1 (D-3): текст черновика ПЕРЕНОСИТСЯ в постоянный ручной
        # портфель. Черновик остаётся до доставленного отчёта, как и раньше.
        manual_text = str(data.get("manual_text") or "")
        if not manual_text.strip():
            manual_text = str((await get_manual_draft(user_id) or {}).get("text") or "")
        saved_line = await _persist_manual_portfolio(user_id, manual_text)
        await state.clear()
        await callback.message.answer(
            "✅ *Портфель принят.*\n\n"
            "Он сохранён — можно выбирать тип анализа."
            + (f"\n\n{saved_line}" if saved_line else ""),
            parse_mode=ParseMode.MARKDOWN,
        )
        await _show_analysis_menu(callback.message, slug)


# ══════════════════════════════════════════════════════════════════════════════
# ГИБРИД PR-1 · ПОСТОЯННЫЙ РУЧНОЙ ПОРТФЕЛЬ: экран, Add / Remove / Trim
# ══════════════════════════════════════════════════════════════════════════════
# Логика правок — чистые функции `portfolio_aggregation.edits`; здесь только
# вид и доставка. Каждая правка — read-modify-write под `user_slot` (S-6):
# без слота два одновременных «+AAPL» прочли бы одну версию и одна правка
# молча потерялась бы.

_MP_EDIT_HELP = (
    "✏️ *Правка ручного портфеля*\n\n"
    "Пришлите одну или несколько строк:\n"
    "`+AAPL 10 150` — добавить (есть — докупить, цена станет средневзвешенной)\n"
    "`-AAPL 5` — уменьшить количество (цена покупки не меняется)\n"
    "`-AAPL` — удалить позицию целиком\n"
    "`+USD 500` / `-USD 500` — кэш; маржа — только явной строкой "
    "`+CASH:USD -1000`\n\n"
    "Ошибка в любой строке — не меняется ничего."
)

#: callback_data правки — недоверенный ввод (S-5): allowlist и формат.
_MP_ACTIONS = frozenset({"show", "add", "rmlist", "del", "delyes", "keep", "back"})
_MP_RM_RE = re.compile(r"^mp:rm:(\d{1,3}):([0-9a-f]{8})$")


def _mp_engine():
    """Движок ТОЛЬКО ради распознавания тикеров (`canonical_ticker`)."""
    return UniversalPortfolioManager(price_source="manual").engine


def _mp_canonical_sync(text: str) -> tuple[str, int, int]:
    """→ (канонический текст, РАЗНЫХ бумаг, отвергнуто строк).

    Счёт — `count_positions`, то же правило, что у правок и `ManualSource`."""
    engine = _mp_engine()
    canon, _lines, rejected = _mp_canonical_text(text, engine)
    return canon, count_positions(_mp_entries_of(canon, engine)), rejected


def _mp_entries_sync(text: str):
    return _mp_entries_of(text, _mp_engine())


def _mp_coverage_note_sync(text: str) -> str:
    """Предупреждение экрана портфеля о бумагах БЕЗ котировок (`§−125`).

    Правки `+…` раньше не проверяли покрытие вовсе: бумага, которой нет в базе
    котировок, сохранялась молча, а узнавал об этом пользователь только по
    отказу отчёта. Подсказка, а не условие: любой сбой — молчание.
    """
    if not str(text or "").strip():
        return ""
    try:
        from finance.manual_portfolio import parse_portfolio_text, preflight_coverage
        engine = _mp_engine()
        report = parse_portfolio_text(text, engine)
        days = env_int("HISTORY_LOOKBACK_DAYS", 1825, lo=90, hi=3650)
        return _manual_coverage_note(preflight_coverage(report, engine, days=days))
    except Exception as exc:                           # noqa: BLE001
        logger.warning("MANUAL STORE: проверка котировок пропущена: %s",
                       type(exc).__name__)
        return ""


def _mp_apply_sync(text: str, op_line: str):
    return _mp_apply_edit(text, op_line, _mp_engine())


def _mp_remove_sync(text: str, index: int, tag: str):
    return _mp_remove_at(text, index, tag, _mp_engine())


async def _persist_manual_portfolio(user_id: int, text: str) -> str:
    """Перенести подтверждённый ввод в постоянный портфель. → строка для чата.

    Хранилище — удобство, а не условие расчёта: сбой (нет мастер-ключа, БД)
    не имеет права отнять у пользователя только что принятый портфель, поэтому
    ошибки только логируются — без содержимого портфеля (S-3).
    """
    if not str(text or "").strip():
        return ""
    loop = asyncio.get_running_loop()
    try:
        canon, n_ok, _n_bad = await loop.run_in_executor(None, _mp_canonical_sync, text)
        if not canon.strip():
            return ""
        limit = manual_max_positions()
        if n_ok > limit:
            logger.info("MANUAL STORE: не сохранён user=%s: %d поз. > %d",
                        user_id, n_ok, limit)
            # Аудит `§−123`: обещание обязано совпадать с тем, что посчитается.
            # Меню источников строит ручной отчёт по СОХРАНЁННОМУ портфелю
            # первым, поэтому при включённом гибриде и уже сохранённом портфеле
            # этот ввод в расчёт НЕ пойдёт — так и говорим.
            prev_text, _unreadable = await _load_manual_portfolio_text(user_id)
            if hybrid_portfolio_enabled(user_id) and prev_text.strip():
                return (f"ℹ️ В этом вводе {n_ok} бумаг при пределе {limit} — он "
                        "*не сохранён*, и отчёты из меню строятся по прежнему "
                        "сохранённому портфелю (/portfolio). Сократите ввод, "
                        "чтобы считать по нему.")
            return (f"ℹ️ Для постоянного хранения портфель длинноват: {n_ok} "
                    f"позиций при пределе {limit} — этот расчёт пройдёт, а "
                    "сохранённым останется прежний портфель.")
        await save_manual_portfolio(user_id, canon)
        logger.info("MANUAL STORE: сохранён user=%s поз.=%d", user_id, n_ok)
        return "💾 Сохранён как ваш ручной портфель — правки: /portfolio"
    except Exception as exc:                           # noqa: BLE001
        logger.warning("MANUAL STORE: не сохранён user=%s: %s", user_id,
                       type(exc).__name__)
        return ""


async def _load_manual_portfolio_text(user_id: int) -> tuple[str, bool]:
    """→ (текст, недоступен_ли_шифротекст). Пустой текст — портфеля нет."""
    try:
        stored = await get_manual_portfolio(user_id)
    except MasterKeyRotatedError:
        logger.warning("MANUAL STORE: шифротекст не читается user=%s", user_id)
        return "", True
    return str((stored or {}).get("text") or ""), False


def kb_mp_screen(has_positions: bool, report_data: str | None = None) -> InlineKeyboardMarkup:
    """Экран ручного портфеля: правка, отчёт по нему, удаление, назад к портфелям."""
    rows = [[_btn("➕ Добавить", "mp:add")]]
    if has_positions:
        rows[0].append(_btn("➖ Убрать", "mp:rmlist"))
        if report_data:
            rows.append([_btn("📊 Отчёт по ручному портфелю", report_data)])
        rows.append([_btn("🗑 Удалить портфель", "mp:del")])
    rows.append([_btn("⬅️ Мой портфель", "mp:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_mp_remove(entries, tag: str) -> InlineKeyboardMarkup:
    """Кнопка на позицию: индекс + хэш ВЕРСИИ (S-5) — старая кнопка не удалит
    другую позицию после правки."""
    rows = [[InlineKeyboardButton(text=f"✖️ {_fit(_md_safe(e.label), 24)}",
                                  callback_data=f"mp:rm:{i}:{tag}")]
            for i, e in enumerate(entries[:_MANUAL_PREVIEW_ROWS])]
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="mp:show")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_mp_forget() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Да, удалить", callback_data="mp:delyes"),
        InlineKeyboardButton(text="❌ Отмена", callback_data="mp:keep"),
    ]])


def _fmt_exact(value: float) -> str:
    """Количество/цена на экране портфеля — БЕЗ округления до двух знаков.

    Аудит `§−123`: `_fmt_amount` печатал 0.005 BTC как «0.01», то есть экран,
    ради проверки которого он существует, показывал не то, что сохранено.
    До 8 знаков после точки, хвостовые нули срезаны, разряды — пробелом.
    """
    s = f"{value:,.8f}".replace(",", " ")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _format_mp_screen(entries, changes: list[str] | None = None,
                      note: str = "") -> str:
    """Экран «Мой ручной портфель». Только вид — числа уже в тексте портфеля."""
    # Счёт — по тому же правилу, что в «Мой портфель» и у лимита: бумаги
    # (`count_positions`), кэш — отдельно (`§−124`).
    n_sec = count_positions(entries)
    has_cash = any(e.is_cash for e in entries)
    count = (_positions(n_sec) + (" + кэш" if has_cash else "") if n_sec
             else ("только кэш" if has_cash else "пуст"))
    lines = [f"✏️ *Мой ручной портфель* · {count}", ""]
    if changes:
        lines.append("*Что изменилось:*")
        lines += [f"  • {_md_safe(c)}" for c in changes]
        lines.append("")
    if not entries:
        lines.append("Портфель пуст — добавьте позиции кнопкой «➕ Добавить».")
    else:
        body = [f"{'Тикер':<14}{'Кол-во':>14}{'Цена':>14} Вал."]
        for e in entries[:_MANUAL_PREVIEW_ROWS]:
            price = "" if e.is_cash else _fit(_fmt_exact(e.price), 14)
            body.append(f"{_fit(_md_safe(e.label), 14):<14}"
                        f"{_fit(_fmt_exact(e.quantity), 14):>14}{price:>14} "
                        f"{_md_safe(e.currency)}")
        rest = len(entries) - _MANUAL_PREVIEW_ROWS
        if rest > 0:
            body.append(f"… и ещё {rest}")
        lines.append("```\n" + "\n".join(body) + "\n```")
        if note:
            lines.append(note)
        lines.append("Правка текстом: `+AAPL 10 150` · `-AAPL 5` · `-AAPL`")
    return _clip_to_telegram_limit("\n".join(lines))


async def _show_manual_portfolio(target: Message | CallbackQuery, user_id: int,
                                 changes: list[str] | None = None, *,
                                 edit: bool = True) -> None:
    """Экран «✏️ Ручной портфель». Нажатие кнопки — правка на месте."""
    text, unreadable = await _load_manual_portfolio_text(user_id)
    if unreadable:
        await _screen(target,
                      "🔐 *Сохранённый портфель недоступен* — ключ шифрования был обновлён.\n\n"
                      "Введите портфель заново: /forget\\_portfolio удалит старую запись.",
                      kb_mp_screen(False), edit=edit)
        return
    loop = asyncio.get_running_loop()
    entries = await loop.run_in_executor(None, _mp_entries_sync, text) if text else []
    note = (await loop.run_in_executor(None, _mp_coverage_note_sync, text)
            if entries else "")
    await _screen(target, _format_mp_screen(entries, changes, note),
                  kb_mp_screen(bool(entries), "src:manual"),
                  edit=edit)


async def _manual_flag_refusal(message: Message, state: FSMContext,
                               user_id: int | None = None) -> None:
    await state.clear()
    await message.answer("ℹ️ Ручной ввод портфеля пока недоступен.",
                         reply_markup=kb_connect_choice(user_id))


async def _mp_error(message: Message, user_id: int, exc: Exception) -> None:
    """Молчание — худший ответ бота (`§−104`): сбой БД/шифра → код поддержки."""
    error_id = uuid.uuid4().hex[:12]
    logger.error("MANUAL STORE: сбой [%s] user=%s: %s", error_id, user_id,
                 type(exc).__name__)
    await message.answer("😔 Ручной портфель сейчас недоступен.\n\n"
                         f"Код ошибки для поддержки: `{error_id}`",
                         parse_mode=ParseMode.MARKDOWN)


async def cmd_portfolio(message: Message, state: FSMContext) -> None:
    """/portfolio — «💼 Мой портфель»: брокер, ручной ввод, демо (`§−124`).

    Раньше команда открывала ТОЛЬКО ручной портфель и только при включённом
    флаге; сменить источник после онбординга было негде вовсе.
    """
    user_id = message.from_user.id
    if not await _require_profile(message, user_id):
        return
    await state.clear()
    await _show_portfolio_hub(message, user_id, edit=False)


async def cmd_forget_portfolio(message: Message, state: FSMContext) -> None:
    """/forget_portfolio — удалить ручной портфель И черновик (после подтверждения).

    Работает и при выключенном флаге ручного ввода: право удалить свои данные
    не зависит от того, включена ли фича (S-9).
    """
    await state.clear()
    await message.answer(
        "🗑 *Удалить ручной портфель?*\n\n"
        "Будут удалены сохранённый портфель и незавершённый черновик ввода. "
        "Отменить удаление нельзя.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_mp_forget(),
    )


async def _forget_manual_portfolio(user_id: int) -> None:
    await delete_manual_portfolio(user_id)
    await delete_manual_draft(user_id)
    logger.info("MANUAL STORE: удалён по запросу user=%s", user_id)


@portfolio_router.callback_query(F.data.startswith("mp:"))
async def cb_manual_portfolio(callback: CallbackQuery, state: FSMContext) -> None:
    """Кнопки экрана ручного портфеля. callback_data — недоверенный ввод (S-5)."""
    await callback.answer()
    try:
        await _cb_manual_portfolio(callback, state)
    except Exception as exc:                           # noqa: BLE001
        await _mp_error(callback.message, callback.from_user.id, exc)


async def _cb_manual_portfolio(callback: CallbackQuery, state: FSMContext) -> None:
    data = str(callback.data or "")
    user_id = callback.from_user.id
    rm = _MP_RM_RE.match(data)
    action = "rm" if rm else data.split(":", 1)[1] if ":" in data else ""
    if action != "rm" and action not in _MP_ACTIONS:
        logger.warning("MANUAL STORE: подделанный callback user=%s", user_id)
        return

    if action == "delyes":
        # Удаление данных — без гейта флага (см. `cmd_forget_portfolio`).
        try:
            async with user_slot(user_id):
                await _forget_manual_portfolio(user_id)
        except SlotBusy:
            await callback.message.answer(
                "⏳ Секунду — идёт другая обработка. Нажмите ещё раз.")
            return
        await state.clear()
        await _screen(callback, "🗑 Ручной портфель и черновик удалены.",
                      kb_nav(_btn("💼 Мой портфель", "home:portfolio")))
        return
    if action == "keep":
        await _screen(callback, "👌 Ничего не удалено.",
                      kb_nav(_btn("💼 Мой портфель", "home:portfolio")))
        return

    # S-4: старая кнопка живёт в чате вечно — флаг проверяется на КАЖДОМ нажатии.
    if not manual_portfolio_enabled(user_id):
        await _manual_flag_refusal(callback.message, state, user_id)
        return

    if action == "show":
        await state.clear()
        await _show_manual_portfolio(callback, user_id)
        return
    if action == "back":
        await state.clear()
        await _show_portfolio_hub(callback, user_id)
        return
    if action == "add":
        await state.set_state(ManualPortfolio.Edit)
        await callback.message.answer(_MP_EDIT_HELP, parse_mode=ParseMode.MARKDOWN)
        return
    if action == "del":
        await _screen(callback, "🗑 *Удалить ручной портфель?*\n\nОтменить удаление нельзя.",
                      kb_mp_forget())
        return

    text, unreadable = await _load_manual_portfolio_text(user_id)
    if action == "rmlist":
        loop = asyncio.get_running_loop()
        entries = await loop.run_in_executor(None, _mp_entries_sync, text) if text else []
        if not entries:
            await _show_manual_portfolio(callback, user_id)
            return
        await state.set_state(ManualPortfolio.Edit)
        await _screen(callback,
                      "➖ *Что убрать?* Кнопка удаляет позицию целиком; чтобы уменьшить, "
                      "пришлите `-ТИКЕР КОЛИЧЕСТВО`.",
                      kb_mp_remove(entries, _mp_version_tag(text)))
        return

    # action == "rm": индекс + версия, правка под слотом.
    index, tag = int(rm.group(1)), rm.group(2)
    try:
        async with user_slot(user_id):
            text, unreadable = await _load_manual_portfolio_text(user_id)
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, _mp_remove_sync, text, index, tag)
            if res.ok:
                await save_manual_portfolio(user_id, res.new_text)
    except SlotBusy:
        await callback.message.answer(
            "⏳ Секунду — идёт другая обработка. Нажмите ещё раз.")
        return
    if not res.ok:
        await callback.message.answer(f"⚠️ {_md_safe(res.error)}")
        return
    await _show_manual_portfolio(callback, user_id, changes=res.applied)


@portfolio_router.message(StateFilter(ManualPortfolio.Edit), F.text,
                          ~F.text.startswith("/"))
async def msg_manual_edit(message: Message, state: FSMContext) -> None:
    """Текстовые правки `+…`/`-…` сохранённого портфеля (D-4)."""
    user_id = message.from_user.id
    if not manual_portfolio_enabled(user_id):
        await _manual_flag_refusal(message, state, user_id)
        return
    ops = message.text or ""
    if len(ops.encode("utf-8")) > MANUAL_DRAFT_MAX_BYTES:
        await message.answer("⚠️ Слишком длинная команда правки.")
        return
    try:
        async with user_slot(user_id):
            text, unreadable = await _load_manual_portfolio_text(user_id)
            if unreadable:
                await message.answer(
                    "🔐 Сохранённый портфель недоступен — ключ шифрования был "
                    "обновлён. Удалите его (/forget\\_portfolio) и введите заново.",
                    parse_mode=ParseMode.MARKDOWN)
                return
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, _mp_apply_sync, text, ops)
            if res.ok:
                await save_manual_portfolio(user_id, res.new_text)
    except SlotBusy:
        await message.answer(
            "⏳ *Секунду — у вас уже идёт обработка.*\n\n"
            "Пришлите правку ещё раз, когда предыдущий запрос завершится.",
            parse_mode=ParseMode.MARKDOWN)
        return
    except Exception as exc:                           # noqa: BLE001
        error_id = uuid.uuid4().hex[:12]
        logger.error("MANUAL STORE: правка упала [%s] user=%s: %s",
                     error_id, user_id, type(exc).__name__)
        await message.answer(
            "😔 Не удалось применить правку.\n\n"
            f"Код ошибки для поддержки: `{error_id}`",
            parse_mode=ParseMode.MARKDOWN)
        return
    if not res.ok:
        await message.answer(
            f"⚠️ Правка не применена: {_md_safe(res.error)}\n\n"
            "Портфель не изменился — исправьте строку и пришлите снова.")
        return
    logger.info("MANUAL STORE: правка user=%s операций=%d", user_id, len(res.applied))
    await _show_manual_portfolio(message, user_id, changes=res.applied, edit=False)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FLOW
# ══════════════════════════════════════════════════════════════════════════════

async def _send_report(
    bot: Bot,
    chat_id: int,
    user_id: int,
    tier: str,
    payload: dict | None = None,
) -> None:
    """
    Render the HTML report, push to GCS, send the user a signed URL.

    The delivery is an interactive HTML page (no PDF, no Chromium); the
    bot just hands the user a signed URL.  "Отчёт" everywhere — both in
    the function name and the user-facing copy.
    """
    report_type = TIER_LABEL[tier]

    # Render + write happen in the executor (`§−122`): Premium-инъекция быстрая,
    # но Jinja-фолбэк и запись 300+ КБ — нет, а loop у бота один на всех.
    # M-1: a template/render failure must degrade gracefully — show the user a
    # soft message with a support error-id (NOT a stack trace), log the full
    # traceback under that id, and signal the caller so the token is refunded.
    loop = asyncio.get_running_loop()

    def _render_and_write() -> str:
        html = render_report_html(payload, user_id=user_id,
                                  report_type=report_type, tier=tier)
        return write_report_html(html, user_id=user_id, tier=tier)

    try:
        local_path = await loop.run_in_executor(None, _render_and_write)
    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Report generation failed [%s] user=%s tier=%s",
                         error_id, user_id, tier)
        await bot.send_message(
            chat_id,
            "😔 Извините, отчёт временно недоступен.\n\n"
            f"Код ошибки для поддержки: `{error_id}`\n"
            "✅ Токен *не списан* — попробуйте позже или напишите в /support.",
            parse_mode=ParseMode.MARKDOWN,
        )
        raise RuntimeError("report_generation_failed") from exc

    # Push to GCS (or fall back to file:// in local-dev mode).  Сеть (PUT +
    # подпись URL) — в executor: секунды заморозки loop'а на КАЖДЫЙ отчёт (`§−122`).
    url = await loop.run_in_executor(
        None, lambda: upload_report(local_path, user_id=user_id, tier=tier))

    # A file:// URL means the GCS upload/signing failed (in production the
    # bucket is always configured).  Telegram rejects file:// links inside a
    # markdown text-link entity, so sending one would crash send_message and
    # the user would get nothing.  Signal the failure to the caller so it can
    # refund tokens and show a clear message instead.
    if url.startswith("file://"):
        raise RuntimeError("report_delivery_failed")

    # Tell the user.  The link is a plain markdown URL — Telegram renders
    # it as a preview card on most clients.
    await bot.send_message(
        chat_id,
        text=(
            f"📊 *{report_type}* готов.\n\n"
            f"[Открыть отчёт]({url})\n\n"
            "Ссылка действительна 48 часов.  Отчёт сформирован институциональным "
            "риск-движком (Euler Decomposition · Bootstrap CVaR · 4-pillar "
            "Scoring · Black-Litterman).  Штурвал всегда у вас."
        ),
        parse_mode               = ParseMode.MARKDOWN,
        disable_web_page_preview = False,
    )


async def cb_analysis_choice(callback: CallbackQuery, state: FSMContext) -> None:
    """Экран цены перед запуском (старое меню тиров, `analysis:<тир>`).

    §−124: цена — по РЕАЛЬНОМУ источнику. Прежде экран обещал демо-пользователю
    «будет списано 1 токен», хотя демо бесплатно (`_effective_cost`).
    """
    await callback.answer()
    _, tier = str(callback.data or "").split(":", 1)
    if tier not in TIER_COST:
        logger.warning("NAV: подделанный callback user=%s", callback.from_user.id)
        return
    user_id = callback.from_user.id
    try:
        source, _stored = await _resolve_portfolio_source(user_id)
    except Exception as exc:                           # noqa: BLE001
        logger.warning("NAV: источник не определён user=%s: %s", user_id,
                       type(exc).__name__)
        source = None
    if source == "undetermined":
        await _screen(callback, "📡 *Сначала выберите портфель.*\n\n"
                                "Откуда брать позиции для отчёта?",
                      kb_connect_choice(user_id))
        return
    priced = source if source in ("freedom", "manual", "demo") else None
    cost = _effective_cost(tier, priced) if priced else TIER_COST[tier]
    balance = await get_balance(user_id)
    ctx = await state.get_data()
    context_slug = ctx.get("context_slug", "menu")

    await state.update_data(tier=tier)
    text, kb = _price_screen(tier, priced, cost, balance,
                             f"confirm:{tier}:{context_slug}")
    await _screen(callback, text, kb)
    await state.set_state(AnalysisFlow.awaiting_approval)


async def cb_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    """
    Flow:
      1. Списать токены.
      2. Получить портфель из брокера (5-10 сек).
      3. Отправить превью-таблицу + сообщить, что анализ займёт 5-10 минут.
      4. Запустить MAC3 анализ как фоновую задачу — handler возвращается СРАЗУ,
         чтобы aiogram long-poll не таймаутил.
      5. По завершении задачи отправить PDF отдельным сообщением.
    """
    await callback.answer()
    _, tier, context_slug = callback.data.split(":", 2)
    await _confirm_flow(callback, state, tier)


#: Причины, с которыми бот предлагает ручной отчёт вместо брокерского (PR-2).
#: Ключ уезжает в callback_data (`fb:manual:<tier>:<код>`) — поэтому allowlist:
#: callback_data — недоверенный ввод (S-5). Значение — текст для CoVe и превью.
FALLBACK_REASON_TEXT: dict[str, str] = {
    "waf_block":   "блокировка запросов по IP на стороне брокера",
    "api_error":   "сбой API брокера",
    "parse_error": "ответ брокера не разобран",
    "timeout":     "брокер не ответил вовремя",
    "auth":        "ключи брокера отклонены",
    "error":       "сбой загрузки портфеля",
    "history":     "не загрузилась история цен",
}
_FB_RE = re.compile(r"^fb:manual:([a-z]{1,12}):([a-z_]{1,16})$")


async def _manual_fallback_offer(user_id: int, tier: str,
                                 reason: str) -> tuple[str, InlineKeyboardMarkup | None]:
    """Хвост сообщения об отказе брокера: что можно сделать ВМЕСТО (PR-2 §3).

    Три ветки: ручной портфель есть → кнопка ручного отчёта; нет → честное
    «пуст» и кнопка ввода; флаг ручного ввода выключен → ничего (I-9: текст и
    клавиатура ровно прежние).
    """
    if not manual_portfolio_enabled(user_id):
        return "", None
    reason = reason if reason in FALLBACK_REASON_TEXT else "error"
    try:
        text, _unreadable = await _load_manual_portfolio_text(user_id)
    except Exception as exc:                           # noqa: BLE001
        logger.warning("FALLBACK: ручной портфель не прочитан user=%s: %s",
                       user_id, type(exc).__name__)
        text = ""
    if text.strip():
        return ("\n\n📈 Можно построить отчёт по вашим *ручным активам* — "
                "котировки возьмём из независимого публичного источника, тариф "
                "тот же, токен спишется только после готового отчёта.",
                InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(
                    text="📈 Сгенерировать отчёт по ручным активам",
                    callback_data=f"fb:manual:{tier}:{reason}")]]))
    return ("\n\nРучной портфель пуст, отчёт сейчас невозможен. Попробуйте "
            "позже или добавьте активы вручную.",
            InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(
                text="➕ Добавить активы вручную", callback_data="mp:add")]]))


class BrokerBudgetExceeded(RuntimeError):
    """Брокер не ответил за `BROKER_FETCH_BUDGET_S` (D-8)."""


async def _answer_fallback_offer(message: Message, user_id: int, tier: str,
                                 reason: str) -> None:
    """То же предложение отдельным сообщением — под уже отправленным отказом."""
    text, kb = await _manual_fallback_offer(user_id, tier, reason or "api_error")
    if kb is not None:
        await message.answer(text.strip(), parse_mode=ParseMode.MARKDOWN,
                             reply_markup=kb)


async def cb_fallback_manual(callback: CallbackQuery, state: FSMContext) -> None:
    """`fb:manual:<tier>:<причина>` — ручной отчёт, когда брокер недоступен.

    Проходит ВЕСЬ путь подтверждения заново (слот, флаг, наличие портфеля,
    баланс), а не прыгает в фоновую задачу: кнопка живёт в чате вечно, и к
    моменту нажатия любое из этих условий могло измениться (S-4).
    """
    await callback.answer()
    m = _FB_RE.match(str(callback.data or ""))
    if not m or m.group(1) not in TIER_COST or m.group(2) not in FALLBACK_REASON_TEXT:
        logger.warning("FALLBACK: подделанный callback user=%s", callback.from_user.id)
        return
    if not manual_portfolio_enabled(callback.from_user.id):
        await callback.message.answer("ℹ️ Ручной ввод портфеля пока недоступен.",
                                      reply_markup=kb_connect_choice(callback.from_user.id))
        return
    await _confirm_flow(callback, state, m.group(1), source_override="manual",
                        fallback_reason=m.group(2))


# ══════════════════════════════════════════════════════════════════════════════
# ГИБРИД PR-3 · АГРЕГИРОВАННЫЙ ОТЧЁТ (брокер + ручной ввод) и меню источников
# ══════════════════════════════════════════════════════════════════════════════
# Сборка состава — `portfolio_aggregation` (L1); здесь только ключи, тексты и
# кнопки. Математика отчёта та же: склейку дублей делает движок.

#: Источники, которые можно выбрать КНОПКОЙ (callback_data — недоверенный ввод).
REPORT_SOURCES = ("freedom", "manual", AGGREGATED_SOURCE, "demo")
#: Кнопки выбора портфеля в «📊 Новый отчёт» (шаг 1 из 2).
_SOURCE_BUTTON = {
    "freedom": "📊 Freedom Broker",
    "manual": "✏️ Ручной портфель",
    AGGREGATED_SOURCE: "🌐 Freedom + ручной",
    "demo": "📋 Демо · бесплатно",
}
_SRC_RE = re.compile(r"^src:([a-z]{1,12})$")
_RPT_RE = re.compile(r"^(rpt|rptgo):([a-z]{1,12}):([a-z]{1,12})$")
_AGG_RE = re.compile(r"^agg:sum:([a-z]{1,12}):([0-9a-f]{8})$")


def _overlap_tag(overlaps: list[str]) -> str:
    """Хэш набора пересечений: подтверждение относится к ЭТОМУ набору."""
    import hashlib

    return hashlib.sha256(",".join(sorted(overlaps)).encode("utf-8")).hexdigest()[:8]


class _BrokerViaBot:
    """Транспорт `FreedomSource` через `_fetch_portfolio_sync` — тот же вызов,
    что у брокерского отчёта (и та же точка подмены в тестах)."""

    def __init__(self, api_key: str, secret_key: str, login: str) -> None:
        self._args = (api_key, secret_key, login)

    def fetch_portfolio(self):
        return _fetch_portfolio_sync(*self._args)


class _ManualViaBot:
    """`ManualSource`, чей движок строится уже на потоке executor'а."""

    name = "manual"

    def __init__(self, text: str) -> None:
        self._text = text

    def load(self):
        return ManualSource(self._text, _mp_engine()).load()


async def _aggregated_refuse(callback: CallbackQuery, state: FSMContext, user_id: int,
                             text: str, kb: InlineKeyboardMarkup | None = None) -> None:
    await _release_user_slot(user_id)
    await callback.message.answer(text + "\n\n✅ Токен *не списан*.",
                                  parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
    await state.clear()


def _kb_manual_report(tier: str, reason: str | None = None) -> InlineKeyboardMarkup:
    data = f"fb:manual:{tier}:{reason}" if reason else f"rptgo:manual:{tier}"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📈 Сгенерировать отчёт по ручным активам",
                              callback_data=data)],
        [InlineKeyboardButton(text="✏️ Мой ручной портфель", callback_data="mp:show")],
    ])


async def _aggregated_step1(callback: CallbackQuery, state: FSMContext, tier: str,
                            user_id: int, *, api_key: str, secret_key: str, login: str,
                            key_origin: str | None, overlap_tag: str | None):
    """Брокер ∥ ручной портфель → гейт I-15 → слияние → экран D-5.

    → `(df, composition, freedom_result)` либо `None`, если отказ уже показан
    (и слот уже снят). Исключений наружу не бросает.
    """
    manual_text, unreadable = await _load_manual_portfolio_text(user_id)
    if unreadable or not manual_text.strip():
        await _aggregated_refuse(
            callback, state, user_id,
            "📝 *Ручной портфель пуст* — агрегированному отчёту нечего добавить "
            "к счёту брокера.",
            InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(
                text="➕ Добавить активы вручную", callback_data="mp:add")]]))
        return None
    if key_origin is None:                              # pragma: no cover — ветка ключей
        await _aggregated_refuse(callback, state, user_id,
                                 "⚠️ *Брокер не подключён.*")
        return None

    budget = broker_fetch_budget_s()
    freedom_src = FreedomSource(api_key, secret_key, login, key_origin=key_origin,
                                connector_factory=_BrokerViaBot)
    freedom, manual = await asyncio.gather(
        load_with_budget(freedom_src, budget),
        load_with_budget(_ManualViaBot(manual_text), budget))
    logger.info("HYBRID: user=%s freedom_ok=%s reason=%s [%s] manual_ok=%s поз.=%d+%d",
                user_id, freedom.ok, freedom.failure_reason or "-",
                freedom.error_id or "-", manual.ok, freedom.positions, manual.positions)

    if not freedom.ok:
        reason = freedom.failure_reason or "error"
        if reason == "empty":
            await _aggregated_refuse(
                callback, state, user_id,
                "📭 *На счёте Freedom нет позиций* — агрегированный отчёт совпал бы "
                "с ручным.", _kb_manual_report(tier))
            return None
        if reason == "auth":
            await _aggregated_refuse(
                callback, state, user_id,
                "⚠️ *Агрегированный отчёт сейчас невозможен.*\n\n"
                "Брокер отклонил ключи — похоже, они неверны или отозваны; "
                "проверьте их в /start → 🔗 Freedom Broker API.",
                _kb_manual_report(tier, "auth"))
            return None
        # D-7: брокер недоступен — причина по `_broker_outage_advice`, а не одна
        # фраза на все случаи (`§−94`).
        if reason == "timeout":
            why = (f"⚠️ Серверы Freedom Broker сейчас недоступны — брокер не "
                   f"ответил за {budget} с. Обычно это проходит за 5–15 минут.")
        else:
            why = _broker_outage_advice(
                reason if reason in ("waf_block", "parse_error") else "api_error",
                freedom.error_id or uuid.uuid4().hex[:12])
        fb_reason = reason if reason in FALLBACK_REASON_TEXT else "error"
        await _aggregated_refuse(
            callback, state, user_id,
            "🌐 *Агрегированный отчёт сейчас невозможен* — Freedom Broker "
            "недоступен, а без живого счёта брокерские котировки показывать "
            "нельзя.\n\n" + why,
            _kb_manual_report(tier, fb_reason))
        return None

    if not manual.ok:
        if manual.failure_reason == "too_many":
            text = (f"📝 В ручном портфеле больше {manual.detail.get('limit')} "
                    "позиций — сократите его в /portfolio.")
        else:
            text = "📝 *Ручной портфель не разобрался* — откройте его в /portfolio."
        await _aggregated_refuse(callback, state, user_id, text,
                                 InlineKeyboardMarkup(inline_keyboard=[[
                                     InlineKeyboardButton(text="✏️ Мой ручной портфель",
                                                          callback_data="mp:show")]]))
        return None

    try:
        merged = PortfolioAggregator().merge(freedom, manual)
    except AggregatedNotPermitted as exc:
        logger.error("HYBRID: I-15 отказ user=%s: %s", user_id, exc.reason)
        await _aggregated_refuse(
            callback, state, user_id,
            "🌐 *Агрегированный отчёт сейчас невозможен* — живой портфель брокера "
            "не подтверждён.", _kb_manual_report(tier, "error"))
        return None
    except AggregationRefused as exc:
        if exc.reason == "currency_conflict":
            text = ("💱 *Одна бумага — в разных валютах.* На счёте Freedom и в "
                    "ручном портфеле указаны разные валюты для: "
                    f"{_md_safe(', '.join(exc.tickers))}. Сложить такие лоты "
                    "нельзя — исправьте валюту в ручном портфеле.")
        elif exc.reason == "too_many":
            text = (f"📚 Вместе получается больше {exc.limit} позиций — это предел "
                    "агрегированного отчёта. Сократите ручной портфель.")
        else:
            text = "📝 *Ручной портфель пуст* — добавьте активы в /portfolio."
        await _aggregated_refuse(callback, state, user_id, text,
                                 InlineKeyboardMarkup(inline_keyboard=[[
                                     InlineKeyboardButton(text="✏️ Исправить ручной портфель",
                                                          callback_data="mp:show")]]))
        return None

    if merged.overlaps and overlap_tag != _overlap_tag(merged.overlaps):
        # D-5: пересечения показываются ДО расчёта, решение — за пользователем.
        await _release_user_slot(user_id)
        await callback.message.answer(
            "🔁 *Эти бумаги есть и на счёте Freedom, и в ручном портфеле:* "
            f"{_md_safe(', '.join(merged.overlaps))}.\n\n"
            "Ручной портфель не должен повторять брокерский — иначе позиция "
            "учтётся дважды.\n\n✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(
                    text="✅ Это разные счета — суммировать",
                    callback_data=f"agg:sum:{tier}:{_overlap_tag(merged.overlaps)}")],
                [InlineKeyboardButton(text="✏️ Исправить ручной портфель",
                                      callback_data="mp:show")],
                [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")],
            ]),
        )
        await state.clear()
        return None
    return merged.frame, merged.composition(), freedom


async def _show_source_menu(target: Message | CallbackQuery, user_id: int, *,
                            edit: bool = False, ov: dict | None = None) -> None:
    """«📊 Новый отчёт» · шаг 1 из 2 (D-9): по какому портфелю.

    Кнопки — только доступные портфели; недоступные названы строкой с причиной,
    чтобы было ясно, что сделать, а не куда пропал пункт. Выключенная фича
    (ручной ввод / «Freedom + ручной») не упоминается вовсе (I-9).
    """
    ov = ov or await _portfolio_overview(user_id)
    rows, closed = [], []
    for src in REPORT_SOURCES:
        if (src == AGGREGATED_SOURCE and not ov["hybrid"]) or (
                src == "manual" and not ov["manual_on"]):
            continue
        why = _source_refusal(src, ov)
        if why is None:
            rows.append([_btn(_SOURCE_BUTTON[src], f"src:{src}")])
        else:
            closed.append(f"• {_SOURCE_BUTTON[src]} — {why}")
    rows.append([_btn("💼 Мой портфель", "home:portfolio"), _btn("🏠 Меню", "home:menu")])
    text = "📊 *Новый отчёт* · шаг 1 из 2\nПо какому портфелю?"
    if ov["hybrid"] and _source_refusal(AGGREGATED_SOURCE, ov) is None:
        text += "\n\n🌐 — один отчёт по счёту Freedom и ручному портфелю вместе."
    if closed:
        text += "\n\n*Недоступно:*\n" + "\n".join(closed)
    await _screen(target, text, InlineKeyboardMarkup(inline_keyboard=rows), edit=edit)


def kb_report_tiers(source: str, *, multi: bool = True) -> InlineKeyboardMarkup:
    """Тиры по портфелю, по одному в строке. Тарифы — `TIER_COST` (D-10, I-4).

    `multi` — у пользователя несколько портфелей: «назад» ведёт к выбору
    портфеля; иначе — в «Мой портфель» (выбирать там не из чего).
    """
    def _label(tier: str) -> str:
        cost = _effective_cost(tier, source)
        price = "бесплатно" if cost == 0 else _tokens(cost)
        return f"{_TIER_ICON[tier]} {_TIER_SHORT[tier]} · {price}"
    nav = (_btn("⬅️ Другой портфель", "home:report") if multi
           else _btn("💼 Мой портфель", "home:portfolio"))
    return InlineKeyboardMarkup(inline_keyboard=[
        [_btn(_label("base"), f"rpt:{source}:base")],
        [_btn(_label("scenario"), f"rpt:{source}:scenario")],
        [_btn(_label("deep"), f"rpt:{source}:deep")],
        [nav, _btn("🏠 Меню", "home:menu")],
    ])


async def _hybrid_menu_refusal(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("ℹ️ Этот пункт сейчас недоступен — откройте меню заново.",
                         reply_markup=kb_nav())


async def cb_report_source(callback: CallbackQuery, state: FSMContext) -> None:
    """`src:<портфель>` → тиры. Доступность — заново на каждом нажатии (S-4)."""
    await callback.answer()
    m = _SRC_RE.match(str(callback.data or ""))
    user_id = callback.from_user.id
    if not m or m.group(1) not in REPORT_SOURCES:
        logger.warning("HYBRID: подделанный callback user=%s", user_id)
        return
    await state.clear()
    await _show_tiers_for(callback, state, user_id, m.group(1))


async def cb_report_tier(callback: CallbackQuery, state: FSMContext) -> None:
    """`rpt:<источник>:<тир>` — экран цены; `rptgo:…` — запуск полного пути."""
    await callback.answer()
    m = _RPT_RE.match(str(callback.data or ""))
    user_id = callback.from_user.id
    if (not m or m.group(2) not in REPORT_SOURCES or m.group(3) not in TIER_COST):
        logger.warning("HYBRID: подделанный callback user=%s", user_id)
        return
    kind, source, tier = m.groups()
    why = _source_refusal(source, await _portfolio_overview(user_id))
    if why is not None:
        await _screen(callback, f"ℹ️ Портфель недоступен: {why}.",
                      _kb_refusal(source, why))
        return
    if kind == "rpt":
        cost = _effective_cost(tier, source)
        balance = await get_balance(user_id)
        text, kb = _price_screen(tier, source, cost, balance, f"rptgo:{source}:{tier}")
        await _screen(callback, text, kb)
        return
    await _confirm_flow(callback, state, tier, source_override=source)


async def cb_aggregated_overlap(callback: CallbackQuery, state: FSMContext) -> None:
    """`agg:sum:<тир>:<хэш набора>` — «это разные счета, суммировать» (D-5).

    Брокер запрашивается ЗАНОВО: I-15 требует живого фетча в том же запросе,
    а состав счёта мог измениться, пока экран висел в чате.
    """
    await callback.answer()
    m = _AGG_RE.match(str(callback.data or ""))
    if not m or m.group(1) not in TIER_COST:
        logger.warning("HYBRID: подделанный callback user=%s", callback.from_user.id)
        return
    if not hybrid_portfolio_enabled(callback.from_user.id):
        await _hybrid_menu_refusal(callback.message, state)
        return
    await _confirm_flow(callback, state, m.group(1), source_override=AGGREGATED_SOURCE,
                        overlap_tag=m.group(2))


async def _confirm_flow(callback: CallbackQuery, state: FSMContext, tier: str, *,
                        source_override: str | None = None,
                        fallback_reason: str | None = None,
                        overlap_tag: str | None = None) -> None:
    """Тело подтверждения отчёта: источник → баланс → загрузка → превью → фон.

    `source_override` — источник, выбранный КНОПКОЙ (`fb:`), а не режимом по
    умолчанию из профиля: `connection_mode` при этом не меняется.
    `fallback_reason` — отчёт строится вместо брокерского (PR-2): причина
    доезжает до превью и до CoVe.
    `overlap_tag` — пользователь подтвердил ИМЕННО этот набор пересечений
    брокер × ручной ввод (экран D-5); другой набор покажет экран заново.
    """
    user_id = callback.from_user.id
    cost    = TIER_COST[tier]

    # Single-flight guard: refuse a second concurrent report from the same
    # user.  Without this, double-tapping the button (or running BASE +
    # DEEP back-to-back) doubles the worker load AND double-charges tokens
    # on the second deduct.  Worker pool is single-instance (Cloud Run
    # max-instances=1), so one greedy user could starve every other user.
    if not await _try_acquire_user_slot(user_id, tier=tier):
        await callback.message.edit_text(
            "⏳ *У вас уже выполняется анализ.*\n\n"
            "Подождите завершения предыдущего отчёта — следующий запрос "
            "обработаем сразу после.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Fix C: the portfolio source decides the price (демо бесплатно), so it is
    # resolved BEFORE the balance gate — a 0-balance user may still run demo.
    # An undetermined source must never produce a (paid) report.
    # Guarded: this touches the tokenomics DB and (when mode isn't 'freedom')
    # the vault file on gcsfuse — an I/O error escaping here would leak the
    # single-flight slot and lock the user out until restart.
    try:
        if source_override is not None:
            source, stored_mode = source_override, None
        else:
            source, stored_mode = await _resolve_portfolio_source(user_id)
    except Exception as exc:                       # noqa: BLE001
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Source resolution failed for %s [%s]: %s",
                         user_id, error_id, exc)
        await _release_user_slot(user_id)
        await callback.message.edit_text(
            "ℹ️ *Не удалось определить источник портфеля.*\n\n"
            f"Код ошибки для поддержки: `{error_id}`\n\n"
            "✅ Токен *не списан*. Попробуйте ещё раз через пару минут.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return
    if source == "undetermined":
        logger.warning(
            "PORTFOLIO SOURCE: undetermined  user=%s  stored_mode=%r — "
            "нет ни явного выбора, ни ключей; отчёт не формируем.",
            user_id, stored_mode,
        )
        await _release_user_slot(user_id)
        # Recovery must be ONE TAP away: a returning user has no other path to
        # the connection screen (/start skips it once a profile exists), so
        # pointing them at /start created a dead loop — attach the source
        # keyboard right here instead (bug 2026-07-16, 2nd user stuck).
        await callback.message.edit_text(
            "⚠️ *Источник портфеля не выбран.*\n\n"
            "Похоже, подключение не завершено или было сброшено. "
            "Выберите источник прямо здесь:\n\n"
            "✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_connect_choice(user_id),
        )
        await state.clear()
        return
    if source == "manual" and not manual_portfolio_enabled(user_id):
        # 🔴 ФЛАГ ОТКАТА, и проверяться он обязан ЗДЕСЬ, а не только на входе
        # в ручной ввод. Режим `manual` хранится в профиле: пользователь, уже
        # выбравший его, приходит сюда напрямую из меню тиров — мимо
        # `cb_manual_action`, где флаг проверяется. Без этой ветки выключение
        # фичи в проде не остановило бы расчёты у тех, кто её уже включил, то
        # есть откат перестал бы быть откатом (`PHASE_06 §1`).
        logger.info("MANUAL: расчёт запрошен при выключенном флаге user=%s",
                    user_id)
        await _release_user_slot(user_id)
        await callback.message.edit_text(
            "🛠 *Расчёт по ручному портфелю временно недоступен.*\n\n"
            "Ваш ввод сохранён — он не потеряется.\n\n"
            "Можно выбрать другой источник:\n\n"
            "✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_connect_choice(user_id),
        )
        await state.clear()
        return
    if source == AGGREGATED_SOURCE and not hybrid_portfolio_enabled(user_id):
        # S-4: кнопка агрегированного отчёта переживает выключение флага.
        logger.info("HYBRID: расчёт запрошен при выключенном флаге user=%s", user_id)
        await _release_user_slot(user_id)
        await callback.message.edit_text(
            "🛠 *Агрегированный отчёт временно недоступен.*\n\n"
            "✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return
    cost = _effective_cost(tier, source)

    # H1 (Phase-3): NO upfront deduction.  Read-only balance pre-check —
    # refuse if the user cannot afford the report, but DO NOT charge until
    # Checkpoint 3 (report successfully rendered + uploaded to GCS).
    balance = await get_balance(user_id)
    if cost > 0 and balance < cost:
        await _release_user_slot(user_id)
        await callback.message.edit_text(
            f"❌ *Недостаточно токенов.*\n\n"
            f"Требуется: *{cost}*, доступно: *{balance}*.\n\n"
            "Пополните баланс командой /topup.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return

    if source == "demo":
        await callback.message.edit_text(
            "⏳ Загружаю *демо-портфель (шаблон)*…\n\n"
            "📋 Отчёт по демо-портфелю *бесплатный* — токены не списываются.",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        # 🔴 Текст про Freedom Broker для ручного ввода — ЛОЖЬ, и не безобидная
        # (`PHASE_06 §3`): пользователь ручного портфеля может не быть клиентом
        # брокера вовсе, а строка утверждала бы обратное. Тот же I-12 на слое
        # текста, что и подпись CoVe (F-6).
        what = ("Собираю ваш портфель" if source == "manual"
                else "Загружаю счёт Freedom Broker и ваш ручной портфель"
                if source == AGGREGATED_SOURCE
                else "Подключаюсь к Freedom Broker и загружаю портфель")
        await callback.message.edit_text(
            f"⏳ {what}…\n\n"
            f"💳 Токен спишется *только после готового отчёта* "
            f"(сейчас на балансе: *{balance}*).",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── Шаг 1 — подгружаем портфель (быстрая часть) ──────────────────────
    loop = asyncio.get_running_loop()
    profile    = await get_profile(user_id)
    bench_tick = _resolve_bench_ticker(profile)

    manual_from_store = False      # ручной отчёт построен по СОХРАНЁННОМУ портфелю
    # I-15: агрегированный отчёт ходит к брокеру ТЕМИ ЖЕ ключами, что и
    # freedom (vault пользователя или сервисные — только администратору), и
    # запоминает их происхождение: это часть доказательства статуса клиента.
    key_origin = None
    if source in ("freedom", AGGREGATED_SOURCE):
        try:
            keys = await loop.run_in_executor(None, _get_keys_sync, user_id)
        except MasterKeyRotatedError as exc:
            # H-7: stored creds can't be decrypted (key rotated/corrupt).
            # Prompt clean re-onboarding instead of crashing — token NOT charged.
            await _release_user_slot(user_id)
            await callback.message.answer(
                f"🔐 {exc.user_message}\n\n"
                "✅ Токен *не списан*.",
                parse_mode=ParseMode.MARKDOWN,
            )
            await state.clear()
            return
        if keys is None:
            # conn_mode == "freedom" but no personal keys in the vault.  Do NOT
            # fall back to the shared service-level FREEDOM_API_KEY for a regular
            # user — that key belongs to the service account, and using it would
            # fetch (and show) SOMEONE ELSE'S portfolio.  Only an admin (the
            # owner of the service key) may use it, for their own testing.
            if _is_admin(user_id):
                api_key    = os.getenv("FREEDOM_API_KEY",    "demo")
                secret_key = os.getenv("FREEDOM_API_SECRET", "")
                login      = os.getenv("FREEDOM_LOGIN",      "")
                key_origin = KEY_ORIGIN_ADMIN_SERVICE
                logger.warning(
                    "KEY SOURCE: env/service  user=%s (admin) — vault пуст, "
                    "используются сервисные ключи.", user_id,
                )
            else:
                logger.warning(
                    "KEY SOURCE: MISSING  user=%s — conn_mode=freedom, но ключей в vault нет; "
                    "сервисный ключ чужому пользователю НЕ подставляем, просим переподключить.",
                    user_id,
                )
                await _release_user_slot(user_id)
                await callback.message.answer(
                    "⚠️ *Брокер не подключён.*\n\n"
                    "Похоже, ваши ключи не сохранились — привяжите счёт заново в "
                    "/start → 🔗 Freedom Broker API.\n\n"
                    "✅ Токен *не списан*.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                await state.clear()
                return
        else:
            login, api_key, secret_key = keys
            login      = (login      or "").strip()
            api_key    = (api_key    or "").strip()
            secret_key = (secret_key or "").strip()
            key_origin = KEY_ORIGIN_VAULT
            logger.info(
                "KEY SOURCE: vault  user=%s  api_key_present=%s  secret_present=%s",
                user_id, bool(api_key), bool(secret_key),
            )
    elif source == "manual":
        # Брокер не спрашивается вовсе — ни ключей, ни клиента (I-12).
        # Демо-ключи здесь подставлять НЕЛЬЗЯ: ниже они привели бы к
        # шаблонному портфелю вместо портфеля пользователя, причём за полную
        # цену тарифа.
        api_key, secret_key, login = "", "", ""
        logger.info("PORTFOLIO SOURCE: manual  user=%s (ввод пользователя; "
                    "цены — независимый источник).", user_id)
    else:
        # source == "demo": the user EXPLICITLY chose the template portfolio
        # (an accidental/default demo is impossible — _resolve_portfolio_source
        # returns 'undetermined' for that and we bailed out above).
        logger.info(
            "PORTFOLIO SOURCE: demo/explicit  user=%s (шаблонный портфель "
            "выбран пользователем; отчёт бесплатный).", user_id,
        )
        api_key, secret_key, login = "demo", "", ""

    try:
        if source == "manual":
            # Черновик старше FSM-состояния: он переживает рестарт контейнера,
            # а состояние — нет (`PHASE_05 §5`).
            manual_text = ("" if source_override == "manual" else
                           str((await state.get_data()).get("manual_text") or ""))
            if source_override == "manual":
                # Кнопочный ручной отчёт (fallback) — это «мой ручной портфель»:
                # сохранённый портфель первым, черновик — запасным.
                manual_text, _unreadable = await _load_manual_portfolio_text(user_id)
                manual_from_store = bool(manual_text.strip())
            if not manual_text.strip():
                # `get_manual_draft` отдаёт СЛОВАРЬ (`text`/`created_at`/
                # `updated_at`), а не строку: `str()` от него дал бы
                # правдоподобный непустой текст, который парсер честно не
                # разберёт, — и отказ выглядел бы как «пользователь ввёл чушь».
                draft = await get_manual_draft(user_id)
                manual_text = str((draft or {}).get("text") or "")
            if not manual_text.strip():
                # Гибрид PR-1: черновик удаляется после доставленного отчёта, а
                # постоянный ручной портфель — нет. Он и есть «мой портфель».
                manual_text, _unreadable = await _load_manual_portfolio_text(user_id)
                manual_from_store = bool(manual_text.strip())
            if not manual_text.strip():
                raise ManualInputUnusable("черновик не найден")
            df = await loop.run_in_executor(
                None, _manual_frame_sync, manual_text
            )
        elif source == AGGREGATED_SOURCE:
            loaded = await _aggregated_step1(
                callback, state, tier, user_id,
                api_key=api_key, secret_key=secret_key, login=login,
                key_origin=key_origin, overlap_tag=overlap_tag)
            if loaded is None:
                return                          # отказ уже показан, слот снят
            df, agg_composition, freedom_proof = loaded
        else:
            # D-8: общий бюджет ожидания брокера. Поток executor'а по таймауту
            # не отменяется (вызов без побочных эффектов) — его поздний
            # результат просто отбрасывается, а слот освобождается ниже ровно
            # один раз.
            try:
                df = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, _fetch_portfolio_sync, api_key, secret_key, login),
                    timeout=broker_fetch_budget_s(),
                )
            except asyncio.TimeoutError as exc:
                # Свой тип, а не голый TimeoutError: в 3.11 им же является
                # `socket.timeout`, и чужой таймаут назвался бы «брокер молчит».
                raise BrokerBudgetExceeded() from exc
    except ManualInputUnusable as exc:
        logger.info("MANUAL: расчёт невозможен user=%s: %s", user_id, exc)
        await _release_user_slot(user_id)
        # Клавиатура ОБЯЗАТЕЛЬНА: без неё пользователь в режиме `manual`
        # заперт — `/start` ведёт в меню тиров, меню приводит сюда, и выхода
        # нет. Ровно та петля, которую закрыли 2026-07-16 для `undetermined`.
        await callback.message.answer(
            "📝 *Портфель не найден.*\n\n"
            "Похоже, ввод не сохранился. Наберите позиции заново — это займёт "
            "минуту.\n\n"
            "✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_connect_choice(user_id),
        )
        await state.clear()
        return
    except BrokerAuthError as exc:
        logger.error("Freedom Broker auth failed for %s: %s", user_id, exc)
        await _release_user_slot(user_id)          # H1: free slot — bg task never spawned
        # PR-2: текст — «ключи неверны», а не «серверы недоступны»; ручной
        # отчёт предлагается так же, как при сбое.
        _offer_text, _offer_kb = await _manual_fallback_offer(user_id, tier, "auth")
        await callback.message.answer(
            "⚠️ *Не удалось подключиться к брокеру.*\n\n"
            "Похоже, API-ключи неверны или отозваны — проверьте их в "
            "/start → 🔗 Freedom Broker API.\n\n"
            "✅ Токен *не списан* — платите только за готовый отчёт." + _offer_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_offer_kb,
        )
        await state.clear()
        return
    except BrokerEmptyPortfolioError as exc:
        await _release_user_slot(user_id)
        await callback.message.answer(
            f"📭 *Портфель пуст*\n\n{exc}\n\n"
            "✅ Токен *не списан* — анализировать пока нечего.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return
    except BrokerBudgetExceeded:
        # D-8: брокер молчит дольше бюджета. «Серверы недоступны» здесь честно —
        # в отличие от `waf_block`/`parse_error` (`_broker_outage_advice`).
        _budget = broker_fetch_budget_s()
        logger.error("PORTFOLIO SOURCE: broker timeout user=%s budget=%ss",
                     user_id, _budget)
        await _release_user_slot(user_id)
        _offer_text, _offer_kb = await _manual_fallback_offer(user_id, tier, "timeout")
        await callback.message.answer(
            "⚠️ *Серверы Freedom Broker сейчас недоступны* — брокер не ответил "
            f"за {_budget} с. Обычно это проходит за 5–15 минут; попробуйте "
            "чуть позже.\n\n"
            "✅ Токен *не списан*." + _offer_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_offer_kb,
        )
        await state.clear()
        return
    except Exception as exc:
        # F-6: never echo str(exc) to the user — arbitrary exception text can
        # carry upstream response bodies (client.py wraps resp.text into
        # BrokerAPIError).  Log under a support id; show only the id.
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Не удалось загрузить портфель для %s [%s]: %s",
                         user_id, error_id, exc)
        await _release_user_slot(user_id)
        _offer_text, _offer_kb = (await _manual_fallback_offer(user_id, tier, "error")
                                  if source == "freedom" else ("", None))
        await callback.message.answer(
            "ℹ️ *Не удалось загрузить портфель прямо сейчас.*\n\n"
            f"Код ошибки для поддержки: `{error_id}`\n\n"
            "✅ Токен *не списан*. Попробуйте ещё раз через пару минут." + _offer_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_offer_kb,
        )
        await state.clear()
        return

    # ── M-6 (2026-07-19): БРОКЕР УПАЛ ≠ «УСПЕШНО ПОЛУЧЕН» ────────────────
    # При сбое Freedom API коннектор возвращает fallback-МОК (шаблонную книгу
    # BTC-USD/AAPL/KSPI) с маркером _ramp_is_fallback.  Live-инцидент 19.07
    # 20:27: превью показало этот мок под «✅ Портфель успешно получен» — юзер
    # увидел ЧУЖИЕ (демо) позиции как свои, и только Шаг 1 упал честно.
    # Останавливаемся ДО превью; движковый гейт (RealPortfolioRequired на
    # _ramp_is_fallback в analyze_all) остаётся второй линией обороны.
    _attrs = getattr(df, "attrs", {}) or {}
    if _attrs.get("_ramp_is_fallback") or (
            source != "demo" and _attrs.get("_ramp_is_mock")):
        _reason = str(_attrs.get("_ramp_fallback_reason") or "")
        _error_id = uuid.uuid4().hex[:12]
        logger.error(
            "PORTFOLIO SOURCE: fallback-mock  user=%s reason=%s [%s] detail=%s — "
            "превью не показываем, отчёт не строим.",
            user_id, _reason or "unknown", _error_id,
            _attrs.get("_ramp_fallback_detail") or "—",
        )
        await _release_user_slot(user_id)
        await callback.message.answer(
            "❌ *Freedom Broker сейчас недоступен.*\n\n"
            + _broker_outage_advice(_reason, _error_id) +
            "\n\n✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await _answer_fallback_offer(callback.message, user_id, tier, _reason)
        await state.clear()
        return

    # ── Шаг 2 — концьерж-уведомление + превью портфеля ──────────────────
    preview_md = _format_portfolio_preview(df)
    source_line = ""
    if source == "demo":
        source_line = ("📋 *Источник: ДЕМО-портфель (шаблон) — отчёт "
                       "бесплатный.*\n\n")
    elif source == AGGREGATED_SOURCE:
        # Названы ОБА источника состава и ОДИН источник цен (I-13, §I.6).
        _comp = agg_composition
        source_line = (
            f"🌐 *Состав: Freedom Broker ({_comp['freedom_positions']} поз.) + "
            f"ручной ввод ({_comp['manual_positions']} поз.); котировки — "
            "Tradernet для всех позиций.*\n"
            + (f"Суммированы бумаги из обоих источников: "
               f"{_md_safe(', '.join(_comp['overlaps']))}.\n" if _comp["overlaps"] else "")
            + "\n")
    elif source == "manual" and fallback_reason:
        # PR-2: отчёт строится ВМЕСТО брокерского — это названо прямо, иначе
        # разница с прошлым брокерским отчётом читалась бы как движение рынка.
        source_line = ("📝 *Отчёт по ручным активам; брокер был недоступен "
                       f"({FALLBACK_REASON_TEXT.get(fallback_reason, 'сбой')}).* "
                       "Котировки — независимый публичный источник.\n\n")
    elif source == "manual":
        # Названо прямо: состав — от пользователя, цены — НЕ от брокера.
        # Умолчание здесь читалось бы как «всё как обычно, через Freedom».
        source_line = ("📝 *Источник: ваш ручной ввод; котировки — "
                       "независимый публичный источник.*\n\n")
    # Заголовок брокерского пути ПИНИТСЯ дословно двумя тестами: он же служит
    # маркером «превью показано» для проверки порядка гейта fallback-мока
    # (`test_phase34_broker_outage_honesty`).  Для ручного ввода «получен»
    # неуместно — портфель не получали, его прислал сам пользователь.
    header = ("✅ *Портфель принят в обработку.*" if source == "manual"
              else "✅ *Портфель успешно получен и принят в обработку.*")
    await callback.message.answer(
        f"{header}\n\n"
        f"{source_line}"
        f"{preview_md}\n\n"
        f"Запускаю *{TIER_LABEL[tier]}* — пришлю ссылку на отчёт через "
        "*5–10 минут*. Можно продолжать пользоваться ботом.",
        parse_mode=ParseMode.MARKDOWN,
    )

    # ── Шаг 3 — запускаем анализ как background task ────────────────────
    asyncio.create_task(_run_analysis_background(
        bot       = callback.message.bot,
        chat_id   = callback.message.chat.id,
        user_id   = user_id,
        tier      = tier,
        cost      = cost,
        df        = df,
        bench_tick= bench_tick,
        source    = source,
        broker_fallback = (FALLBACK_REASON_TEXT.get(fallback_reason)
                           if fallback_reason else None),
        # Аудит `§−123`: отчёт по СОХРАНЁННОМУ портфелю не трогает черновик —
        # иначе доставка fallback-отчёта стирала незаконченный ввод.
        **({"delete_draft": False}
           if source == "manual" and manual_from_store else {}),
        **({"aggregated_composition": agg_composition,
            "freedom_proof": freedom_proof}
           if source == AGGREGATED_SOURCE else {}),
    ))
    await state.clear()


class _SkipSnapshot(Exception):
    """Снимок для сравнения месяц-к-месяцу не пишется (другой состав книги)."""


async def _send_history_fallback_offer(bot, chat_id: int, user_id: int,
                                       tier: str, source: str) -> None:
    """Шаг 1 брокерского отчёта упал — предложить ручной отчёт ОТДЕЛЬНОЙ кнопкой.

    Только для `freedom`: у ручного отчёта брокера нет, а у агрегированного
    своя ветка отказа (D-7). Сбой здесь не имеет права уронить обработку
    ошибки, ради которой он вызван.
    """
    if source != "freedom":
        return
    try:
        text, kb = await _manual_fallback_offer(user_id, tier, "history")
        if kb is not None:
            await bot.send_message(chat_id, text.strip(),
                                   parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
    except Exception as exc:                           # noqa: BLE001
        logger.warning("FALLBACK: предложение не отправлено user=%s: %s",
                       user_id, type(exc).__name__)


#: `§−126`: расчёты, идущие в ЭТОМ процессе: user_id → (chat_id, tier).
#: Редеплой убивает процесс посреди расчёта (Cloud Run шлёт SIGTERM, через
#: 10 с — SIGKILL), и раньше пользователь навсегда оставался с «⏳ Шаг 4/4»:
#: сообщение о прерывании отправить было некому и не через что.
_INFLIGHT_REPORTS: dict[int, tuple[int, str]] = {}

_INTERRUPTED_TEXT = (
    "⚠️ *Отчёт прерван: сервис перезапустился (обновление).*\n\n"
    "✅ Токен *не списан* — вы платите только за готовый отчёт.\n"
    "Запустите отчёт заново через минуту."
)


async def notify_interrupted_reports(send) -> int:
    """Сказать владельцам незавершённых расчётов, что расчёт прерван (`§−126`).

    `send(chat_id, text, reply_markup)` — корутина отправки. Отдельный
    параметр, потому что на остановке основная сессия Telegram уже закрыта
    (освобождение getUpdates за 2–3 с — жёсткое правило против 409), и слать
    приходится через свою, короткую. Сбой одного адресата не мешает другим.
    → число доставленных уведомлений.
    """
    pending = dict(_INFLIGHT_REPORTS)
    if not pending:
        return 0
    kb = kb_nav(_btn("📊 Новый отчёт", "home:report"), new_message=True)
    results = await asyncio.gather(
        *(send(chat_id, _INTERRUPTED_TEXT, kb) for chat_id, _tier in pending.values()),
        return_exceptions=True)
    for (uid, _), res in zip(pending.items(), results):
        if isinstance(res, Exception):
            logger.warning("Уведомление о прерванном отчёте не ушло user=%s: %s",
                           uid, type(res).__name__)
    return sum(1 for r in results if not isinstance(r, Exception))


async def _run_analysis_background(
    *,
    bot,
    chat_id: int,
    user_id: int,
    tier: str,
    cost: int,
    df,
    bench_tick: str | None,
    source: str = "freedom",
    broker_fallback: str | None = None,
    aggregated_composition: dict | None = None,
    freedom_proof=None,
    delete_draft: bool = True,
) -> None:
    """
    Фоновая задача с поэтапными уведомлениями в Telegram.

    Каждый этап MAC3-pipeline публикует прогресс и ошибки сразу как они
    случаются — пользователь видит, что делается, и где именно сломалось.
    На критических ошибках токены возвращаются автоматически.
    """
    from finance.investment_logic import UniversalPortfolioManager

    loop = asyncio.get_running_loop()
    # Sprint-5 UX: ONE status message, edited in place across all 4 steps —
    # replaces the old "Шаг 1…/Шаг 2…/Шаг 3…/Шаг 4…" message spam (6+ messages
    # per run) with a single, calmly-updating concierge line.
    status_msg = None

    async def step(emoji: str, text: str):
        """Update the single in-place status message (create it on first call)."""
        nonlocal status_msg
        body = f"{emoji} {text}"
        try:
            if status_msg is None:
                status_msg = await bot.send_message(
                    chat_id, body, parse_mode=ParseMode.MARKDOWN)
            else:
                await status_msg.edit_text(body, parse_mode=ParseMode.MARKDOWN)
            return status_msg
        except Exception as e:
            # An "edit with identical text" or transient error must never break
            # the pipeline — the status line is cosmetic.
            logger.warning("Не удалось обновить статус-сообщение: %s", e)
            return status_msg

    async def refund(reason: str) -> None:
        """
        H1 (Phase-3): nothing is deducted upfront anymore — the token is
        charged ONLY after Checkpoint 3 (report rendered + uploaded).  So a
        failure before that point means the user simply WASN'T charged; we
        just reassure them.  (Name kept to minimise call-site churn.)
        """
        try:
            await bot.send_message(
                chat_id,
                "✅ Токен *не списан* — вы платите только за готовый отчёт.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as exc:
            logger.warning("Не удалось отправить уведомление о неспискании для %s: %s", user_id, exc)

    _gate_held = False
    _INFLIGHT_REPORTS[user_id] = (chat_id, tier)
    try:
        # `§−122`: общий потолок одновременных расчётов — ДО первой тяжёлой стадии.
        await _enter_report_gate(bot, chat_id)
        _gate_held = True
        # ── Step 1: load market history ──────────────────────────────────
        await step("⏳", "*Шаг 1/4:* Интеграция рыночных данных и FX-трансформация цен…")

        # Источник ЦЕН: демо считается на локальной детерминированной витрине
        # (одинаково у всех, всегда, без сети); остальные — на живом фиде.
        #
        # Источник передаётся КАК ЕСТЬ, а не сводится к паре demo/freedom.
        # Прежняя запись `... else "freedom"` означала, что любой новый источник
        # МОЛЧА поедет на брокерском фиде — то есть ручной портфель нарушил бы
        # I-12 (данные Tradernet не-клиентам Freedom) без единого признака.
        # `provider_for_source` — fail-closed: неизвестный ему источник честно
        # отказывает ДО сетевых вызовов, и это правильный конец такой ветки.
        #
        # I-15: менеджер с `aggregated` (а с ним и клиент Tradernet для ручных
        # тикеров) создаётся ТОЛЬКО за гейтом живого фетча по ключам vault.
        if source == AGGREGATED_SOURCE:
            manager = aggregated_manager(freedom_proof, UniversalPortfolioManager)
        else:
            manager = UniversalPortfolioManager(price_source=source)

        # Wrap the heavy parts to detect WHERE we fail.
        def _stage_market_data():
            # H-3: candidate tickers come from the portfolio frame; the engine
            # FACADE (prefetch_market_data) does the NON_RISK filtering, the
            # load, and the resolve/internal bookkeeping — the bot no longer
            # reaches into manager.engine.* private config.
            candidates = (
                df["Ticker"].astype(str).tolist() if "Ticker" in df.columns
                else df.index.astype(str).tolist()
            )
            return manager.prefetch_market_data(candidates)

        try:
            preview = await loop.run_in_executor(None, _stage_market_data)
        except Exception as exc:
            # F-6: support-id instead of raw exception text (info disclosure).
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Stage 1 (market data) failed [%s]: %s", error_id, exc)
            await bot.send_message(
                chat_id,
                "❌ *Шаг 1 не удался:* не получилось загрузить исторические цены.\n\n"
                f"Код ошибки для поддержки: `{error_id}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("market_data_error")
            await _send_history_fallback_offer(bot, chat_id, user_id, tier, source)
            raise

        # Unpack the facade summary — no engine internals touched in the bot.
        all_data           = preview.data
        risky_tickers      = preview.risky_tickers
        history_result     = preview.history_result
        loaded_count       = preview.loaded_count
        internal_tickers   = preview.internal_tickers   # factor ETFs + benchmark infra
        resolved_portfolio = preview.resolved_portfolio
        portfolio_loaded   = preview.portfolio_loaded
        portfolio_total    = preview.portfolio_total

        # `§−125`: ручной портфель оценивается ТОЛЬКО по базе котировок — цены
        # брокера у его строк нет. Ни одной котировки → движок выбросит все
        # бумаги и скажет «стоимость = 0, проверьте подключение к брокеру»,
        # хотя брокера здесь нет вовсе. Останавливаемся ДО движка и называем
        # тикеры с причиной, которую уже знает провайдер.
        if source == "manual" and (loaded_count == 0
                                   or (portfolio_total and portfolio_loaded == 0)):
            await bot.send_message(
                chat_id,
                _manual_no_quotes_text(resolved_portfolio,
                                      dict(history_result.failed or {}),
                                      base_down=loaded_count == 0),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_nav(_btn("✏️ Ручной портфель", "mp:show"),
                                    new_message=True),
            )
            await refund("manual_no_quotes")
            raise RuntimeError("manual_no_quotes")

        if loaded_count == 0:
            await bot.send_message(
                chat_id,
                "❌ *Шаг 1 — критично:* Freedom API не вернул ни одной серии цен.\n\n"
                "*Возможные причины:*\n"
                "• *Временный сбой Freedom API* — сервер обрывает соединения "
                "(обычно проходит за 5–15 минут); это самая частая причина, "
                "когда не грузятся даже базовые ETF\n"
                "• Ваш API-ключ Freedom Broker не имеет доступа к Market Data "
                "— это **отдельная подписка** на стороне брокера\n\n"
                "*Что делать:*\n"
                "1. Подождите 5–15 минут и просто повторите запрос\n"
                "2. Если повторяется: Личный кабинет Freedom Broker → API → "
                "Market Data — активируйте подписку на исторические данные\n"
                "3. Либо поддержка брокера: попросите включить доступ к "
                "`getHloc` для вашего API-ключа",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("no_market_data")
            await _send_history_fallback_offer(bot, chat_id, user_id, tier, source)
            raise RuntimeError("market_data_subscription_required")

        # Sprint-5 UX: the verbose per-ticker load diagnostics (proxy map,
        # retries, per-ticker failures) used to be a SEPARATE chat message.
        # That broke the "one calm status line" directive, so the detail now
        # goes to the SERVER LOG only — the user just sees the status line move
        # on to Step 2.  Hard failures (loaded_count == 0) are still surfaced
        # above via the dedicated subscription-required message.
        portfolio_failed = {
            t: r for t, r in (history_result.failed or {}).items()
            if t not in internal_tickers
        }
        logger.info(
            "Load diagnostics user=%s: %d/%d series, proxies=%s, retried=%s, failed=%s",
            user_id, portfolio_loaded, portfolio_total,
            dict(preview.proxy_map),
            [t for t in (history_result.retried or []) if t not in internal_tickers],
            portfolio_failed,
        )

        # ── Step 2: full MAC3 analysis (includes SEC EDGAR) ────────────────
        await step("⏳", "*Шаг 2/4:* Факторное моделирование и декомпозиция рисков по Эйлеру…")
        # H4: resolve the investor's risk mandate from their profile so the
        # composite-risk score + AI narrative are calibrated to it.
        _profile_h4   = await get_profile(user_id)
        _mandate_name = (_profile_h4 or {}).get("profile_name")
        try:
            results = await loop.run_in_executor(
                None, _analyze_existing_portfolio_sync, df, bench_tick, _mandate_name,
                source,
            )
        except (RealPortfolioRequired, DataQualityBlocked, AggregatedNotPermitted):
            # `§−125`: штатный ОТКАЗ движка — не сбой. Причину назовёт внешний
            # обработчик одним сообщением; раньше перед ней шло «движок упал» +
            # код поддержки, и пользователь читал два противоречащих ответа.
            raise
        except Exception as exc:
            # F-6: support-id instead of raw exception text (info disclosure).
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Stage 2 (MAC3) failed [%s]: %s", error_id, exc)
            await bot.send_message(
                chat_id,
                "❌ *Шаг 2 не удался:* движок риск-анализа упал.\n\n"
                f"Код ошибки для поддержки: `{error_id}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("mac3_failure")
            raise

        await step("✅", "Факторная модель и декомпозиция рисков рассчитаны.")

        # PR-2: отчёт построен ВМЕСТО брокерского — CoVe обязан это назвать
        # (`data_lineage._manual_source_status`). Ключ ставит слой доставки:
        # движок о брокере ничего не знает и знать не должен.
        if broker_fallback:
            results["broker_fallback_reason"] = broker_fallback
        # Аудит `§−123`: бумага без цены (нет ряда у провайдера, нет цены
        # брокера — так бывает у РУЧНОЙ строки) выпадает в движке молча. Числа
        # не трогаем — называем выпавшее пользователю и в CoVe.
        if source in ("manual", AGGREGATED_SOURCE):
            _lost = unpriced_positions(df, results)
            if _lost:
                results["unpriced_positions"] = _lost
                logger.warning("UNPRICED user=%s: %d поз. вне расчёта",
                               user_id, len(_lost))
                await bot.send_message(
                    chat_id,
                    "⚠️ *Не вошли в расчёт — нет рыночной цены:* "
                    f"{_md_safe(', '.join(_lost))}.\n\n"
                    "Отчёт посчитан по остальным позициям; доли и риск — без них.",
                    parse_mode=ParseMode.MARKDOWN,
                )
        # PR-3 §I.6: состав агрегированного отчёта (N + M позиций, пересечения)
        # — для строки CoVe `_aggregated_source_status`.
        if source == AGGREGATED_SOURCE and aggregated_composition:
            results["aggregated_composition"] = dict(aggregated_composition)

        # Note: an intermediate "MAC3 Risk Summary" dump used to surface
        # raw vol / Sharpe / Sortino / CVaR / VaR / positive-days here.
        # Concierge-tone redesign removed it — all of those numbers belong
        # to the final report (gauge, KPI strip, performance table) and
        # double-posting them as in-chat metric blasts undermines the
        # "polished, minimalist" UX directive.  The progress chain stays:
        #   ✅ portfolio received  →  ⏳ 1/4  →  2/4  →  3/4  →  4/4  →  ✅ done.

        # ── Step 3: gatekeeper (advisory only, non-blocking) ──────────────
        await step("⏳", "*Шаг 3/4:* Оптимизация целевых весов по модели Блэка-Литтермана…")

        profile = await get_profile(user_id)
        gate = run_gatekeeper_for(results, profile)

        # Sprint-5 (Task 3 — silent Gatekeeper): the loud ⛔/⚠️ risk-limit blasts
        # in chat are gone.  Breaches are SERVER-LOGGED only here; the user sees
        # the same findings inside the polished report (gatekeeper drives the
        # mandate-compliance panel + risk hotspots), not as a scary chat dump.
        if gate["critical"] or gate["warnings"]:
            logger.info(
                "Gatekeeper (advisory) user=%s: %d critical, %d warnings | %s",
                user_id, len(gate["critical"]), len(gate["warnings"]),
                "; ".join(gate["critical"][:5] + gate["warnings"][:5]),
            )

        # Concierge-tone redesign: the sector exposure used to render an
        # ASCII-bar chart in chat (`█████ Technology: 55%`).  The same data
        # is already a polished pie chart in the report — duplicating it
        # as raw bars in chat looked unfinished.  Removed here; the
        # progress-message chain carries the user straight to the report.

        # ── Step 4: report assembly ───────────────────────────────────────
        await step("⏳", "*Шаг 4/4:* Сборка интерактивного интерфейса и валидация данных…")

        # Fetch previous snapshot for month-over-month delta.
        # Аудит `§−123`: снимки ключуются (пользователь, тир), а не источником.
        # Агрегированный и fallback-отчёт — ДРУГОЙ состав книги, и дельта
        # «риск-индекс против прошлого месяца» сравнивала бы разные портфели.
        # Такие отчёты историю не читают и не пишут — брокерская остаётся чистой.
        keep_history = not (source == AGGREGATED_SOURCE or broker_fallback)
        prev_snapshot = (await get_last_report_snapshot(user_id, tier)
                         if keep_history else None)

        # Resolve user's risk profile name for AI stock-pick context
        profile       = await get_profile(user_id)
        profile_name  = (profile or {}).get("profile_name", "Moderate")

        # `§−122`: RAG (ChromaDB + ONNX) и вызов Anthropic (30–120 с для DEEP)
        # живут внутри `_build_pdf_payload` — в executor, как и сценарная
        # ветка. Прямой вызов держал event loop: пока модель писала нарратив
        # одному пользователю, бот не читал апдейты Telegram ни для кого.
        payload = await loop.run_in_executor(
            None,
            lambda: _build_pdf_payload(
                results, tier,
                user_bench_ticker=bench_tick,
                prev_snapshot=prev_snapshot,
                user_risk_profile=profile_name,
                user_profile=profile,
            ),
        )
        # CHECKPOINT 3 — render + upload.  `_send_report` raises
        # RuntimeError("report_delivery_failed") if the GCS upload fails, so
        # reaching the next line means the report is genuinely delivered.
        await _send_report(bot, chat_id, user_id, tier, payload)
        # Отчёт доставлен — прерывание после этой строки уже не «прервало отчёт».
        _INFLIGHT_REPORTS.pop(user_id, None)

        # ── H1 BILLING: deduct the token ONLY now (post-Checkpoint-3) ───────
        # This is the single charge point in the whole flow.  Any earlier
        # failure (broker / engine / render) never reaches here, so the user
        # is never charged for a report they didn't receive.
        # cost == 0 → демо-отчёт (Fix C): бесплатен, deduct пропускается
        # (deduct_tokens отвергает amount <= 0).
        if cost > 0:
            try:
                await deduct_tokens(user_id, cost, reason=f"{tier}_analysis")
            except InsufficientFundsError:
                # Extremely unlikely (balance was pre-checked + single-flight),
                # but the report is already delivered — log and don't double-bill.
                logger.error("Post-CP3 deduct failed for %s (report already sent).", user_id)
        balance_after = await get_balance(user_id)

        # Черновик ручного ввода живёт до ДОСТАВЛЕННОГО отчёта и удаляется
        # только здесь (§5).  Удалить его раньше — на подтверждении или на
        # старте расчёта — значило бы: отчёт упал по нашей вине, а двадцать
        # позиций пользователь набирает заново.
        if source == "manual" and delete_draft:
            try:
                await delete_manual_draft(user_id)
            except Exception as draft_exc:             # noqa: BLE001
                logger.warning("MANUAL: черновик не удалён для %s: %s",
                               user_id, draft_exc)

        # Persist this report's key metrics for future MoM comparison
        metrics = results.get("portfolio_metrics") or {}
        try:
            if not keep_history:
                raise _SkipSnapshot()
            await save_report_snapshot(
                telegram_id = user_id,
                tier        = tier,
                risk_score  = payload.get("risk_pct"),
                sharpe      = metrics.get("Sharpe_Ratio"),
                cvar        = metrics.get("CVaR_95_Daily"),
                volatility  = metrics.get("Total_Volatility_Ann"),
                total_value = results.get("total_value"),
            )
        except _SkipSnapshot:
            pass
        except Exception as snap_exc:
            logger.warning("Failed to save report snapshot: %s", snap_exc)

        billing_line = (
            f"💳 С баланса списан *{cost}* токен · остаток: "
            f"*{balance_after}* токен(а)."
            if cost > 0 else
            "📋 Отчёт по *демо-портфелю* — *бесплатно*, токены не списаны."
        )
        await bot.send_message(
            chat_id,
            "✅ *Расчёты успешно завершены.* Отчёт — по ссылке выше 👆\n\n"
            f"{billing_line}",
            parse_mode=ParseMode.MARKDOWN,
            # Под BASE/DEEP навигацию несёт следующее сообщение (CTA сценария).
            # Меню — НОВЫМ сообщением: строку списания затирать нельзя.
            reply_markup=None if tier in (TIER_BASE, TIER_DEEP) else kb_nav(new_message=True),
        )

        # Follow-up CTA: предложить сценарную диагностику ОДНИМ тапом.  Сценарный
        # отчёт считается ИЗ ЭТОГО ЖЕ `results` (без повторной загрузки/ИИ), так
        # что кэшируем его и вешаем inline-кнопку.  Только для BASE/DEEP — под
        # самим сценарным отчётом кнопка не нужна.
        if tier in (TIER_BASE, TIER_DEEP):
            try:
                is_demo = cost == 0
                if is_demo:
                    # Приватный маркер для cb_scenario_cached: сценарный отчёт
                    # по демо-портфелю тоже бесплатен (консистентность Fix C).
                    results["_demo_portfolio"] = True
                _cache_results_for_scenario(user_id, results)
                scenario_price = "*бесплатно* (демо)" if is_demo else "*1 токен*"
                await bot.send_message(
                    chat_id,
                    "🎯 *Сценарный анализ этого же портфеля* — вклад позиций в "
                    "риск, 3 макро-сценария, бэктест. "
                    f"{scenario_price}, мгновенно: данные уже загружены.",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=_kb_scenario_cta(free=is_demo),
                )
            except Exception as cta_exc:
                logger.warning("Scenario CTA/cache failed for %s: %s", user_id, cta_exc)

    except AggregatedNotPermitted as exc:
        # I-15 на второй линии: без доказательства живого фетча расчёт не
        # начинается вовсе — до первого сетевого вызова.
        logger.error("HYBRID: I-15 отказ в фоне user=%s: %s", user_id, exc.reason)
        await bot.send_message(
            chat_id,
            "🌐 *Агрегированный отчёт невозможен* — живой портфель брокера не "
            "подтверждён в этом запросе. Запустите отчёт заново из меню /start.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await refund("aggregated_not_permitted")
    except RealPortfolioRequired as exc:
        await bot.send_message(
            chat_id,
            "⚠️ *Анализ невозможен: нет реальных данных портфеля.*\n\n"
            f"{exc}",
            parse_mode=ParseMode.MARKDOWN,
        )
        await refund("no_real_portfolio")
    except DataQualityBlocked as exc:
        # Ф-3 (2026-08-02): данные не годятся для честного расчёта. Та же ветка
        # «отказ без списания», что у RealPortfolioRequired: списание живёт
        # после CHECKPOINT 3, поэтому сюда мы попадаем с нетронутым балансом.
        # Текст берём у отчёта чекеров — он человеческий по построению
        # (`PHASE_03 §4.1`): без кодов проверок и без str(exc).
        logger.warning("Ф-3: отчёт не построен для %s — %s", user_id,
                       "; ".join(f"{f.id}:{f.message}" for f in exc.report.blocking))
        await bot.send_message(
            chat_id,
            "⚠️ *Отчёт не построен — данные неполные.*\n\n"
            f"{exc.report.user_message()}",
            parse_mode=ParseMode.MARKDOWN,
        )
        await refund("data_quality_blocked")
    except RuntimeError as exc:
        if str(exc) in ("market_data_subscription_required", "manual_no_quotes"):
            # Already reported + refunded above.
            pass
        elif str(exc) == "report_generation_failed":
            # M-1: _send_report already showed the user a soft message + error
            # id.  Just refund here — no second message, no raw traceback.
            await refund("report_generation_failed")
        elif str(exc) == "report_delivery_failed":
            logger.error("Report generated but upload/delivery failed for %s", user_id)
            await bot.send_message(
                chat_id,
                "⚠️ *Отчёт сформирован, но не удалось загрузить его в облачное "
                "хранилище.*\n\n"
                "Это сбой на нашей стороне — повторите анализ позже или "
                "обратитесь в /support.",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("report_delivery_failed")
        else:
            # M-2: never echo a raw exception string to the user — log the full
            # traceback under a support id and show only that id.
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Background analysis runtime error [%s]: %s", error_id, exc)
            await bot.send_message(
                chat_id,
                "😔 Извините, отчёт временно недоступен.\n\n"
                f"Код ошибки для поддержки: `{error_id}`\n"
                "✅ Токен *не списан*.",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("runtime_error")
    except Exception as exc:
        # M-2: graceful catch-all — soft message + support id, full trace to log.
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Unexpected analysis failure [%s] for %s: %s", error_id, user_id, exc)
        await bot.send_message(
            chat_id,
            "😔 Извините, произошла непредвиденная ошибка.\n\n"
            f"Код ошибки для поддержки: `{error_id}`\n"
            "✅ Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await refund("unexpected_error")
    finally:
        # Always release the single-flight slot, regardless of success /
        # failure / cancellation.  Without this a single hung task locks
        # the user out forever (they would only see "анализ уже идёт").
        _INFLIGHT_REPORTS.pop(user_id, None)
        if _gate_held:
            _leave_report_gate()
        await _release_user_slot(user_id)


async def cb_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.clear()
    await callback.message.edit_text("❌ Отменено — токены не списаны.",
                                     reply_markup=kb_nav())


# ── Utility commands ──────────────────────────────────────────────────────────

async def cmd_balance(message: Message) -> None:
    await _show_balance(message, message.from_user.id, edit=False)


async def cmd_topup(message: Message) -> None:
    await _show_topup(message, edit=False)


async def cmd_grant(message: Message) -> None:
    """
    Admin-only token grant for testing.

    Usage:
      /grant            → credit 10 tokens to YOURSELF
      /grant 25         → credit 25 tokens to YOURSELF
      /grant 25 <uid>   → credit 25 tokens to user <uid>

    Restricted to IDs in the ADMIN_USER_IDS env var.  Anyone else gets a
    generic "unknown command" so the command stays invisible to regular
    users.
    """
    caller = message.from_user.id
    if not _is_admin(caller):
        # Stay invisible to non-admins.
        await message.answer("Неизвестная команда. Откройте меню: /start")
        return

    parts = (message.text or "").split()
    amount = 10
    target = caller
    try:
        if len(parts) >= 2:
            amount = int(parts[1])
        if len(parts) >= 3:
            target = int(parts[2])
    except ValueError:
        await message.answer("Формат: `/grant [кол-во] [user_id]`",
                             parse_mode=ParseMode.MARKDOWN)
        return

    if amount <= 0 or amount > 10_000:
        await message.answer("Количество должно быть 1…10000.")
        return

    # Make sure the target row exists (init_user is a no-op grant if already
    # registered; it only seeds the welcome bonus on the very first call).
    await init_user(target)
    await credit_tokens(target, amount, reason=f"admin_grant_by_{caller}")
    bal = await get_balance(target)
    who = "вам" if target == caller else f"пользователю `{target}`"
    logger.info("ADMIN %s granted %s tokens to %s", caller, amount, target)
    await message.answer(
        f"✅ Начислено *{amount}* токен(ов) {who}.\n"
        f"Текущий баланс: *{bal}* токен(а).",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_support(message: Message) -> None:
    await message.answer(
        f"🛟 *Поддержка {branding.bot_name()}*\n\n"
        f"По всем вопросам пишите: {md_safe(branding.support_contact())}\n"
        "Часы работы: пн–пт, 09:00–18:00 (UTC+5).",
        parse_mode=ParseMode.MARKDOWN,
    )


async def msg_text_fallback(message: Message, state: FSMContext) -> None:
    """Sprint-5 Task 1 — hard text-input filter (registered LAST, StateFilter None).

    The bot is fully button-driven.  Stray free-text (outside the Freedom
    credential FSM, which carries an active state and is handled earlier) is
    softly redirected to the inline-button menu rather than interpreted.  In a
    1:1 chat Telegram forbids bots from deleting the USER's message, so the
    delete is best-effort and the gentle nudge is the real mitigation.
    """
    try:
        await message.delete()
    except Exception:
        pass
    await message.answer(
        f"🔘 {branding.bot_name()} работает через кнопки — печатать ничего не нужно.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_nav(),
    )


async def cmd_mandate(message: Message, state: FSMContext) -> None:
    """B1 (2026-07-17): меню мандата вместо принудительной ре-анкеты.

    Показывает текущий мандат + точечные правки (бенчмарк / классы активов /
    риск-профиль / полная анкета).  Смена настроек БЕСПЛАТНА — никакого
    биллинга в этом флоу нет и быть не должно (guardrail §5.3 ТЗ).
    """
    await _show_mandate_menu(message, state, message.from_user.id)


async def _start_requiz(target: Message | CallbackQuery, state: FSMContext) -> None:
    """Полная ре-анкета (прежнее поведение /mandate) — один из пунктов меню."""
    await state.clear()
    await state.set_state(Onboarding.Q1)
    q   = QUESTIONS[0]
    msg = target.message if _is_callback(target) else target
    sent = await msg.answer(
        "🔄 *Обновление инвестиционного мандата*\n\n"
        "Пройдите анкетирование заново, чтобы обновить ваш профиль.\n\n"
        + q["text"],
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_question(q["options"], q_num=1),
    )
    await state.update_data(ob_message_id=sent.message_id)


async def cb_mandate_action(callback: CallbackQuery, state: FSMContext) -> None:
    """Роутер меню /mandate (`mandate:*`).  Билинг здесь ЗАПРЕЩЁН."""
    await callback.answer()
    action  = callback.data
    user_id = callback.from_user.id

    if action == "mandate:close":
        # Кнопка «Закрыть» из прошлых сообщений: теперь это «🏠 Меню».
        await state.clear()
        await _show_home(callback, user_id)
        return

    if action == "mandate:report":
        await _open_report(callback, state, user_id)
        return

    if action == "mandate:back":
        await _show_mandate_menu(callback, state, user_id)
        return

    if action == "mandate:edit:requiz":
        await _start_requiz(callback, state)
        return

    profile = await get_profile(user_id)
    if profile is None:
        await callback.message.answer(
            "⚠️ У вас ещё нет профиля. Используйте /start для регистрации.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if action == "mandate:edit:bench":
        current = _resolve_bench_ticker(profile)
        await state.update_data(edit_mode=True, benchmark_ticker=current)
        await state.set_state(Onboarding.Benchmark)
        await _edit_or_answer(
            callback, state,
            "🎯 *Смена бенчмарка*\n\n"
            f"Текущий: *{BENCHMARK_LIST.get(current or '', current or '—')}*.\n\n"
            "ℹ️ Бенчмарк — эталон, с которым отчёт сравнивает ваш портфель "
            "(доходность, Tracking Error и факторное разложение).\n\n"
            "Выберите новый и нажмите «Продолжить»:",
            kb_benchmark(current=current),
        )
        return

    if action == "mandate:edit:universe":
        selected = list(profile.get("selected_assets") or [])
        await state.update_data(edit_mode=True, universe=selected)
        await state.set_state(Onboarding.Universe)
        await _edit_or_answer(
            callback, state,
            "🧬 *Классы активов*\n\n"
            "Отметьте классы, которые хотите включить в стратегию. "
            "Баллы риск-профиля не меняются — только вселенная и лимиты.",
            kb_universe(set(selected)),
        )
        return

    if action == "mandate:edit:profile":
        await state.update_data(edit_mode=True)
        await state.set_state(MandateEdit.Profile)
        await _edit_or_answer(
            callback, state,
            "⚖️ *Риск-профиль (экспертный режим)*\n\n"
            "Профиль задаёт целевую волатильность, Tracking Error и лимиты "
            "классов активов. Обычно он определяется анкетой — меняйте "
            "вручную, только если понимаете последствия.\n\n"
            "Ваш выбранный бенчмарк сохранится без изменений.",
            kb_mandate_profile(profile.get("profile_name")),
        )
        return

    if action.startswith("mandate:profile:"):
        try:
            score = int(action.rsplit(":", 1)[1])
        except ValueError:
            return
        score    = max(6, min(18, score))
        new_prof = RiskProfileManager.score_to_profile(score)
        universe = list(profile.get("selected_assets") or ASSET_KEYS)
        limits   = RiskProfileManager.apply_universe(new_prof, universe)
        # Бенчмарк пользователя НЕ перетираем (решение по умолчанию №2 ТЗ) —
        # дефолт нового профиля лишь подсказываем текстом.
        kept_bench    = profile.get("benchmark_ticker")
        default_bench = PROFILE_BENCH_TICKER.get(new_prof["name"])
        await save_profile(
            telegram_id       = user_id,
            score             = score,
            profile_name      = new_prof["name"],
            target_volatility = new_prof["target_vol"],
            target_te         = new_prof["target_te"],
            selected_assets   = universe,
            limits_dict       = limits,
            benchmark_ticker  = kept_bench,
        )
        await approve_mandate(user_id)
        hint = ""
        if default_bench and kept_bench and default_bench != kept_bench:
            hint = (f"\n💡 Для профиля «{new_prof['name']}» мы обычно рекомендуем "
                    f"бенчмарк *{BENCHMARK_LIST.get(default_bench, default_bench)}* — "
                    "сменить его можно в «🎯 Сменить бенчмарк».\n")
        updated = await get_profile(user_id)
        await _edit_or_answer(
            callback, state,
            f"✅ *Риск-профиль обновлён: {new_prof['name']}.*\n{hint}\n"
            + _mandate_overview_text(updated or {}),
            kb_mandate_changed(),
        )
        await _reset_state_keep_message(state)
        return


async def cmd_help(message: Message) -> None:
    """Короткая карта бота (`§−124`): три шага + токены + поддержка."""
    await _show_help(message, message.from_user.id, edit=False)


# ── Beta-access middleware ────────────────────────────────────────────────────
# Closed-beta gating: when TG_ALLOWED_USERS is set (comma-separated Telegram
# user IDs), the bot silently ignores any update from a non-listed user
# AFTER a single polite "beta closed" reply.  Empty/unset env → open mode
# (matches the pre-beta behaviour).  Used to safely onboard the first 10
# users without spam exposure on a public bot.
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

# Sentinel for "cache never populated yet".  We can't use None for this
# slot because None ALSO means "gating intentionally disabled" — the two
# states must be distinguishable.  Without this sentinel, an empty
# TG_ALLOWED_USERS env var was cached as set() and the second `_allowed_users`
# call returned set() (truthy-checked as non-None) → every user except the
# very first request was blocked by the middleware.
_UNINIT = object()
_ALLOWED_USERS_CACHE: object = _UNINIT
_DENIED_NOTIFIED: set[int] = set()   # tell each user ONCE they're not in the beta


def _allowed_users() -> set[int] | None:
    """Return whitelist set, or None when no gating is configured."""
    global _ALLOWED_USERS_CACHE
    if _ALLOWED_USERS_CACHE is not _UNINIT:
        return _ALLOWED_USERS_CACHE  # type: ignore[return-value]
    raw = (os.getenv("TG_ALLOWED_USERS") or "").strip()
    if not raw:
        _ALLOWED_USERS_CACHE = None      # disabled — DO NOT cache as set()
        return None
    out: set[int] = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token.lstrip("-").isdigit():
            out.add(int(token))
    _ALLOWED_USERS_CACHE = out
    logger.info("Beta whitelist active: %d user(s)", len(out))
    return out


def _admin_users() -> set[int]:
    """
    Parse ADMIN_USER_IDS (comma/semicolon-separated Telegram IDs).

    Admins can self-credit tokens with /grant for testing.  Empty/unset →
    no admins → /grant is refused for everyone (safe default in prod).
    """
    raw = (os.getenv("ADMIN_USER_IDS") or "").strip()
    out: set[int] = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token.lstrip("-").isdigit():
            out.add(int(token))
    return out


def _is_admin(user_id: int) -> bool:
    return user_id in _admin_users()


class WhitelistMiddleware(BaseMiddleware):
    """Drop updates from non-whitelisted users (with a one-time notice)."""

    async def __call__(self, handler, event: TelegramObject, data: dict):
        allowed = _allowed_users()
        # Defence in depth: also treat an explicitly empty set as "disabled".
        # An empty whitelist would otherwise block every single user — that
        # is never the intended config; misconfiguration should fail OPEN
        # for the beta, not lock everyone out.
        if not allowed:                           # None or empty set
            return await handler(event, data)
        user = getattr(event, "from_user", None)
        if user is None:
            inner = getattr(event, "message", None) or getattr(event, "callback_query", None)
            user = getattr(inner, "from_user", None)
        if user is None or user.id in allowed:
            return await handler(event, data)
        # Non-allowed: send a one-shot reply, then ignore further updates.
        if user.id not in _DENIED_NOTIFIED:
            _DENIED_NOTIFIED.add(user.id)
            try:
                bot = data.get("bot")
                chat_id = (getattr(event, "chat", None) and event.chat.id) or user.id
                if bot is not None:
                    await bot.send_message(
                        chat_id,
                        "🔒 *Закрытая бета* — доступ только по списку. "
                        "Если хотите подключиться, напишите владельцу.",
                        parse_mode=ParseMode.MARKDOWN,
                    )
            except Exception as exc:           # noqa: BLE001
                logger.debug("beta-deny notice send failed: %s", exc)
        logger.info("Beta gate: blocked update from user_id=%s", user.id)
        return None   # short-circuit: handler never runs


# ── Per-user single-flight guard ─────────────────────────────────────────────
# Prevents the same user from queuing multiple report jobs (intentional or
# fat-finger).  Background analysis jobs run as asyncio tasks in the polling
# process — without this guard one user spamming "DEEP" would stack parallel
# jobs and starve everyone else.
#
# A-4 (2026-08-02): аренда ПЕРЕЕХАЛА В SQLite.  Прежний `set()` защищал только
# от параллели ВНУТРИ процесса, а токен списывается ПОСЛЕ доставки отчёта
# (CHECKPOINT 3) — значит два инстанса, взявшие одного пользователя, доставят
# ДВА отчёта и спишут ОДИН токен (второй `deduct_tokens` упадёт
# `InsufficientFundsError`, но отчёт уже отправлен).  Сегодня окна нет только
# потому, что `cloudbuild.yaml` задаёт `--max-instances=1`; эта правка —
# ПРЕДУСЛОВИЕ его снятия, а не следствие.
#
# Владелец аренды — идентификатор ПРОЦЕССА: release снимает только свою строку,
# поэтому запоздавший release умирающего инстанса не выбьет нового держателя.
_INSTANCE_ID: str = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

#: In-memory зеркало.  Роль двойная: (1) быстрый отказ без похода в БД,
#: (2) ДЕГРАДАЦИЯ — если БД недоступна, гард продолжает работать ровно так,
#: как работал до A-4 (в пределах процесса).  Отказывать в отчёте из-за сбоя
#: БД смысла нет: следующий же шаг всё равно идёт в неё за балансом и вернёт
#: пользователю честную ошибку.
_IN_FLIGHT_USERS: set[int] = set()

# ── Global gate: сколько отчётов считается ОДНОВРЕМЕННО (`§−122`) ────────────
# Слот выше — ПО ПОЛЬЗОВАТЕЛЮ; без общего потолка десять пользователей дали бы
# десять параллельных конвейеров на одном CPU: каждый в 10 раз медленнее, а
# пул executor'а (≈5 потоков) целиком занят ожиданием ответов Anthropic.
# Лишние ждут в очереди и видят, сколько перед ними. Дефолт 2: на 1 vCPU
# третий параллельный расчёт уже не ускоряет, а только растягивает всех.
MAX_CONCURRENT_REPORTS: int = env_int("MAX_CONCURRENT_REPORTS", 2, lo=1, hi=16)
_REPORT_GATE = asyncio.Semaphore(MAX_CONCURRENT_REPORTS)
_REPORTS_RUNNING: int = 0
_REPORTS_WAITING: int = 0
#: Через сколько секунд ожидания в очереди писать WARNING (и повторять). Гейт,
#: который никто не отпустил, — ТИХИЙ отказ: пользователи «в очереди», а
#: отчётов нет. Периодическое предупреждение делает его видимым в логах и
#: алерте (`READINESS_10_USERS §5`).
_QUEUE_WARN_S: float = 600.0


async def _enter_report_gate(bot, chat_id: int) -> None:
    """Дождаться места в конвейере; пока ждём — сказать пользователю, сколько
    расчётов перед ним. Молчание здесь читается как зависание."""
    global _REPORTS_RUNNING, _REPORTS_WAITING
    if _REPORT_GATE.locked():
        _REPORTS_WAITING += 1
        ahead = _REPORTS_WAITING - 1
        try:
            await bot.send_message(
                chat_id,
                "⏳ *Сейчас считаются отчёты других пользователей.* Ваш в очереди"
                + (f" — перед вами ещё {ahead}" if ahead else "")
                + ". Он начнётся автоматически, ничего нажимать не нужно.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as exc:                   # noqa: BLE001
            logger.warning("Не удалось сообщить об очереди %s: %s", chat_id, exc)
        acquire = asyncio.ensure_future(_REPORT_GATE.acquire())
        waited = 0.0
        try:
            while True:
                done, _ = await asyncio.wait({acquire}, timeout=_QUEUE_WARN_S)
                if done:
                    acquire.result()
                    break
                waited += _QUEUE_WARN_S
                logger.warning(
                    "Очередь отчётов стоит: chat=%s ждёт %.0f с, в работе %d, "
                    "ждут %d — если так долго, гейт не отпущен.",
                    chat_id, waited, _REPORTS_RUNNING, _REPORTS_WAITING)
        except BaseException:
            acquire.cancel()
            raise
        finally:
            _REPORTS_WAITING -= 1
    else:
        await _REPORT_GATE.acquire()
    _REPORTS_RUNNING += 1


def _leave_report_gate() -> None:
    global _REPORTS_RUNNING
    _REPORTS_RUNNING -= 1
    _REPORT_GATE.release()


async def _try_acquire_user_slot(user_id: int, tier: str | None = None) -> bool:
    """Взять слот на построение отчёта. False — у пользователя уже идёт расчёт."""
    if user_id in _IN_FLIGHT_USERS:
        return False
    try:
        won = await acquire_report_lock(user_id, _INSTANCE_ID, tier=tier)
    except Exception as exc:                       # noqa: BLE001
        logger.warning(
            "A-4: аренда отчёта недоступна (%s) — гард деградировал до "
            "внутрипроцессного; при max-instances>1 это ОКНО.", exc)
        _IN_FLIGHT_USERS.add(user_id)
        return True
    if won:
        _IN_FLIGHT_USERS.add(user_id)
    return won


async def _release_user_slot(user_id: int) -> None:
    """Освободить слот. Идемпотентно — вызывается из десятка веток выхода."""
    _IN_FLIGHT_USERS.discard(user_id)
    try:
        await release_report_lock(user_id, _INSTANCE_ID)
    except Exception as exc:                       # noqa: BLE001
        # Не роняем доставку отчёта из-за housekeeping: просроченную аренду
        # всё равно перехватит следующий `acquire_report_lock`.
        logger.warning("A-4: снятие аренды не удалось для %s: %s", user_id, exc)


class SlotBusy(RuntimeError):
    """Слот пользователя занят — расчёт уже идёт."""


@asynccontextmanager
async def user_slot(user_id: int, tier: str | None = None):
    """Слот на тяжёлую работу пользователя, снимаемый ГАРАНТИРОВАННО (Т-9).

    Зачем контекст-менеджер, если есть пара функций: в `cb_confirm` слот
    освобождается ДЕВЯТЬЮ отдельными вызовами — по одному на каждую ветку
    отказа.  Ручной флоу добавляет свои ветки (ошибка разбора, отмена на экране
    подтверждения, недоступный курс), и повторять этот приём означает почти
    наверняка однажды забыть вызов: забытый `release` запирает пользователя до
    истечения аренды (30 минут), а видит он при этом «у вас уже выполняется
    анализ» — то есть ошибку, которую сам исправить не может.

    Существующие девять веток `cb_confirm` этой фазой НЕ переписываются: они
    сегодня корректны, а их переписывание — отдельный рефакторинг с
    собственным риском (`PHASE_05 §6`).  Новый код использует только этот
    менеджер.
    """
    if not await _try_acquire_user_slot(user_id, tier=tier):
        raise SlotBusy(str(user_id))
    try:
        yield
    finally:
        await _release_user_slot(user_id)


# ── Per-user results cache for the one-tap Scenario button ───────────────────
# After a BASE/DEEP report is delivered, we offer «🎯 Сценарный анализ» as an
# inline button.  Каждый сценарный отчёт детерминирован и считается ИЗ УЖЕ
# посчитанного `results` (0 LLM-API, без повторной загрузки цен) — поэтому мы
# кэшируем последний `results` пользователя и billing-ем 1 токен только за
# отрисованный отчёт.  Bounded + TTL: max-instances=1, beta-масштаб → память
# под контролем; протухший кэш честно просит перезапустить из меню.
import time as _time
from collections import OrderedDict

_SCENARIO_CACHE: "OrderedDict[int, tuple[dict, float]]" = OrderedDict()
_SCENARIO_CACHE_TTL_SEC = 3600.0     # 1 час — ссылка на отчёт живёт 48ч, но
                                     # держать тяжёлый results дольше часа незачем
_SCENARIO_CACHE_MAX = 64


def _cache_results_for_scenario(user_id: int, results: dict) -> None:
    _SCENARIO_CACHE[user_id] = (results, _time.monotonic())
    _SCENARIO_CACHE.move_to_end(user_id)
    while len(_SCENARIO_CACHE) > _SCENARIO_CACHE_MAX:
        _SCENARIO_CACHE.popitem(last=False)


def _get_cached_results(user_id: int) -> dict | None:
    item = _SCENARIO_CACHE.get(user_id)
    if item is None:
        return None
    results, ts = item
    if _time.monotonic() - ts > _SCENARIO_CACHE_TTL_SEC:
        _SCENARIO_CACHE.pop(user_id, None)
        return None
    return results


def _kb_scenario_cta(free: bool = False) -> InlineKeyboardMarkup:
    """Inline-кнопка «Сценарный анализ» под готовым BASE/DEEP отчётом + меню."""
    price = "бесплатно" if free else "1 токен"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"🎯 Сценарный анализ · {price}",
                              callback_data="scenario:cached")],
        # Меню — НОВЫМ сообщением: это сообщение — часть истории отчёта.
        [InlineKeyboardButton(text="🏠 Меню", callback_data="home:open")],
    ])


async def cb_scenario_cached(callback: CallbackQuery, state: FSMContext) -> None:
    """One-tap сценарный отчёт из закэшированного `results` последнего BASE/DEEP
    прогона.  Billing 1 токен ТОЛЬКО после успешной доставки (как в основном
    флоу).  Кэш протух → просим перезапустить из меню."""
    await callback.answer()
    user_id = callback.from_user.id
    results = _get_cached_results(user_id)
    if results is None:
        await callback.message.answer(
            "⌛ Данные прошлого отчёта уже устарели.\n\n"
            "Запустите *🎯 Сценарный анализ* из меню /start — он посчитается "
            "с нуля.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    if not await _try_acquire_user_slot(user_id):
        await callback.message.answer(
            "⏳ *У вас уже выполняется анализ.* Дождитесь его завершения.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    # Fix C: сценарный отчёт по кэшу ДЕМО-прогона тоже бесплатен (маркер
    # ставится в _run_analysis_background перед кэшированием).
    cost = _effective_cost(
        TIER_SCENARIO, "demo" if results.get("_demo_portfolio") else "freedom")
    try:
        balance = await get_balance(user_id)
        if cost > 0 and balance < cost:
            await callback.message.answer(
                f"❌ *Недостаточно токенов.* Нужен *{cost}*, доступно *{balance}*.\n\n"
                "Пополните баланс: /topup.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        await callback.message.answer(
            "🎯 Собираю *сценарную диагностику* этого портфеля "
            "(Euler-MCTR · 3 макро-режима · бэктест)…",
            parse_mode=ParseMode.MARKDOWN,
        )
        loop = asyncio.get_running_loop()
        # Детерминированная сборка (numpy/бэктест) — в executor, чтобы не
        # блокировать long-poll event-loop.
        payload = await loop.run_in_executor(
            None, _build_pdf_payload, results, TIER_SCENARIO)
        await _send_report(callback.message.bot, callback.message.chat.id,
                           user_id, TIER_SCENARIO, payload)
        if cost > 0:
            await deduct_tokens(user_id, cost, reason="scenario_analysis")
            bal = await get_balance(user_id)
            billing = f"💳 Списан *{cost}* токен · остаток *{bal}*."
        else:
            billing = "📋 Демо-портфель — *бесплатно*, токены не списаны."
        await callback.message.answer(
            f"✅ Сценарный анализ готов. {billing}",
            parse_mode=ParseMode.MARKDOWN,
        )
    except RuntimeError as exc:
        # _send_report сигналит сбой рендера/загрузки — токен НЕ списан.
        logger.warning("Scenario-from-cache delivery failed for %s: %s", user_id, exc)
        await callback.message.answer(
            "😔 Не удалось сформировать сценарный отчёт. Токен *не списан* — "
            "попробуйте позже или запустите из меню /start.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Scenario-from-cache error [%s] user=%s: %s",
                         error_id, user_id, exc)
        await callback.message.answer(
            f"😔 Ошибка сценарного анализа. Код: `{error_id}`. Токен *не списан*.",
            parse_mode=ParseMode.MARKDOWN,
        )
    finally:
        await _release_user_slot(user_id)


# ── Dispatcher assembly ───────────────────────────────────────────────────────

def bot_commands() -> list:
    """Команды «меню ⋮» Telegram — разделы главного меню, тем же языком (`§−124`).

    Список ОДИН на всех (команды не знают пользователя), поэтому
    `/forget_portfolio` в нём — только при ручном вводе «для всех» (`on`); на
    ступени `admins` администратор найдёт удаление в «✏️ Ручной портфель» и в
    /help. `/topup` из меню убран — пополнение открывается из «💳 Баланс»,
    сама команда по-прежнему работает.
    """
    from aiogram.types import BotCommand

    commands = [
        BotCommand(command="start",     description="Главное меню"),
        BotCommand(command="report",    description="Новый отчёт"),
        BotCommand(command="portfolio", description="Мой портфель: брокер, ручной ввод, демо"),
        BotCommand(command="mandate",   description="Мандат: риск-профиль и бенчмарк"),
        BotCommand(command="balance",   description="Баланс и пополнение"),
        BotCommand(command="help",      description="Как пользоваться"),
        BotCommand(command="support",   description="Поддержка"),
    ]
    if rollout_mode(MANUAL_PORTFOLIO_ENV) == FLAG_ON:
        commands.append(BotCommand(command="forget_portfolio",
                                   description="Удалить ручной портфель"))
    return commands


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher(storage=MemoryStorage())

    # Beta whitelist runs FIRST so non-allowed users are filtered out before
    # any handler / FSM transition.  Registered on both message and callback
    # buses so neither path bypasses it.
    dp.message.middleware(WhitelistMiddleware())
    dp.callback_query.middleware(WhitelistMiddleware())

    # Routers first — StateFilter guards prevent cross-fire with AnalysisFlow.
    dp.include_router(onboarding_router)
    dp.include_router(portfolio_router)

    # Message commands
    dp.message.register(cmd_start,    CommandStart())
    dp.message.register(cmd_balance,  F.text == "/balance")
    dp.message.register(cmd_topup,    F.text == "/topup")
    dp.message.register(cmd_support,  F.text == "/support")
    dp.message.register(cmd_help,     F.text == "/help")
    # Admin-only token grant for testing (ADMIN_USER_IDS env gate).
    dp.message.register(cmd_grant,    F.text.startswith("/grant"))
    dp.message.register(cmd_mandate,  F.text == "/mandate")
    # Гибрид PR-1: сохранённый ручной портфель (экран + удаление по запросу).
    dp.message.register(cmd_portfolio,        F.text == "/portfolio")
    dp.message.register(cmd_forget_portfolio, F.text == "/forget_portfolio")
    # §−124: навигация — главное меню, «Мой портфель», прямой вход в отчёт.
    dp.message.register(cmd_report,           F.text == "/report")

    # Analysis flow callbacks
    dp.callback_query.register(cb_analysis_choice, F.data.startswith("analysis:"))
    dp.callback_query.register(cb_confirm,          F.data.startswith("confirm:"))
    # Гибрид PR-2: ручной отчёт вместо брокерского (кнопка из сообщения об отказе).
    dp.callback_query.register(cb_fallback_manual,  F.data.startswith("fb:"))
    # Гибрид PR-3: меню источников (D-9) и подтверждение пересечений (D-5).
    dp.callback_query.register(cb_report_source,      F.data.startswith("src:"))
    dp.callback_query.register(cb_report_tier,        F.data.startswith("rpt"))
    dp.callback_query.register(cb_aggregated_overlap, F.data.startswith("agg:"))
    dp.callback_query.register(cb_home,             F.data.startswith("home:"))
    dp.callback_query.register(cb_portfolio_card,   F.data.startswith("pf:"))
    dp.callback_query.register(cb_scenario_cached,  F.data == "scenario:cached")
    dp.callback_query.register(cb_cancel,           F.data == "cancel")
    # /mandate menu (B1 2026-07-17) — free mandate edits, no billing here.
    dp.callback_query.register(cb_mandate_action,   F.data.startswith("mandate:"))

    # Sprint-5 (Task 1 — hard text-input filter) MUST be LAST.  Bot navigation
    # is button-driven; the ONLY legitimate free-text inputs are the Freedom
    # broker credentials (portfolio_router FSM states Login/ApiKey/SecretKey) —
    # those carry an active FSM state, so `StateFilter(None)` here never swallows
    # them.  Any OTHER stray, non-command text (state is None) is softly bounced
    # back to the inline-button flow instead of being mis-parsed as a command.
    dp.message.register(
        msg_text_fallback,
        StateFilter(None),
        F.text,
        ~F.text.startswith("/"),
    )

    return dp


_CONFLICT_MAX_RETRIES = 5
_CONFLICT_RETRY_DELAY = 10  # seconds

# aiogram session — bounded request timeout so a hung Telegram call can't
# wedge the long-poll loop indefinitely (root of the TelegramNetworkError
# "Request timeout" + TimeoutError seen in the logs).
_SESSION_TIMEOUT_S = 60


class _RetryingSession(AiohttpSession):
    """
    AiohttpSession that retries transient Telegram transport failures
    (TelegramNetworkError "Request timeout", TelegramServerError "Bad
    Gateway"/5xx) with exponential backoff.  TelegramConflictError is NOT
    retried here — it is a multi-instance condition handled in main().
    """

    _MAX_RETRIES   = 3
    _BACKOFF_BASE  = 1.0   # seconds: 1s, 2s, 4s

    async def make_request(self, bot, method, timeout=None):  # type: ignore[override]
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                return await super().make_request(bot, method, timeout=timeout)
            except (TelegramNetworkError, TelegramServerError) as exc:
                last_exc = exc
                if attempt == self._MAX_RETRIES - 1:
                    break
                delay = self._BACKOFF_BASE * (2 ** attempt)
                logger.warning("Telegram transport error (%s) — retry %d/%d in %.0fs",
                               type(exc).__name__, attempt + 1, self._MAX_RETRIES, delay)
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc


async def main() -> None:
    # M-9: refuse to boot if balances / broker creds would land on ephemeral
    # storage in production — fail loud instead of silently losing money state.
    assert_persistent_state()
    await init_db()

    # Bounded-timeout, auto-retrying transport for Telegram API calls.
    session = _RetryingSession(timeout=_SESSION_TIMEOUT_S)
    bot = Bot(token=BOT_TOKEN, session=session)
    dp  = build_dispatcher()

    # B1 (2026-07-17): регистрируем команды в «меню ⋮» Telegram-клиента.
    # Non-fatal: сбой сети здесь не должен ронять бота на старте.
    try:
        await bot.set_my_commands(bot_commands())
    except Exception as exc:                           # noqa: BLE001
        logger.warning("set_my_commands failed: %s", exc)

    # Graceful shutdown: Cloud Run sends SIGTERM before tearing the
    # container down on a new deploy.  Stopping the poller and CLOSING the
    # session makes Telegram release the getUpdates lock immediately, so the
    # incoming instance does not collide → no TelegramConflictError on deploy.
    stop_event = asyncio.Event()

    def _request_stop(signame: str) -> None:
        logger.info("Получен сигнал %s — graceful shutdown.", signame)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(_sig, _request_stop, _sig.name)
        except (NotImplementedError, RuntimeError):
            pass   # add_signal_handler unsupported (e.g. non-main thread)

    async def _watch_shutdown() -> None:
        await stop_event.wait()
        logger.info("Сигнал остановки — graceful shutdown (release getUpdates ≤3с).")
        # 1) Ask the poller to stop — takes effect after the in-flight long-poll
        #    returns (which can be many seconds on an idle bot).
        try:
            await dp.stop_polling()
        except Exception:
            pass
        # 2) Cloud Run HARD rule: the getUpdates lock must be released within
        #    2-3s or the incoming revision collides → 409 Conflict.  Closing the
        #    aiohttp session ABORTS the in-flight getUpdates immediately, so
        #    Telegram frees the lock now instead of waiting out the long-poll.
        #    No long background work here — just stop + close.
        try:
            await asyncio.wait_for(bot.session.close(), timeout=2.5)
            logger.info("Сессия Telegram закрыта (graceful, getUpdates освобождён).")
        except Exception:
            pass
        # 3) `§−122`: аренды слотов ЭТОГО инстанса. Расчёт, застигнутый
        #    редеплоем, всё равно погибнет с процессом, а его аренда жила бы в
        #    SQLite ещё до 30 минут — пользователь видел бы «анализ уже идёт»
        #    и не мог повторить. Снимаем только свои (owner = _INSTANCE_ID):
        #    чужие держатели при max-instances>1 не задеваются.
        # 4) `§−126`: владельцам незавершённых расчётов — сообщение о
        #    прерывании, параллельно с (3). Основная сессия уже закрыта,
        #    поэтому своя, короткая; до SIGKILL у Cloud Run 10 с.
        async def _release_leases() -> None:
            n = await release_report_locks_for_owner(_INSTANCE_ID)
            if n:
                logger.info("Сняты аренды отчётов этого инстанса: %d.", n)

        async def _notify_owners() -> None:
            if not _INFLIGHT_REPORTS:
                return
            notifier = Bot(token=BOT_TOKEN, session=AiohttpSession(timeout=3))
            try:
                async def _send(chat_id, text, kb):
                    return await notifier.send_message(
                        chat_id, text, parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb)
                n = await notify_interrupted_reports(_send)
                logger.info("Прерванные редеплоем отчёты: уведомлено %d из %d.",
                            n, len(_INFLIGHT_REPORTS))
            finally:
                await notifier.session.close()

        results = await asyncio.gather(
            asyncio.wait_for(_release_leases(), timeout=3.0),
            asyncio.wait_for(_notify_owners(), timeout=4.0),
            return_exceptions=True)
        for label, res in zip(("Аренды отчётов", "Уведомления о прерывании"), results):
            if isinstance(res, Exception):
                logger.warning("%s при остановке не завершены: %s", label, res)

    logger.info("%s Bot запущен.", branding.bot_name())
    watcher = asyncio.create_task(_watch_shutdown())
    try:
        for attempt in range(_CONFLICT_MAX_RETRIES):
            try:
                # Make sure no webhook is registered (webhook + polling = 409)
                # and drop the backlog so a redeploy starts clean.
                await bot.delete_webhook(drop_pending_updates=True)
                await dp.start_polling(
                    bot,
                    allowed_updates=dp.resolve_used_update_types(),
                    drop_pending_updates=True,
                    handle_signals=False,   # we manage SIGTERM ourselves
                )
                break
            except TelegramConflictError:
                if stop_event.is_set():
                    break
                if attempt < _CONFLICT_MAX_RETRIES - 1:
                    logger.warning(
                        "Конфликт (409), жду %ds перед попыткой %d/%d",
                        _CONFLICT_RETRY_DELAY, attempt + 2, _CONFLICT_MAX_RETRIES,
                    )
                    await asyncio.sleep(_CONFLICT_RETRY_DELAY)
                else:
                    logger.error(
                        "Конфликт не разрешился за %d попыток. Завершение.",
                        _CONFLICT_MAX_RETRIES,
                    )
                    raise
            except Exception:
                # During graceful shutdown _watch_shutdown closes the session,
                # which aborts the in-flight getUpdates — that surfaces here as a
                # transport/closed-session error.  Expected: exit cleanly.
                if stop_event.is_set():
                    break
                raise
    finally:
        watcher.cancel()
        await bot.session.close()
        logger.info("Сессия Telegram закрыта.")


# if __name__ == "__main__":
#     asyncio.run(main())
