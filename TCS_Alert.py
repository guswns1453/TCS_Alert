# ==========================================================
# TCS Daily Signal Notifier (robust ver. for GitHub Actions)
# - 지표: MA20/MA60, 일목 SpanA/B, 구름두께(4분위), MA 교차
# - 동작: 미국 거래일마다 1건 텔레그램 알림
# - 안정성: yfinance 재시도 + Stooq CSV 폴백
# - 시크릿: TELEGRAM_TOKEN, TELEGRAM_CHAT (GitHub Secrets 권장)
# ==========================================================
import os
import io
import time
import json
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, IchimokuIndicator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

# -----------------------------
# 설정
# -----------------------------
TICKERS = ["QQQ", "SPY", "SOXX"]
START_DATE = "2010-01-01"  # 지표 안정화 + 회귀 대비
NEAR_BAND = 0.01           # MA20 대비 ±1%는 'MA근처'
RETRY_MAX = 4
BACKOFF_SEC = 3            # 3, 6, 9, 12s

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT", "").strip()

pd.set_option("display.width", 120)

# -----------------------------
# 텔레그램
# -----------------------------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("[알림 비활성화] TELEGRAM_TOKEN/CHAT 미설정")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": text},
            timeout=15,
        )
        print("Telegram:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram Error:", e)

# -----------------------------
# 날짜 유틸 (미국 세션 확정 기준)
# -----------------------------
def is_business_day(d: datetime.date) -> bool:
    return d.weekday() < 5  # Mon~Fri

def prev_business_day(d: datetime.date) -> datetime.date:
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d

def get_target_us_session_date(now_et: datetime | None = None) -> datetime.date:
    """
    '확정된' 일봉을 대상으로 알림을 보내기 위해
    ET 18:00 이전엔 전 거래일, 이후엔 당일을 타깃으로 삼는다.
    주말이면 직전 거래일.
    """
    if now_et is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    today = now_et.date()

    if not is_business_day(today):
        return prev_business_day(today - timedelta(days=1))

    cutoff = now_et.replace(hour=18, minute=0, second=0, microsecond=0)
    if now_et < cutoff:
        d = today - timedelta(days=1)
        return prev_business_day(d)
    return today

# -----------------------------
# 데이터 로드 (yfinance → Stooq 폴백)
# -----------------------------
def _stooq_csv_url(ticker: str) -> str:
    # Stooq 포맷: 소문자 + .us (미국 종목)
    return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

def _download_stooq(ticker: str, start: str) -> pd.DataFrame:
    url = _stooq_csv_url(ticker)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    # Stooq 컬럼: Date, Open, High, Low, Close, Volume
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.loc[df.index >= pd.to_datetime(start)]
    return df

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 정규화 (Yahoo/Stooq 공통 처리)
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for need in ["open", "high", "low", "close", "adj close", "volume"]:
        if need in cols:
            mapping[cols[need]] = need.title() if need != "adj close" else "Adj Close"
    out = df.rename(columns=mapping)
    # 1D 보정
    for col in ["Close", "High", "Low"]:
        s = out[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        out[col] = pd.Series(np.asarray(s).ravel(), index=out.index, name=col)
    return out.dropna().copy()

def load_price(ticker: str, start: str) -> pd.DataFrame:
    # 1) yfinance 재시도
    for attempt in range(1, RETRY_MAX + 1):
        print(f"📥 {ticker} 데이터 불러오는 중... (시도 {attempt}/{RETRY_MAX})")
        try:
            df = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,  # 러너 안정성
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and len(df) > 0:
                return _normalize_ohlc(df)
        except Exception as e:
            print(f"⚠️  {ticker} yfinance 오류: {e!s}")

        # 백오프
        sleep_s = attempt * BACKOFF_SEC
        print(f"⚠️  {ticker} 다운로드 실패 → {sleep_s}s 대기 후 재시도")
        time.sleep(sleep_s)

    print(f"❌ yfinance 최종 실패 → Stooq 폴백 시도: {ticker}")
    # 2) Stooq 폴백
    try:
        df = _download_stooq(ticker, start)
        if len(df):
            return _normalize_ohlc(df)
    except Exception as e:
        print(f"❌ Stooq 폴백 실패({ticker}): {e!s}")

    # 3) 최종 실패
    print(f"❌ {ticker} 다운로드 완전 실패")
    return pd.DataFrame()

# -----------------------------
# 지표 계산 & 상태 분류
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = SMAIndicator(out["Close"], 20).sma_indicator()
    out["ma60"] = SMAIndicator(out["Close"], 60).sma_indicator()

    ichi = IchimokuIndicator(high=out["High"], low=out["Low"], window1=9, window2=26, window3=52)
    out["senkou_a"] = ichi.ichimoku_a()
    out["senkou_b"] = ichi.ichimoku_b()

    out["cloud_top"] = np.maximum(out["senkou_a"], out["senkou_b"])
    out["cloud_bot"] = np.minimum(out["senkou_a"], out["senkou_b"])
    out["cloud_thickness"] = (out["cloud_top"] - out["cloud_bot"]).abs()

    # 구름두께 4분위
    try:
        out["cloud_strength"] = pd.qcut(
            out["cloud_thickness"].rank(method="first"),
            q=[0, 0.25, 0.5, 0.75, 1],
            labels=["매우 얇음", "얇음", "두꺼움", "매우 두꺼움"],
        )
    except Exception:
        out["cloud_strength"] = "얇음"

    # MA 교차
    out["ma_cross"] = np.where(
        (out["ma20"].shift(1) <= out["ma60"].shift(1)) & (out["ma20"] > out["ma60"]),
        "Golden Cross",
        np.where(
            (out["ma20"].shift(1) >= out["ma60"].shift(1)) & (out["ma20"] < out["ma60"]),
            "Dead Cross",
            "Hold",
        ),
    )
    return out

def classify_state(df: pd.DataFrame) -> pd.DataFrame:
    ma_state = np.where(
        df["ma20"] > df["ma60"], "정배열",
        np.where(df["ma20"] < df["ma60"], "역배열", "횡보"),
    )
    above = df["Close"] > df["ma20"] * (1 + NEAR_BAND)
    below = df["Close"] < df["ma20"] * (1 - NEAR_BAND)
    ma_pos = np.where(above, "MA위", np.where(below, "MA아래", "MA근처"))

    ich_state = np.where(
        df["Close"] > df["cloud_top"], "구름 위",
        np.where(df["Close"] < df["cloud_bot"], "구름 아래", "구름 내부"),
    )

    mapping = {
        ("정배열", "MA위",   "구름 위"):   "A1",
        ("정배열", "MA위",   "구름 내부"): "A2",
        ("정배열", "MA아래", "구름 위"):   "B1",
        ("정배열", "MA근처", "구름 내부"): "B2",
        ("횡보",   "MA위",   "구름 내부"): "C1",
        ("횡보",   "MA아래", "구름 아래"): "C2",
        ("역배열", "MA아래", "구름 내부"): "D1",
        ("역배열", "MA아래", "구름 아래"): "D2",
        ("역배열", "MA위",   "구름 내부"): "E1",
        ("역배열", "MA위",   "구름 위"):   "E2",
    }
    combo = list(zip(ma_state, ma_pos, ich_state))
    codes = [mapping.get(c, np.nan) for c in combo]
    desc_map = {v: f"({a} · {b} · {c})" for (a, b, c), v in mapping.items()}
    descs = [desc_map.get(code, "") for code in codes]

    out = df.copy()
    out["state"] = codes
    out["state_desc"] = descs
    return out

def decision_from_state(code: str) -> str:
    if code in ["A1", "B1", "E2"]:
        return "BUY"
    if code in ["D2", "D1"]:
        return "SELL"
    return "NEUTRAL"

# -----------------------------
# 메인
# -----------------------------
if __name__ == "__main__":
    # GitHub Actions는 UTC가 기본 → ET 고정
    os.environ.setdefault("TZ", "America/New_York")

    target_date = get_target_us_session_date()
    print("Target US session date:", target_date)

    any_sent = False
    for t in TICKERS:
        df = load_price(t, START_DATE)
        if df.empty:
            print(f"{t}: 데이터 비어있음 → 스킵")
            continue

        df = compute_indicators(df)
        df = classify_state(df)

        latest_ts = df.index[-1]
        latest_date = latest_ts.date()
        if latest_date != target_date:
            print(f"{t}: 스킵 (latest={latest_date}, target={target_date})")
            continue

        row = df.iloc[-1]
        decision = decision_from_state(row["state"])

        msg = (
            f"[{t}] {decision}\n"
            f"상태코드: {row['state']} {row['state_desc']}\n"
            f"구름두께: {row['cloud_strength']}\n"
            f"이동평균선교차: {row['ma_cross']}\n"
            f"날짜: {latest_ts.strftime('%Y-%m-%d')}"
        )
        print(msg)
        send_telegram(msg)
        any_sent = True
        # 깃허브 러너에서 호출 과도 방지
        time.sleep(1.5)

    if not any_sent:
        print("No messages sent (holiday, data not ready, or all skipped).")
