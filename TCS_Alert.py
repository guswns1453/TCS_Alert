# ==========================================================
# TCS Daily Signal Notifier (proxy+yf → AlphaVantage → Stooq)
# - 우선순위: yfinance(프록시) → Alpha Vantage → Stooq
# - 지표: MA20/MA60, 일목 SpanA/B, 구름두께(4분위), MA 교차
# - 동작: 미국 거래일마다 1건 텔레그램 알림 (완료된 일봉 기준)
# - 시크릿: TELEGRAM_TOKEN, TELEGRAM_CHAT, ALPHA_VANTAGE_KEY, PROXY_URL(선택)
# ==========================================================
import os
import io
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, IchimokuIndicator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------
# 설정
# -----------------------------
TICKERS = ["QQQ", "SPY", "SOXX"]
START_DATE = "2010-01-01"
NEAR_BAND = 0.01         # MA20 대비 ±1%는 'MA근처'
RETRY_MAX = 4
BACKOFF_SEC = 3          # 3, 6, 9, 12s

TELEGRAM_TOKEN      = (os.getenv("TELEGRAM_TOKEN", "") or "").strip()
TELEGRAM_CHAT       = (os.getenv("TELEGRAM_CHAT", "") or "").strip()
ALPHA_VANTAGE_KEY   = (os.getenv("ALPHA_VANTAGE_KEY", "") or "").strip()
PROXY_URL           = (os.getenv("PROXY_URL", "") or "").strip()  # 예: http://user:pass@host:port

pd.set_option("display.width", 120)

# -----------------------------
# 프록시 (yfinance/requests가 인식하는 환경변수로 설정)
# -----------------------------
if PROXY_URL:
    os.environ["HTTP_PROXY"]  = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

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
    return d.weekday() < 5

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
# 소스3: Stooq CSV
# -----------------------------
def _stooq_csv_url(ticker: str) -> str:
    return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

def _download_stooq(ticker: str, start: str) -> pd.DataFrame:
    url = _stooq_csv_url(ticker)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.loc[df.index >= pd.to_datetime(start)]
    return df

# -----------------------------
# 소스2: Alpha Vantage
# -----------------------------
def _download_alpha_vantage(ticker: str, start: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_KEY:
        raise RuntimeError("ALPHA_VANTAGE_KEY missing")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_KEY}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    # 실패/쿼터초과 처리
    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"AlphaVantage bad payload: {json.dumps(data)[:200]}")
    ts = data["Time Series (Daily)"]
    # DataFrame 변환
    df = pd.DataFrame.from_dict(ts, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
        }
    )
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.loc[df.index >= pd.to_datetime(start)]
    return df.dropna()

# -----------------------------
# 소스1: yfinance (프록시 적용, 재시도)
# -----------------------------
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 정규화 (Yahoo/Stooq/AV 공통 처리)
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for need in ["open", "high", "low", "close", "adj close", "volume"]:
        if need in cols:
            mapping[cols[need]] = need.title() if need != "adj close" else "Adj Close"
    out = df.rename(columns=mapping)
    for col in ["Close", "High", "Low"]:
        s = out[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        out[col] = pd.Series(np.asarray(s).ravel(), index=out.index, name=col)
    return out.dropna().copy()

def _download_yfinance(ticker: str, start: str) -> pd.DataFrame:
    for attempt in range(1, RETRY_MAX + 1):
        print(f"📥 {ticker} (yfinance) ... 시도 {attempt}/{RETRY_MAX}")
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
            print(f"⚠️  yfinance 오류({ticker}): {e!s}")
        # 백오프
        sleep_s = attempt * BACKOFF_SEC
        print(f"   → {sleep_s}s 대기 후 재시도")
        time.sleep(sleep_s)
    raise RuntimeError("yfinance 최종 실패")

# -----------------------------
# 통합 로더 (yf → AV → Stooq)
# -----------------------------
def load_price(ticker: str, start: str) -> pd.DataFrame:
    # 1) yfinance (프록시 포함)
    try:
        return _download_yfinance(ticker, start)
    except Exception as e:
        print(f"❌ yfinance 실패({ticker}): {e!s}")

    # 2) Alpha Vantage
    try:
        print(f"→ Alpha Vantage 폴백 시도: {ticker}")
        df = _download_alpha_vantage(ticker, start)
        if len(df):
            return _normalize_ohlc(df)
    except Exception as e:
        print(f"❌ Alpha Vantage 실패({ticker}): {e!s}")

    # 3) Stooq
    try:
        print(f"→ Stooq 폴백 시도: {ticker}")
        df = _download_stooq(ticker, start)
        if len(df):
            return _normalize_ohlc(df)
    except Exception as e:
        print(f"❌ Stooq 실패({ticker}): {e!s}")

    print(f"❌ {ticker} 모든 소스 실패")
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
        "골든크로스",
        np.where(
            (out["ma20"].shift(1) >= out["ma60"].shift(1)) & (out["ma20"] < out["ma60"]),
            "데드크로스",
            "유지",
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
        time.sleep(1.0)  # 텔레그램 rate-limit 완충

    if not any_sent:
        print("No messages sent (holiday, data not ready, or all skipped).")
