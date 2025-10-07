# ==========================================================
# TCS Daily Signal Notifier (Daily push on US trading days)
# - 지표: MA20/MA60, 일목(SpanA/B), 구름두께(사분위), MA 교차
# - 동작: 거래일(미국 현지)마다 1건 텔레그램 알림 전송
# - 타임존 이슈 해결: ET 18:00 기준으로 타깃 세션 날짜 판정
# - CI 안정화: yfinance 재시도 + 빈 데이터 스킵
# ==========================================================
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from ta.trend import SMAIndicator, IchimokuIndicator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

TICKERS = ["QQQ", "SPY", "SOXX"]
START_DATE = "2010-01-01"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT", "")  # Actions의 secret 이름과 일치

pd.set_option("display.width", 120)

# -----------------------------
# 텔레그램
# -----------------------------
def send_telegram(text: str):
    token = (TELEGRAM_TOKEN or "").strip()
    chat  = (TELEGRAM_CHAT or "").strip()
    if not token or not chat:
        print("[알림 비활성화] TELEGRAM_TOKEN/CHAT 미설정")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text},
            timeout=10
        )
        print("Telegram:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram Error:", e)

# -----------------------------
# 미국 세션 날짜 판정
# -----------------------------
def is_business_day(d):  # 월=0..일=6
    return d.weekday() < 5

def prev_business_day(d):
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d

def get_target_us_session_date(now_et=None):
    """
    '완전히 확정된' 일봉이 존재해야 하는 미국 세션 날짜를 반환.
    - 주말이면: 직전 영업일
    - 평일이면: ET 18:00(6pm) 이전엔 전 영업일, 이후엔 당일
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
# 데이터 로드 (재시도 + 방어)
# -----------------------------
def _postprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Close", "High", "Low"]:
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            df[col] = pd.Series(np.asarray(s).ravel(), index=df.index, name=col)
    return df

def load_price(ticker: str, start: str, max_retries: int = 4) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"📥 {ticker} 데이터 불러오는 중... (시도 {attempt}/{max_retries})")
            df = yf.download(
                ticker, start=start, progress=False, auto_adjust=False, actions=False, threads=False
            )
            if df is None or len(df) == 0:
                raise RuntimeError("empty dataframe")
            df = _postprocess_columns(df).dropna().copy()
            if len(df) == 0:
                raise RuntimeError("empty after postprocess")
            return df
        except Exception as e:
            last_err = e
            wait = 3 * attempt
            print(f"⚠️  {ticker} 다운로드 실패: {e} → {wait}s 대기 후 재시도")
            time.sleep(wait)
    print(f"❌ {ticker} 다운로드 최종 실패: {last_err}")
    return pd.DataFrame()  # 빈 DF 반환하여 상위에서 스킵

# -----------------------------
# 지표 계산
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
    try:
        out["cloud_strength"] = pd.qcut(
            out["cloud_thickness"].rank(method="first"),
            q=[0, 0.25, 0.5, 0.75, 1],
            labels=["매우 얇음", "얇음", "두꺼움", "매우 두꺼움"]
        )
    except Exception:
        out["cloud_strength"] = "얇음"

    out["ma_cross"] = np.where(
        (out["ma20"].shift(1) <= out["ma60"].shift(1)) & (out["ma20"] > out["ma60"]),
        "Golden Cross",
        np.where(
            (out["ma20"].shift(1) >= out["ma60"].shift(1)) & (out["ma20"] < out["ma60"]),
            "Dead Cross",
            "Hold"
        )
    )
    return out

# -----------------------------
# 상태 분류 + 설명
# -----------------------------
def classify_state(df: pd.DataFrame) -> pd.DataFrame:
    ma_state = np.where(
        df["ma20"] > df["ma60"], "정배열",
        np.where(df["ma20"] < df["ma60"], "역배열", "횡보")
    )
    near_band = 0.01
    above = df["Close"] > df["ma20"] * (1 + near_band)
    below = df["Close"] < df["ma20"] * (1 - near_band)
    ma_pos = np.where(above, "MA위", np.where(below, "MA아래", "MA근처"))

    ich_state = np.where(
        df["Close"] > df["cloud_top"], "구름 위",
        np.where(df["Close"] < df["cloud_bot"], "구름 아래", "구름 내부")
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
    state_codes = [mapping.get(c, np.nan) for c in combo]
    desc_map = {v: f"({a} · {b} · {c})" for (a, b, c), v in mapping.items()}
    state_desc = [desc_map.get(code, "") for code in state_codes]

    out = df.copy()
    out["state"] = state_codes
    out["state_desc"] = state_desc
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
    target_date = get_target_us_session_date()
    any_sent = False

    for t in TICKERS:
        try:
            px = load_price(t, START_DATE)
            if px.empty:
                print(f"{t}: 데이터 비어있음 → 스킵")
                continue

            px = compute_indicators(px)
            px = classify_state(px)

            latest_ts = px.index[-1]
            latest_date = latest_ts.date()

            if latest_date != target_date:
                print(f"{t}: 비거래일 스킵 (latest={latest_date}, target={target_date})")
                continue

            latest = px.iloc[-1]
            latest_state = latest["state"]
            latest_desc  = latest["state_desc"]
            cloud_strength = latest["cloud_strength"]
            ma_cross       = latest["ma_cross"]
            date_str       = latest_ts.strftime("%Y-%m-%d")

            decision = decision_from_state(latest_state)
            msg = (
                f"[{t}] {decision}\n"
                f"상태코드: {latest_state} {latest_desc}\n"
                f"구름두께: {cloud_strength}\n"
                f"이동평균선교차: {ma_cross}\n"
                f"날짜: {date_str}"
            )
            print(msg)
            send_telegram(msg)
            any_sent = True
        except Exception as e:
            # 어떤 티커가 실패해도 전체 잡은 계속 진행
            print(f"❗ {t} 처리 중 예외: {e} → 스킵")

    if not any_sent:
        # 주말/휴일/데이터 이슈로 아무 것도 못 보냈을 때 로그만 남김
        print("No messages sent (holiday or data not ready).")

    # 예외 없이 여기까지 오면 Exit 0
