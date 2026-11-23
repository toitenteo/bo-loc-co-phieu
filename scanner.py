import os
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np

# Giả lập trình duyệt để lấy dữ liệu ổn định từ API công khai
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ==============================================================================
# BẢNG THÔNG SỐ LỌC (BẠN CÓ THỂ SỬA CÁC CON SỐ NÀY TRỰC TIẾP TRÊN GITHUB)
# ==============================================================================
CONFIG = {
    # 1. Lớp lọc thanh khoản chung
    "MIN_AVG_VAL_20_BASE": 10.0,      # Giá trị giao dịch TB 20 phiên >= 10 tỷ VNĐ

    # 2. Bộ tiêu chí Swing Trading
    "SWING_MIN_VAL_20": 15.0,         # Thanh khoản TB 20 phiên >= 15 tỷ/phiên
    "SWING_MIN_RS_RATING": 75,        # Sức mạnh tương đối RS >= 75/99
    "SWING_MIN_ATR_PCT": 2.2,         # Biên độ dao động ATR(14) >= 2.2%/phiên

    # 3. Bộ tiêu chí Đầu tư cơ bản
    "INVEST_MIN_ROE": 15.0,           # ROE >= 15%
    "INVEST_MIN_REV_GROWTH": 10.0,    # Tăng trưởng doanh thu >= 10%
    "INVEST_MIN_PROFIT_GROWTH": 15.0, # Tăng trưởng lợi nhuận sau thuế >= 15%
    "INVEST_MAX_DE": 1.5,             # Nợ vay / Vốn chủ sở hữu <= 1.5
    "INVEST_MAX_PE": 22.0,            # Chỉ số P/E <= 22
}


def get_all_stocks():
    """Lấy danh sách cổ phiếu niêm yết trên cả 3 sàn HOSE, HNX, UPCOM."""
    url = (
        "https://finfo-api.vndirect.com.vn/v4/stocks"
        "?q=type:stock~status:listed~floor:HOSE,HNX,UPCOM&size=2000"
    )
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        data = res.json().get("data", [])
        stocks = [
            {"symbol": item["code"], "floor": item["floor"]}
            for item in data
            if item.get("code") and len(item["code"]) == 3
        ]
        print(f"Đã tải danh sách {len(stocks)} mã cổ phiếu.")
        return stocks
    except Exception as exc:
        print(f"[Lỗi] Không tải được danh sách mã: {exc}")
        return []


def calculate_indicators(df):
    """Tính toán các đường MA, ATR, độ siết nền giá và điểm RS thô."""
    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    vol = float(df["volume"].iloc[-1])

    # Thanh khoản trung bình 20 phiên (Tỷ VNĐ)
    avg_vol_20 = float(df["volume"].tail(20).mean())
    avg_val_20 = (avg_vol_20 * close) / 1e9

    if avg_val_20 < CONFIG["MIN_AVG_VAL_20_BASE"]:
        return None

    # Các đường trung bình động
    ma20 = float(df["close"].tail(20).mean())
    ma50 = float(df["close"].tail(50).mean())
    ma200 = float(df["close"].tail(200).mean())
    ma200_10d_ago = float(df["close"].iloc[-210:-10].mean())

    # Tính biến động ATR(14) theo %
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_pct = (float(true_range.tail(14).mean()) / close) * 100

    # Tính điểm sức mạnh giá thô (trọng số 3 tháng - 6 tháng - 9 tháng - 12 tháng)
    c_63 = float(df["close"].iloc[-63]) if len(df) >= 63 else float(df["close"].iloc[0])
    c_126 = float(df["close"].iloc[-126]) if len(df) >= 126 else float(df["close"].iloc[0])
    c_189 = float(df["close"].iloc[-189]) if len(df) >= 189 else float(df["close"].iloc[0])
    c_252 = float(df["close"].iloc[-252]) if len(df) >= 252 else float(df["close"].iloc[0])

    raw_rs = (
        0.4 * ((close - c_63) / c_63)
        + 0.2 * ((close - c_126) / c_126)
        + 0.2 * ((close - c_189) / c_189)
        + 0.2 * ((close - c_252) / c_252)
    )

    # Biên độ dao động 5 phiên gần nhất & khoảng cách tới đỉnh 1 năm
    range_5d = ((float(df["high"].tail(5).max()) - float(df["low"].tail(5).min())) / close) * 100
    high_52w = float(df["high"].tail(252).max())
    pct_from_high = ((close - high_52w) / high_52w) * 100

    return {
        "close": close,
        "prev_close": prev_close,
        "avg_val_20": round(avg_val_20, 1),
        "ma20": round(ma20, 0),
        "ma50": round(ma50, 0),
        "ma200": round(ma200, 0),
        "ma200_up": ma200 > ma200_10d_ago,
        "atr_pct": round(atr_pct, 2),
        "vol_ratio": round(vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        "range_5d": round(range_5d, 2),
        "pct_from_high": round(pct_from_high, 1),
        "raw_rs": raw_rs,
    }


def fetch_stock_data(stock_info):
    """Tải dữ liệu lịch sử giá và chỉ số tài chính của 1 cổ phiếu."""
    symbol = stock_info["symbol"]
    floor = stock_info["floor"]
    now_sec = int(time.time())
    from_sec = now_sec - (400 * 24 * 60 * 60)

    try:
        price_url = (
            f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
            f"?ticker={symbol}&type=stock&resolution=D&from={from_sec}&to={now_sec}"
        )
        p_res = requests.get(price_url, headers=HEADERS, timeout=10)
        if p_res.status_code != 200:
            return None

        bars = p_res.json().get("data", [])
        if len(bars) < 210:
            return None

        df = pd.DataFrame(bars)
        tech = calculate_indicators(df)
        if tech is None:
            return None

        fa_url = (
            f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/"
            f"{symbol}/financialratio?yearly=0&isAll=false"
        )
        f_res = requests.get(fa_url, headers=HEADERS, timeout=10)
        fa = f_res.json()[0] if (f_res.status_code == 200 and f_res.json()) else {}

        return {
            "symbol": symbol,
            "floor": floor,
            **tech,
            "roe": round((fa.get("roe") or 0) * 100, 1),
            "rev_growth": round((fa.get("revenueGrowth") or 0) * 100, 1),
            "profit_growth": round((fa.get("postTaxProfitGrowth") or 0) * 100, 1),
            "de": round(fa.get("debtOnEquity") or 0, 2),
            "pe": round(fa.get("pe") or 0, 1),
            "pb": round(fa.get("pb") or 0, 1),
        }
    except Exception:
        return None


def build_swing_signals(row):
    """Gắn nhãn nhận diện hành động giá nổi bật cho cổ phiếu Swing."""
    signals = []
    if row["vol_ratio"] >= 1.5 and row["close"] > row["prev_close"]:
        signals.append(f"Bùng nổ Vol ({row['vol_ratio']}x)")
    if row["range_5d"] <= 6.0:
        signals.append("Nền giá siết chặt")
    if abs(row["close"] - row["ma20"]) / row["ma20"] <= 0.02:
        signals.append("Test hỗ trợ MA20")
    if row["pct_from_high"] >= -5.0:
        signals.append("Sát đỉnh 52W")
    return " | ".join(signals) if signals else "Đạt chuẩn Trend"


def export_markdown_report(df_all, swing_df, invest_df, timestamp_str):
    """Xuất bảng kết quả ra file KET_QUA_LOC.md để xem trực tiếp trên GitHub."""
    lines = [
        f"# Báo Cáo Quét Cổ Phiếu Tự Động ({timestamp_str} GMT+7)\n",
        f"- Tổng số mã đủ điều kiện thanh khoản phân tích: **{len(df_all)} mã**",
        f"- Số mã đạt chuẩn **Swing Trading**: **{len(swing_df)} mã**",
        f"- Số mã đạt chuẩn **Đầu tư cơ bản**: **{len(invest_df)} mã**\n",
        "---\n",
        "## 1. Danh Sách Swing Trading (Xu hướng tăng + RS >= 75)\n",
        "| Mã | Sàn | Giá | RS Rating | GTGD 20P (Tỷ) | ATR (%) | Vol/TB20 | Biên độ 5P | Tín hiệu nổi bật |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    if swing_df.empty:
        lines.append("| *Không có mã nào đạt tiêu chí Swing phiên này* | - | - | - | - | - | - | - | - |")
    else:
        for _, r in swing_df.iterrows():
            lines.append(
                f"| **{r['symbol']}** | {r['floor']} | {r['close']:,.0f} | **{r['rs_rating']}** | "
                f"{r['avg_val_20']} | {r['atr_pct']}% | {r['vol_ratio']}x | {r['range_5d']}% | {r['signals']} |"
            )

    lines.extend([
        "\n---\n",
        "## 2. Danh Sách Đầu Tư Cơ Bản (ROE >= 15%, Tăng trưởng lợi nhuận >= 15%)\n",
        "| Mã | Sàn | Giá | ROE (%) | TT Doanh thu | TT Lợi nhuận | Nợ/VCSH | P/E | P/B | Đánh giá |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if invest_df.empty:
        lines.append("| *Không có mã nào đạt tiêu chí Đầu tư phiên này* | - | - | - | - | - | - | - | - | - |")
    else:
        for _, r in invest_df.iterrows():
            note = "Định giá hấp dẫn + Tăng trưởng" if r["pe"] <= 14 else "Tăng trưởng chất lượng"
            lines.append(
                f"| **{r['symbol']}** | {r['floor']} | {r['close']:,.0f} | **{r['roe']}%** | "
                f"{r['rev_growth']}% | {r['profit_growth']}% | {r['de']} | {r['pe']} | {r['pb']} | {note} |"
            )

    with open("KET_QUA_LOC.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Đã xuất file KET_QUA_LOC.md thành công.")


def send_telegram_alert(swing_df, invest_df, timestamp_str):
    """Tự động gửi tin nhắn Telegram nếu bạn đã điền Token trong GitHub Secrets."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    top_swing = (
        "\n".join(
            f"• *{r['symbol']}* (RS {r['rs_rating']}): {r['signals']}"
            for _, r in swing_df.head(12).iterrows()
        )
        or "Không có mã đạt chuẩn."
    )

    top_invest = (
        "\n".join(
            f"• *{r['symbol']}* (ROE {r['roe']}%, P/E {r['pe']})"
            for _, r in invest_df.head(12).iterrows()
        )
        or "Không có mã đạt chuẩn."
    )

    msg = (
        f"📊 *BÁO CÁO QUÉT CỔ PHIẾU ({timestamp_str})*\n\n"
        f"🚀 *SWING TRADING ({len(swing_df)} mã):*\n{top_swing}\n\n"
        f"💎 *ĐẦU TƯ CƠ BẢN ({len(invest_df)} mã):*\n{top_invest}"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


def main():
    vn_tz = timezone(timedelta(hours=7))
    now_str = datetime.now(vn_tz).strftime("%d/%m/%Y %H:%M")

    stocks = get_all_stocks()
    results = []

    if stocks:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fetch_stock_data, s) for s in stocks]
            for future in as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

    print(f"Số mã đạt chuẩn thanh khoản phân tích: {len(results)}")

    if not results:
        empty_df = pd.DataFrame()
        export_markdown_report(empty_df, empty_df, empty_df, now_str)
        return

    df_all = pd.DataFrame(results)
    df_all["rs_rating"] = (df_all["raw_rs"].rank(pct=True) * 98 + 1).round(0).astype(int)

    # Lọc danh sách Swing Trading
    swing_df = df_all[
        (df_all["avg_val_20"] >= CONFIG["SWING_MIN_VAL_20"])
        & (df_all["close"] > df_all["ma20"])
        & (df_all["ma20"] > df_all["ma50"])
        & (df_all["ma50"] > df_all["ma200"])
        & (df_all["ma200_up"] == True)
        & (df_all["rs_rating"] >= CONFIG["SWING_MIN_RS_RATING"])
        & (df_all["atr_pct"] >= CONFIG["SWING_MIN_ATR_PCT"])
    ].copy()

    if not swing_df.empty:
        swing_df["signals"] = swing_df.apply(build_swing_signals, axis=1)
        swing_df = swing_df.sort_values(by=["rs_rating", "vol_ratio"], ascending=False)

    # Lọc danh sách Đầu tư cơ bản
    invest_df = df_all[
        (df_all["roe"] >= CONFIG["INVEST_MIN_ROE"])
        & (df_all["rev_growth"] >= CONFIG["INVEST_MIN_REV_GROWTH"])
        & (df_all["profit_growth"] >= CONFIG["INVEST_MIN_PROFIT_GROWTH"])
        & (df_all["de"] <= CONFIG["INVEST_MAX_DE"])
        & (df_all["pe"] > 0)
        & (df_all["pe"] <= CONFIG["INVEST_MAX_PE"])
    ].sort_values(by=["roe", "profit_growth"], ascending=False)

    export_markdown_report(df_all, swing_df, invest_df, now_str)
    send_telegram_alert(swing_df, invest_df, now_str)


if __name__ == "__main__":
    main()
