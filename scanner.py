import os
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np

# Giả lập trình duyệt chuẩn để hạn chế bị tường lửa chặn
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ==============================================================================
# BẢNG THÔNG SỐ LỌC (BẠN CÓ THỂ SỬA CÁC CON SỐ NÀY TRỰC TIẾP TRÊN GITHUB)
# ==============================================================================
CONFIG = {
    # 1. Lớp lọc thanh khoản chung
    "MIN_AVG_VAL_20_BASE": 10.0,      # Giá trị giao dịch TB 20 phiên >= 10 tỷ VNĐ

    # 2. Bộ tiêu chí Swing Trading
    "SWING_MIN_VAL_20": 15.0,         # Thanh khoản TB 20 phiên >= 15 tỷ/phiên
    "SWING_MIN_RS_RATING": 70,        # Sức mạnh tương đối RS >= 70/99
    "SWING_MIN_ATR_PCT": 2.0,         # Biên độ dao động ATR(14) >= 2.0%/phiên

    # 3. Bộ tiêu chí Đầu tư cơ bản
    "INVEST_MIN_ROE": 15.0,           # ROE >= 15%
    "INVEST_MIN_REV_GROWTH": 10.0,    # Tăng trưởng doanh thu >= 10%
    "INVEST_MIN_PROFIT_GROWTH": 15.0, # Tăng trưởng lợi nhuận sau thuế >= 15%
    "INVEST_MAX_DE": 1.5,             # Nợ vay / Vốn chủ sở hữu <= 1.5
    "INVEST_MAX_PE": 22.0,            # Chỉ số P/E <= 22
}

# Danh sách dự phòng ~150 mã thanh khoản cao nhất 3 sàn (dùng khi API danh sách bị chặn IP)
FALLBACK_STOCKS = [
    # Ngân hàng
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "HDB", "LPB",
    "VIB", "TPB", "SHB", "MSB", "OCB", "EIB", "NAB", "BAB", "ABB", "NVB",
    # Chứng khoán
    "SSI", "VND", "VCI", "HCM", "SHS", "MBS", "FTS", "BSI", "CTS", "VIX",
    "AGR", "ORS", "BVS", "VDS", "TVS", "DSC", "AAS",
    # Thép & Vật liệu
    "HPG", "HSG", "NKG", "VGS", "TVN", "SMC", "BMP", "NTP", "HT1", "BCC",
    "PTB", "VCS", "KSB", "DHA", "VLB",
    # Bất động sản & KCN
    "VHM", "VIC", "VRE", "KDH", "NLG", "PDR", "DXG", "DIG", "CEO", "NVL",
    "TCH", "HDG", "HDC", "AGG", "SCR", "NTL", "DXS", "CRE", "HPX", "LDG",
    "KBC", "IDC", "SZC", "VGC", "GVR", "PHR", "DPR", "BCM", "SIP", "LHG", "NTC",
    # Công nghệ & Viễn thông
    "FPT", "CMG", "ELC", "CTR", "VGI", "FOX", "ITD", "SGT", "VTP",
    # Bán lẻ & Tiêu dùng
    "MWG", "PNJ", "FRT", "DGW", "PET", "MSN", "VNM", "SAB", "KDC", "MCH",
    "DBC", "HAG", "BAF", "PAN", "SBT", "QNS", "ANV", "VHC", "IDI", "FMC", "CMX",
    # Dầu khí & Hóa chất & Phân bón
    "GAS", "PVD", "PVS", "BSR", "PLX", "OIL", "PVC", "PVB", "PVT", "CNG",
    "DGC", "DCM", "DPM", "CSV", "LAS", "BFC", "DDV",
    # Cảng biển, Vận tải & Hàng không
    "GMD", "HAH", "VOS", "VSC", "PHP", "SGP", "ACV", "HVN", "VJC", "SCS", "NCT",
    # Điện, Xây dựng & Hạ tầng
    "REE", "PC1", "POW", "NT2", "GEG", "BCG", "TV2", "QTP", "HND",
    "VCG", "HHV", "LCG", "FCN", "CII", "CTD", "C4G", "GEX", "DPG", "HTN",
    # Dệt may, Bảo hiểm, Dược & Khác
    "TNG", "MSH", "GIL", "TCM", "STK", "VGT", "DHC", "HAX", "DRC", "CSM",
    "BVH", "MIG", "PVI", "BMI", "DHG", "IMP", "DBD", "TRA",
]


def get_all_stocks():
    """Lấy danh sách cổ phiếu với cơ chế 3 tầng dự phòng."""
    # Tầng 1: Vietcap (VCI)
    try:
        url = "https://trading.vietcap.com.vn/api/price/symbols/getAll"
        headers = {**HEADERS, "Referer": "https://trading.vietcap.com.vn/"}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            stocks = [
                {
                    "symbol": item["symbol"],
                    "floor": "HOSE" if item.get("board") == "HSX" else item.get("board", "HOSE"),
                }
                for item in data
                if item.get("type") == "STOCK"
                and item.get("board") in ("HSX", "HNX", "UPCOM")
                and item.get("symbol")
                and len(item["symbol"]) == 3
            ]
            if len(stocks) >= 100:
                print(f"[Danh sách] Lấy thành công {len(stocks)} mã từ Vietcap.")
                return stocks, f"Vietcap ({len(stocks)} mã)"
    except Exception as exc:
        print(f"[Danh sách] Vietcap lỗi: {exc}")

    # Tầng 2: VNDirect
    try:
        url = (
            "https://finfo-api.vndirect.com.vn/v4/stocks"
            "?q=type:stock~status:listed~floor:HOSE,HNX,UPCOM&size=2000"
        )
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            stocks = [
                {"symbol": item["code"], "floor": item.get("floor", "HOSE")}
                for item in data
                if item.get("code") and len(item["code"]) == 3
            ]
            if len(stocks) >= 100:
                print(f"[Danh sách] Lấy thành công {len(stocks)} mã từ VNDirect.")
                return stocks, f"VNDirect ({len(stocks)} mã)"
    except Exception as exc:
        print(f"[Danh sách] VNDirect lỗi: {exc}")

    # Tầng 3: Danh sách dự phòng tích hợp sẵn
    unique_codes = sorted(set(FALLBACK_STOCKS))
    stocks = [{"symbol": code, "floor": "VN"} for code in unique_codes]
    print(f"[Danh sách] Dùng danh sách dự phòng ({len(stocks)} mã thanh khoản cao).")
    return stocks, f"Danh sách dự phòng ({len(stocks)} mã thanh khoản cao)"


def fetch_ohlcv_dnse(symbol, from_sec, to_sec):
    """Nguồn giá 1: DNSE / Entrade Chart API (đơn vị gốc là nghìn VNĐ -> nhân 1000)."""
    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
        f"?from={from_sec}&to={to_sec}&symbol={symbol}&resolution=1D"
    )
    res = requests.get(url, headers=HEADERS, timeout=8)
    if res.status_code != 200:
        return None
    data = res.json()
    closes = data.get("c") or []
    if len(closes) < 205:
        return None

    df = pd.DataFrame({
        "open": pd.to_numeric(data.get("o", []), errors="coerce") * 1000.0,
        "high": pd.to_numeric(data.get("h", []), errors="coerce") * 1000.0,
        "low": pd.to_numeric(data.get("l", []), errors="coerce") * 1000.0,
        "close": pd.to_numeric(closes, errors="coerce") * 1000.0,
        "volume": pd.to_numeric(data.get("v", []), errors="coerce"),
    }).dropna()
    return df if len(df) >= 205 else None


def fetch_ohlcv_tcbs(symbol, from_sec, to_sec):
    """Nguồn giá 2: TCBS Public API."""
    url = (
        f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
        f"?ticker={symbol}&type=stock&resolution=D&from={from_sec}&to={to_sec}"
    )
    headers = {**HEADERS, "Referer": "https://tcinvest.tcbs.com.vn/"}
    res = requests.get(url, headers=headers, timeout=8)
    if res.status_code != 200:
        return None
    bars = res.json().get("data") or []
    if len(bars) < 205:
        return None

    df = pd.DataFrame(bars)[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(df) < 205:
        return None

    # Chuẩn hóa nếu giá đang ở đơn vị nghìn VNĐ
    if float(df["close"].median()) < 500:
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * 1000.0
    return df


def fetch_ohlcv_yahoo(symbol):
    """Nguồn giá 3: Yahoo Finance (không bao giờ chặn IP GitHub Actions)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.VN?range=18mo&interval=1d"
    res = requests.get(url, headers=HEADERS, timeout=8)
    if res.status_code != 200:
        return None
    chart = res.json().get("chart", {})
    results = chart.get("result") or []
    if not results:
        return None
    quotes = results[0].get("indicators", {}).get("quote", [{}])[0]
    if not quotes or not quotes.get("close"):
        return None

    df = pd.DataFrame({
        "open": pd.to_numeric(quotes.get("open", []), errors="coerce"),
        "high": pd.to_numeric(quotes.get("high", []), errors="coerce"),
        "low": pd.to_numeric(quotes.get("low", []), errors="coerce"),
        "close": pd.to_numeric(quotes.get("close", []), errors="coerce"),
        "volume": pd.to_numeric(quotes.get("volume", []), errors="coerce"),
    }).dropna()
    return df if len(df) >= 205 else None


def fetch_ohlcv_with_fallback(symbol, from_sec, to_sec):
    """Thử lần lượt DNSE -> TCBS -> Yahoo Finance cho tới khi lấy được dữ liệu."""
    for source_name, fn in [
        ("DNSE", lambda: fetch_ohlcv_dnse(symbol, from_sec, to_sec)),
        ("TCBS", lambda: fetch_ohlcv_tcbs(symbol, from_sec, to_sec)),
        ("Yahoo", lambda: fetch_ohlcv_yahoo(symbol)),
    ]:
        try:
            df = fn()
            if df is not None and len(df) >= 205:
                return df, source_name
        except Exception:
            continue
    return None, None


def fetch_fundamental_data(symbol):
    """Lấy chỉ số tài chính cơ bản từ TCBS (trả về dict rỗng nếu không lấy được)."""
    try:
        url = (
            f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/"
            f"{symbol}/financialratio?yearly=0&isAll=false"
        )
        headers = {**HEADERS, "Referer": "https://tcinvest.tcbs.com.vn/"}
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            payload = res.json()
            if isinstance(payload, list) and len(payload) > 0:
                return payload[0], True
    except Exception:
        pass
    return {}, False


def calculate_indicators(df):
    """Tính toán các đường MA, ATR, độ siết nền giá và điểm RS thô."""
    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    vol = float(df["volume"].iloc[-1])

    if close <= 0:
        return None

    # Thanh khoản trung bình 20 phiên (Tỷ VNĐ)
    avg_vol_20 = float(df["volume"].tail(20).mean())
    avg_val_20 = (avg_vol_20 * close) / 1e9

    if avg_val_20 < CONFIG["MIN_AVG_VAL_20_BASE"]:
        return None

    # Các đường trung bình động
    ma20 = float(df["close"].tail(20).mean())
    ma50 = float(df["close"].tail(50).mean())
    ma200 = float(df["close"].tail(200).mean())
    ma200_10d_ago = (
        float(df["close"].iloc[-210:-10].mean())
        if len(df) >= 210
        else float(df["close"].head(200).mean())
    )

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
    from_sec = now_sec - (420 * 24 * 60 * 60)

    df, price_source = fetch_ohlcv_with_fallback(symbol, from_sec, now_sec)
    if df is None:
        return None

    tech = calculate_indicators(df)
    if tech is None:
        return None

    fa, fa_ok = fetch_fundamental_data(symbol)

    return {
        "symbol": symbol,
        "floor": floor,
        "price_source": price_source,
        "fa_ok": fa_ok,
        **tech,
        "roe": round((fa.get("roe") or 0) * 100, 1),
        "rev_growth": round((fa.get("revenueGrowth") or 0) * 100, 1),
        "profit_growth": round((fa.get("postTaxProfitGrowth") or 0) * 100, 1),
        "de": round(fa.get("debtOnEquity") or 0, 2),
        "pe": round(fa.get("pe") or 0, 1),
        "pb": round(fa.get("pb") or 0, 1),
    }


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


def export_markdown_report(df_all, swing_df, invest_df, timestamp_str, diag_info):
    """Xuất bảng kết quả ra file KET_QUA_LOC.md để xem trực tiếp trên GitHub."""
    lines = [
        f"# Báo Cáo Quét Cổ Phiếu Tự Động ({timestamp_str} GMT+7)\n",
        f"- Tổng số mã đủ điều kiện thanh khoản phân tích: **{len(df_all)} mã**",
        f"- Số mã đạt chuẩn **Swing Trading**: **{len(swing_df)} mã**",
        f"- Số mã đạt chuẩn **Đầu tư cơ bản**: **{len(invest_df)} mã**\n",
        "---\n",
        "## 1. Danh Sách Swing Trading (Xu hướng tăng + RS >= 70)\n",
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

    lines.extend([
        "\n---\n",
        "### Nhật ký nguồn dữ liệu (Chẩn đoán kỹ thuật)",
        f"- **Nguồn danh sách mã:** {diag_info['list_source']}",
        f"- **Nguồn dữ liệu giá thành công:** {diag_info['price_sources']}",
        f"- **Số mã lấy được chỉ số cơ bản (FA):** {diag_info['fa_count']} mã",
    ])

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

    stocks, list_source = get_all_stocks()
    results = []

    if stocks:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_stock_data, s) for s in stocks]
            for future in as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

    print(f"Số mã đạt chuẩn thanh khoản phân tích: {len(results)}")

    if not results:
        empty_df = pd.DataFrame()
        diag_info = {
            "list_source": list_source,
            "price_sources": "Không lấy được dữ liệu giá từ cả 3 nguồn",
            "fa_count": 0,
        }
        export_markdown_report(empty_df, empty_df, empty_df, now_str, diag_info)
        return

    df_all = pd.DataFrame(results)
    df_all["rs_rating"] = (df_all["raw_rs"].rank(pct=True) * 98 + 1).round(0).astype(int)

    # Thống kê nguồn dữ liệu cho phần chẩn đoán
    source_counts = df_all["price_source"].value_counts().to_dict()
    price_sources_str = ", ".join(f"{k}: {v} mã" for k, v in source_counts.items())
    diag_info = {
        "list_source": list_source,
        "price_sources": price_sources_str,
        "fa_count": int(df_all["fa_ok"].sum()),
    }

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

    export_markdown_report(df_all, swing_df, invest_df, now_str, diag_info)
    send_telegram_alert(swing_df, invest_df, now_str)


if __name__ == "__main__":
    main()
