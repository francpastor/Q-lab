#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estrategia FUSIÓN RSI + Bollinger Bands
- Entra en contra-tendencia (mean-reversion).
- Cuando está flat (señal 0), el capital se asigna a Oro (GC=F).
- Dos modos de entrada: 
    * 'instant'  -> compra/venta inmediatamente al ver sobreventa + fuera de banda
    * 'confirm'  -> espera confirmación (salida de sobreventa y cierre de vuelta dentro de bandas)
Requiere: pandas, numpy, yfinance, matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =======================
# Parámetros editables
# =======================
TICKER = "SPY"       # Activo a operar
CASH_TICKER = "GC=F" # Oro como "cash" cuando estamos flat
PERIOD = "max"

BB_LEN = 8
BB_STD = 2.0

RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

ENTRY_MODE = "instant"   # "instant" o "confirm"
EXIT_RULE = "midband"    # "midband" (salir en la media) o "neutral" (salir cuando no hay condición)

# =======================
# Indicadores
# =======================
def compute_bollinger(close: pd.Series, length=20, nstd=2.0) -> pd.DataFrame:
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return pd.DataFrame({
        "MA": ma,
        "BB_upper": ma + nstd * sd,
        "BB_lower": ma - nstd * sd
    })

def rsi_simple(close: pd.Series, length=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =======================
# Señales
# =======================
def fused_signal(close: pd.Series,
                 bb_len=BB_LEN, bb_std=BB_STD,
                 rsi_len=RSI_LENGTH,
                 overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD,
                   entry_mode=ENTRY_MODE, exit_rule=EXIT_RULE) -> pd.Series:
    bb = compute_bollinger(close, bb_len, bb_std)
    rsi = rsi_simple(close, rsi_len)

    long_raw  = (close < bb["BB_lower"]) & (rsi < oversold)
    short_raw = (close > bb["BB_upper"]) & (rsi > overbought)

    sig = pd.Series(0, index=close.index, dtype=float)

    if entry_mode == "instant":
        # Entra en el mismo día de la condición (pero ejecutaremos al día siguiente)
        sig[long_raw] = 1.0
        sig[short_raw] = -1.0

    elif entry_mode == "confirm":
        # Espera confirmación: salida de sobreventa/sobrecompra y vuelta dentro de bandas
        # Largo si AYER estaba fuera por abajo y en sobreventa,
        # y HOY cierra de vuelta dentro (close >= lower) y RSI cruza arriba de oversold
        cond_y_long = long_raw.shift(1).fillna(False)
        cond_hoy_long = (close >= bb["BB_lower"]) & (rsi.shift(1) < oversold) & (rsi >= oversold)
        sig[cond_y_long & cond_hoy_long] = 1.0

        # Corto simétrico
        cond_y_short = short_raw.shift(1).fillna(False)
        cond_hoy_short = (close <= bb["BB_upper"]) & (rsi.shift(1) > overbought) & (rsi <= overbought)
        sig[cond_y_short & cond_hoy_short] = -1.0

    # Regla de salida (opcional):
    # - "midband": cuando largo, salimos si cierra por encima de la media; cuando corto, salimos si cierra por debajo de la media.
    # - "neutral": si ya no se cumple la condición de entrada (para instant) o si perdemos la confirmación (para confirm), volvemos a 0.
    if exit_rule == "midband":
        in_long = sig.replace(0, np.nan).ffill() == 1.0
        in_short = sig.replace(0, np.nan).ffill() == -1.0
        exit_long = (close >= bb["MA"]) & in_long
        exit_short = (close <= bb["MA"]) & in_short
        # Volvemos a 0 en esos puntos
        sig[exit_long | exit_short] = 0.0

    # Ejecutamos al día siguiente para evitar look-ahead
    return sig.shift(1).fillna(0.0)

# =======================
# Métricas
# =======================
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def sharpe_ratio(returns: pd.Series, periods=252) -> float:
    mu = returns.mean() * periods
    sigma = returns.std(ddof=0) * np.sqrt(periods)
    return np.nan if sigma == 0 else mu / sigma

# =======================
# Backtest
# =======================
def backtest_with_gold(data: pd.DataFrame, ticker: str, cash_ticker: str) -> pd.DataFrame:
    df = data[[ticker, cash_ticker]].dropna().copy()
    close = df[ticker]

    sig = fused_signal(close)
    r_asset = close.pct_change().fillna(0.0)
    r_gold  = df[cash_ticker].pct_change().fillna(0.0)

    # Retorno de la estrategia:
    # si pos=0 -> retorno del oro; si pos=±1 -> ±retorno del activo
    pos = sig
    r_strat = pos * r_asset + (1 - pos.abs()) * r_gold

    eq_asset = (1 + r_asset).cumprod()
    eq_strat = (1 + r_strat).cumprod()

    out = pd.DataFrame({
        "Position": pos,
        "Ret_Asset": r_asset,
        "Ret_Gold": r_gold,
        "Ret_Strategy": r_strat,
        "Eq_Asset": eq_asset,
        "Eq_Strategy": eq_strat
    }, index=df.index)
    return out
# --- Filtrado temporal y costes ---
   # end exclusivo: cubre todo 2024
USE_COSTS  = True           # pon False si quieres ver resultados brutos
 
# Costes (ajusta a tu broker/ETF)
FEE_SPY          = 0.0002   # 2 bps por cambiar exposición en SPY
FEE_GOLD         = 0.0002   # 2 bps por cambiar exposición en Oro
SLIPPAGE_SPY     = 0.0001   # 1 bp de slippage SPY
SLIPPAGE_GOLD    = 0.0001   # 1 bp de slippage Oro
BORROW_RATE_SHORT= 0.03     # 3% anual por mantener cortos (SPY)
GOLD_MGMT_FEE    = 0.004    # 0.40% anual (si usas ETF tipo GLD; con GC=F puedes poner 0)
 
def main():
    # --- Descarga solo 2024 (diario) ---
    data = yf.download([TICKER, CASH_TICKER],start="2024-01-01",
                 end="2024-12-01")['Close'].ffill()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # --- Backtest bruto ---
    res = backtest_with_gold(data, TICKER, CASH_TICKER)

    # --- Costes (se calculan DESPUÉS, usando columnas de res) ---
    if USE_COSTS:
        pos     = res["Position"]
        r_strat = res["Ret_Strategy"]

        # Turnover de SPY (cambios de posición)
        dpos = pos.diff().fillna(pos.iloc[0])
        turnover_spy  = dpos.abs()

        # Turnover de Oro = cambio del peso en oro (1-|pos|)
        turnover_gold = pos.abs().diff().abs().fillna(pos.abs().iloc[0])

        # Costes por trade (comisión + slippage)
        trade_costs = turnover_spy  * (FEE_SPY  + SLIPPAGE_SPY) \
                    + turnover_gold * (FEE_GOLD + SLIPPAGE_GOLD)

        # Costes de mantenimiento diarios
        borrow_daily   = (BORROW_RATE_SHORT/252.0) * (pos < 0).astype(float) * pos.abs()
        gold_fee_daily = (GOLD_MGMT_FEE/252.0)     * (1.0 - pos.abs())

        # Retornos y equity netos
        r_strat_net = r_strat - trade_costs - borrow_daily - gold_fee_daily
        eq_strat_net = (1.0 + r_strat_net).cumprod()

        res["Ret_Strategy_Net"] = r_strat_net
        res["Eq_Strategy_Net"]  = eq_strat_net
    else:
        res["Ret_Strategy_Net"] = res["Ret_Strategy"]
        res["Eq_Strategy_Net"]  = res["Eq_Strategy"]

    # --- Métricas y gráfico (bruto vs neto) ---
    def max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        return float((equity/peak - 1.0).min())

    def sharpe_ratio(returns: pd.Series, periods=252) -> float:
        mu = returns.mean()*periods
        sd = returns.std(ddof=0)*np.sqrt(periods)
        return float("nan") if sd == 0 else mu/sd

    def cagr(eq: pd.Series, periods=252) -> float:
        n = len(eq)
        return float("nan") if n == 0 else float(eq.iloc[-1]**(periods/n) - 1)

    eq_a = res["Eq_Asset"]
    r_a  = res["Ret_Asset"]
    eq_s_g = res["Eq_Strategy"]
    r_s_g  = res["Ret_Strategy"]
    eq_s_n = res["Eq_Strategy_Net"]
    r_s_n  = res["Ret_Strategy_Net"]

    print(f"Modo entrada: {ENTRY_MODE} | Salida: {EXIT_RULE}")
    print(f"Estrategia: {TICKER} (flat en {CASH_TICKER})  BB={BB_LEN}x{BB_STD}  RSI={RSI_LENGTH} ({RSI_OVERSOLD}/{RSI_OVERBOUGHT})")
    print("---- 2024 ----")
    print(f"Buy&Hold   | CAGR: {cagr(eq_a):.2%} | Sharpe: {sharpe_ratio(r_a):.2f} | MaxDD: {max_drawdown(eq_a):.2%} | Equity x{eq_a.iloc[-1]:.2f}")
    print(f"Estrate BR | CAGR: {cagr(eq_s_g):.2%} | Sharpe: {sharpe_ratio(r_s_g):.2f} | MaxDD: {max_drawdown(eq_s_g):.2%} | Equity x{eq_s_g.iloc[-1]:.2f}")
    print(f"Estrate NET| CAGR: {cagr(eq_s_n):.2%} | Sharpe: {sharpe_ratio(r_s_n):.2f} | MaxDD: {max_drawdown(eq_s_n):.2%} | Equity x{eq_s_n.iloc[-1]:.2f}")

    # Curva de capital (neto vs buy&hold)
    plt.figure(figsize=(10,5))
    plt.plot(eq_a.index, eq_a, label=f"Buy&Hold {TICKER}")
    plt.plot(eq_s_n.index, eq_s_n, label="RSI+BB NET (flat→Gold)")
    plt.title("Curva de capital 2024")
    plt.xlabel("Fecha"); plt.ylabel("Equity (inicio=1)")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
