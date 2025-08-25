#Market Data analysis

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#Tickers
TICKERS= ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD"]
START = "2015-01-01"
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "outputs"
DATADIR = BASE_DIR / "data"
OUTDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)

print(f"Tickers: {TICKERS}")
print(f"start date:{START}")

#download Prices
print("Downloading Price data")
prices = yf.download(TICKERS, start=START, auto_adjust=True, progress=True)["Close"]
print("Download Complete. Preview of Data")
print(prices.tail())

#saving prices
prices.to_csv(DATADIR / "close_prices.csv")
print(f"Saved prices to {DATADIR}/ close_prices.csv")

#calc the returns
print("Calculating Daily returns....")
returns = prices.pct_change().dropna()
returns.to_csv(DATADIR / "daily_returns.csv")
print(f"Saved daily returns to {DATADIR}/daily_returns.csv")

#annualized metrics 
print("calculating annualized metrics")
annual_factor = 252 #Trading Days
annual_return = returns.mean() * annual_factor
annual_volatility = returns.std() * np.sqrt(annual_factor)
sharpe_ratio = annual_return / annual_volatility

metrics = pd.DataFrame({
"Annual Return": annual_return,
"Annual Volatility": annual_volatility,
"Sharpe Ratio": sharpe_ratio,
}).sort_values("Sharpe Ratio", ascending =False)

print("\n== Annualized Metrics (sorted by Sharpe Ratio) ===")
print(metrics)

metrics.to_csv(OUTDIR / "asset_metrics.csv")
print(f"Saved metrics to {OUTDIR} / asset_metrics.csv")
#Equal Weighted Portfolio
print("Building equal-weight portfolio...")
weights = np.repeat(1/len(TICKERS), len(TICKERS))
portfolio_returns = returns @ weights
portfolio_cumulative = (1 + portfolio_returns).cumprod()

print('generating plots')

# Cumulative growth 
plt.figure()
portfolio_cumulative.plot(title="Equal Weight Portfolio: Cumulative Growth")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.tight_layout()
plt.savefig(OUTDIR / "portfolio_cum_growth.png", dpi=160) 
plt.show()
plt.close()
# Annualized ret & vol
plt.figure()
metrics[["Annual Return", "Annual Volatility"]].plot(kind="bar", title="Annualized Return & Volatility")
plt.tight_layout()
plt.savefig(OUTDIR / "ann_return_vol.png", dpi=160)
plt.show()
plt.close()
print(f"Saved plot: {OUTDIR}/ann_return_vol.png")
