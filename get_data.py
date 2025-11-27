import yfinance as yf
import pandas as pd
from time import sleep

# List of NSE tickers
tickers = [
    "BYND",
    "GWH",
    "TE",
    "VSTS",
    "LU",
    "LAR",
    "NRGV",
    "NAT",
    "SES",
    "CANG",
    "BAK",
    "VOC",
    "PSQH",
    "BHR",
    "AMPY",
    "BDN",
    "TOVX",
    "YYAI",
    "CJET",
    "SAVA"
]
tickers = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp.
    "AMZN",  # Amazon.com Inc.
    "GOOGL", # Alphabet Inc. (Class A)
    "NVDA",  # NVIDIA Corp.
    "META",  # Meta Platforms Inc. (formerly Facebook)
    "TSLA",  # Tesla Inc.
    "ADBE",  # Adobe Inc.
    "NFLX",  # Netflix Inc.
    "INTU"   # Intuit Inc.
]
# tickers = [  
#     "SMCI",   # Super Micro Computer — very high beta / volatility 
# # ::contentReference[oaicite:0]{index=0}
  
#     "PLTR",   # Palantir — speculative, tied to AI + gov contracts :contentReference[oaicite:1]{index=1}  
#     "NVDA",   # Nvidia — big moves, high beta :contentReference[oaicite:2]{index=2}  
#     "TSLA",   # Tesla — well-known for large price swings :contentReference[oaicite:3]{index=3}  
#     "MSTR",   # MicroStrategy — tied to bitcoin, very levered :contentReference[oaicite:4]{index=4}  
#     "ARM",    # Arm Holdings — very high beta on WallStreetZen :contentReference[oaicite:5]{index=5}  
#     "COIN",   # Coinbase — volatile because of crypto linkage :contentReference[oaicite:6]{index=6}  
#     "HOOD",   # Robinhood — mentioned for high beta / speculative nature :contentReference[oaicite:7]{index=7}  
#     "VSH",    # Vishay Intertechnology — listed in volatile-stock list :contentReference[oaicite:8]{index=8}  
#     "POWI"    # Power Integrations — also in Investing.com’s volatile stocks list :contentReference[oaicite:9]{index=9}  
# ]  
# Add NSE suffix
nse_tickers = [ticker for ticker in tickers]

# Dictionary to hold each stock's data
all_data = {}

for ticker in nse_tickers:
    print(f"Downloading data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start='2015-01-01',end='2025-01-01', interval="3mo")
        data = data[['Close']].rename(columns={'Close': ticker})
        print(f"Got {len(data)} rows for {ticker}")
        all_data[ticker] = data
        sleep(0.5)
        print("Complete: ", ticker)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
combined_df = pd.concat(all_data.values(), axis=1)
combined_df.dropna(how='any', inplace=True)

combined_df.reset_index(inplace=True)
combined_df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)

# Save to CSV
combined_df.to_csv("nse_data.csv", index=False)

print("Data saved to nse_data.csv")
