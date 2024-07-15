from fastapi import FastAPI, HTTPException
import yfinance as yf
import requests
from fastapi.middleware.cors import CORSMiddleware
#from routes import alphavantage_api
from scraping import get_symbol

app = FastAPI()
#app.include_router(alphavantage_api.router)
app.include_router(get_symbol.router)

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    # 他に許可したいオリジンがあれば、ここに追加します
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/stock/")
def get_stock_data(tickers:list[str],period:str):
    '''
    tickerシンボルのリストを受け取って、１か月分の日時データを返すためのAPI

    tickers: 値を取りたいシンボルのリスト
    （sbiの株の番号+.Tで値を取得することもできる。）

    perod: 1日（1d）、5日（5d）、1ヶ月（1mo）、3ヶ月（3mo）、6ヶ月（6mo）、1年（1y）、2年（2y）、5年（5y）、10年（10y）、年初来（ytd）、最大（max）
    '''
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        data.append(hist.to_dict(orient="index"))
    return data
