from fastapi import FastAPI, HTTPException
import yfinance as yf
import requests
from fastapi.middleware.cors import CORSMiddleware
#from routes import alphavantage_api
from scraping import get_symbol
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse, FileResponse

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

@app.post("/stock/vision")
def get_stock_data(tickers:list[str],period:str):
    '''
    tickerシンボルのリストを受け取って、データを可視化するAPI

    tickers: 値を取りたいシンボルのリスト
    （sbiの株の番号+.Tで値を取得することもできる。）

    perod: 1日（1d）、5日（5d）、1ヶ月（1mo）、3ヶ月（3mo）、6ヶ月（6mo）、1年（1y）、2年（2y）、5年（5y）、10年（10y）、年初来（ytd）、最大（max）
    '''
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        data.append(hist)
    # 必要なカラムだけを選択
    if isinstance(data[0], pd.DataFrame):
        data = data[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        raise TypeError("The data is not in the expected format. Please check the data source.")

    print(data)
    # 欠損値の処理
    data.fillna(method='ffill', inplace=True)

    # 移動平均の計算
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # 日次リターンの計算
    data['Daily_Return'] = data['Close'].pct_change()

    # ボラティリティの計算
    data['Volatility'] = data['Daily_Return'].rolling(window=50).std()

    # 取得したデータの最初の5行を表示
    print(data.head())

    # プロットを作成
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['SMA_50'], label='50-Day SMA')
    plt.plot(data['SMA_200'], label='200-Day SMA')
    plt.title(f'{tickers[0]} Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # プロット画像の保存
    plot_filename = 'stock_plot.png'
    plt.savefig(plot_filename)
    plt.close()

    # HTMLで画像を表示
    html_content = f"""
    <html>
        <head>
            <title>Stock Plot</title>
        </head>
        <body>
            <h1>{tickers[0]} Stock Price and Moving Averages</h1>
            <img src="/plot/{plot_filename}" alt="Stock Plot">
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/stock/statics")
def get_stock_data(tickers:list[str],period:str):
    '''
    tickerシンボルを受け取って、基本的な統計量を返すAPI

    tickers: 値を取りたいシンボルのリスト
    （sbiの株の番号+.Tで値を取得することもできる。）

    perod: 1日（1d）、5日（5d）、1ヶ月（1mo）、3ヶ月（3mo）、6ヶ月（6mo）、1年（1y）、2年（2y）、5年（5y）、10年（10y）、年初来（ytd）、最大（max）
    '''
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        data.append(hist)
    # 必要なカラムを選択
    data = data[0][['Open', 'High', 'Low', 'Close', 'Volume']]

    # 欠損値の処理
    data.fillna(method='ffill', inplace=True)

    # 基本的な統計量の計算
    statistics = data.describe()
    print(statistics)
    
    return statistics