from fastapi import FastAPI, HTTPException
import yfinance as yf
import requests
from fastapi.middleware.cors import CORSMiddleware
#from routes import alphavantage_api
from scraping import get_symbol
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse, FileResponse
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mangum import Mangum
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")
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
handler = Mangum(app)
class StockRequest(BaseModel):
    ticker: str
    period: str = '1y'
    steps: int = 30

class ForecastResponse(BaseModel):
    forecast: List[float]


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

@app.post("/stock/correlation")
def get_stock_data(tickers:list[str],period:str):
    '''
    tickerシンボルを受け取って、相関分析を返すAPI

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
    correlation_kihon = data.corr()

    data['Daily_Return'] = data['Close'].pct_change()

    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    data['Volatility'] = data['Daily_Return'].rolling(window=50).std()
    #　相関行列の計算
    correlation_matrix = data.corr()
    print(correlation_matrix)

    plt.figure(figsize=(12,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5)
    plt.title('Correlation Matrix')
    # プロット画像の保存
    now = datetime.now()
    plot_filename = f'correlation_matrix{now}.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return {'自己相関':correlation_kihon,'相関行列':correlation_matrix}

@app.post("/stock/ARIMA")
def get_stock_data(tickers:list[str],period:str):
    '''
    tickerシンボルを受け取って、自己回帰移動平均モデルの予測を返すAPI

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
    data = data[0][['Close']]

    # 欠損値の処理
    data = data.dropna()

    # ARIMAモデルの定義と適用
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()

    # 予測の実施
    forecast = model_fit.forecast(steps=30)

    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Observed')
    plt.plot(forecast, label='Forecast')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # プロット画像の保存
    now = datetime.now()
    plot_filename = f'forecast{now}.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return {'モデルサマリー': model_fit.summary()}

@app.post("/predict_arima", response_model=ForecastResponse)
def predict_stock(request: StockRequest):
    try:
        # データ取得
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()
        
        # ARIMAモデルの定義と適用
        model = ARIMA(data, order=(5, 1, 0))  # p=5, d=1, q=0のモデルを使用
        model_fit = model.fit()
        
        # 予測の実施
        forecast = model_fit.forecast(steps=request.steps)

        # 予測結果のタイムインデックスを設定
        forecast_index = pd.date_range(start=data.index[-1], periods=request.steps + 1, closed='right')
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # 予測結果をリストに変換
        forecast_list = forecast.tolist()

         # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Observed')
        plt.plot(forecast.index, forecast_series, label='Forecast', color='red')
        plt.title(f'{request.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        now = datetime.now()
        plot_filename = f'predict_arima{now}.png'
        plt.savefig(plot_filename)
        plt.close()
        
        return FileResponse(plot_filename)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fastapi.responses import FileResponse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ForecastResponse(BaseModel):
    file_path: str


@app.post("/predict_lstm", response_model=ForecastResponse)
def predict_stock(request: StockRequest):
    try:
        # データ取得
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()

        # データのスケーリング
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # データセットの準備
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)

        # トレーニングデータとテストデータの分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = Y[:train_size], Y[train_size:]

        # データの形状変更
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTMモデルの構築
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # モデルのコンパイル
        model.compile(optimizer='adam', loss='mean_squared_error')

        # モデルの訓練
        model.fit(X_train, y_train, batch_size=1, epochs=1)

        # 予測の実施
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # データの逆スケーリング
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        actual_data = scaler.inverse_transform(scaled_data)

        # 評価指標の計算
        train_mae = mean_absolute_error(y_train, train_predict)
        train_mse = mean_squared_error(y_train, train_predict)
        test_mae = mean_absolute_error(y_test, test_predict)
        test_mse = mean_squared_error(y_test, test_predict)

        # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, actual_data, label='Observed')
        plt.plot(data.index[time_step:time_step+len(train_predict)], train_predict, label='Train Predict')
        plt.plot(data.index[time_step+len(train_predict)+1:time_step+len(train_predict)+1+len(test_predict)], test_predict, label='Test Predict')

        # 未来予測のプロット
        future_steps = request.steps
        future_predictions = []
        last_sequence = X[-1]

        for _ in range(future_steps):
            next_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
            future_predictions.append(next_prediction[0][0])
            last_sequence = np.append(last_sequence[1:], next_prediction[0])
            last_sequence = last_sequence.reshape(time_step, 1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='B')[1:]
        plt.plot(future_dates, future_predictions, label='Future Predict', color='red')

        plt.title(f'{request.ticker} Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        plot_file = 'static/prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # ファイルを返す
        return FileResponse(plot_file)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ForecastMeanResponse(BaseModel):
    file_path: str
    train_mae: float
    train_mse: float
    test_mae: float
    test_mse: float

@app.post("/predict_lstm_mean", response_model=ForecastMeanResponse)
def predict_stock(request: StockRequest):
    try:
        # データ取得
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()

        # データのスケーリング
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # データセットの準備
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)

        # トレーニングデータとテストデータの分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = Y[:train_size], Y[train_size:]

        # データの形状変更
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTMモデルの構築
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # モデルのコンパイル
        model.compile(optimizer='adam', loss='mean_squared_error')

        # モデルの訓練
        model.fit(X_train, y_train, batch_size=1, epochs=1)

        # 予測の実施
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # データの逆スケーリング
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        actual_data = scaler.inverse_transform(scaled_data)

        # 評価指標の計算
        train_mae = mean_absolute_error(y_train, train_predict)
        train_mse = mean_squared_error(y_train, train_predict)
        test_mae = mean_absolute_error(y_test, test_predict)
        test_mse = mean_squared_error(y_test, test_predict)

        # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, actual_data, label='Observed')
        plt.plot(data.index[time_step:time_step+len(train_predict)], train_predict, label='Train Predict')
        plt.plot(data.index[time_step+len(train_predict)+1:time_step+len(train_predict)+1+len(test_predict)], test_predict, label='Test Predict')

        # 未来予測のプロット
        future_steps = request.steps
        future_predictions = []
        last_sequence = X[-1]

        for _ in range(future_steps):
            next_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
            future_predictions.append(next_prediction[0][0])
            last_sequence = np.append(last_sequence[1:], next_prediction[0])
            last_sequence = last_sequence.reshape(time_step, 1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='B')[1:]
        plt.plot(future_dates, future_predictions, label='Future Predict', color='red')

        plt.title(f'{request.ticker} Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        plot_file = 'static/prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # ファイルを返す
        return ForecastMeanResponse(file_path=plot_file, train_mae=train_mae, train_mse=train_mse, test_mae=test_mae, test_mse=test_mse)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict_plot")
def predict_stock(request: StockRequest):
    try:
        # データ取得
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()
        
        # ARIMAモデルの定義と適用
        model = ARIMA(data, order=(5, 1, 0))  # p=5, d=1, q=0のモデルを使用
        model_fit = model.fit()
        
        # 予測の実施
        forecast = model_fit.get_forecast(steps=request.steps)
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=request.steps, freq='B')
        forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
        
        # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Observed')
        plt.plot(forecast_series.index, forecast_series, label='Forecast', color='red')
        plt.title(f'{request.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        plot_file = 'static/prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # ファイルを返す
        return FileResponse(plot_file)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_random_forest")
def predict_stock(request: StockRequest):
    try:
        # データ取得
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()

        # データのスケーリング
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # データセットの準備
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)

        # トレーニングデータとテストデータの分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = Y[:train_size], Y[train_size:]

        # ランダムフォレストモデルの構築
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 予測の実施
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # 未来の値の予測
        future_steps = request.steps
        future_predictions = []
        last_sequence = X[-1]

        for _ in range(future_steps):
            next_prediction = model.predict(last_sequence.reshape(1, -1))
            future_predictions.append(next_prediction[0])
            last_sequence = np.append(last_sequence[1:], next_prediction)

        # データの逆スケーリング
        train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
        test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        actual_data = scaler.inverse_transform(scaled_data)

        # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, actual_data, label='Observed')
        plt.plot(data.index[time_step:time_step+len(train_predict)], train_predict, label='Train Predict')
        plt.plot(data.index[time_step+len(train_predict)+1:time_step+len(train_predict)+1+len(test_predict)], test_predict, label='Test Predict')

        # 未来予測のプロット
        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1)[1:]
        plt.plot(future_dates, future_predictions, label='Future Predict', color='red')

        plt.title(f'{request.ticker} Stock Price Prediction using RandomForest')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        plot_file = 'static/prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # ファイルを返す
        return FileResponse(plot_file)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ForecastAdjustResponse(BaseModel):
    file_path: str
    train_mae: float
    train_mse: float
    train_r2: float
    test_mae: float
    test_mse: float
    test_r2: float

@app.post("/predict_adjust", response_model=ForecastAdjustResponse)
def predict_stock(request: StockRequest):
    try:
        # データ取得
        print("Fetching stock data...")
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        if data.empty:
            print("No data found for ticker:", request.ticker)
            raise HTTPException(status_code=404, detail="No data found for ticker")

        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()

        # データのスケーリング
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # データセットの準備
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)

        # トレーニングデータとテストデータの分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = Y[:train_size], Y[train_size:]

        # データの形状変更
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTMモデルの構築
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        # モデルのコンパイル
        model.compile(optimizer='adam', loss='mean_squared_error')

        # モデルの訓練
        print("Training model...")
        model.fit(X_train, y_train, batch_size=16, epochs=1, validation_split=0.2)

        # 予測の実施
        print("Predicting...")
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # データの逆スケーリング
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        actual_data = scaler.inverse_transform(scaled_data)

        # 評価指標の計算
        train_mae = mean_absolute_error(y_train, train_predict)
        train_mse = mean_squared_error(y_train, train_predict)
        train_r2 = r2_score(y_train, train_predict)
        test_mae = mean_absolute_error(y_test, test_predict)
        test_mse = mean_squared_error(y_test, test_predict)
        test_r2 = r2_score(y_test, test_predict)

        print(f"train_mae: {train_mae}, train_mse: {train_mse}, train_r2: {train_r2}")
        print(f"test_mae: {test_mae}, test_mse: {test_mse}, test_r2: {test_r2}")

        # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, actual_data, label='Observed')
        plt.plot(data.index[time_step:time_step+len(train_predict)], train_predict, label='Train Predict')
        plt.plot(data.index[time_step+len(train_predict)+1:time_step+len(train_predict)+1+len(test_predict)], test_predict, label='Test Predict')

        # 未来予測のプロット
        future_steps = request.steps
        future_predictions = []
        last_sequence = X[-1]

        for _ in range(future_steps):
            next_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
            future_predictions.append(next_prediction[0][0])
            last_sequence = np.append(last_sequence[1:], next_prediction[0])
            last_sequence = last_sequence.reshape(time_step, 1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='B')[1:]
        plt.plot(future_dates, future_predictions, label='Future Predict', color='red')

        plt.title(f'{request.ticker} Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        plot_file = 'static/prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # ファイルを返す
        return ForecastAdjustResponse(
            file_path=plot_file, 
            train_mae=train_mae, 
            train_mse=train_mse, 
            train_r2=train_r2, 
            test_mae=test_mae, 
            test_mse=test_mse, 
            test_r2=test_r2
        )
    
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/predict_random_forest_adjust", response_model=ForecastAdjustResponse)
def predict_stock(request: StockRequest):
    try:
        # データ取得
        stock = yf.Ticker(request.ticker)
        data = stock.history(period=request.period)
        
        # 必要なカラムを選択
        data = data[['Close']]
        
        # 欠損値の処理
        data = data.dropna()

        # データのスケーリング
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # データセットの準備
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)

        # トレーニングデータとテストデータの分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = Y[:train_size], Y[train_size:]

        # ランダムフォレストモデルの構築
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 予測の実施
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # データの逆スケーリング
        train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
        test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        actual_data = scaler.inverse_transform(scaled_data)

        # 評価指標の計算
        train_mae = mean_absolute_error(y_train, train_predict)
        train_mse = mean_squared_error(y_train, train_predict)
        train_r2 = r2_score(y_train, train_predict)
        test_mae = mean_absolute_error(y_test, test_predict)
        test_mse = mean_squared_error(y_test, test_predict)
        test_r2 = r2_score(y_test, test_predict)

        # プロットの作成
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, actual_data, label='Observed')
        plt.plot(data.index[time_step:time_step+len(train_predict)], train_predict, label='Train Predict')
        plt.plot(data.index[time_step+len(train_predict)+1:time_step+len(train_predict)+1+len(test_predict)], test_predict, label='Test Predict')

        # 未来予測のプロット
        future_steps = request.steps
        future_predictions = []
        last_sequence = X[-1]

        for _ in range(future_steps):
            next_prediction = model.predict(last_sequence.reshape(1, -1))
            future_predictions.append(next_prediction[0])
            last_sequence = np.append(last_sequence[1:], next_prediction)
            last_sequence = last_sequence.reshape(time_step, 1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='B')[1:]
        plt.plot(future_dates, future_predictions, label='Future Predict', color='red')

        plt.title(f'{request.ticker} Stock Price Prediction using RandomForest')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # 画像の保存
        plot_file = 'static/prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # ファイルを返す
        return ForecastAdjustResponse(
            file_path=plot_file, 
            train_mae=train_mae, 
            train_mse=train_mse, 
            train_r2=train_r2, 
            test_mae=test_mae, 
            test_mse=test_mse, 
            test_r2=test_r2
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))