from fastapi import FastAPI, HTTPException
import yfinance as yf
import requests
from fastapi.middleware.cors import CORSMiddleware
#from routes import alphavantage_api

app = FastAPI()
#app.include_router(alphavantage_api.router)

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

@app.get("/stock/{ticker}")
async def get_stock_data(ticker: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    data = hist.to_dict(orient="index")
    return data
