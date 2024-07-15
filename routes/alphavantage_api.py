from fastapi import APIRouter, Depends
import requests
API_KEY = "667KDUBSCVNV0GF6"  # 新しいAPIキーをここに設定
router = APIRouter(
    prefix="/alphavantage",
    responses={404: {"discription": "Not Found"}},
)

@router.get("/stock/{symbol}")
def get_stock_price(symbol: str):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Alpha Vantage")
    data = response.json()
    
    if "Error Message" in data:
        raise HTTPException(status_code=404, detail="Stock symbol not found")
    
    try:
        last_refreshed = data["Meta Data"]["3. Last Refreshed"]
        close_price = data["Time Series (Daily)"][last_refreshed]["4. close"]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Unexpected response format from API: {str(e)}")
    
    return {"symbol": symbol, "last_refreshed": last_refreshed, "close_price": close_price, "data": data["Time Series (Daily)"]}

@router.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




"""
from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
API_KEY = "667KDUBSCVNV0GF6"

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



@app.get("/stock/{symbol}")
def get_stock_price(symbol: str):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "Error Message" in data:
        raise HTTPException(status_code=404, detail="Stock symbol not found")
    
    try:
        last_refreshed = data["Meta Data"]["3. Last Refreshed"]
        close_price = data["Time Series (Daily)"][last_refreshed]["4. close"]
    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response format from API")
    data_dairy = data["Time Series (Daily)"]
    return {"symbol": symbol, "last_refreshed": last_refreshed, "close_price": close_price, "data": data_dairy}

@app.get("/")
def get_stock(symbol: str):
    return {"Hello": "Hello"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""