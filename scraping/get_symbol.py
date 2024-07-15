import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter
from yahooquery import Screener
router = APIRouter(
    prefix="/get_symbol",
    responses={404: {"discription": "Not Found"}},
)
@router.get("/scraping/")
def get_symbol():
    '''
    yfinanceで使えるシンボルを得るためのAPI。
    '''
    # YahooファイナンスのURL
    url = 'https://finance.yahoo.com/screener/predefined/ms_technology'

    # ページを取得
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    print(soup)
    # ティッカーシンボルを含む要素を取得
    symbols = soup.find_all('a', {'class': 'Fw(600) C($linkColor)'})
    #print(symbols)
    # ティッカーシンボルをリストに格納
    ticker_symbols = [symbol.text for symbol in symbols]
    return ticker_symbols

@router.get("/yahooquery/")
def get_symbol_yahooquey():
    # テクノロジーセクターのスクリーナー設定
    s = Screener()
    data = s.get_screeners("advertising_agencies", count=100)
    print(data)
    # ティッカーシンボルのリストを取得
    tickers = [item['symbol'] for item in data["advertising_agencies"]['quotes']]
    return tickers

@router.get("/screener/")
def get_screener():
    # Screenerインスタンスを作成
    s = Screener()

    # 利用可能なスクリーナーを取得
    available_screeners = s.available_screeners
    return available_screeners