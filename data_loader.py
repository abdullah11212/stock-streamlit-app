import yfinance as yf

def load_data(ticker, end_date="2025-12-18"):
    data = yf.download(ticker, period="max", end=end_date)

    if 'Adj Close' in data.columns:
        data['Price'] = data['Adj Close']
    else:
        data['Price'] = data['Close']

    return data
