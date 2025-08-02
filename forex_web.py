import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from scipy.stats import norm  # Para probabilidade

# CSS para melhorar o design (mais bonito e profissional)
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #1f1f1f;
    }
    h1 {
        color: #4a90e2;
        text-align: center;
    }
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .stSelectbox {
        border-radius: 5px;
    }
    .success {
        color: green;
        font-weight: bold;
    }
    .error {
        color: red;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Configurações
API_KEY = 'SUA_CHAVE_ALPHA_VANTAGE'  # Substitua pela sua chave real
PAIRS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'AUD/USD', 'USD/CAD']
INTERVAL = '60min'  # Mude para 'daily' se quota exceder
LOOKBACK = 30
HORIZONS = ['1 Hora', '4 Horas', '1 Dia', '1 Semana', '1 Mês']
HORIZON_STEPS = {'1 Hora': 1, '4 Horas': 4, '1 Dia': 24, '1 Semana': 168, '1 Mês': 720}
MC_SAMPLES = 10

# Funções (adaptadas para web, sem Tkinter)
def fetch_forex_data(pair):
    from_symbol, to_symbol = pair.split('/')
    function = 'FX_INTRADAY' if INTERVAL != 'daily' else 'FX_DAILY'
    url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_symbol}&to_symbol={to_symbol}&interval={INTERVAL if INTERVAL != "daily" else ""}&apikey={API_KEY}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    if 'Error Message' in data:
        raise ValueError(data['Error Message'])
    if 'Note' in data or 'Information' in data:
        raise ValueError(data.get('Note') or data.get('Information'))
    key = f'Time Series FX ({INTERVAL})'
    if key not in data:
        raise ValueError(f"Chave não encontrada: {key}")
    df = pd.DataFrame.from_dict(data[key], orient='index')
    df = df.astype(np.float32)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = ['open', 'high', 'low', 'close']
    return df

def fetch_news_sentiment(pair):
    _, to_symbol = pair.split('/')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=FOREX:{to_symbol}&topics=financial_markets&apikey={API_KEY}&limit=50'
    response = requests.get(url)
    data = response.json()
    if 'Error Message' in data:
        raise ValueError(data['Error Message'])
    feed = data.get('feed', [])
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(item['title'] + ' ' + item['summary'])['compound'] for item in feed]
    return np.float32(np.mean(sentiments) if sentiments else 0)

def add_indicators(df):
    df['SMA_10'] = df['close'].rolling(window=10).mean().astype(np.float32)
    df['RSI'] = compute_rsi(df['close'], 14).astype(np.float32)
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean().astype(np.float32)
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().astype(np.float32)
    rs = gain / loss
    return (100 - (100 / (1 + rs))).astype(np.float32)

class ForexDataset(Dataset):
    def __init__(self, data, lookback):
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(data).astype(np.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.lookback], dtype=torch.float32), torch.tensor(self.data[idx+self.lookback], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_predict(df, sentiment, steps):
    df = df.copy()
    df['sentiment'] = np.float32(sentiment)
    data = df[['close', 'SMA_10', 'RSI', 'sentiment']].values.astype(np.float32)

    dataset = ForexDataset(data, LOOKBACK)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(5):
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y[:, 0].unsqueeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    current_input = torch.tensor(dataset.scaler.transform(data[-LOOKBACK:]).astype(np.float32), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            step_preds = []
            for _ in range(MC_SAMPLES):
                model.train()
                pred = model(current_input)
                step_preds.append(pred.item())
            mean_pred = np.mean(step_preds)
            predictions.append(mean_pred)
            new_input = np.array([[mean_pred, data[-1][1], data[-1][2], sentiment]], dtype=np.float32)
            new_input_scaled = dataset.scaler.transform(new_input).astype(np.float32)
            current_input = torch.cat((current_input[:, 1:, :], torch.tensor(new_input_scaled, dtype=torch.float32).unsqueeze(0)), dim=1)

    final_mean = np.mean(predictions[-MC_SAMPLES:])
    final_std = np.std(predictions[-MC_SAMPLES:])
    final_pred = dataset.scaler.inverse_transform(np.array([[final_mean, 0, 0, 0]], dtype=np.float32))[0][0]
    return final_pred, final_std

def calculate_probability(current_close, predicted_close, std):
    if std == 0:
        return 100 if predicted_close > current_close else 0
    z = (current_close - predicted_close) / std
    if predicted_close > current_close:
        prob = (1 - norm.cdf(z)) * 100
    else:
        prob = norm.cdf(z) * 100
    return round(prob, 2)

# Interface Web com Streamlit
st.title("Sistema de Análise Forex - Online")

pair = st.selectbox("Selecione o Par de Moedas", PAIRS)
horizon = st.selectbox("Selecione o Horizonte de Previsão", HORIZONS)

if st.button("Atualizar Agora"):
    with st.spinner("Atualizando..."):
        try:
            df = fetch_forex_data(pair)
            df = add_indicators(df)
            sentiment = fetch_news_sentiment(pair)
            steps = HORIZON_STEPS[horizon]
            predicted_close, std = train_predict(df.tail(LOOKBACK + steps), sentiment, steps)

            current_close = df["close"].iloc[-1]
            st.write(f"Close Atual ({pair}): {current_close:.4f}")
            st.write(f"Previsão ({pair}, {horizon}): {predicted_close:.4f}")

            if predicted_close > current_close:
                trend = "Alta Prevista"
                color = "green"
                scenario = "alta"
            else:
                trend = "Baixa Prevista"
                color = "red"
                scenario = "baixa"
            st.markdown(f"<h3 style='color: {color};'>Tendência ({pair}): {trend}</h3>", unsafe_allow_html=True)

            prob = calculate_probability(current_close, predicted_close, std)
            st.write(f"Probabilidade de {scenario}: {prob}%")

            fig, ax = plt.subplots()
            ax.plot(df.index, df['close'], label='Histórico')
            ax.axvline(df.index[-1], color='red', linestyle='--', label='Início da Previsão')
            ax.set_title(f"Gráfico de Tendências - {pair} ({horizon})")
            ax.legend()
            st.pyplot(fig)

            st.success("Atualizado com sucesso!")
        except Exception as e:
            st.error(f"Erro: {str(e)}")