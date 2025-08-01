import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
import threading
from scipy.stats import norm  # Para calcular probabilidade via CDF

# Configurações
API_KEY = 'VZ6XL34A1G4VCKP3'  # Substitua pela sua chave
PAIRS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'AUD/USD', 'USD/CAD']  # Adicione mais pares aqui
INTERVAL = '60min'  # Para horárias; mude para 'daily' se quota exceder
LOOKBACK = 30  # Reduzido para acelerar
HORIZONS = ['1 Hora', '4 Horas', '1 Dia', '1 Semana', '1 Mês']  # Restauradas
HORIZON_STEPS = {'1 Hora': 1, '4 Horas': 4, '1 Dia': 24, '1 Semana': 168, '1 Mês': 720}  # Steps em horas
MC_SAMPLES = 10  # Número de amostras para Monte Carlo Dropout (para probabilidade)

# Funções de dados e análise (com melhor erro handling)
def fetch_forex_data(pair):
    print("Debug: Baixando dados forex para", pair)
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
        raise ValueError(f"Chave de dados não encontrada: {key}. Resposta da API: {data}")
    df = pd.DataFrame.from_dict(data[key], orient='index')
    df = df.astype(np.float32)  # Força float32 desde o início
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = ['open', 'high', 'low', 'close']
    return df

def fetch_news_sentiment(pair):
    print("Debug: Baixando notícias para", pair)
    _, to_symbol = pair.split('/')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=FOREX:{to_symbol}&topics=financial_markets&apikey={API_KEY}&limit=50'
    response = requests.get(url)
    data = response.json()
    if 'Error Message' in data:
        raise ValueError(data['Error Message'])
    if 'Note' in data or 'Information' in data:
        raise ValueError(data.get('Note') or data.get('Information'))
    feed = data.get('feed', [])
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(item['title'] + ' ' + item['summary'])['compound'] for item in feed]
    return np.float32(np.mean(sentiments) if sentiments else 0)  # Float32

def add_indicators(df):
    print("Debug: Calculando indicadores...")
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
        self.data = self.scaler.fit_transform(data).astype(np.float32)  # Força float32
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.lookback], dtype=torch.float32), torch.tensor(self.data[idx+self.lookback], dtype=torch.float32)  # float32 para y também

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, dropout=0.2):  # num_layers=2 para dropout funcionar sem warning
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_predict(df, sentiment, steps):
    print("Debug: Treinando e prevendo para", steps, "steps...")
    df = df.copy()  # Corrige SettingWithCopyWarning
    df['sentiment'] = np.float32(sentiment)  # Float32
    data = df[['close', 'SMA_10', 'RSI', 'sentiment']].values.astype(np.float32)  # Força float32

    dataset = ForexDataset(data, LOOKBACK)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMModel(dropout=0.2)  # Adicionado dropout para MC
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(5):  # Reduzido para acelerar
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y[:, 0].unsqueeze(1))
            loss.backward()
            optimizer.step()

    # Previsão com Monte Carlo Dropout para estimar incerteza
    model.eval()  # Mas ativamos dropout manualmente para MC
    predictions = []
    current_input = torch.tensor(dataset.scaler.transform(data[-LOOKBACK:]).astype(np.float32), dtype=torch.float32).unsqueeze(0)  # float32
    with torch.no_grad():
        for _ in range(steps):
            # Rodar MC_SAMPLES vezes para cada step
            step_preds = []
            for _ in range(MC_SAMPLES):
                model.train()  # Ativa dropout durante inferência para MC
                pred = model(current_input)
                step_preds.append(pred.item())
            mean_pred = np.mean(step_preds)
            predictions.append(mean_pred)
            new_input = np.array([[mean_pred, data[-1][1], data[-1][2], sentiment]], dtype=np.float32)  # float32
            new_input_scaled = dataset.scaler.transform(new_input).astype(np.float32)
            current_input = torch.cat((current_input[:, 1:, :], torch.tensor(new_input_scaled, dtype=torch.float32).unsqueeze(0)), dim=1)

    final_mean = np.mean(predictions[-MC_SAMPLES:])  # Média das últimas amostras
    final_std = np.std(predictions[-MC_SAMPLES:])  # Std dev para incerteza
    final_pred = dataset.scaler.inverse_transform(np.array([[final_mean, 0, 0, 0]], dtype=np.float32))[0][0]
    return final_pred, final_std  # Retorna previsão e std para probabilidade

def calculate_probability(current_close, predicted_close, std):
    if std == 0:  # Evitar divisão por zero
        return 100 if predicted_close > current_close else 0
    # Assume distribuição normal; prob de preço > atual (para alta)
    z = (current_close - predicted_close) / std
    if predicted_close > current_close:
        prob = (1 - norm.cdf(z)) * 100  # % de chance de alta
    else:
        prob = norm.cdf(z) * 100  # % de chance de baixa
    return round(prob, 2)

# Interface Gráfica (adicionada label para probabilidade)
class ForexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análise Forex - Múltiplos Pares e Previsões")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
        style.configure("TCombobox", font=("Arial", 12))

        self.title_label = ttk.Label(root, text="Selecione um Par e Horizonte de Previsão", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        self.pair_var = tk.StringVar(value=PAIRS[0])
        self.pair_combo = ttk.Combobox(root, textvariable=self.pair_var, values=PAIRS, state="readonly")
        self.pair_combo.pack(pady=5)

        self.horizon_var = tk.StringVar(value=HORIZONS[0])
        self.horizon_combo = ttk.Combobox(root, textvariable=self.horizon_var, values=HORIZONS, state="readonly")
        self.horizon_combo.pack(pady=5)

        self.status_label = ttk.Label(root, text="Status: Pronto", foreground="blue")
        self.status_label.pack()

        self.close_label = ttk.Label(root, text="Close Atual: Carregando...")
        self.close_label.pack()

        self.pred_label = ttk.Label(root, text="Previsão: Carregando...")
        self.pred_label.pack()

        self.prob_label = ttk.Label(root, text="Probabilidade: Carregando...", font=("Arial", 12, "italic"))
        self.prob_label.pack()

        self.trend_label = ttk.Label(root, text="Tendência: Carregando...", font=("Arial", 14, "bold"))
        self.trend_label.pack(pady=10)

        self.update_button = ttk.Button(root, text="Atualizar Agora", command=self.update_data)
        self.update_button.pack(pady=5)

        self.quit_button = ttk.Button(root, text="Sair", command=root.quit)
        self.quit_button.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.running = True
        # Comente a linha abaixo se quiser desativar atualização automática
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def update_data(self):
        pair = self.pair_var.get()
        horizon = self.horizon_var.get()
        steps = HORIZON_STEPS[horizon]
        self.status_label.config(text=f"Atualizando {pair} para {horizon}...", foreground="orange")
        try:
            df = fetch_forex_data(pair)
            df = add_indicators(df)
            sentiment = fetch_news_sentiment(pair)
            predicted_close, std = train_predict(df.tail(LOOKBACK + steps), sentiment, steps)  # Agora retorna std

            current_close = df["close"].iloc[-1]
            self.close_label.config(text=f"Close Atual ({pair}): {current_close:.4f}")
            self.pred_label.config(text=f"Previsão ({pair}, {horizon}): {predicted_close:.4f}")

            if predicted_close > current_close:
                trend = "Alta Prevista"
                color = "green"
                scenario = "alta"
            else:
                trend = "Baixa Prevista"
                color = "red"
                scenario = "baixa"
            self.trend_label.config(text=f"Tendência ({pair}): {trend}", foreground=color)

            prob = calculate_probability(current_close, predicted_close, std)
            self.prob_label.config(text=f"Probabilidade de {scenario}: {prob}%")

            self.ax.clear()
            self.ax.plot(df.index, df['close'], label='Histórico', color='blue')
            self.ax.axvline(df.index[-1], color='red', linestyle='--', label='Início da Previsão')
            self.ax.set_title(f"Gráfico de Tendências - {pair} ({horizon})")
            self.ax.legend()
            self.canvas.draw()

            self.status_label.config(text="Atualizado com sucesso!", foreground="green")
        except Exception as e:
            self.status_label.config(text=f"Erro: {str(e)}", foreground="red")
            print("Debug Erro completo:", str(e))  # Mostra no terminal para depuração

    def monitor(self):
        while self.running:
            self.update_data()
            time.sleep(3600)

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = ForexApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()