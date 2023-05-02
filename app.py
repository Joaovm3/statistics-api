# from flask import Flask

# app = Flask(__name__)

# @app.route('/hello')
# def hello_world():
#   return '<b>Hello, world!</b>'

#if __name__ == '__main__':
#  app.run(port=5000, debug=True, host='localhost')

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

class StatisticalAnalysis:
  def __init__(self, data, k=0, population=True):
    self.data = np.array(data)
    self.population = population
    self.freq_table = self.create_freq_table(k)
    self.stats = self.calculate_stats()

  def create_freq_table(self, k):
    n = len(self.data)
    k = k if k > 0 else (5 if n <= 20 else int(np.ceil(np.sqrt(n))))
    min_val, max_val = min(self.data), max(self.data)
    h = np.ceil((max_val - min_val) / k)
    intervals = [(np.floor(min_val) + i * h, np.floor(min_val) + (i + 1) * h) for i in range(k)]

    freq_table_data = {key: [sum(interval[0] <= val < interval[1] for val in self.data) for interval in intervals] for key in ["Fi"]}
    freq_table_data["Xi"] = [int(np.mean(interval)) for interval in intervals]
    freq_table_data["Fi * Xi"] = [fi * xi for fi, xi in zip(freq_table_data["Fi"], freq_table_data["Xi"])]
    freq_table_data["Fi * Xi^2"] = [fi * (xi ** 2) for fi, xi in zip(freq_table_data["Fi"], freq_table_data["Xi"])]

    freq_table = pd.DataFrame(freq_table_data)
    freq_table["Intervalo"] = intervals
    freq_table["Frequência Acumulada"] = freq_table["Fi"].cumsum()
    freq_table["Frequência Acumulada (%)"] = freq_table["Frequência Acumulada"] / len(self.data) * 100
    freq_table["Frequência Relativa"] = freq_table["Fi"] / len(self.data)
    freq_table["Frequência Relativa (%)"] = freq_table["Frequência Relativa"] * 100
    freq_table = freq_table[["Intervalo", "Fi", "Xi", "Frequência Acumulada", "Frequência Acumulada (%)", "Frequência Relativa", "Frequência Relativa (%)", "Fi * Xi", "Fi * Xi^2"]]

    return freq_table

  def calculate_stats(self):
    n = np.sum(self.freq_table["Fi"])
    mean = np.average(self.freq_table["Xi"], weights=self.freq_table["Fi"])
    median = self._calculate_median()
    mode = self._calculate_mode()
    variance = self._calculate_variance(mean)
    std_deviation = np.sqrt(variance)
    coeff_variation = std_deviation / mean * 100

    return {
      "Média": mean,
      "Mediana": median,
      "Moda": mode,
      "Variância (S²)": variance,
      "Desvio Padrão (S)": std_deviation,
      "Coeficiente de Variação (CV)": coeff_variation,
    }

  def _calculate_median(self):
    n = np.sum(self.freq_table["Fi"])
    cum_freq = self.freq_table["Fi"].cumsum()
    median_idx = np.where(cum_freq >= n / 2)[0][0]
    L1 = self.freq_table["Intervalo"][median_idx][0]
    fa = cum_freq[median_idx - 1] if median_idx > 0 else 0
    fm = self.freq_table["Fi"][median_idx]
    h = self.freq_table["Intervalo"][median_idx][1] - self.freq_table["Intervalo"][median_idx][0]

    median = L1 + (n / 2 - fa) / fm * h
    return median

  def _calculate_mode(self):
    max_freq = np.max(self.freq_table["Fi"])
    mode_idx = np.where(self.freq_table["Fi"] == max_freq)[0][0]
    L1 = self.freq_table["Intervalo"][mode_idx][0]
    d1 = self.freq_table["Fi"][mode_idx] - self.freq_table["Fi"][mode_idx - 1] if mode_idx > 0 else self.freq_table["Fi"][mode_idx]
    d2 = self.freq_table["Fi"][mode_idx] - self.freq_table["Fi"][mode_idx + 1] if mode_idx < len(self.freq_table) - 1 else self.freq_table["Fi"][mode_idx]
    h = self.freq_table["Intervalo"][mode_idx][1] - self.freq_table["Intervalo"][mode_idx][0]

    mode = L1 + d1 / (d1 + d2) * h
    return mode

  def _calculate_variance(self, mean):
    E_xi_2 = np.sum(self.freq_table["Fi * Xi^2"]) / np.sum(self.freq_table["Fi"])
    E_xi = mean
    variance = E_xi_2 - E_xi ** 2

    if not self.population:
      n = np.sum(self.freq_table["Fi"])
      variance = variance * n / (n - 1)

    return variance


@app.route('/api/v1/analyze', methods=['POST'])
def analyze():
  data = request.json.get('data')
  k = request.json.get('k', 0)
  population = request.json.get('population', True)

  if not data:
    return jsonify({'error': 'O campo data é obrigatório'}), 400

  stats_analysis = StatisticalAnalysis(data, k, population)

  result = {
    'freq_table': stats_analysis.freq_table.to_dict(orient='records'),
    'stats': stats_analysis.stats,
  }

  return jsonify(result), 200


if __name__ == '__main__':
  app.run(debug=True)