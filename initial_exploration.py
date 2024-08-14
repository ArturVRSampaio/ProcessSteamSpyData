import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def save_graphics(name,
                  coluna,
                  total_rows,
                  contagem_valores,
                  media,
                  mediana,
                  percentil_25,
                  percentil_50,
                  percentil_75,
                  variancia,
                  skewness,
                  kurtosis_value,
                  is_log = True):

    if not os.path.exists('out/initial/' + name):
        os.makedirs('out/initial/' + name)

    plt.figure(figsize=(10, 5))
    plt.hist(coluna, bins=14, edgecolor='k', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    if is_log:
        plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('out/initial/' + name + '/histograma.png')

    plt.figure(figsize=(10, 5))
    plt.boxplot(coluna, vert=False)
    plt.title('Boxplot')
    plt.xlabel('Valores')
    if is_log:
        plt.xscale('log')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('out/initial/' + name + '/boxplot.png')

    plt.figure(figsize=(12, 6))
    contagem_valores.plot(kind='bar')
    plt.title('Contagem de Valores Únicos')
    plt.xlabel('Valores')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig('out/initial/' + name + '/contagem_valores.png')

    estatisticas = {
        'Total de Linhas': total_rows,
        'Média': media,
        'Mediana': mediana,
        'Percentil 25': percentil_25,
        'Percentil 50': percentil_50,
        'Percentil 75': percentil_75,
        'Variância': variancia,
        'Skewness': skewness,
        'Kurtosis': kurtosis_value
    }

    plt.figure(figsize=(12, 6))
    bars = plt.bar(estatisticas.keys(), estatisticas.values(), color='skyblue')
    plt.title('Resumo Estatístico')
    plt.xlabel('Estatística')
    if is_log:
        plt.yscale('log')
    plt.ylabel('Valor')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    plt.savefig('out/initial/' + name + '/resumo_estatistico.png')

    plt.figure(figsize=(12, 3))
    plt.scatter(coluna, np.zeros_like(coluna), alpha=0.5, label='Valores', color='lightgray')
    plt.scatter([media] * 2, [1, 1], color='red', label='Média', marker='o', s=100)
    plt.scatter([mediana] * 2, [2, 2], color='blue', label='Mediana', marker='x', s=100)
    plt.scatter([percentil_25] * 2, [3, 3], color='green', label='Percentil 25', marker='v', s=100)
    plt.scatter([percentil_50] * 2, [4, 4], color='purple', label='Percentil 50', marker='^', s=100)
    plt.scatter([percentil_75] * 2, [5, 5], color='orange', label='Percentil 75', marker='s', s=100)

    plt.axvline(media - np.sqrt(variancia), color='gray', linestyle='--', label='± Variância')
    plt.axvline(media + np.sqrt(variancia), color='gray', linestyle='--')

    plt.title('Distribuição dos Valores com Estatísticas')
    plt.xlabel('Valores')
    plt.ylabel('Estatísticas')
    plt.yticks([1, 2, 3, 4, 5], ['Média', 'Mediana', 'Percentil 25', 'Percentil 50', 'Percentil 75'])
    if is_log:
        plt.xscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('out/initial/' + name + '/distribuicao_estatisticas.png')


data = pd.read_csv("dataset/output.csv")

# !!!!!!!!!!!!!!!!!!!! owner first exploration

coluna = data['owners']

total_rows = len(coluna)
contagem_valores = data['owners'].value_counts()
media = np.mean(coluna)
mediana = np.median(coluna)
percentil_25 = np.percentile(coluna, 25)
percentil_50 = np.percentile(coluna, 50)
percentil_75 = np.percentile(coluna, 75)
variancia = np.var(coluna)
skewness = skew(coluna)
kurtosis_value = kurtosis(coluna)

print(f'Total de linhas: {total_rows}')
print(f'Contagem valores:\n{contagem_valores}')
print(f'Média: {media}')
print(f'Mediana: {mediana}')
print(f'Percentil 25: {percentil_25}')
print(f'Percentil 50: {percentil_50}')
print(f'Percentil 75: {percentil_75}')
print(f'Variância: {variancia}')
print(f'Skewness: {skewness}')
print(f'Kurtosis: {kurtosis_value}')

save_graphics('default_owner', coluna, total_rows, contagem_valores, media, mediana, percentil_25, percentil_50, percentil_75, variancia, skewness, kurtosis_value)

# !!!!!!!!!!!!!!!!!!!! owner second exploration

owners_to_class = {
    100000000: 0,
    200000000: 0,
    500000000: 0,
    50000000: 0,
    20000000: 0,
    10000000: 0,
    5000000: 0,
    2000000: 0,
    1000000: 0,
    500000: 0,
    200000: 0,
    100000: 0,
    20000: 1,
    50000: 1
}

data['owners_class'] = data['owners'].map(owners_to_class)

# balance data
class_counts = data['owners_class'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

df_minority = data[data['owners_class'] == minority_class]
df_majority = data[data['owners_class'] == majority_class]

df_majority_sampled = df_majority.sample(len(df_minority), random_state=42)

data_balanced = pd.concat([df_majority_sampled, df_minority])

coluna = data_balanced["owners"]
# balance data


total_rows = len(coluna)
contagem_valores = data_balanced['owners_class'].value_counts()
media = np.mean(coluna)
mediana = np.median(coluna)
percentil_25 = np.percentile(coluna, 25)
percentil_50 = np.percentile(coluna, 50)
percentil_75 = np.percentile(coluna, 75)
variancia = np.var(coluna)
skewness = skew(coluna)
kurtosis_value = kurtosis(coluna)

print(f'Total de linhas: {total_rows}')
print(f'Contagem valores:\n{contagem_valores}')
print(f'Média: {media}')
print(f'Mediana: {mediana}')
print(f'Percentil 25: {percentil_25}')
print(f'Percentil 50: {percentil_50}')
print(f'Percentil 75: {percentil_75}')
print(f'Variância: {variancia}')
print(f'Skewness: {skewness}')
print(f'Kurtosis: {kurtosis_value}')

save_graphics('balanced_owner', coluna, total_rows, contagem_valores, media, mediana, percentil_25, percentil_50, percentil_75, variancia, skewness, kurtosis_value)



# !!!!!!!!!!!!!!!!!!!!!!! first user_Score exploration
coluna = data['user_score']

total_rows = len(coluna)
contagem_valores = data['owners'].value_counts()
media = np.mean(coluna)
mediana = np.median(coluna)
percentil_25 = np.percentile(coluna, 25)
percentil_50 = np.percentile(coluna, 50)
percentil_75 = np.percentile(coluna, 75)
variancia = np.var(coluna)
skewness = skew(coluna)
kurtosis_value = kurtosis(coluna)

print(f'Total de linhas: {total_rows}')
print(f'Contagem valores:\n{contagem_valores}')
print(f'Média: {media}')
print(f'Mediana: {mediana}')
print(f'Percentil 25: {percentil_25}')
print(f'Percentil 50: {percentil_50}')
print(f'Percentil 75: {percentil_75}')
print(f'Variância: {variancia}')
print(f'Skewness: {skewness}')
print(f'Kurtosis: {kurtosis_value}')

save_graphics('default_user_score', coluna, total_rows, contagem_valores, media, mediana, percentil_25, percentil_50, percentil_75, variancia, skewness, kurtosis_value, False)



# !!!!!!!!!!!!!!!!!!!!!!! second user_Score exploration
coluna = data_balanced['user_score']

total_rows = len(coluna)
contagem_valores = data_balanced['owners'].value_counts()
media = np.mean(coluna)
mediana = np.median(coluna)
percentil_25 = np.percentile(coluna, 25)
percentil_50 = np.percentile(coluna, 50)
percentil_75 = np.percentile(coluna, 75)
variancia = np.var(coluna)
skewness = skew(coluna)
kurtosis_value = kurtosis(coluna)

print(f'Total de linhas: {total_rows}')
print(f'Contagem valores:\n{contagem_valores}')
print(f'Média: {media}')
print(f'Mediana: {mediana}')
print(f'Percentil 25: {percentil_25}')
print(f'Percentil 50: {percentil_50}')
print(f'Percentil 75: {percentil_75}')
print(f'Variância: {variancia}')
print(f'Skewness: {skewness}')
print(f'Kurtosis: {kurtosis_value}')

save_graphics('balanced_user_score', coluna, total_rows, contagem_valores, media, mediana, percentil_25, percentil_50, percentil_75, variancia, skewness, kurtosis_value, False)
