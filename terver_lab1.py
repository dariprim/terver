import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
# Данные
data = np.array([2.36, 12.25, 14.35, 6.98, 14.52, 14.41, -0.29, 2.61, 5.44, 13.42, 
             7.00, -0.24, -6.07, -2.90, -0.48, 13.78, 3.36, 9.86, 2.04, 2.08,
             -4.37, 8.24, 13.46, -1.67, 9.62, 8.98, 4.79, 10.16, 1.65, 7.04, 
             1.87, 9.72, 18.17, 2.99, 0.97, 5.92, 15.38, -3.31, 2.32, 3.12,
             13.64, 10.07, -0.40, -0.87, 12.64, 10.28, 4.95, -3.34, -3.48, 0.19])

# Вариационный ряд
sorted_data = np.sort(data)
print("Вариационный ряд:", sorted_data)

# Подсчет частот каждого уникального значения
data_counts = Counter(data)

# Преобразование в DataFrame для удобства отображения
stat_series = pd.DataFrame.from_dict(data_counts, orient='index', columns=['Частота']).sort_index()

# Вывод статистического ряда
print("Статистический ряд:")
print(stat_series)


# Размах выборки
range_ = np.ptp(data) #max-min
print("Размах выборки:", range_)

# Число интервалов
intervals = 7

# Минимум и максимум данных
min_val = np.min(data)
max_val = np.max(data)

# Шаг интервала
interval_length = (max_val - min_val) / intervals

# Создание интервалов
bins = np.linspace(min_val, max_val, intervals + 1)

# Частоты
freq, bins = np.histogram(data, bins=bins)

# Таблица частот
freq_table = pd.DataFrame({'Интервал': pd.IntervalIndex.from_breaks(bins), 'Частота': freq})
print(freq_table)



# Эмпирическая функция распределения
sorted_data = np.sort(data)
y_vals = np.arange(1, len(sorted_data)+1) / len(sorted_data)
plt.step(sorted_data, y_vals, where="post")
plt.title('Эмпирическая функция распределения')
plt.xlabel('Значение')
plt.ylabel('Вероятность')
plt.show()

hist, bin_edges = np.histogram(data, bins = 7)
plt.hist(data, bins = 7)
middle = [] 
for i in range(7):
    middle.append(bin_edges[i]+(bin_edges[i+1]-bin_edges[i])/2)
plt.title("Гистограмма и полигон частот")
plt.plot(middle, hist, 'pink')
plt.grid()
plt.show()


# Математическое ожидание
mean = np.mean(data)
print("Математическое ожидание (среднее):", mean)

# Смещенная дисперсия
variance_biased = np.var(data)
print("Смещенная дисперсия:", variance_biased)

# Несмещенная дисперсия
variance_unbiased = np.var(data, ddof=1)
print("Несмещенная дисперсия:", variance_unbiased)

# Медиана
median = np.median(data)
print("Медиана:", median)

from collections import Counter

# Поиск моды с помощью Counter
data_counts = Counter(data)
mode_data = data_counts.most_common(1)  # Находит наиболее частый элемент
print("Мода:", mode_data[0][0])


# Средние значения интервалов
mid_points = (bins[:-1] + bins[1:]) / 2

# Математическое ожидание для группированной выборки
grouped_mean = np.sum(mid_points * freq) / np.sum(freq)
print("Математическое ожидание (группированная выборка):", grouped_mean)

# Дисперсия для группированной выборки (смещенная и несмещенная)
grouped_variance_biased = np.sum(freq * (mid_points - grouped_mean)**2) / np.sum(freq)
grouped_variance_unbiased = np.sum(freq * (mid_points - grouped_mean)**2) / (np.sum(freq) - 1)

print("Смещенная дисперсия (группированная выборка):", grouped_variance_biased)
print("Несмещенная дисперсия (группированная выборка):", grouped_variance_unbiased)

# Медиана для группированной выборки
cum_freq = np.cumsum(freq)
n = np.sum(freq)
median_interval_idx = np.where(cum_freq >= n / 2)[0][0]
L = bins[median_interval_idx]
F = cum_freq[median_interval_idx - 1] if median_interval_idx > 0 else 0
h = interval_length
median_grouped = L + (n / 2 - F) / freq[median_interval_idx] * h
print("Медиана (группированная выборка):", median_grouped)

# Мода для группированной выборки
mode_interval_idx = np.argmax(freq)
mode_grouped = bins[mode_interval_idx] + h * (freq[mode_interval_idx] - freq[mode_interval_idx-1]) / \
               ((freq[mode_interval_idx] - freq[mode_interval_idx-1]) + (freq[mode_interval_idx] - freq[mode_interval_idx+1]))
print("Мода (группированная выборка):", mode_grouped)

