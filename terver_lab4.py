import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Исходные данные
X = np.array([
    9.75, 10.53, 12.16, 7.51, 7.89, 7.81, 12.21, 6.54, 11.23, 12.03,
    7.35, 7.28, 10.27, 7.53, 8.19, 12.13, 8.38, 7.21, 12.18, 11.26,
    10.03, 11.96, 9.37, 9.24, 6.58, 8.56, 11.11, 7.85, 11.12, 7.89,
    9.65, 9.0, 9.41, 8.29, 9.66, 8.17, 7.66, 9.19, 11.7, 7.13,
    10.92, 9.88, 11.26, 11.38, 9.84, 12.04, 11.95, 10.42, 12.45, 9.52
])
Y = np.array([
    78.61, 76.45, 87.37, 38.4, 51.92, 50.08, 71.29, 45.9, 68.66, 72.03,
    44.06, 60.32, 64.67, 50.46, 71.55, 66.25, 69.92, 43.09, 89.22, 78.22,
    71.04, 79.54, 47.04, 65.96, 38.08, 45.82, 77.99, 60.26, 89.11, 57.97,
    64.17, 62.0, 54.03, 65.72, 77.79, 53.81, 51.47, 52.11, 71.49, 42.69,
    54.96, 72.43, 81.53, 87.79, 63.47, 79.1, 82.61, 50.39, 84.26, 69.89
])

#средние значения и отклонение
meanX = np.mean(X)
stdX = np.std(X, ddof=1) 
meanY = np.mean(Y)
stdY = np.std(Y, ddof=1) 

# 1. Построение диаграммы рассеивания
plt.scatter(X, Y)
plt.title('Диаграмма рассеивания X и Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# 2. Проверка гипотезы H0: корреляция = 0
r, p_value = stats.pearsonr(X, Y)
print(f"Коэффициент корреляции: {r:.4f}")

# 3. Интервальная оценка коэффициента корреляции
alpha = 0.05
z = np.arctanh(r)
se = 1 / np.sqrt(len(X) - 3)
z_critical = stats.norm.ppf(1 - alpha/2)
z_interval = [z - z_critical * se, z + z_critical * se]
r_interval = np.tanh(z_interval)
print(f"95%-ный доверительный интервал для коэффициента корреляции: {r_interval}")

# 4. Уравнение линейной регрессии Y на X
X_reshaped = X.reshape(-1, 1)
reg_model_Y_on_X = LinearRegression().fit(X_reshaped, Y)
print(f"Уравнение регрессии Y на X: Y = {reg_model_Y_on_X.intercept_:.4f} + {reg_model_Y_on_X.coef_[0]:.4f} * X")

# Параметры регрессии
a = reg_model_Y_on_X.intercept_
b = reg_model_Y_on_X.coef_[0]
n = len(X)

# Уравнение регрессии X на Y
Y_reshaped = Y.reshape(-1, 1)
reg_model_X_on_Y = LinearRegression().fit(Y_reshaped, X)
print(f"Уравнение регрессии X на Y: X = {reg_model_X_on_Y.intercept_:.4f} + {reg_model_X_on_Y.coef_[0]:.4f} * Y")

# 5. Нанесение графиков выборочных регрессионных прямых на диаграмму рассеивания
plt.scatter(X, Y, label='Данные')
plt.plot(X, reg_model_Y_on_X.predict(X_reshaped), color='red', label='Регрессия Y на X')
plt.plot(reg_model_X_on_Y.predict(Y_reshaped), Y, color='green', label='Регрессия X на Y')
plt.title('Диаграмма рассеивания с регрессионными прямыми')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# 6. Проверка значимости регрессии Y на X
ss_total = np.sum((Y - np.mean(Y))**2)
ss_residual = np.sum((Y - reg_model_Y_on_X.predict(X_reshaped))**2)
r_squared = 1 - (ss_residual / ss_total)
f_statistic = (r_squared / (1 - r_squared)) * ((len(X) - 2) / 1)
p_value_f = 1 - stats.f.cdf(f_statistic, 1, len(X) - 2)
print(f"F-статистика: {f_statistic:.4f}")

# 7. Проверка гипотезы mX = mY
t_stat, p_value_t = stats.ttest_ind(X, Y)
print(f"t-статистика: {t_stat:.4f}")

# Предсказанные значения Y
Y_pred = reg_model_Y_on_X.predict(X_reshaped)

# 1. Оценка дисперсии ошибок наблюдений s^2
s_squared = np.sum((Y - Y_pred) ** 2) / (n - 2)
print(f"Оценка дисперсии ошибок наблюдений s^2: {s_squared:.4f}")

# 2. Коэффициент детерминации R^2
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print(f"Коэффициент детерминации R^2: {r_squared:.4f}")

# 3. Доверительные интервалы для параметров a и b
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, n - 2)
se_b = np.sqrt(s_squared / np.sum((X - np.mean(X)) ** 2))
se_a = se_b * np.sqrt(np.sum(X**2) / n)

ci_b = [b - t_critical * se_b, b + t_critical * se_b]
ci_a = [a - t_critical * se_a, a + t_critical * se_a]
print(f"Доверительный интервал для b: {ci_b}")
print(f"Доверительный интервал для a: {ci_a}")

# 4. Доверительный интервал для дисперсии ошибок наблюдений s^2
chi2_lower = stats.chi2.ppf(alpha / 2, n - 2)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 2)
ci_s2 = [(n - 2) * s_squared / chi2_upper, (n - 2) * s_squared / chi2_lower]
print(f"Доверительный интервал для дисперсии ошибок s^2: {ci_s2}")

# 5. Доверительный интервал для среднего значения Y при x = x_0
x_0 = 1.0  # Пример значения x_0
y_0_pred = a + b * x_0
se_y_0 = np.sqrt(s_squared * (1/n + (x_0 - np.mean(X))**2 / np.sum((X - np.mean(X))**2)))
ci_y_0 = [y_0_pred - t_critical * se_y_0, y_0_pred + t_critical * se_y_0]
print(f"Доверительный интервал для среднего значения Y при x = {x_0}: {ci_y_0}")

# Сгруппируем данные по X
# Для этого разделим X на интервалы (бинами)
bins = np.linspace(min(X), max(X), 10)  # Создаем 10 групп по X
df = pd.DataFrame({'X': X, 'Y': Y})
df['X_binned'] = pd.cut(df['X'], bins)

# Группировка данных
grouped = df.groupby('X_binned', observed=False).mean().dropna()
X_grouped = grouped['X'].values.reshape(-1, 1)
Y_grouped = grouped['Y'].values

# Линейная регрессия Y на X для сгруппированных данных
reg_model_Y_on_X_grouped = LinearRegression().fit(X_grouped, Y_grouped)

# Предсказанные значения
Y_grouped_pred = reg_model_Y_on_X_grouped.predict(X_grouped)

# SSR: Сумма квадратов регрессии
SSR = np.sum((Y_grouped_pred - np.mean(Y_grouped)) ** 2)

# SSE: Сумма квадратов остатков
SSE = np.sum((Y_grouped - Y_grouped_pred) ** 2)

# Количество наблюдений
n_grouped = len(X_grouped)

# F-статистика
k = 1  # Количество параметров в модели
F_statistic = (SSR / k) / (SSE / (n_grouped - k - 1))
print(f"F-статистика: {F_statistic:.4f}")

# Критическое значение F для уровня значимости alpha = 0.05
alpha = 0.05
df1 = k  # Степени свободы числителя
df2 = n_grouped - k - 1  # Степени свободы знаменателя
F_critical = stats.f.ppf(1 - alpha, df1, df2)
print(f"Критическое значение F: {F_critical:.4f}")

# Проверка гипотезы
if F_statistic > F_critical:
    print("Модель адекватна на уровне значимости 0.05")
else:
    print("Модель неадекватна на уровне значимости 0.05")