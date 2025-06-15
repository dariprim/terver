import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

#исходная негруппированная выборка
selection = [1, 4, 5, 9, 3, 0, 4, 4, 5, 0, 7, 4, 10, 3, 4, 2, 4, 3, 4, 2, 5, 8, 5, 5, 9, 7, 6, 4, 5, 0, 4, 1, 5, 6, 7, 4, 3, 8, 5, 11, 6, 4, 2, 4, 4, 5, 4, 6, 2, 4]
n = 50
#вариационный ряд
var_series = np.sort(selection)
print("Вариационный ряд:", var_series)

print('------------Негруппированная выборка----------------')
#выборочное среднее
mx = np.mean(var_series) #MO
print(f'Выборочное среднее (оценка математического ожидания): {mx:.2f}')
sample_variance = np.var(var_series, ddof=1)  # Выборочная дисперсия
print(f'Выборочная дисперсия: {sample_variance:.2f}')

#Вычисление стандартного отклонения
std_dev = np.std(var_series, ddof=1)  # ddof=1 для выборочного стандартного отклонения

alpha = [0.05, 0.10, 0.01, 0.20, 0.15]

#Определение t-критического значения
for i in range(len(alpha)):
    t_alpha = stats.t.ppf(1 - alpha[i]/2, df=n-1)  # Двусторонний тест
    margin_of_error = t_alpha * (std_dev / np.sqrt(n))
    confidence_interval = (mx - margin_of_error, mx + margin_of_error)
    print(f'Доверительный интервал для математического ожидания при уровне значимости {alpha[i]}: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})')

confidence_interval_for_m0 = (mx - stats.t.ppf(1 - alpha[0]/2, df=n-1) * (std_dev / np.sqrt(n)), mx + stats.t.ppf(1 - alpha[0]/2, df=n-1) * (std_dev / np.sqrt(n)))
#Определение критических значений хи-квадрат
for i in range(len(alpha)):
    chi2_lower = stats.chi2.ppf(1 - alpha[i]/2, df=n-1)  # Для верхнего предела
    chi2_upper = stats.chi2.ppf(alpha[i]/2, df=n-1)     # Для нижнего предела
    confidence_interval_variance = ((n-1) * sample_variance / chi2_lower,(n-1) * sample_variance / chi2_upper)
    print(f'Доверительный интервал для дисперсии при уровне значимости {alpha[i]}: ({confidence_interval_variance[0]:.2f}, {confidence_interval_variance[1]:.2f})')
confidence_interval_variance_for_a0 = ((n-1) * sample_variance / stats.chi2.ppf(1 - alpha[0]/2, df=n-1),(n-1) * sample_variance / stats.chi2.ppf(alpha[0]/2, df=n-1))


print('---------------Группированная выборка----------------------')
#мат ожидание
Z = [-4.34, -0.88, 2.59, 6.05, 9.51, 12.96, 16.44]
N = [6, 7, 11, 7, 8, 9, 2]
z_n = []
for i in range(len(Z)):
    z_n.append(Z[i]*N[i])
sum_zn = sum(z_n)
Mx = sum_zn/n
print(f'Выборочное среднее: {Mx:.2f}')

#выборочная смещенная дисперсия
z2_n = []
for i in range(len(Z)):
    z2_n.append(((Z[i])**2) * N[i])
sum2_zn = sum(z2_n)
Dx = (sum2_zn - n * Mx**2)/n

#выборочная несмещенная дисперсия
s2 = (Dx * 50)/49
print(f'Выборочная несмещенная дисперсия: {s2:.2f}')


#Определение t-критического значения
for i in range(len(alpha)):
    t_alpha = stats.t.ppf(1 - alpha[i]/2, df=n-1)  # Двусторонний тест
    margin_of_error = t_alpha * (np.sqrt(s2/n))
    confidence_interval1 = (Mx - margin_of_error, Mx + margin_of_error)
    print(f'Доверительный интервал для математического ожидания при уровне значимости {alpha[i]}: ({confidence_interval1[0]:.2f}, {confidence_interval1[1]:.2f})')

#Определение критических значений хи-квадрат
for i in range(len(alpha)):
    chi2_lower = stats.chi2.ppf(1 - alpha[i]/2, df=n-1)  # Для верхнего предела
    chi2_upper = stats.chi2.ppf(alpha[i]/2, df=n-1)     # Для нижнего предела
    confidence_interval_variance1 = ((n-1) * s2 / chi2_lower,(n-1) * s2 / chi2_upper)
    print(f'Доверительный интервал для дисперсии при уровне значимости {alpha[i]}: ({confidence_interval_variance1[0]:.2f}, {confidence_interval_variance1[1]:.2f})')

#Проверка гипотез
M0 = mx + 0.5 * std_dev
if M0 <= confidence_interval_for_m0[1] and M0 >= confidence_interval_for_m0[0]:
    print(f'Гипотеза M0 = mx + 0.5 принимается, M0 = {M0:.2f}')
else:
    print(f'Гипотеза M0 = mx + 0.5 отклоняется, M0 = {M0:.2f}')

A0 = 2 * std_dev**2
if A0 <= confidence_interval_variance_for_a0[1] and A0 >= confidence_interval_variance_for_a0[0]:
    print(f'Гипотеза A0 = 2 * s^2 принимается, A0 = {A0:.2f}')
else:
    print(f'Гипотеза A0 = 2 * s^2 отклоняется, A0 = {A0:.2f}')


#перегруппировка, т.к. должно быть ni>=5
seredina = [-4.34, -0.88, 2.59, 6.05, 9.51, 14.71]
ni1, ni2, ni3, ni4, ni5, ni6 = 0, 0, 0, 0, 0, 0
for i in range(len(var_series)):
    if var_series[i] >= -6.07 and var_series[i] < -2.61:
        ni1+=1
    if var_series[i] >= -2.61 and var_series[i] < 0.86:
        ni2+=1
    if var_series[i] >= 0.86 and var_series[i] < 4.32:
        ni3+=1
    if var_series[i] >= 4.32 and var_series[i] < 7.78:
        ni4+=1
    if var_series[i] >= 7.78 and var_series[i] < 11.24:
        ni5+=1
    if var_series[i] >= 11.24 and var_series[i] <= 18.17:
        ni6+=1
Ni = [ni1, ni2, ni3, ni4, ni5, ni6]
z_n_phi = []
for i in range(len(seredina)):
    z_n_phi.append(seredina[i]*Ni[i])
sum_z_n_phi = sum(z_n_phi)
Mx_phi = sum_z_n_phi/n
print(f'Выборочное среднее: {Mx_phi:.2f}')

#выборочная смещенная дисперсия
z2_n_phi = []
for i in range(len(seredina)):
    z2_n_phi.append(((seredina[i])**2) * Ni[i])
sum2_zn_phi = sum(z2_n_phi)
Dx_phi = (sum2_zn_phi - n * Mx_phi**2)/n

#выборочная несмещенная дисперсия
s2_phi = (Dx_phi * 50)/49
print(f'Выборочная несмещенная дисперсия: {s2_phi:.2f}')
s_phi = np.sqrt(s2_phi)

print('Выборочные вероятности pi попадания случайной величины в каждый из интервалов выбранного разбиения:')
z1 = (-2.61 - Mx_phi) / s_phi
p1 = norm.cdf(z1)
print('P{X<-2.61}', f'= Ф((-2.61 - 5.53)/6.43) = Ф(-1.27) = {p1:.3f}')

z2_1 = (-2.61 - Mx_phi) / s_phi
z2_2 = (0.86 - Mx_phi) / s_phi
p2_1 = norm.cdf(z2_1)
p2_2 = norm.cdf(z2_2)
razn2 = p2_2 - p2_1
print('P{-2.61<=X<0.86}', f'= Ф({z2_2:.2f}) - Ф({z2_1:.2f}) = {razn2:.3f}')

z3_1 = (0.86 - Mx_phi) / s_phi
z3_2 = (4.32 - Mx_phi) / s_phi
p3_1 = norm.cdf(z3_1)
p3_2 = norm.cdf(z3_2)
razn3 = p3_2 - p3_1
print('P{0.86<=X<4.32}', f'= Ф({z3_2:.2f}) - Ф({z3_1:.2f}) = {razn3:.3f}')

z4_1 = (4.32 - Mx_phi) / s_phi
z4_2 = (7.78 - Mx_phi) / s_phi
p4_1 = norm.cdf(z4_1)
p4_2 = norm.cdf(z4_2)
razn4 = p4_2 - p4_1
print('P{4.32<=X<7.78}', f'= Ф({z4_2:.2f}) - Ф({z4_1:.2f}) = {razn4:.3f}')

z5_1 = (7.78 - Mx_phi) / s_phi
z5_2 = (11.24 - Mx_phi) / s_phi
p5_1 = norm.cdf(z5_1)
p5_2 = norm.cdf(z5_2)
razn5 = p5_2 - p5_1
print('P{7.78<=X<11.24}', f'= Ф({z5_2:.2f}) - Ф({z5_1:.2f}) = {razn5:.3f}')

z6 = (11.24 - Mx_phi)/s_phi
p6 = norm.cdf(z6)
p6 = 1 - p6
print('P{X>=11.24} = 1 - P{X<11.24}', f'= 1 - Ф({z6:.2f}) = {p6:.2f}')

print('Найдем теоретические частоты:')
n11 = round(p1*50 , 3)
print(f"n1' = {n11}")
n22 = round(razn2*50 , 3)
print(f"n2' = {n22}")
n33 = round(razn3*50 , 3)
print(f"n3' = {n33}")
n44 = round(razn4*50 , 3)
print(f"n4' = {n44}")
n55 = round(razn5*50 , 3)
print(f"n5' = {n55}")
n66 = round(p6*50 , 3)
print(f"n6' = {n66}")

n_n = [n11, n22, n33, n44, n55, n66]
phi = 0
for i in range(len(Ni)):
    phi += ((Ni[i] - n_n[i])**2/n_n[i])
print(f'Phi_B = {phi:.2f}')