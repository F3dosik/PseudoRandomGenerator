from math import gcd, log2, ceil
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from scipy.stats import chi2, kstest


# Для проверки простых делителей
def prime_factors(n):
    factors = []
    # Проверяем делимость на 2
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    # Теперь n нечётное, перебираем до sqrt(n)
    d = 3
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n = n // d
        d += 2
    # Если осталось простое число > 2
    if n > 1:
        factors.append(n)
    return factors


# Проверка параметров для Линейного конгруэнтного генератора
def check_parameters(N, a, c):
    print(f"Взаимная простота с и N: {gcd(c, N) == 1}.")
    for p in prime_factors(N):
        if (a - 1) % p != 0:
            print(f"a - 1 = {a - 1} не кратно простому делителю N {p}.")
    print(f"a - 1 = {a - 1} кратно всем простым делителям N.")
    if N % 4 == 0:
        if (a - 1) % 4 == 0:
            print(f"a - 1 и N кратно 4.")
        else:
            print(f"a - 1 не кратно 4, а N кратно.")


# Линейный конгруэнтный генератор
class LCG:
    def __init__(self, seed=datetime.now().microsecond, a=1103515245, c=12345, m=1 << 32):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def generate_float(self, n):
        return [self.next() for _ in range(n)]

    def generate(self, n, low, high):
        return [low + int(self.next() * (high - low)) for _ in range(n)]


# Генератор Фибоначчи с запаздыванием
class LFG:
    def __init__(self, seed_arr, l=17, k=31, m=1 << 32):
        if len(seed_arr) < k:
            raise ValueError("Длина начального массива должна быть не меньше k")
        self.l = l
        self.k = k
        self.m = m
        self.state_arr = deque([int(x * self.m) for x in seed_arr[-k:]], maxlen=k)

    def next(self):
        state = (self.state_arr[-self.l] + self.state_arr[-self.k]) % self.m
        self.state_arr.append(state)
        return state / self.m

    def generate_float(self, n):
        return [self.next() for _ in range(n)]

    def generate(self, n, low, high):
        return [low + int(self.next() * (high - low)) for _ in range(n)]


# Отображение графиков распределения значений
def visualisation(stat, low, high, bins=None):
    if bins is None:
        bins = high - low

    plt.hist(stat, bins=bins, range=(low, high), edgecolor='black')
    plt.title("Распределение чисел")
    plt.xlabel("Значения")
    plt.ylabel("Частота")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Алгоритм выбора n случайных чисел из N членов последовательности
def sample_by_acceptance(seq, n):
    N = len(seq)
    selected = []
    t = 0  # просмотрено
    m = 0  # отобрано
    lcg = LCG()

    while m < n:
        u = lcg.next()
        if (N - t) * u > (n - m):
            t += 1
        else:
            selected.append(seq[t])
            t += 1
            m += 1
    return selected


# Алгоритм (резервуар выборки) выбор случайных n чисел из последовательности неизвестной длины
def reservoir_sample(seq, n):
    V = seq[:n]  # начальные n элементов
    gen = LCG()

    for i in range(n, len(seq)):
        j = int(gen.next() * (i + 1))  # случайный индекс от 0 до i
        if j < n:
            V[j] = seq[i]
    return V


# Тест Хи^2
def chi_squared_test(data):
    n = len(data)
    bins = ceil(log2(n) + 1)
    print(bins)
    expected = n // bins
    observed = [0] * bins

    for value in data:
        index = min(int(value * bins), bins - 1)  # чтобы не выйти за границы
        observed[index] += 1

    chi2_stat = sum((o - expected) ** 2 / expected for o in observed)
    df = bins - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)  # Получаем уровень значимости по значению Хи^2
    return chi2_stat, p_value


# print("Проверка параметров для линейного конгруэнтного генератора:\na = 1103515245, c = 12345, N = 2^32")
# check_parameters(1 << 32, 1103515245, 12345)

lcg = LCG()
lcg_seq = lcg.generate_float(100000)

seed_arr = lcg.generate_float(55)
lfg = LFG(seed_arr)
lfg_seq = lfg.generate_float(100000)

sample1 = reservoir_sample(lfg_seq, 10000)
sample2 = sample_by_acceptance(lfg_seq, 10000)


# lcg_val = lcg.generate(100000, 0, 64)
# lfg_val = lfg.generate(1000000, 0, 64)
# visualisation(lcg_val, 0, 64)
# visualisation(lfg_val, 0, 64)
#

_, p = chi_squared_test(lcg_seq)
print(f"Тест Хи^2: p-value = {p:.4f} для ЛКГ")
_, p = chi_squared_test(lfg_seq)
print(f"Тест Хи^2: p-value = {p:.4f} для генератор Фибоначчи с запаздыванием")
_, p = chi_squared_test(sample1)
print(f"Тест Хи^2: p-value = {p:.4f} для выборки по первому алгоритму")
_, p = chi_squared_test(sample2)
print(f"Тест Хи^2: p-value = {p:.4f} для выборки по второму алгоритму")
#
#
_, p = kstest(lcg_seq, 'uniform')
print(f"K-S тест: p-value = {p:.4f} для ЛКГ")  # p > 0.05 ⇒ скорее равномерно
_, p = kstest(lfg_seq, 'uniform')
print(f"K-S тест: p-value = {p:.4f} для генератор Фибоначчи с запаздыванием")  # p > 0.05 ⇒ скорее равномерно
_, p = kstest(sample1, 'uniform')
print(f"K-S тест: p-value = {p:.4f} для выборки по первому алгоритму")
_, p = kstest(sample2, 'uniform')
print(f"K-S тест: p-value = {p:.4f} для выборки по второму алгоритму")
