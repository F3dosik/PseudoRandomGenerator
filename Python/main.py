from math import gcd, log2, ceil
import math
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time 
from collections import deque
import numpy as np
from scipy import stats
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
    
# print("Проверка параметров для линейного конгруэнтного генератора:\na = 1103515245, c = 12345, N = 2^32")
# check_parameters(1 << 32, 1103515245, 12345)


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

# lcg_val = lcg.generate(100000, 0, 64)
# lfg_val = lfg.generate(1000000, 0, 64)
# visualisation(lcg_val, 0, 64)
# visualisation(lfg_val, 0, 64)

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
def chi_squared_uniform_test(data):
    n = len(data)
    bins = int(np.sqrt(n))  
    expected = n / bins # Количесвто элементов в каждом интервале 
    observed = [0] * bins 

    for value in data:
        index = min(int(value * bins), bins - 1)  # чтобы не выйти за границы
        observed[index] += 1

    chi2_stat = sum((o - expected) ** 2 / expected for o in observed)
    df = bins - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)  # Получаем уровень значимости по значению Хи^2
    return chi2_stat, p_value

def kolmogorov_smirnov_uniform_test(data):
    data = np.sort(data) # Сортируем, чтобы легче считать эмпирическую функцию распределения
    N = len(data)

    # Считаем статистику Dn
    D_plus = max(i/N - data[i - 1] for i in range(1, N + 1))
    D_minus = max(data[i - 1] - (i-1)/N for i in range(1, N + 1))
    Dn = max(D_plus, D_minus)

    # Считаем p_value
    s = sum((-1)**(k-1) * math.exp(-2 * k**2 * N * Dn**2) for k in range(1, 1000))
    p_value = 2 * s

    return Dn, p_value

def f(block):
    block = list(block)
    t = len(block)
    f_cur = 0

    for r in range(t, 1, -1):
        prefix = block[:r]
        s = prefix.index(max(prefix))
        f_cur = r * f_cur + s
        block[r-1], block[s] = block[s], block[r-1]

    return f_cur

def observe(data, t = 4):
    N = len(data)
    M = N // t # кол-во блоков
    K = math.factorial(t) # t! категорий

    counts = [0] * K

    for j in range(M):
        block = data[j*t:(j+1)*t]
        idx = f(block)
        counts[idx] += 1
    return counts, M

def permutations_test(data, t=4):
    obs, M = observe(data)
    T = math.factorial(t)
    expected = M / T

    if expected < 5:
        raise ValueError("Недостаточно данных для применения χ²-критерия")
    
    chi2_stat = sum(
        (obs[i] - expected) ** 2 / expected
        for i in range(T)
    )
    df = T - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p_value

def series_test(data):
    N = len(data)
    n = N // 2 # Число непересекающихся пар
    d = int(np.sqrt(n/5))

    X = [int(d * Yi) for Yi in data]

    k = d * d 
    O = [0] * k

    for j in range(n):
        r = X[2*j]
        g = X[2*j + 1]
        idx = r * d + g
        O[idx] += 1

    E = n / k 

    chi2_stat = sum((Oi - E)**2 / E for Oi in O)
    df = k - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value


def runs_test(data):
    dirs = []
    for i in range(len(data) - 1):
        if data[i+1] > data[i]:
            dirs.append(1)
        elif data[i+1] < data[i]:
            dirs.append(-1)

    N = len(dirs)
    if N < 2:
        raise ValueError("Недостаточно данных")

    R = 1
    for i in range(1, N):
        if dirs[i] != dirs[i-1]:
            R += 1

    mean = (2*N - 1) / 3
    var = (16*N - 29) / 90

    z = (R - mean) / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value

def check_generators():
    lcg = LCG()
    seed_arr = lcg.generate_float(55)
    lfg = LFG(seed_arr)

    pKS_1, pKS_2 = [], []
    pChi_1, pChi_2 = [], []
    pPerm_1, pPerm_2 = [],[]
    pSer_1, pSer_2 = [],[]
    pRuns_1, pRuns_2 = [], []

    for i in range(1000):
        seq = lfg.generate_float(100000)

        s1 = reservoir_sample(seq, 10000)
        s2 = sample_by_acceptance(seq, 10000)
        
        pChi_1.append(chi_squared_uniform_test(s1)[1])
        pKS_1.append(kolmogorov_smirnov_uniform_test(s1)[1])
        pPerm_1.append(permutations_test(s1)[1])
        pSer_1.append(series_test(s1)[1])
        pRuns_1.append(runs_test(s1)[1])

        pChi_2.append(chi_squared_uniform_test(s2)[1])
        pKS_2.append(kolmogorov_smirnov_uniform_test(s2)[1])
        pPerm_2.append(permutations_test(s2)[1])
        pSer_2.append(series_test(s2)[1])
        pRuns_2.append(runs_test(s2)[1])

    # Конвертация в NumPy
    pChi_1 = np.array(pChi_1)
    pChi_2 = np.array(pChi_2)
    pKS_1  = np.array(pKS_1)
    pKS_2  = np.array(pKS_2)
    pPerm_1 = np.array(pPerm_1)
    pPerm_2 = np.array(pPerm_2)
    pSer_1 = np.array(pSer_1)
    pSer_2 = np.array(pSer_2)
    pRuns_1 = np.array(pRuns_1)
    pRuns_2 = np.array(pRuns_2)

    # Средние p-value
    print("Доля p_value < 0.05")
    print(f"Chi2: 1 алгоритм {np.mean(pChi_1 < 0.05)}, 2 алгоритм {np.mean(pChi_2 < 0.05)}")
    print(f"KS: 1 алгоритм {np.mean(pKS_1 < 0.05)}, 2 алгоритм {np.mean(pKS_2 < 0.05)}")
    print(f"Критерий перестановок: 1 алгоритм {np.mean(pPerm_1 < 0.05)}, 2 алгоритм {np.mean(pPerm_2 < 0.05)}")
    print(f"Критерий серий: 1 алгоритм {np.mean(pSer_1 < 0.05)}, 2 алгоритм {np.mean(pSer_2 < 0.05)}")
    print(f"Критерий монотонности: 1 алгоритм {np.mean(pRuns_1 < 0.05)}, 2 алгоритм {np.mean(pRuns_2 < 0.05)}")


# check_generators()

# Метод полярных координат
class Normal:
    def __init__(self):
        seed1 = int(time.time_ns() % (1 << 32))
        time.sleep(0.001)
        seed2 = int(time.time_ns() % (1 << 32))

        self.u1 = LCG(seed=seed1)
        self.u2 = LCG(seed=seed2)

    def next(self):
        while True:
            v1 = 2 * self.u1.next() - 1
            v2 = 2 * self.u2.next() - 1
            s = v1 * v1 + v2 * v2

            if 0 < s < 1: 
                break
        
        factor = math.sqrt((-2 * math.log(s)) / s)
        x1 = v1 * factor
        x2 = v2 * factor

        return x1, x2
    
    def generate(self, n):
        seq = []
        if n % 2 == 0:
            for i in range(n // 2):
                x1, x2 = self.next()
                seq.append(x1)
                seq.append(x2)
        else:
            for i in range((n-1)//2):
                x1, x2 = self.next()
                seq.append(x1)
                seq.append(x2)
            seq.append(self.next()[0])

        return seq


def visualisationNormal(stat, low, high, bins=50):
    plt.hist(
        stat,
        bins=bins,
        range=(low, high),
        density=True,
        edgecolor='black',
        alpha=0.6,
        label="Эмпирическая гистограмма"
    )

    x = np.linspace(low, high, 1000)
    pdf = (1 / math.sqrt(2 * math.pi)) * np.exp(-x**2 / 2)

    plt.plot(x, pdf, 'r', linewidth=2, label="N(0,1)")

    plt.title("Нормальное распределение")
    plt.xlabel("x")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def chi_squared_test_normal(data, M, sigma):
    N = len(data)
    k = 1 + math.ceil(math.log2(N))
    L = 3

    h = (2 * L * sigma) / k

    borders = [-np.inf]
    for i in range(k + 1):
        borders.append(M - L * sigma + i * h)
    borders.append(np.inf)

    observed, _ = np.histogram(data, bins=borders)

    expected = []
    for i in range(len(borders) - 1):
        p = (
            stats.norm.cdf(borders[i + 1], loc=M, scale=sigma)
            - stats.norm.cdf(borders[i], loc=M, scale=sigma)
        )
        expected.append(N * p)

    while expected[0] < 5: 
        observed[1] += observed[0]
        observed.pop(0)
        expected[1] += expected[0]
        expected.pop(0)

    while expected[-1] < 5: 
        observed[-2] += observed[-1]
        observed.pop()
        expected[-2] += expected[-1]
        expected.pop()

    chi_squared = sum(
        (observed[i] - expected[i])**2 / expected[i]
        for i in range(len(expected))
        if expected[i] > 0
    )

    df = len(expected) - 3
    p_value = 1 - stats.chi2.cdf(chi_squared, df)

    return chi_squared, p_value

def kolmogorov_smirnov_test_normal(data, M=0, sigma=1):
    data = np.sort(data)
    N = len(data)

    Fn = np.arange(1, N+1) / N
    F = stats.norm.cdf(data, loc=M, scale=sigma)

    D = np.max(np.abs(Fn-F))

    p_value = stats.kstest(data,'norm',args=(M,sigma)).pvalue

    return D, p_value

norm = Normal()
data = norm.generate(100000)
visualisationNormal(data, low=-5, high=5, bins=60)
print(f"Chi2: {chi_squared_test_normal(data, 0, 1)[1]:.4f}, KS: {kolmogorov_smirnov_test_normal(data)[1]:.4f}")



