import numpy as np
from scipy.optimize import fsolve

def equation(r, n, S):
    # 根据给出的r (y/x)，n 和 S (x+y) 计算原方程两边的差
    x = S / (1 + r)
    y = r * x
    lhs = n * y * (y - 1)  # n*y*(y-1)
    rhs = x * (x - 1)      # x*(x-1)
    return lhs - rhs

def solve_ratio(n, S):
    # 使用fsolve求解方程，初始猜测r=1
    initial_guess = 1
    ratio = fsolve(equation, initial_guess, args=(n, S))
    return ratio[0]

# 示例：使用 n=2 和 S=10
n = 2
S = 10
ratio = solve_ratio(n, S)
print("The ratio y/x for n={} and x+y={} is approximately {:.4f}".format(n, S, ratio))


