import time

import sympy

x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")
m = sympy.Symbol("m")
n = sympy.Symbol("n")
d1 = sympy.Symbol("d1")
d1.copy()
a = time.time()
expss = (m - n) ** 3 + z * ((x - 1) * m + y)
b = time.time()

c = time.time()
expss3 = expss.xreplace({m: d1})
d = time.time()

print(d - c)
