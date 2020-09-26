# from bgp.scores import CheckCoef
# a = time.time()
# ass = CheckCoef(["a","b","c"],{"d":2,"e":3,"f":5})
# e=ass.ind
# b=time.time()
# p=[0.1]*ass.num
# c=time.time()
# p = ass.group(p)
# d = time.time()
# print(d-c,c-b,b-a)
from sympy import Symbol

x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
expr = z * (x + y)
