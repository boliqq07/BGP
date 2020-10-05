import math
import random

import numpy as np
import sympy

from bgp.base import SymbolSet, SymbolTree
from bgp.calculation.translate import general_expr_dict, compile_context

if __name__ == "__main__":
    x = np.full((10, 4), fill_value=0.1)
    y = np.ones((10,))

    height = 2
    hbgp = 1
    # bgp
    group = 2
    pset = SymbolSet()
    pset.add_features(x, y, x_group=group, )
    pset.add_accumulative_operation(categories=("MAdd", "MMul", "MSub", "MDiv", "Conv", "Self"),
                                    special_prob={"MAdd": 0.16, "MMul": 0.16, "MSub": 0.16,
                                                  "MDiv": 0.16, "Conv": 0.16, "Self": 0.16})
    pset.add_operations(categories=("Add", "Mul", "Sub", "Div"))

    s = pset.free_symbol[1]
    ss = []
    for si in s:
        if isinstance(si, sympy.Symbol):
            ss.append(si)
        else:
            ss.extend(si)

    target = (ss[0] + ss[1]) * (ss[2] - ss[3])
    target = sympy.simplify(target)
    # a = time.time()
    random.seed(4)
    population = [SymbolTree.genFull(pset, int(height - 1), int(height - 1) + 1) for _ in range(5000)]
    for n, i in enumerate(population):
        i = compile_context(i, pset.context, pset.gro_ter_con, simplify=True)
        expr = general_expr_dict(i, pset.expr_init_map, pset.free_symbol,
                                 pset.gsym_map, simplifying=True)
        # print(expr)
        if expr == target:
            print(n)
            break

    ####GP
    pset = SymbolSet()
    pset.add_features(x, y)
    pset.add_operations(categories=("Add", "Mul", "Sub", "Div"))

    s = pset.free_symbol[1]
    ss = s

    target = (ss[0] + ss[1]) * (ss[2] - ss[3])
    target = sympy.simplify(target)
    # a = time.time()
    random.seed(0)
    height = int(height - 1) + math.ceil(np.log2(group))
    population = [SymbolTree.genFull(pset, height, height + 1, ) for _ in range(5000)]
    for n, i in enumerate(population):
        i = compile_context(i, pset.context, pset.gro_ter_con, simplify=True)
        expr = general_expr_dict(i, pset.expr_init_map, pset.free_symbol,
                                 pset.gsym_map, simplifying=True)
        # print(expr)
        if expr == target:
            print(n)
            break
