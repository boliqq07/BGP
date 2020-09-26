"""
Notes: the translation process
    the three function should be the same key.
    1.
    sym_vector_map: repr of SymbolTree to sympy.Function
    sym_dispose_map: repr of SymbolTree to "group" sympy.Function.
    2.
    np_map(): repr of "group" sympy.Function to numpy function
    3.
    dim_map(): repr of "group" sympy.Function to Dim function
    4.
    gsym_map():repr of "group" sympy.Function to universal sympy.Function
"""

__all__ = ["dimfunc", "gsymfunc", "newfunc", "npfunc", "symfunc"]
