import unittest

import sympy
from sympy.core.numbers import nan
from sympy.core.numbers import oo as Inf

from bgp.functions.gsymfunc import NewArray
from bgp.functions.gsymfunc import gsym_map as sym_map


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.a = sympy.Symbol("a") * sympy.Symbol("a")
        self.nan = nan
        self.n = NewArray([1.0, 1, Inf])
        self.x = sympy.Symbol("x")
        self.k = sympy.Symbol("k")
        self.y = sympy.Symbol("y")
        self.z = sympy.Symbol("z")
        self.arr3 = NewArray([self.y, self.z, self.x])
        self.arr3_ = NewArray([self.x, self.z, self.k])
        self.arr3_NaN = NewArray([self.x, self.z, nan])
        self.arr4 = NewArray([self.x, self.y, self.z, self.k])

    def test_mul(self):
        self.assertEqual(self.a * self.arr3, NewArray([self.y * self.a, self.z * self.a, self.x * self.a]))
        self.assertEqual(self.arr3 * self.a, NewArray([self.y * self.a, self.z * self.a, self.x * self.a]))

        arrs = NewArray(
            [i * j for i, j in zip(self.arr3_, self.arr3)]
        )
        self.assertEqual(self.arr3_ * self.arr3, arrs)

        self.assertNotEqual(self.arr3_NaN * self.arr3, arrs)

        arrs2 = NewArray(
            [i * j for i, j in zip(self.n, self.arr3)]
        )
        self.assertEqual(self.n * self.arr3, arrs2)

    def test_add(self):
        self.assertEqual(self.a + self.arr3, NewArray([self.y + self.a, self.z + self.a, self.x + self.a]))
        self.assertEqual(self.arr3 + self.a, NewArray([self.y + self.a, self.z + self.a, self.x + self.a]))

        arrs = NewArray(
            [i + j for i, j in zip(self.arr3_, self.arr3)]
        )
        self.assertEqual(self.arr3_ + self.arr3, arrs)

        self.assertNotEqual(self.arr3_NaN + self.arr3, arrs)

        arrs2 = NewArray(
            [i + j for i, j in zip(self.n, self.arr3)]
        )
        self.assertEqual(self.arr3 + self.n, arrs2)

    def test_sub(self):
        self.assertEqual(self.arr3 - self.a, NewArray([self.y - self.a, self.z - self.a, self.x - self.a]))

        arrs = NewArray(
            [i - j for i, j in zip(self.arr3_, self.arr3)]
        )
        self.assertEqual(self.arr3_ - self.arr3, arrs)

        self.assertNotEqual(self.arr3_NaN - self.arr3, arrs)

        arrs2 = NewArray(
            [i - j for i, j in zip(self.n, self.arr3)]
        )
        self.assertEqual(self.n - self.arr3, arrs2)

    def test_rsub(self):
        self.assertEqual(self.a - self.arr3, NewArray([self.a - self.y, self.a - self.z, self.a - self.x]))
        self.assertEqual(self.a - self.arr3, -NewArray([self.y - self.a, self.z - self.a, self.x - self.a]))

        arrs = NewArray(
            [i - j for i, j in zip(self.arr3_, self.arr3)]
        )
        self.assertEqual(self.arr3_ - self.arr3, arrs)

        self.assertNotEqual(self.arr3_NaN - self.arr3, arrs)

        arrs2 = NewArray(
            [i - j for i, j in zip(self.n, self.arr3)]
        )
        self.assertEqual(self.n - self.arr3, arrs2)

    def test_rdiv(self):
        self.assertEqual(self.a / self.arr3, NewArray([self.a / self.y, self.a / self.z, self.a / self.x]))
        # self.assertEqual(self.a / self.arr3, 1/NewArray([self.y / self.a, self.z / self.a, self.x / self.a]))

        arrs = NewArray(
            [i / j for i, j in zip(self.arr3_, self.arr3)]
        )
        self.assertEqual(self.arr3_ / self.arr3, arrs)

        self.assertNotEqual(self.arr3_NaN / self.arr3, arrs)

        arrs2 = NewArray(
            [i / j for i, j in zip(self.n, self.arr3)]
        )
        self.assertEqual(self.n / self.arr3, arrs2)

    def test_div(self):
        self.assertEqual(self.arr3 / self.a, NewArray([self.y / self.a, self.z / self.a, self.x / self.a]))
        # self.assertEqual(self.arr3/self.a, 1/NewArray([self.a/self.y ,self.a/ self.z , self.a/self.x]))

        arrs = NewArray(
            [i / j for i, j in zip(self.arr3_, self.arr3)]
        )
        self.assertEqual(self.arr3_ / self.arr3, arrs)

        self.assertNotEqual(self.arr3_NaN / self.arr3, arrs)

        arrs2 = NewArray(
            [i / j for i, j in zip(self.n, self.arr3)]
        )
        self.assertEqual(self.n / self.arr3, arrs2)

    def test_exp(self):
        f = sym_map()["exp"]
        self.assertEqual(f(self.arr3), NewArray([f(self.y), f(self.z), f(self.x)]))

    def test_sin(self):
        f = sym_map()["exp"]
        self.assertEqual(f(self.arr3), NewArray([f(self.y), f(self.z), f(self.x)]))

    def test_Flat(self):
        f = sym_map()["MAdd"]
        self.assertEqual(f(self.arr3), sum([self.y, self.z, self.x]))

    def test_comp(self):
        f = sym_map()["MMul"]
        self.assertEqual(f(self.arr3), self.y * self.z * self.x)

    def test_conv(self):
        f = sym_map()["Conv"]
        self.assertEqual(f(self.arr3), self.arr3)
        assert True == (f(NewArray([self.y, self.z])) == NewArray([self.z, self.y]))

    def test_msub(self):
        f = sym_map()["MSub"]
        self.assertEqual(f(self.arr3), self.arr3)
        self.assertEqual(f(NewArray([self.y, self.z])), self.y - self.z)

    def test_mdiv(self):
        f = sym_map()["MDiv"]
        self.assertEqual(f(self.arr3), self.arr3)
        self.assertEqual(f(NewArray([self.y, self.z])), self.y / self.z)
    #
    # def test_ppp(self):
    #     syk_map


if __name__ == '__main__':
    unittest.main()
