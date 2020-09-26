import unittest

import numpy as np

from bgp.probability.preference import PreMap


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.premap = PreMap.from_shape(14)

    def test_down_self(self):
        self.premap.down_other_point(*[0, 1, 0.9])
        # self.premap.down_others(*[0, 5, 0.4])
        sums1 = np.sum(self.premap, axis=0)
        sums2 = np.sum(self.premap, axis=1)
        print(sums1)
        print(sums2)
        # [self.assertEqual(str(round(sumsi,3)), str(0.895)) for sumsi in sums2]
