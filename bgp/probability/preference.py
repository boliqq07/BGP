#!/usr/bin/python
# coding:utf-8

# @author: wangchangxin
# @contact: 986798607@qq.com
# @software: PyCharm
# @file: preference.py
# @time: 2020/5/26 19:50

from itertools import combinations

import numpy as np
from numpy import random
from numpy.core import numeric


class PreMap(numeric.ndarray):
    """2D probability map"""

    def __new__(cls, data):

        assert isinstance(data, numeric.ndarray)
        dtype = np.float32
        arr = numeric.array(data, dtype=dtype, copy=True)
        shape = arr.shape
        ret = numeric.ndarray.__new__(cls, shape, dtype=np.float32,
                                      buffer=arr,
                                      order='c')
        return ret

    @classmethod
    def from_shape(cls, shape):
        """
        Generation.

        Parameters
        ----------
        shape:int
            shape of premap.

        Returns
        -------
        PreMap
        """
        shape = (shape, shape)
        ret = numeric.ndarray.__new__(cls, shape, dtype=np.float16)
        val = np.full(shape, (shape[0] - 0.01) / (shape[0] * (shape[0] - 1)), dtype=np.float16)
        ret[:] = val
        for i in range(ret.shape[0]):
            ret[i, i] = 0.01 / shape[0]
        return cls(ret)

    def down_other_point(self, *sv):

        """
        Use for binding.rate the others and add the subbed value to the [index1,index2]
        the rate are [0,1).
            
        Parameters
        ----------
        sv:[index1,index2,rate]
            site to set value

        """
        e = self.copy()
        a = sv[0]
        b = sv[1]
        c = sv[2]
        if c == 1:
            c -= 0.0001

        self[(a, b), :] *= (1 - c)
        self[:, (a, b)] *= (1 - c)
        self[a, a] /= (1 - c)
        self[b, b] /= (1 - c)
        self[a, b] /= (1 - c) ** 2
        self[b, a] /= (1 - c) ** 2
        nv = e - self
        self[a, b] = (np.sum(nv) + nv[a, a] + nv[b, b]) / 4 + e[a, b]
        self[b, a] = self[a, b]

        decend = np.sum(nv[(a, b), :], axis=0)
        ratio = decend / np.sum(e, axis=0)

        e_add = e * (1 + ratio)
        e_add[(a, b), :] = self[(a, b), :]
        e_add[:, (a, b)] = self[:, (a, b)]

        self[:] = e_add

    def set_sigle_point(self, *sv):
        """
        set the value of [index1,index2]
        the rate are [0,1)
            
        Parameters
        ----------
        sv:[index1,index2,rate]
            site to set value


        """
        a = sv[0]
        b = sv[1]
        c = sv[2]
        if c == 1:
            c -= 0.001
        self[a, b] = c
        self[b, a] = c

    def set_ratio(self, *sv):
        """
        Rate the [index1,index2] to sum and add the subbed value to the others.
        under check.

        Parameters
        ----------
        sv:[index1,index2,rate]
            rate in [0,n), if [0,1) down, if [1,n) up.
        """

        a = sv[0]
        b = sv[1]
        c = sv[2]
        if c == 1:
            c -= 0.001

        suma = np.sum(self[a])
        sumb = np.sum(self[b])
        st = c * (suma + sumb) / 2
        vale = st - self[a, b]
        base = (suma + sumb - 2 * self[a, b]) / 2
        coef = 1 - vale / base

        self[(a, b), :] *= coef
        self[:, (a, b)] *= coef
        self[a, a] /= coef
        self[b, b] /= coef

        self[a, b] = st
        self[b, a] = st

    def set_ratio_point(self, *sv):
        """
        Rate the [index1,index2] to self and add the subbed value to the others.
        under check.

        Parameters
        ----------
        sv:[index1,index2,rate]\n
            in [0,n)\n
            [0,1) down,\n
            [1,n) up\n
        """

        a = sv[0]
        b = sv[1]
        c = sv[2]
        if c == 1:
            c -= 0.00001

        st = c * self[a, b]
        vale = (1 - c) * self[a, b]
        base = (np.sum(self[a]) + np.sum(self[b]) - 2 * self[a, b]) / 2
        coef = 1 + vale / base

        self[(a, b), :] *= coef
        self[:, (a, b)] *= coef
        self[a, a] /= coef
        self[b, b] /= coef

        self[a, b] = st
        self[b, a] = st

    def noise(self):
        """add noise with 1% scale"""
        ran = random.random_sample(size=self.shape)
        rant = ran.T
        noise = ran + rant - 1
        self[:] = self * (1 + noise * 0.1)

    def add_new(self):
        """add new features to self"""
        npself = self
        ave = npself.mean(axis=0)
        se = np.concatenate((npself, ave.reshape(-1, 1)), axis=1)
        ave = np.append(ave, 0.0)
        se = np.concatenate((se, ave.reshape(1, -1)), axis=0)
        se *= (1 - 1 / se.shape[0])

        return PreMap(se)

    def update(self, ind, pset, ratio=0.5):
        """

        Parameters
        ----------
        ind: SymbolTree
            individual
        pset: SymbolSet
            SymbolSet
        ratio: float [0,1]
            change ratio
            
        Returns
        -------
        self
        """
        ratio = ratio / self.shape[0]
        ters = pset.terminals_and_constants
        pri = [prii for prii in ind if prii.arity == 0]
        indexs = [ters.index(prii) for prii in pri]
        iters = combinations(indexs, 2)
        se = self.copy()
        [self.down_other_point(i, j, ratio) for i, j in iters]
        self[:] += se
        self[:] /= 2
        self.noise()

    def get_indexes_value(self, indexes, weight=None):
        """

        Parameters
        ----------
        indexes:tuple, indexes
            get the value of average of indexes affect
        weight: None, tuple,np.ndarray
            the same size with self.shape[0]
            
        Returns
        -------
        probability list
        """
        if isinstance(indexes, int):
            indexes = (indexes,)
        indexes = tuple(indexes)
        var = self[indexes, :]
        if var.ndim == 2:
            if weight is not None:
                weight = np.array(weight)
                assert var.shape[0] == weight.shape[0]
            var = np.average(var, weights=weight, axis=0)

        return var / (np.sum(var))

    def get_ind_value(self, ind, pset):
        """
        get the value according to ind

        Parameters
        ----------
        ind: SymbolTree
        pset:SymbolSet
        
        Returns
        ----------
        probability list
        """
        ters = pset.terminals_and_constants
        pri = [prii for prii in ind if prii.arity == 0]
        if len(pri) >= 1:
            indexes = [ters.index(i) for i in pri]

            return self.get_indexes_value(indexes, weight=None)
        else:
            return None

    def get_one_node_value(self, ind=None, pset=None, node=None, site=None, ):
        """
        get affect value except site node.
        
        Parameters
        ----------
        ind: SymbolTree
        pset:SymbolSet
        node:Terminals
        site:site of Terminals

        Returns
        -------
        probability list
        """
        if not node:
            node = ind[site]

        ters = pset.terminals_and_constants

        pri = [prii for prii in ind if prii.arity == 0]
        if len(pri) > 1:
            weight = list(range(len(pri)))

            if site:
                ast = len([k for k, j in enumerate(ind[:site]) if node == j])
            else:
                ast = 0
            loc = [k for k, j in enumerate(pri) if node == j][ast]  # choice first 0ne

            weight = np.array([len(weight) - abs(i - loc) for i in weight])
            weight = weight / weight.sum(keepdims=True)
            indexes = [ters.index(prii) for prii in pri]
            indexes = np.delete(indexes, loc)
            weight = np.delete(weight, loc)

            prob = self.get_indexes_value(indexes, weight=weight)
        else:
            prob = None
        return prob

    def get_nodes_value(self, ind=None, pset=None, node=None, site=None, ):
        """
        get affect value except sites nodes.
        
        Parameters
        ----------
        ind: SymbolTree
        pset:SymbolSet
        node:Terminals
        site:site of Terminals

        Returns
        -------
        probability list
        """
        if isinstance(site, int):
            site = (site,)
        if isinstance(node, int):
            node = (node,)

        if not node:
            node = [ind[i] for i in site]

        ters = pset.terminals_and_constants

        pri = [prii for prii in ind if prii.arity == 0]
        if len(pri) > 1:
            weight = list(range(len(pri)))
            if site is not None:
                ast = [len([k for k, j in enumerate(ind[:sitei]) if nodei == j]) for nodei, sitei in zip(node, site)]
            else:
                ast = [0] * len(node)
            locs = [[k for k, j in enumerate(pri) if nodei == j][asti] for nodei, asti in
                    zip(node, ast)]  # choice first 0ne

            weight = np.array([[len(weight) - abs(i - loci) for i in weight] for loci in locs])

            if weight.ndim == 1:
                weight = weight / weight.sum()
            elif weight.ndim == 2:
                weight = weight.mean(axis=0)
            else:
                pass
            indexes = [ters.index(prii) for prii in pri]
            indexes = np.delete(indexes, locs)
            weight = np.delete(weight, locs)

            prob = self.get_indexes_value(indexes, weight=weight)
        else:
            prob = None
        return prob

# if __name__ == "__main__":
#    # import copy
#    # from bgp.gp import mutNodeReplacementVerbose, mutDifferentReplacementVerbose
#    # from numpy import random
#    # from sklearn.datasets import load_boston
#    #
#    # from bgp.base import SymbolSet
#    # from bgp.base import SymbolTree
#    # from bgp.dim import dless, Dim
#    #
#    # random.seed(3)
#    # # data
#    # data = load_boston()
#    # x = data["data"]
#    # y = data["target"]
#    # c = [6, 3, 4]
#    # # unit
#    # from sympy.physics.units import kg
#    #
#    # x_u = [kg] * 13
#    # y_u = kg
#    # c_u = [dless, dless, dless]
#    #
#    # x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
#    # y, y_dim = Dim.convert_xi(y, y_u)
#    # c, c_dim = Dim.convert_x(c, c_u)
#    #
#    # # symbolset
#    # pset0 = SymbolSet()
#    # pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, group=[[1, 2], [4, 5]])
#    # pset0.add_constants(c, dim=c_dim, prob=None)
#    # pset0.add_operations(power_categories=(2, 3, 0.5),
#    #                      categories=("Add", "Mul", "Neg", "Abs"),
#    #                      self_categories=None)
#    # pset0.personal_maps([[0,1,0.9]])
#    # pset0.personal_maps([[0,5,0.7]])
#    # pp = np.array(pset0.premap)
#    # for i in range(100):
#    #     a = SymbolTree.genGrow(pset0, 2, 4, per=True)
#    #     mutNodeReplacementVerbose(a, pset=pset0)
#    #     mutDifferentReplacementVerbose(a, pset=pset0)
#    # print(np.sum(pset0.premap))
#    # pset0.personal_preference([[3, 4, 0.8]])
#    # print(np.sum(pset0.premap))
#    # pset0.add_tree_to_features(a)
#    # print(np.sum(pset0.premap))
#    # pset0.premap.noise()
#    # print(np.sum(pset0.premap))
#    # values = pset0.premap.get_indexes_value(4)
#    # values2 = pset0.premap.get_ind_value(a, pset0)
#    premap = PreMap.from_shape(10)
#    premap.down_other_point(*[0, 1, 1])
#    premap.down_other_point(*[0, 2, 0.50])
#    premap.down_other_point(*[0, 3, 0.33])
#    premap.down_other_point(*[0, 4, 0.25])
#    # premap.down_other_point(*[0, 3, 0.5])
#    # premap.set_ratio(*[0, 1, 1])
#    # premap.set_ratio(*[0, 3, 0.5])
#    # premap.set_ratio(*[0, 5, 0.5])
#    # premap.down_others(*[0, 2, 0.25])
#    pp = np.array(premap)
#
#    sums1 = np.sum(premap, axis=0)
#    sums2 = np.sum(premap, axis=1)
