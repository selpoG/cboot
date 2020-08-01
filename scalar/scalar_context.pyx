from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import sys
from collections import Counter
from functools import reduce

import cython
import numpy as np
from sage.matrix.constructor import matrix
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.real_mpfr import RealField

from sage.cboot.context_object import (
    damped_rational, get_dimG, is_integer, prefactor_numerator)

if sys.version_info.major == 2:
    from future_builtins import ascii, filter, hex, map, oct, zip


@cython.cfunc
@cython.returns(cython.bint)
@cython.locals(num=RealNumber)
def RealNumber_is_zero(num):
    return mpfr_zero_p(cython.cast(RealNumber, num).value) != 0


class poleData(object):
    """
    poleData(type, k, ell, a, b, context):
    A class containing the information about the poles and residues of
    conformal block w.r.t. \\Delta.

    type = 1 or 2 or 3 refers to the rows of the Section 4, Table 1 of 1406.4858.
    k, ell has the same meaning, epsilon (here) = \\nu (there)
    a = - \\Delta_{12} / 2
    b = + \\Delta_{34} / 2
    """

    def __init__(self, type, k, ell, a, b, context):
        self.type = type
        self.ell = context(ell)
        if k > ell and type == 3:
            raise NotImplementedError
        self.k = k
        self.epsilon = context.epsilon
        self.a = context(a)
        self.b = context(b)
        self.context = context

    def S(self):
        return self.a + self.b

    def P(self):
        return 2 * self.a * self.b

    def descendant_level(self):
        if self.type == 1:
            return self.k
        if self.type == 2:
            return 2 * self.k
        if self.type == 3:
            return self.k
        raise NotImplementedError

    def polePosition(self):
        if self.type == 1:
            return self.context(1 - self.ell - self.k)
        if self.type == 2:
            return self.context(1 + self.epsilon - self.k)
        if self.type == 3:
            return self.context(1 + self.ell + 2 * self.epsilon - self.k)
        raise NotImplementedError

    def residueDelta(self):
        if self.type == 1:
            return 1 - self.ell
        if self.type == 2:
            return 1 + self.epsilon + self.k
        if self.type == 3:
            return 1 + self.ell + 2 * self.epsilon
        raise NotImplementedError

    def residueEll(self):
        if self.type == 1:
            return self.ell + self.k
        if self.type == 2:
            return self.ell
        if self.type == 3:
            return self.ell - self.k
        raise NotImplementedError

    def coeff(self):
        def poch(x, n):
            return self.context.pochhammer(x, n)

        local_sign = -1 if self.k % 2 != 0 else 1
        p0 = poch(1, self.k)
        common_factor = -local_sign * self.k / (p0 ** 2)
        if self.type == 1:
            p1 = poch(self.ell + 2 * self.epsilon, self.k)
            p2 = poch((1 - self.k + 2 * self.a) / 2, self.k)
            p3 = poch((1 - self.k + 2 * self.b) / 2, self.k)
            p4 = poch(self.ell + self.epsilon, self.k)
            return common_factor * p1 * p2 * p3 / p4
        if self.type == 2:
            p1 = poch(self.epsilon - self.k, 2 * self.k)
            p2 = poch(self.ell + self.epsilon - self.k, 2 * self.k)
            p3 = poch(self.ell + self.epsilon + 1 - self.k, 2 * self.k)
            tmp = (1 - self.k + self.ell + self.epsilon) / 2
            p4 = poch(tmp - self.a, self.k)
            p5 = poch(tmp + self.a, self.k)
            p6 = poch(tmp - self.b, self.k)
            p7 = poch(tmp + self.b, self.k)
            return common_factor * p1 / (p2 * p3) * p4 * p5 * p6 * p7
        if self.type == 3:
            if self.k > self.ell:
                raise RuntimeError(
                    "pole identifier k must be k <= ell for type 3 pole.")
            p1 = poch(self.ell + 1 - self.k, self.k)
            p2 = poch((1 - self.k + 2 * self.a) / 2, self.k)
            p3 = poch((1 - self.k + 2 * self.b) / 2, self.k)
            p4 = poch(1 + self.ell + self.epsilon - self.k, self.k)
            return common_factor * p1 * p2 * p3 / p4
        raise NotImplementedError("pole type unrecognizable.")

    def residue_of_h(self):
        return self.coeff() * self.context.h_times_rho_k(
            self.descendant_level(), self.residueEll(), self.residueDelta(),
            self.S(), self.P())


class rational_approx_data_generic_dim(object):
    """
    rational_aprrox_data(self, cutoff, epsilon, ell, Delta_1_2=0, Delta_3_4=0, approximate_poles=True)
    computes and holds rational approximation of conformal block datum.
    """

    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4):
        is_correlator_multiple = Delta_1_2 != 0 or Delta_3_4 != 0
        self.epsilon = context.epsilon
        self.ell = ell
        self.cutoff = cutoff
        self.a = -context(Delta_1_2) / 2
        self.b = context(Delta_3_4) / 2
        self.S = self.a + self.b
        self.P = 2 * self.a * self.b
        self.context = context
        self.approx_poles = []
        if is_correlator_multiple:
            self.poles = [
                poleData(1, x, ell, self.a, self.b, context)
                for x in range(1, self.cutoff + 1)
            ] + [
                poleData(2, x, ell, self.a, self.b, context)
                for x in range(1, cutoff // 2 + 1)
            ] + [
                poleData(3, x, ell, self.a, self.b, context)
                for x in range(1, min(ell, cutoff // 2) + 1)
            ]
        else:
            self.poles = [
                poleData(1, x, ell, self.a, self.b, context)
                for x in range(2, 2 * (cutoff // 2) + 2, 2)
            ] + [
                poleData(2, x, ell, self.a, self.b, context)
                for x in range(1, cutoff // 2 + 1)
            ] + [
                poleData(3, x, ell, self.a, self.b, context)
                for x in range(2, 2 * (min(ell, cutoff) // 2) + 2, 2)
            ]
        if ell == 0:
            unitarity_bound = self.epsilon
        else:
            unitarity_bound = ell + 2 * self.epsilon
        unitarity_bound += context(0.5) ** 10
        dim_approx_base = len(self.poles)
        self.approx_column = lambda x: [
            1 / (unitarity_bound - x.polePosition()) ** i
            for i in range(1, dim_approx_base // 2 + 2)
        ] + [
            x.polePosition() ** i
            for i in range((dim_approx_base + 1) // 2 - 1)
        ]
        self.approx_matrix = matrix(map(self.approx_column, self.poles))
        step = 1 if is_correlator_multiple else 2
        if is_correlator_multiple:
            start1 = cutoff + 1
            start3 = min(ell, cutoff // 2) + 1
        else:
            start1 = 2 * (cutoff // 2) + 2
            start3 = 2 * (min(ell, cutoff) // 2) + 2
        self.approx_poles = [
            poleData(1, x, ell, self.a, self.b, context)
            for x in range(start1, 2 * cutoff + 1, step)
        ] + [
            poleData(2, x, ell, self.a, self.b, context)
            for x in range(cutoff // 2 + 1, 2 * (cutoff // 2) + 1)
        ] + [
            poleData(3, x, ell, self.a, self.b, context)
            for x in range(start3, ell + 1, step)
        ]

    def prefactor(self):
        return damped_rational(
            Counter(x.polePosition() for x in self.poles),
            4 * self.context.rho, 1, self.context)

    def approx_h(self):
        res = self.context.h_asymptotic_form(self.S) * reduce(
            lambda y, w: y * w,
            [(self.context.Delta - x.polePosition()) for x in self.poles])
        _polys = [reduce(
            lambda y, z: y * z,
            [(self.context.Delta - w.polePosition())
             for w in self.poles if w != x]) for x in self.poles]
        res += reduce(
            lambda y, w: y + w,
            map(lambda x, y: x.residue_of_h() * y, self.poles, _polys))
        if self.approx_poles != []:
            approx_matrix_inv = self.approx_matrix.transpose().inverse()
            approx_target = matrix(
                map(self.approx_column, self.approx_poles)).transpose()
            approx_polys = list(
                (matrix(_polys) * approx_matrix_inv * approx_target)[0])
            for x, y in zip(self.approx_poles, approx_polys):
                res += x.residue_of_h() * y
        return res

    def approx_g(self):
        return self.context.c2_expand(
            self.context.univariate_func_prod(
                self.context.rho_to_delta,
                self.approx_h()),
            self.ell, self.context.Delta, self.S, self.P)

    def approx(self):
        pref = self.prefactor()
        body = self.approx_g()
        return prefactor_numerator(pref, body, self.context)


def context_for_scalar(epsilon=0.5, Lambda=15, Prec=800, nMax=250):
    return scalar_context(Lambda, Prec, nMax, epsilon)


@cython.cclass
class scalar_context(cb_universal_context):
    """
    Context object for the bootstrap.
    Frequently used quantities are stored here.
    """

    def __init__(self, Lambda, Prec, nMax, epsilon):
        cb_universal_context.__init__(self, Lambda, Prec, nMax)
        self.epsilon = self(epsilon)

    @cython.ccall
    @cython.locals(k="unsigned long", array="mpfr_t*")
    def h_times_rho_k(self, k, ell, Delta, S, P):
        ell_c = self(ell)
        Delta_c = self(Delta)
        S_c = self(S)
        P_c = self(P)
        sig_on()
        array = hBlock_times_rho_n(
            k,
            cython.cast(RealNumber, self.epsilon).value,
            cython.cast(RealNumber, ell_c).value,
            cython.cast(RealNumber, Delta_c).value,
            cython.cast(RealNumber, S_c).value,
            cython.cast(RealNumber, P_c).value,
            self.c_context)
        sig_off()
        return mpfr_move_to_ndarray_1(array, self.Lambda + 1, self.field)

    @cython.ccall
    @cython.locals(array="mpfr_t*")
    def h_asymptotic_form(self, S):
        S_c = self(S)
        array = h_asymptotic(
            cython.cast(RealNumber, self.epsilon).value,
            cython.cast(RealNumber, S_c).value,
            self.c_context)
        return mpfr_move_to_ndarray_1(array, self.Lambda + 1, self.field)

    @cython.ccall
    @cython.locals(array="mpfr_t*")
    def gBlock(self, ell, Delta, Delta_1_2, Delta_3_4):
        """
        gBlock(epsilon, ell, Delta, Delta_1_2, Delta_3_4, self=self):
        computes conformal block in the notation of arXiv/1305.1321
        """

        ell_c = self(ell)
        Delta_c = self(Delta)
        S_c = (self(-Delta_1_2) + self(Delta_3_4)) / 2
        P_c = self(-Delta_1_2) * self(Delta_3_4) / 2

        # In case Delta and ell = 0, return the identity_vector.
        if RealNumber_is_zero(ell_c) and RealNumber_is_zero(Delta_c):
            if RealNumber_is_zero(S_c) and RealNumber_is_zero(P_c):
                return self.identity_vector()
            raise ValueError(
                "Delta, ell = 0 while "
                "Delta_1_2 = {0} and Delta_3_4 = {1}".format(
                    Delta_1_2, Delta_3_4))

        sig_on()
        array = gBlock_full(
            cython.cast(RealNumber, self.epsilon).value,
            cython.cast(RealNumber, ell_c).value,
            cython.cast(RealNumber, Delta_c).value,
            cython.cast(RealNumber, S_c).value,
            cython.cast(RealNumber, P_c).value,
            self.c_context)
        sig_off()
        return mpfr_move_to_ndarray_1(array, get_dimG(self.Lambda), self.field)

    def c2_expand(self, array_real, ell, Delta, S, P):
        """
        c2_expand(array_real, self.epsilon, ell, Delta, S, P)
        computes y-derivatives of scalar conformal blocks
        from x-derivatives of conformal block called array_real,
        which is a (self.Lambda + 1)-dimensional array.
        """
        local_c2 = ell * (ell + 2 * self.epsilon) + \
            Delta * (Delta - 2 - 2 * self.epsilon)

        def aligned_index(y_del, x_del):
            return (self.Lambda + 2 - y_del) * y_del + x_del

        ans = np.ndarray(get_dimG(self.Lambda), dtype='O')
        ans[0:self.Lambda + 1] = array_real
        for i in range(1, (self.Lambda // 2) + 1):
            for j in range(self.Lambda - 2 * i + 1):
                val = self(0)
                common_factor = self(2 * self.epsilon + 2 * i - 1)
                if j >= 3:
                    val += ans[aligned_index(i, j - 3)] * 16 * common_factor
                if j >= 2:
                    val += ans[aligned_index(i, j - 2)] * 8 * common_factor
                if j >= 1:
                    val += -ans[aligned_index(i, j - 1)] * 4 * common_factor
                val += (4 * (2 * P + 8 * S * i + 4 * S * j - 8 * S + 2 * local_c2 + 4 * self.epsilon * i + 4 * self.epsilon *
                             j - 4 * self.epsilon + 4 * i ** 2 + 8 * i * j - 2 * i + j ** 2 - 5 * j - 2) / i) * ans[aligned_index(i - 1, j)]
                val += (-self((j + 1) * (j + 2)) / i) * \
                    ans[aligned_index(i - 1, j + 2)]
                val += (2 * (j + 1) * (2 * S + 2 * self.epsilon - 4 *
                                       i - j + 6) / i) * ans[aligned_index(i - 1, j + 1)]
                if j >= 1:
                    val += (8 * (2 * P + 8 * S * i + 2 * S * j - 10 * S - 4 * self.epsilon * i + 2 * self.epsilon * j + 2 *
                                 self.epsilon + 12 * i ** 2 + 12 * i * j - 34 * i + j ** 2 - 13 * j + 22) / i) * ans[aligned_index(i - 1, j - 1)]
                if i >= 2:
                    val += (4 * self((j + 1) * (j + 2)) / i) * \
                        ans[aligned_index(i - 2, j + 2)]
                    val += (8 * (j + 1) * self(2 * S - 2 * self.epsilon +
                                               4 * i + 3 * j - 6) / i) * ans[aligned_index(i - 2, j + 1)]

                ans[aligned_index(i, j)] = val / (2 * common_factor)
        return ans

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0):
        return rational_approx_data_generic_dim(
            self, cutoff, ell, Delta_1_2, Delta_3_4)

    def approx_cb(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0):
        return self.rational_approx_data(
            cutoff, ell, Delta_1_2, Delta_3_4).approx()

    def __str__(self):
        return "Conformal bootstrap scalar context in " \
            "{0} dimensional space-time with " \
            "Lambda = {1}, precision = {2}, nMax = {3}".format(
                float(2 + 2 * self.epsilon),
                self.Lambda, self.precision, self.maxExpansionOrder)

    def __repr__(self):
        return "scalar_context" \
            "(Lambda={0}, Prec={1}, nMax={2}, epsilon={3})".format(
                repr(self.Lambda), repr(self.precision),
                repr(self.maxExpansionOrder), repr(self.epsilon))
