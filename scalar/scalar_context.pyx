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


class k_poleData(object):
    """
    poleData(type, k, a, b, context):
    A class containing the information about the poles and residues of
    chiral conformal blockm k_{\\beta, \\Delta_{1, 2}, \\Delta_{3, 4}} w.r.t. \\beta.

    k, ell has the same meaning, epsilon (here) = \\nu (there)
    a = - \\Delta_{12} / 2
    b = + \\Delta_{34} / 2
    """

    def __init__(self, k, a, b, context):
        self.k = k
        self.a = context(a)
        self.b = context(b)
        self.context = context

    def S(self):
        return self.a + self.b

    def P(self):
        return 2 * self.a * self.b

    def descendant_level(self):
        return self.k

    def polePosition(self):
        return self.context(1 - self.k) / 2

    def residueDelta(self):
        return self.context(1 + self.k) / 2

    def coeff(self):
        local_sign = -1 if self.k % 2 != 0 else 1
        p1 = self.context.pochhammer(1, self.k)
        p2 = self.context.pochhammer((1 - self.k + 2 * self.a) / 2, self.k)
        p3 = self.context.pochhammer((1 - self.k + 2 * self.b) / 2, self.k)
        return (-local_sign * self.k / (2 * p1 ** 2)) * p2 * p3

    def residue_of_h(self):
        return self.coeff() * self.context.chiral_h_times_rho_to_n(
            self.descendant_level(), self.residueDelta(),
            -2 * self.a, 2 * self.b)


class k_rational_approx_data(object):
    """
    rational_aprrox_data(self, cutoff, epsilon, ell, Delta_1_2=0, Delta_3_4=0, is_correlator_multiple=True, approximate_poles=0)
    computes and holds rational approximation of conformal block datum.
    """

    def __init__(self, context, cutoff, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles):
        self.cutoff = cutoff
        self.a = -context(Delta_1_2) / 2
        self.b = context(Delta_3_4) / 2
        self.S = self.a + self.b
        self.P = 2 * self.a * self.b
        self.context = context
        self.approx_poles = []
        if is_correlator_multiple:
            self.poles = [k_poleData(x, self.a, self.b, context)
                          for x in range(1, cutoff + 1)]
        else:
            self.poles = [k_poleData(x, self.a, self.b, context)
                          for x in range(2, cutoff + 2, 2)]
        if approximate_poles:
            unitarity_bound = context(0.5) ** 10
            dim_approx_base = len(self.poles)
            self.approx_column = lambda x: [
                1 / (unitarity_bound - x.polePosition()) ** i
                for i in range(1, dim_approx_base // 2 + 2)
            ] + [
                x.polePosition() ** i
                for i in range((dim_approx_base + 1) // 2 - 1)
            ]
            self.approx_matrix = matrix(map(self.approx_column, self.poles))
            if is_correlator_multiple:
                self.approx_poles = [
                    k_poleData(x, self.a, self.b, context)
                    for x in range(cutoff + 1, 2 * cutoff + 1)]
            else:
                self.approx_poles = [
                    k_poleData(x, self.a, self.b, context)
                    for x in range(2 * (cutoff // 2) + 2, 2 * cutoff + 2, 2)]

    def get_poles(self):
        return Counter(x.polePosition() for x in self.poles)

    def prefactor(self):
        return damped_rational(
            self.get_poles(), 4 * self.context.rho, 1, self.context)

    def approx_chiral_h(self):
        res = self.context.chiral_h_asymptotic(self.S) * reduce(
            lambda y, w: y * w,
            [(self.context.Delta - x.polePosition()) for x in self.poles])
        _polys = [
            reduce(
                lambda y, z: y * z,
                [
                    self.context.Delta - w.polePosition()
                    for w in self.poles if w != x])
            for x in self.poles]
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

    def approx_k(self):
        return self.context.univariate_func_prod(
            self.context.rho_to_delta, self.approx_chiral_h())


class rational_approx_data_generic_dim(object):
    """
    rational_aprrox_data(self, cutoff, epsilon, ell, Delta_1_2=0, Delta_3_4=0, approximate_poles=True)
    computes and holds rational approximation of conformal block datum.
    """

    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles):
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
        if approximate_poles:
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


class g_rational_approx_data_two_d(object):
    """
    We use a different class for d = 2,
    utilizing the exact formula by Dolan and Osborn.
    The usage is similar to .
    """

    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles=True):
        self.cutoff = cutoff
        self.ell = ell
        self.a = -context(Delta_1_2) / 2
        self.b = context(Delta_3_4) / 2
        self.S = self.a + self.b
        self.P = 2 * self.a * self.b
        self.context = context
        self.chiral_approx_data = k_rational_approx_data(
            context, cutoff, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def prefactor(self):
        __chiral_poles = set(self.chiral_approx_data.get_poles())
        __q = [2 * x - self.ell for x in __chiral_poles] + \
              [2 * x + self.ell for x in __chiral_poles]
        return damped_rational(
            Counter(__q),
            4 * self.context.rho,
            self.context(4) ** len(__chiral_poles),
            self.context)

    def approx_g(self):
        __chiral_block = self.chiral_approx_data.approx_k()
        left_contribution = [x((self.context.Delta + self.ell) / 2)
                             for x in __chiral_block]
        right_contribution = [x((self.context.Delta - self.ell) / 2)
                              for x in __chiral_block]
        __zz_res = []
        for i in range(self.context.Lambda + 1):
            for j in range(i, self.context.Lambda - i + 1):
                __zz_res.append(
                    (left_contribution[i] * right_contribution[j] +
                     left_contribution[j] * right_contribution[i]) / 2)
        return self.context.zzbar_to_xy_marix.dot(np.array(__zz_res))

    def approx(self):
        pref = self.prefactor()
        body = self.approx_g()
        return prefactor_numerator(pref, body, self.context)


class g_rational_approx_data_four_d(object):
    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles=True):
        self.cutoff = cutoff
        self.ell = ell
        self.a = -context(Delta_1_2) / 2
        self.b = context(Delta_3_4) / 2
        self.S = self.a + self.b
        self.P = 2 * self.a * self.b
        self.context = context
        self.chiral_approx_data = k_rational_approx_data(
            context.k_context, cutoff, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def prefactor(self):
        __chiral_poles = set(self.chiral_approx_data.get_poles())
        __q = [2 * x - self.ell for x in __chiral_poles] + \
              [2 * x + self.ell + 2 for x in __chiral_poles]
        return damped_rational(
            Counter(__q),
            4 * self.context.rho,
            self.context(4) ** len(__chiral_poles) /
            (4 * self.context.rho * self.context(self.ell + 1)),
            self.context)

    def approx_g(self):
        __chiral_block = self.chiral_approx_data.approx_k()
        __chiral_block_with_z = self.context(
            0.5) * __chiral_block + np.insert(__chiral_block[:-1], 0, 0)
        # z-multiply!!!!
        left_contribution = [x((self.context.Delta + self.ell) / 2)
                             for x in __chiral_block_with_z]
        right_contribution = [x((self.context.Delta - self.ell - 2) / 2)
                              for x in __chiral_block_with_z]
        __zz_res = []
        for i in range(self.context.Lambda // 2 + 2):
            for j in range(i + 1, self.context.Lambda - i + 2):
                __zz_res.append(
                    (-left_contribution[i] * right_contribution[j] +
                     left_contribution[j] * right_contribution[i]))
        return self.context.zzbar_anti_symm_to_xy_matrix.dot(
            np.array(__zz_res))

    def approx(self):
        pref = self.prefactor()
        body = self.approx_g()
        return prefactor_numerator(pref, body, self.context)


def zzbar_anti_symm_to_xy_matrix(Lambda, field=RealField(400)):
    # type: (int, RealField_class) -> np.ndarray
    q = ZZ[str('x')]
    dimG = get_dimG(Lambda)
    result = np.full((dimG, dimG), field(0))
    qplus = q(str('x + 1'))
    qminus = q(str('x - 1'))
    for i in range(Lambda // 2 + 2):
        for j in range(i + 1, Lambda + 2 - i):
            coeff = ((qplus ** j) * (qminus ** i) -
                     (qminus ** j) * (qplus ** i)).padded_list()
            column_position = (Lambda + 2 - i) * i + (j - i - 1)
            parity = (i + j + 1) % 2
            xypositions = [
                (Lambda + 2 - (i + j - x - 1) // 2) * (i + j - x - 1) // 2 + x
                for x in range(parity, len(coeff), 2)]
            for p, c in zip(xypositions, coeff[parity::2]):
                result[column_position][int(p)] = field(c) / 2
    return result.transpose()


def context_for_scalar(epsilon=0.5, Lambda=15, Prec=800, nMax=250):
    if not is_integer(epsilon):
        return scalar_cb_context_generic(Lambda, Prec, nMax, epsilon)
    epsilon = Integer(epsilon)
    if epsilon == 0:
        return scalar_cb_2d_context(Lambda, Prec, nMax)
    if epsilon == 1:
        return scalar_cb_4d_context(Lambda, Prec, nMax)
    raise RuntimeError(
        "Sorry, space-time dimensions d={0} is unsupported. Create it yourself and let me know!".format(
            2 + 2 * epsilon))


@cython.cclass
class scalar_cb_context_generic(cb_universal_context):
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

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0,
                             is_correlator_multiple=False,
                             approximate_poles=True):
        return rational_approx_data_generic_dim(
            self, cutoff, ell, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def approx_cb(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0,
                  include_odd=False, approximate_poles=True):
        if not include_odd:
            if Delta_1_2 != 0 or Delta_3_4 != 0:
                include_odd = True
        g = self.rational_approx_data(
            cutoff, ell, Delta_1_2, Delta_3_4,
            is_correlator_multiple=include_odd,
            approximate_poles=approximate_poles).approx()
        return g

    def __str__(self):
        return "Conformal bootstrap scalar context in " \
            "{0} dimensional space-time with " \
            "Lambda = {1}, precision = {2}, nMax = {3}".format(
                float(2 + 2 * self.epsilon),
                self.Lambda, self.precision, self.maxExpansionOrder)

    def __repr__(self):
        return "scalar_cb_context_generic" \
            "(Lambda={0}, Prec={1}, nMax={2}, epsilon={3})".format(
                repr(self.Lambda), repr(self.precision),
                repr(self.maxExpansionOrder), repr(self.epsilon))


@cython.cclass
class scalar_cb_2d_context(scalar_cb_context_generic):

    @cython.locals(Lambda=cython.int, Prec=mpfr_prec_t, nMax=long)
    def __init__(self, Lambda, Prec, nMax):
        scalar_cb_context_generic.__init__(self, Lambda, Prec, nMax, 0)

    @cython.ccall
    @cython.locals(array="mpfr_t*")
    def chiral_h_asymptotic(self, S):
        S_c = self(S)
        array = chiral_h_asymptotic_c(
            cython.cast(RealNumber, S_c).value, self.c_context)
        return mpfr_move_to_ndarray_1(array, self.Lambda + 1, self.field)

    @cython.ccall
    @cython.locals(n=long, array="mpfr_t*")
    def chiral_h_times_rho_to_n(self, n, h, Delta_1_2=0, Delta_3_4=0):
        S_c = self(-Delta_1_2 + Delta_3_4) / 2
        P_c = self(-Delta_1_2 * Delta_3_4) / 2
        h_c = self(h)
        sig_on()
        array = chiral_h_times_rho_to_n_c(
            cython.cast("unsigned long", n),
            cython.cast(RealNumber, h_c).value,
            cython.cast(RealNumber, S_c).value,
            cython.cast(RealNumber, P_c).value,
            self.c_context)
        sig_off()
        return mpfr_move_to_ndarray_1(array, self.Lambda + 1, self.field)

    @cython.ccall
    @cython.locals(array="mpfr_t*")
    def k_table(self, h, Delta_1_2, Delta_3_4):
        S_c = self(-Delta_1_2 + Delta_3_4) / 2
        P_c = self(-Delta_1_2 * Delta_3_4) / 2
        h_c = self(h)
        array = k_table_c(
            cython.cast(RealNumber, h_c).value,
            cython.cast(RealNumber, S_c).value,
            cython.cast(RealNumber, P_c).value,
            self.c_context)
        return mpfr_move_to_ndarray_1(array, self.Lambda + 1, self.field)

    def k_rational_approx_data(self, cutoff, Delta_1_2=0, Delta_3_4=0,
                               is_correlator_multiple=False,
                               approximate_poles=True):
        return k_rational_approx_data(
            self, cutoff, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0,
                             is_correlator_multiple=True,
                             approximate_poles=True):
        return g_rational_approx_data_two_d(
            self, cutoff, ell, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def __str__(self):
        return "Conformal bootstrap scalar context in " \
            "two dimensional space-time with " \
            "Lambda = {0}, precision = {1}, nMax = {2}".format(
                self.Lambda, self.precision, self.maxExpansionOrder)

    def __repr__(self):
        return "scalar_cb_2d_context" \
            "(Lambda={0}, Prec={1}, nMax={2})".format(
                repr(self.Lambda), repr(self.precision),
                repr(self.maxExpansionOrder))


@cython.cclass
class scalar_cb_4d_context(scalar_cb_context_generic):

    @cython.locals(Lambda=cython.int, Prec=mpfr_prec_t, nMax=long)
    def __init__(self, Lambda, Prec, nMax):
        scalar_cb_context_generic.__init__(self, Lambda, Prec, nMax, 1)
        self.k_context = scalar_cb_2d_context(Lambda + 1, Prec, nMax)
        self.epsilon = self(1)
        self.zzbar_anti_symm_to_xy_matrix = zzbar_anti_symm_to_xy_matrix(
            Lambda, field=self.field)

    def chiral_h_asymptotic(self, S):
        return self.k_context.chiral_h_asymptotic(S)

    @cython.ccall
    @cython.locals(n=long)
    def chiral_h_times_rho_to_n(self, n, h, Delta_1_2=0, Delta_3_4=0):
        return self.k_context.chiral_h_times_rho_to_n(
            n, h, Delta_1_2=Delta_1_2, Delta_3_4=Delta_3_4)

    def k_table(self, h, Delta_1_2, Delta_3_4):
        return self.k_context.k_table(h, Delta_1_2, Delta_3_4)

    def k_rational_approx_data(self, cutoff, Delta_1_2=0, Delta_3_4=0,
                               is_correlator_multiple=True,
                               approximate_poles=True):
        return self.k_context.k_rational_approx_data(
            cutoff, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles=approximate_poles)

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0,
                             is_correlator_multiple=True,
                             approximate_poles=True):
        return g_rational_approx_data_four_d(
            self, cutoff, ell, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def __str__(self):
        return "Conformal bootstrap scalar context in " \
            "four dimensional space-time with " \
            "Lambda = {0}, precision = {1}, nMax = {2}".format(
                self.Lambda, self.precision, self.maxExpansionOrder)

    def __repr__(self):
        return "scalar_cb_4d_context" \
            "(Lambda={0}, Prec={1}, nMax={2})".format(
                repr(self.Lambda), repr(self.precision),
                repr(self.maxExpansionOrder))
