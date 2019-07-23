from __future__ import print_function, unicode_literals
from __future__ import division
import sys
if sys.version_info.major == 2:
    from future_builtins import ascii, filter, hex, map, oct, zip
import numpy as np
from libcpp cimport bool
from sage.cboot.context_object cimport *
from sage.cboot.scalar.scalar_context cimport *
from sage.cboot.context_object import SDP, get_dimG, format_poleinfo
from sage.all import matrix, ZZ, Integer, cached_method
from cysignals.signals cimport sig_on, sig_off
from collections import Counter
from functools import reduce

class k_poleData:
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
        self.a = context.field(a)
        self.b = context.field(b)
        self.context = context

    def S(self):
        return self.a + self.b

    def P(self):
        return 2 * self.a * self.b

    def descendant_level(self):
        return self.k

    def polePosition(self):
        return self.context.field(1 - self.k) / 2

    def residueDelta(self):
        return self.context.field(1 + self.k) / 2

    def coeff(self):
        local_sign = -1 if self.k % 2 != 0 else 1
        p1 = self.context.pochhammer(1, self.k)
        p2 = self.context.pochhammer((1 - self.k + 2 * self.a) / 2, self.k)
        p3 = self.context.pochhammer((1 - self.k + 2 * self.b) / 2, self.k)
        return (-local_sign * self.k / (2 * p1 ** 2)) * p2 * p3

    def residue_of_h(self):
        return self.coeff() * self.context.chiral_h_times_rho_to_n(
            self.descendant_level(), self.residueDelta(), -2 * self.a, 2 * self.b)


class k_rational_approx_data:
    """
    rational_aprrox_data(self, cutoff, epsilon, ell, Delta_1_2=0, Delta_3_4=0, is_correlator_multiple=True, scheme="no approx pole", cutoff_for_approximating_pole=0)
    computes and holds rational approximation of conformal block datum.
    """

    def __init__(self, context, cutoff, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles):
        self.cutoff = cutoff
        self.a = -context.field(Delta_1_2) / 2
        self.b = context.field(Delta_3_4) / 2
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
            unitarity_bound = context.field(0.5) ** 10
            dim_approx_base = len(self.poles)
            self.approx_column = lambda x: [
                1 / (unitarity_bound - x.polePosition()) ** i
                for i in range(1, dim_approx_base // 2 + 2)
            ] + [
                (x.polePosition()) ** i
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
        return format_poleinfo(Counter(x.polePosition() for x in self.poles))

    def prefactor(self):
        return damped_rational(self.get_poles(), 4 * self.context.rho,
                               self.context.field(1), self.context)

    def approx_chiral_h(self):
        res = self.context.chiral_h_asymptotic(self.S) * reduce(
            lambda y, w: y * w, [(self.context.Delta - x.polePosition()) for x in self.poles])
        _polys = [reduce(lambda y, z: y * z, [self.context.Delta - w.polePosition()
                                             for w in self.poles if w != x]) for x in self.poles]
        res += reduce(lambda y, w: y + w, map(lambda x, y: x.residue_of_h() * y, self.poles, _polys))
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


class g_rational_approx_data_two_d:
    """
    We use a different class for d = 2,
    utilizing the exact formula by Dolan and Osborn.
    The usage is similar to .
    """

    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles=True):
        self.cutoff = cutoff
        self.ell = ell
        self.a = -context.field(Delta_1_2) / 2
        self.b = context.field(Delta_3_4) / 2
        self.S = self.a + self.b
        self.P = 2 * self.a * self.b
        self.context = context
        self.chiral_approx_data = k_rational_approx_data(
            context,
            cutoff,
            Delta_1_2,
            Delta_3_4,
            is_correlator_multiple,
            approximate_poles)

    def prefactor(self):
        __chiral_poles = set(self.chiral_approx_data.get_poles())
        __q = [2 * x + self.ell for x in __chiral_poles] + \
              [2 * x - self.ell for x in __chiral_poles]
        return damped_rational(format_poleinfo(Counter(__q)),
                               4 * self.context.rho,
                               self.context.field(4) ** len(__chiral_poles),
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
                     left_contribution[j] *right_contribution[i]) / 2)
        return self.context.zzbar_to_xy_marix.dot(np.array(__zz_res))

    def approx(self):
        pref = self.prefactor()
        body = self.approx_g()
        return prefactor_numerator(pref, body, self.context)


cdef class scalar_cb_context_generic(cb_universal_context):
    """
    Context object for the bootstrap.
    Frequently used quantities are stored here.
    """

    cdef public object epsilon
    def __init__(self, Lambda, Prec, nMax, epsilon):
        cb_universal_context.__init__(self, Lambda, Prec, nMax)
        self.epsilon = self.field(epsilon)
    def h_times_rho_k(self, unsigned long k,  ell, Delta, S, P):
        ell_c = self.field(ell)
        Delta_c = self.field(Delta)
        S_c = self.field(S)
        P_c = self.field(P)
        cdef mpfr_t* array
        sig_on()
        array = hBlock_times_rho_n(k, <mpfr_t>(<RealNumber>self.epsilon).value, <mpfr_t>(<RealNumber>ell_c).value, <mpfr_t>(<RealNumber>Delta_c).value, <mpfr_t>(<RealNumber>S_c).value, <mpfr_t>(<RealNumber>P_c).value, <cb_context>self.c_context)
        sig_off()
        res = np.ndarray(self.Lambda + 1, dtype='O')
        for i in range(self.Lambda + 1):
            res[i] = <RealNumber>(<RealField_class>self.field)._new()
            (<RealNumber>res[i])._parent = self.field
            mpfr_init2(<mpfr_t>(<RealNumber>res[i]).value, self.precision)
            mpfr_set(<mpfr_t>(<RealNumber>res[i]).value, array[i],  MPFR_RNDN)
            mpfr_clear(array[i])
        free(array)
        return res

    cpdef h_asymptotic_form(self, S):
        S_c = self.field(S)
        cdef mpfr_t* _array
        _array = h_asymptotic(<mpfr_t>(<RealNumber>self.epsilon).value,  <mpfr_t>(<RealNumber>S_c).value, <cb_context>(self.c_context))
        res = np.ndarray(self.Lambda + 1, dtype='O')
        for i in range(self.Lambda + 1):
            res[i] = <RealNumber>(<RealField_class>self.field)._new()

            (<RealNumber>res[i])._parent = self.field
            mpfr_init2(<mpfr_t>(<RealNumber>res[i]).value, <mpfr_prec_t>self.precision)
            mpfr_set(<mpfr_t>(<RealNumber>res[i]).value, _array[i],  MPFR_RNDN)
            mpfr_clear(_array[i])
        return np.array(res)

    def gBlock(self, ell, Delta, Delta_1_2, Delta_3_4):
        """
        gBlock(epsilon, ell, Delta, Delta_1_2, Delta_3_4, self=self):
        computes conformal block in the notation of arXiv/1305.1321
        """

        ell_c = self.field(ell)
        Delta_c = self.field(Delta)
        S_c = (self.field(-Delta_1_2) + self.field(Delta_3_4)) / 2
        P_c = self.field(-Delta_1_2) * self.field(Delta_3_4) / 2

        # In case Delta and ell = 0, return the identity_vector.
        if (mpfr_zero_p(<mpfr_t>(<RealNumber>ell_c).value)) != 0 and mpfr_zero_p(<mpfr_t>(<RealNumber>Delta_c).value) != 0:
            if mpfr_zero_p(<mpfr_t>(<RealNumber>S_c).value) != 0 and mpfr_zero_p(<mpfr_t>(<RealNumber>P_c).value) != 0:
                return self.identity_vector()
            raise ValueError("Delta, ell = 0 while Delta_1_2 = {0} and Delta_3_4 = {1}".format(Delta_1_2, Delta_3_4))

        cdef mpfr_t* array
        sig_on()
        array = gBlock_full(<mpfr_t>(<RealNumber>self.epsilon).value, <mpfr_t>(<RealNumber>ell_c).value, <mpfr_t>(<RealNumber>Delta_c).value, <mpfr_t>(<RealNumber>S_c).value, <mpfr_t>(<RealNumber>P_c).value, <cb_context>self.c_context)
        sig_off()

        dimG = get_dimG(self.Lambda)
        res = np.ndarray(dimG, dtype='O')
        for i in range(dimG):
            res[i] = <RealNumber>(<RealField_class>self.field)._new()
            (<RealNumber>res[i])._parent = self.field
            mpfr_init2(<mpfr_t>(<RealNumber>res[i]).value, self.precision)
            mpfr_set(<mpfr_t>(<RealNumber>res[i]).value, array[i],  MPFR_RNDN)
            mpfr_clear(array[i])

        free(array)
        return res

    def c2_expand(self, array_real, ell, Delta, S, P):
        """
        c2_expand(array_real, self.epsilon, ell, Delta, S, P)
        computes y-derivatives of scalar conformal blocks
        from x-derivatives of conformal block called array_real,
        which is a (self.Lambda + 1)-dimensional array.
        """
        local_c2 = ell * (ell + 2 * self.epsilon) + Delta * (Delta - 2 - 2 * self.epsilon)
        aligned_index = lambda y_del, x_del: (self.Lambda + 2 - y_del) * y_del + x_del
        ans = np.ndarray(get_dimG(self.Lambda), dtype='O')
        ans[0:self.Lambda + 1] = array_real
        for i in range(1, (self.Lambda // 2) + 1):
            for j in range(self.Lambda - 2 * i + 1):
                val = self.field(0)
                common_factor = self.field(2 * self.epsilon + 2 * i - 1)
                if j >= 3:
                    val += ans[aligned_index(i, j - 3)] * 16 * common_factor
                if j >= 2:
                    val += ans[aligned_index(i, j - 2)] * 8 * common_factor
                if j >= 1:
                    val += -ans[aligned_index(i, j - 1)] * 4 * common_factor
                val += (4 * (2 * P + 8 * S * i + 4 * S * j - 8 * S + 2 * local_c2 + 4 * self.epsilon * i + 4 * self.epsilon * j - 4 * self.epsilon + 4 * i ** 2 + 8 * i * j - 2 * i + j ** 2 - 5 * j - 2) / i) * ans[aligned_index(i - 1, j)]
                val += (-self.field((j + 1) * (j + 2)) / i) * ans[aligned_index(i - 1, j + 2)]
                val += (2 * (j + 1) * (2 * S + 2 * self.epsilon - 4 * i - j + 6) / i) * ans[aligned_index(i - 1, j + 1)]
                if j >= 1:
                    val += (8 * (2 * P + 8 * S * i + 2 * S * j - 10 * S - 4 * self.epsilon * i + 2 * self.epsilon * j + 2 * self.epsilon + 12 * i ** 2 + 12 * i * j - 34 * i + j ** 2 - 13 * j + 22) / self.field(i)) * ans[aligned_index(i - 1, j - 1)]
                if i >= 2:
                    val += (4 * self.field((j + 1) * (j + 2)) / i) * ans[aligned_index(i - 2, j + 2)]
                    val += (8 * (j + 1) * self.field(2 * S - 2 * self.epsilon + 4 * i + 3 * j - 6) / i) * ans[aligned_index(i - 2, j + 1)]

                ans[aligned_index(i, j)] = val / (2 * common_factor)
        return ans

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0,
                             is_correlator_multiple=False, approximate_poles=True):
        return rational_approx_data_generic_dim(
            self, cutoff, ell, Delta_1_2, Delta_3_4,
            is_correlator_multiple, approximate_poles)

    def approx_cb(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0, include_odd=False, approximate_poles=True):
        if not include_odd:
            if Delta_1_2 != 0 or Delta_3_4 != 0:
                include_odd = True
        g = self.rational_approx_data(
            cutoff, ell, Delta_1_2, Delta_3_4,
            is_correlator_multiple=include_odd, approximate_poles=approximate_poles).approx()
        return g


class poleData:
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
        self.ell = context.field(ell)
        if k > ell and type == 3:
            raise NotImplementedError
        self.k = k
        self.epsilon = context.epsilon
        self.a = context.field(a)
        self.b = context.field(b)
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
            return self.context.field(1 - self.ell - self.k)
        if self.type == 2:
            return self.context.field(1 + self.epsilon - self.k)
        if self.type == 3:
            return self.context.field(1 + self.ell + 2 * self.epsilon - self.k)
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
        local_sign = -1 if self.k % 2 != 0 else 1
        p0 = self.context.pochhammer(1, self.k)
        common_factor = -local_sign * self.k / (p0 ** 2)
        if self.type == 1:
            p1 = self.context.pochhammer(self.ell + 2 * self.epsilon, self.k)
            p2 = self.context.pochhammer((1 - self.k + 2 * self.a) / 2, self.k)
            p3 = self.context.pochhammer((1 - self.k + 2 * self.b) / 2, self.k)
            p4 = self.context.pochhammer(self.ell + self.epsilon, self.k)
            return common_factor * p1 * p2 * p3 / p4
        if self.type == 2:
            p1 = self.context.pochhammer(self.epsilon - self.k, 2 * self.k)
            p2 = self.context.pochhammer(self.ell + self.epsilon - self.k, 2 * self.k)
            p3 = self.context.pochhammer(self.ell + self.epsilon + 1 - self.k, 2 * self.k)
            p4 = self.context.pochhammer((1 - self.k + self.ell - 2 * self.a + self.epsilon) / 2, self.k)
            p5 = self.context.pochhammer((1 - self.k + self.ell + 2 * self.a + self.epsilon) / 2, self.k)
            p6 = self.context.pochhammer((1 - self.k + self.ell - 2 * self.b + self.epsilon) / 2, self.k)
            p7 = self.context.pochhammer((1 - self.k + self.ell + 2 * self.b + self.epsilon) / 2, self.k)
            return common_factor * p1 / (p2 * p3) * p4 * p5 * p6 * p7
        if self.type == 3:
            if self.k > self.ell:
                raise RuntimeError(
                    "pole identifier k must be k <= ell for type 3 pole.")
            p1 = self.context.pochhammer(self.ell + 1 - self.k, self.k)
            p2 = self.context.pochhammer((1 - self.k + 2 * self.a) / 2, self.k)
            p3 = self.context.pochhammer((1 - self.k + 2 * self.b) / 2, self.k)
            p4 = self.context.pochhammer(1 + self.ell + self.epsilon - self.k, self.k)
            return common_factor * p1 * p2 * p3 / p4
        raise NotImplementedError("pole type unrecognizable.")

    def residue_of_h(self):
        return self.coeff() * self.context.h_times_rho_k(self.descendant_level(),
                                                         self.residueEll(), self.residueDelta(), self.S(), self.P())


class rational_approx_data_generic_dim:
    """
    rational_aprrox_data(self, cutoff, epsilon, ell, Delta_1_2=0, Delta_3_4=0, approximate_poles=True)
    computes and holds rational approximation of conformal block datum.
    """

    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles):
        self.epsilon = context.epsilon
        self.ell = ell
        self.cutoff = cutoff
        self.a = -context.field(Delta_1_2) / 2
        self.b = context.field(Delta_3_4) / 2
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
            unitarity_bound += context.field(0.5) ** 10
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
                    poleData(1, x, ell, self.a, self.b, context)
                    for x in range(cutoff + 1, 2 * cutoff + 1)
                ] + [
                    poleData(2, x, ell, self.a, self.b, context)
                    for x in range(cutoff // 2 + 1, 2 * (cutoff // 2) + 1)
                ] + [
                    poleData(3, x, ell, self.a, self.b, context)
                    for x in range(min(ell, cutoff // 2) + 1, ell + 1)
                ]
            else:
                self.approx_poles = [
                    poleData(1, x, ell, self.a, self.b, context)
                    for x in range(2 * (cutoff // 2) + 2, 2 * cutoff + 2, 2)
                ] + [
                    poleData(2, x, ell, self.a, self.b, context)
                    for x in range(cutoff // 2 + 1, 2 * (cutoff // 2) + 1)
                ] + [
                    poleData(3, x, ell, self.a, self.b, context)
                    for x in range(2 * (min(ell, cutoff) // 2) + 2, 2 * (ell // 2) + 2, 2)
                ]

    def prefactor(self):
        return damped_rational(
            format_poleinfo(Counter(x.polePosition() for x in self.poles)),
            4 * self.context.rho, self.context.field(1), self.context)

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
            self.ell,
            self.context.Delta,
            self.S,
            self.P)

    def approx(self):
        pref = self.prefactor()
        body = self.approx_g()
        return prefactor_numerator(pref, body, self.context)


def context_for_scalar(epsilon=0.5, Lambda=15, Prec=800, nMax=250):
    try:
        temp = Integer(epsilon)
        if temp == 0:
            return scalar_cb_2d_context(Lambda, Prec, nMax)
        if temp == 1:
            return scalar_cb_4d_context(Lambda, Prec, nMax)
        raise RuntimeError(
            "Sorry, space-time dimensions d={0} is unsupported. Create it yourself and let me know!".format(2 + 2 * epsilon))
    except TypeError:
        return scalar_cb_context_generic(Lambda, Prec, nMax, epsilon)


def zzbar_anti_symm_to_xy_matrix(Lambda, field=RealField(400)):
    if not isinstance(field, RealField_class):
        raise TypeError("field must be instance of RealField_class, but it is {0}.".format(type(field)))
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
            xypositions = [(Lambda + 2 - (i + j - x - 1) // 2) * (i + j - x - 1) //
                           2 + x for x in range(parity, len(coeff), 2)]
            for p, c in zip(xypositions, coeff[parity::2]):
                result[column_position][int(p)] = field(c) / 2
    return result.transpose()


cdef class scalar_cb_2d_context(scalar_cb_context_generic):
    def __init__(self, int Lambda, mpfr_prec_t Prec, long nMax):
        scalar_cb_context_generic.__init__(self, Lambda, Prec, nMax, 0)
        k_context = cb_universal_context(Lambda, Prec, nMax)

    def chiral_h_asymptotic(self, S):
        S_c = self.field(S)
        cdef mpfr_t* _array
        _array = chiral_h_asymptotic_c(<mpfr_t>(<RealNumber>S_c).value, <cb_context>(self.c_context))
        res = np.ndarray(self.Lambda + 1, dtype='O')
        for i in range(self.Lambda + 1):
            res[i] = <RealNumber>(<RealField_class>self.field)._new()
            (<RealNumber>res[i])._parent = self.field
            mpfr_init2(<mpfr_t>(<RealNumber>res[i]).value, <mpfr_prec_t>self.precision)
            mpfr_set(<mpfr_t>(<RealNumber>res[i]).value, _array[i],  MPFR_RNDN)
            mpfr_clear(_array[i])
        return np.array(res)

    cpdef chiral_h_times_rho_to_n(self, long n, h, Delta_1_2=0, Delta_3_4=0):
        S_c = self.field(-Delta_1_2 + Delta_3_4) / 2
        P_c = self.field(-Delta_1_2 * Delta_3_4) / 2
        h_c = self.field(h)
        cdef mpfr_t* _array
        sig_on()
        _array = chiral_h_times_rho_to_n_c(<unsigned long>n, <mpfr_t>(<RealNumber>h_c).value, <mpfr_t>(<RealNumber>S_c).value, <mpfr_t>(<RealNumber>P_c).value, <cb_context>(self.c_context))
        sig_off()
        res = np.ndarray(self.Lambda + 1, dtype='O')
        for i in range(self.Lambda + 1):
            res[i] = <RealNumber>(<RealField_class>self.field)._new()
            (<RealNumber>res[i])._parent = self.field
            mpfr_init2(<mpfr_t>(<RealNumber>res[i]).value, <mpfr_prec_t>self.precision)
            mpfr_set(<mpfr_t>(<RealNumber>res[i]).value, _array[i],  MPFR_RNDN)
            mpfr_clear(_array[i])
        return np.array(res)

    cpdef k_table(self, h, Delta_1_2, Delta_3_4):
        S_c = self.field(-Delta_1_2 + Delta_3_4) / 2
        P_c = self.field(-Delta_1_2 * Delta_3_4) / 2
        h_c = self.field(h)
        cdef mpfr_t* _array
        _array = k_table_c(<mpfr_t>(<RealNumber>h_c).value, <mpfr_t>(<RealNumber>S_c).value, <mpfr_t>(<RealNumber>P_c).value, <cb_context>(self.c_context))
        res = np.ndarray(self.Lambda + 1, dtype='O')
        for i in range(self.Lambda + 1):
            res[i] = <RealNumber>(<RealField_class>self.field)._new()
            (<RealNumber>res[i])._parent = self.field
            mpfr_init2(<mpfr_t>(<RealNumber>res[i]).value, <mpfr_prec_t>self.precision)
            mpfr_set(<mpfr_t>(<RealNumber>res[i]).value, _array[i],  MPFR_RNDN)
            mpfr_clear(_array[i])
        return np.array(res)

    def k_rational_approx_data(self, cutoff, Delta_1_2=0, Delta_3_4=0, is_correlator_multiple=False, approximate_poles=True):
        return k_rational_approx_data(self, cutoff, Delta_1_2, Delta_3_4, is_correlator_multiple, approximate_poles)

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0, is_correlator_multiple=True, approximate_poles=True):
        return g_rational_approx_data_two_d(self, cutoff, ell, Delta_1_2, Delta_3_4, is_correlator_multiple, approximate_poles)


cdef class scalar_cb_4d_context(scalar_cb_context_generic):
    cdef public scalar_cb_2d_context k_context
    cdef public object zzbar_anti_symm_to_xy_matrix

    def __init__(self, int Lambda, mpfr_prec_t Prec, long nMax):
        scalar_cb_context_generic.__init__(self, Lambda, Prec, nMax, 1)
        self.k_context = scalar_cb_2d_context(Lambda + 1, Prec, nMax)
        self.epsilon = self.field(1)
        self.zzbar_anti_symm_to_xy_matrix = zzbar_anti_symm_to_xy_matrix(Lambda, field=self.field)

    def chiral_h_asymptotic(self, S):
        return self.k_context.chiral_h_asymptotic(S)
    def chiral_h_times_rho_to_n(self, long n, h, Delta_1_2=0, Delta_3_4=0):
        return self.k_context.chiral_h_times_rho_to_n(n, h, Delta_1_2=Delta_1_2, Delta_3_4=Delta_3_4)
    def k_table(self, h, Delta_1_2, Delta_3_4):
        return self.k_context.k_table(h, Delta_1_2, Delta_3_4)

    def k_rational_approx_data(self, cutoff, Delta_1_2=0, Delta_3_4=0, is_correlator_multiple=True, approximate_poles=True):
        return self.k_context.k_rational_approx_data(cutoff, Delta_1_2, Delta_3_4, is_correlator_multiple, approximate_poles=approximate_poles)

    def rational_approx_data(self, cutoff, ell, Delta_1_2=0, Delta_3_4=0, is_correlator_multiple=True, approximate_poles=True):
        return g_rational_approx_data_four_d(self, cutoff, ell, Delta_1_2, Delta_3_4, is_correlator_multiple, approximate_poles)


class g_rational_approx_data_four_d:
    def __init__(self, context, cutoff, ell, Delta_1_2, Delta_3_4,
                 is_correlator_multiple, approximate_poles=True):
        self.cutoff = cutoff
        self.ell = ell
        self.a = -context.field(Delta_1_2) / 2
        self.b = context.field(Delta_3_4) / 2
        self.S = self.a + self.b
        self.P = 2 * self.a * self.b
        self.context = context
        self.chiral_approx_data = k_rational_approx_data(
            context.k_context,
            cutoff,
            Delta_1_2,
            Delta_3_4,
            is_correlator_multiple,
            approximate_poles)

    def prefactor(self):
        __chiral_poles = set(self.chiral_approx_data.get_poles())
        __q = [2 * x - self.ell for x in __chiral_poles] + \
              [2 * x + self.ell + 2 for x in __chiral_poles]
        return damped_rational(
            format_poleinfo(Counter(__q)),
            4 * self.context.rho,
            self.context.field(4) ** len(__chiral_poles)
            / (4 * self.context.rho * self.context.field(self.ell + 1)),
            self.context)

    def approx_g(self):
        __chiral_block = self.chiral_approx_data.approx_k()
        __chiral_block_with_z = self.context.field(
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
                    (-left_contribution[i] * right_contribution[j]
                     + left_contribution[j] * right_contribution[i]))
        return self.context.zzbar_anti_symm_to_xy_matrix.dot(
            np.array(__zz_res))

    def approx(self):
        pref = self.prefactor()
        body = self.approx_g()
        return prefactor_numerator(pref, body, self.context)
