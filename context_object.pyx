from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import cython
import re
import sys
from collections import Counter
from functools import reduce

import numpy as np
from sage.all import Integer, is_square, log, Matrix, sqrt
from sage.functions.gamma import gamma
from sage.rings.real_mpfr import RealField, RealNumber, RR

if sys.version_info.major == 2:
    from future_builtins import ascii, filter, hex, map, oct, zip

from libcpp cimport bool
cimport numpy as np
from sage.libs.mpfr cimport (
    mpfr_prec_t, mpfr_t, mpfr_init2, mpfr_set, mpfr_clear, mpfr_mul,
    mpfr_add, mpfr_add_ui, mpfr_set_ui, MPFR_RNDN, mpfr_set_zero)
from sage.rings.real_mpfr cimport RealField_class

def is_integer(x):
    try:
        Integer(x)
    except TypeError:
        return False
    else:
        return True

def get_dimG(Lambda):
    # type: int -> int
    if Lambda % 2 != 0:
        return (Lambda + 1) * (Lambda + 3) // 4
    return ((Lambda + 2) ** 2) // 4


def z_zbar_derivative_to_x_y_derivative_Matrix(Lambda, field=RealField(400)):
    # type: (int, RealField_class) -> np.ndarray
    # ret.shape = (dimG, dimG), ret.dtype = RealNumber
    """
    z_zbar_derivative_to_x_y_derivative_Matrix(Lambda, field=RealField(400))
    returns the matrix to convert the derivatives of real function
    w.r.t. z and z_bar to those with x and y.
    Assuming the derivatives to be ordered as
    f, D_z f, D_z
    """
    assert isinstance(field, RealField_class), "field must be instance of RealField_class, but is {0}.".format(type(field))
    q = field[str('x')]
    dimG = get_dimG(Lambda)
    result = np.full((dimG, dimG), field(0))
    def set_ij_elements(x, a, b, i, j):
        result[(Lambda + 2 - a) * a + b][(Lambda + 2 - i) * i + j] = x
    qplus = q(str('x + 1'))
    qminus = q(str('x - 1'))
    for i in range(Lambda // 2 + 1):
        for j in range(i, Lambda + 1 - i):
            if i == j:
                coeff = ((qplus ** i) * (qminus ** i)).padded_list()
            else:
                coeff = ((qplus ** j) * (qminus ** i) +
                         (qminus ** j) * (qplus ** i)).padded_list()
            parity = (i + j) % 2
            for c, p in zip(coeff[parity::2], range(parity, len(coeff), 2)):
                set_ij_elements(c, (i + j - p) // 2, p, i, j - i)
    return result


@cython.cclass
class cb_universal_context(object):
    """
    Class to store a bunch of frequently used datum, like
    precision, cutoff parameter Lambda, and the matrix representing
    the change variables, e.g. {z, z_bar} -> (x, y) and r -> x.
    """

    @cython.locals(Lambda=cython.int, Prec=mpfr_prec_t, nMax=cython.int)
    def __cinit__(self, Lambda, Prec, nMax, epsilon=0.0):
        # argument epsilon is not used, but it is needed (why?)
        self.c_context = cython.cast(cb_context, context_construct(nMax, Prec, Lambda))
        self.precision = cython.cast(mpfr_prec_t, Prec)
        self.field = cython.cast(RealField_class, RealField(Prec))
        self.Delta_Field = self.field[str('Delta')]  # type: sage.rings.polynomial.polynomial_ring.PolynomialRing_field_with_category
        self.Delta = self.Delta_Field(str('Delta'))  # type: sage.rings.polynomial.polynomial_real_mpfr_dense.PolynomialRealDense
        self.Lambda = Lambda
        self.maxExpansionOrder = nMax
        self.rho_to_z_matrix = np.ndarray((Lambda + 1, Lambda + 1), dtype='O')

        for i in range(Lambda + 1):
            for j in range(Lambda + 1):
                r = cython.cast(RealNumber, cython.cast(RealField_class, self.field)._new())
                r._parent = self.field
                mpfr_init2(r.value, cython.cast(mpfr_prec_t, Prec))
                mpfr_set(r.value, cython.cast(mpfr_t, self.c_context.rho_to_z_matrix[i * (Lambda + 1) + j]), MPFR_RNDN)
                self.rho_to_z_matrix[i][j] = r

    def __deallocate__(self):
        clear_cb_context(cython.cast(cb_context, cython.cast(cb_universal_context, self).c_context))

    @cython.locals(Lambda=cython.int, Prec=mpfr_prec_t, nMax=cython.int)
    def __init__(self, Lambda, Prec, nMax):
        self.polynomial_vector_evaluate = np.vectorize(lambda x, value: self.Delta_Field(x)(value)) # type: Function[[np.ndarray, RealNumber], np.ndarray]
        self.polynomial_vector_shift = np.vectorize(lambda x, shift: self.Delta_Field(x)(self.Delta + shift)) # type: Function[[np.ndarray, RealNumber], np.ndarray]
        self.rho = 3 - 2 * self.field(2).sqrt()  # 3 - 2 sqrt(2)

        self.zzbar_to_xy_marix = z_zbar_derivative_to_x_y_derivative_Matrix(self.Lambda, self.field)
        self.index_list = reduce(
            lambda x, y: x + y,
            ([np.array([i, j]) for j in range(self.Lambda + 1 - 2 * i)]
            for i in range(self.Lambda // 2 + 1)))
        self.rho_to_delta = np.ndarray(self.Lambda + 1, dtype='O')
        self.rho_to_delta[0] = self.Delta_Field(1)
        for i in range(1, self.Lambda + 1):
            self.rho_to_delta[i] = self.rho_to_delta[i - 1] * (self.Delta + 1 - i) / (self.rho * i)
        self.rho_to_delta = self.rho_to_z_matrix.dot(self.rho_to_delta)
        self.null_ftype = np.full(self.dim_f(), self.field(0))
        self.null_htype = np.full(self.dim_h(), self.field(0))

    # dim_f + dim_h = dim_G
    def dim_f(self):
        return int(((self.Lambda + 1) // 2) * (((self.Lambda + 1) // 2) + 1) // 2)

    def dim_h(self):
        return int(((self.Lambda + 2) // 2) * (((self.Lambda + 2) // 2) + 1) // 2)

    def __call__(self, x):
        """
        The default action of this class is
        to convert numbers into real numbers with the default precision.
        """
        return self.field(x)

    def __repr__(self):
        return "Conformal bootstrap context with Lambda = {0}, precision = {1}, nMax = {2}".format(self.Lambda, self.precision, self.maxExpansionOrder)

    def identity_vector(self):
        res = np.concatenate([self.null_ftype, self.null_htype])
        res[0] = self.field(1)
        return res

    def __v_to_d(self, d):
        """
        compute the table of derivative of v = (z z_bar) ^ d
        in the x_y basis
        """
        local_table = [d.parent(1)]
        local_res = []
        for i in range(1, self.Lambda + 1):
            local_table.append(local_table[-1] * (d.parent(-2) * (d - d.parent(len(local_table) - 1))) / d.parent(len(local_table)))

        for i in range(self.Lambda + 1):
            for j in range(i, self.Lambda + 1 - i):
                local_res.append((local_table[i] * local_table[j] + local_table[j] * local_table[i]) / 2)
        return self.zzbar_to_xy_marix.dot(np.array(local_res))

    def __F_matrix_impl(self, d, parity):
        """
        compute a numpy matrix corresponding to
        v ^ d multiplication followed by x <-> -x (anti-)symmetrization.
        For example, the vector for
        F^{+}_{d, \\Delta, l}(x, y) is computed by
        F_minus = F_minus_matrix(d).dot(gBlock(ell, Delta, S, P))
        """
        aligned_index = lambda x: (self.Lambda + 2 - x[0]) * x[0] + x[1]
        local_v = self.__v_to_d(d)
        return ((self.field(1) / 4) ** d) * np.array([
            np.array([
                local_v[aligned_index(i - m)] if (i - m)[0] >= 0 and (i - m)[1] >= 0 else d.parent(0)
                for m in self.index_list])
            for i in self.index_list if i[1] % 2 == parity])

    def F_minus_matrix(self, d):
        """
        x <-> -x anti-symmetrization
        """
        return self.__F_matrix_impl(d, 1)

    def F_plus_matrix(self, d):
        """
        x <-> -x symmetrization
        """
        return self.__F_matrix_impl(d, 0)

    def univariate_func_prod(self, x, y):
        return np.array([x[0:i + 1].dot(y[i::-1]) for i in range(self.Lambda + 1)])

    def SDP(self, normalization, objective, pvm):
        return SDP(normalization, objective, pvm, context=self)

    def damped_rational(self, poles, c=1):
        return damped_rational(poles, 4 * self.rho, c, self)

    def prefactor_numerator(self, pref, array):
        return prefactor_numerator(pref, array, self)

    @cython.ccall
    @cython.locals(n="unsigned long", temp1=mpfr_t)
    def pochhammer(self, x, n):
        x_c = self.field(x)
        mpfr_init2(temp1, self.precision)
        result = cython.cast(RealNumber, cython.cast(RealField_class, self.field)._new())
        cython.cast(RealNumber, result)._parent = self.field
        mpfr_init2(cython.cast(mpfr_t, cython.cast(RealNumber, result).value), self.precision)
        mpfr_set_ui(cython.cast(mpfr_t, cython.cast(RealNumber, result).value), 1, MPFR_RNDN)
        for j in range(n):
            mpfr_add_ui(temp1, cython.cast(mpfr_t, cython.cast(RealNumber, x_c).value), j, MPFR_RNDN)
            mpfr_mul(cython.cast(mpfr_t, cython.cast(RealNumber, result).value), cython.cast(mpfr_t, cython.cast(RealNumber, result).value), temp1, MPFR_RNDN)
        mpfr_clear(temp1)
        return result

    def vector_to_prefactor_numerator(self, vector):
        """
        Convert a constant (i.e., non-polynomial) vector into positive_matrix_with_prefactor.
        """
        pref = self.damped_rational([], 1)
        return self.prefactor_numerator(pref, vector)

    def join(self, l):
        dims = dict()
        pns = []
        pnindices = []
        nBlock = len(l)
        bodies = dict()
        nrow = None
        for n, mat in enumerate(l):
            if nrow is None:
                nrow = len(mat)
            elif nrow != len(mat):
                raise RuntimeError("unequal dim")
            for i, row in enumerate(mat):
                if nrow != len(row):
                    raise RuntimeError("unequal dim")
                for j, x in enumerate(row):
                    if isinstance(x, prefactor_numerator):
                        len_x = x.matrix.shape[0]
                        pns.append(x)
                        pnindices.append((n, i, j))
                    else:
                        len_x = int(x)
                        bodies[(n, i, j)] = np.full(len_x, self(0), dtype='O')
                    if n not in dims:
                        dims[n] = len_x
                    elif dims[n] != len_x:
                        raise RuntimeError("Input has inconsistent dimensions.")
        res_pref_poles = damped_rational.poles_max(pn.prefactor for pn in pns)
        res_pref = self.damped_rational(res_pref_poles)
        for ind, pn in zip(pnindices, pns):
            bodies[ind] =  res_pref.div(pn.prefactor).denominator(self.Delta) * pn.matrix
        res = np.ndarray((nrow, nrow, sum(dims[x] for x in dims)), dtype='O')
        for i in range(nrow):
            for j in range(nrow):
                v = tuple(bodies[(n, i, j)].reshape((dims[n], )) for n in range(nBlock))
                vv = np.concatenate(v)
                res[i, j] = vv
        return prefactor_numerator(res_pref, res, self)

    def sumrule_to_SDP(self, normalization, objective, svs):
        # svs must be a list of lists of matrices
        # a matrix is one component, or a list of lists of component (which must be a square matrix)
        # svs[m][n][i][j]: m-th block, n-th equation, i-th row, j-th column
        n_block = len(svs[0])
        dims = dict()  # type: Dict[int, int]
        res = []  # type: List[List[List[List[prefactor_numerator]]]]
        tbs = {n: [] for n in range(n_block)}
        for m, sv in enumerate(svs):
            if len(sv) != n_block:
                raise RuntimeError("Sum rule vector has in equal dimensions!")
            psv = []  # type: List[List[List[prefactor_numerator]]]
            for n, component in enumerate(sv):
                # pcomponent: List[List[prefactor_numerator]]
                if not isinstance(component, list):
                    pcomponent = [[component]]
                else:
                    pcomponent = component
                assert len(pcomponent) == len(pcomponent[0]), "Each component must be a square matrix."
                for i, row in enumerate(pcomponent):
                    for j, x in enumerate(row):
                        if isinstance(x, prefactor_numerator):
                            if n not in dims:
                                dims[n] = x.matrix.shape[0]
                            elif dims[n] != x.matrix.shape[0]:
                                raise RuntimeError("Found inconsistent dimension.")
                        elif isinstance(x, np.ndarray):
                            pcomponent[i][j] = self.vector_to_prefactor_numerator(x)
                            if n not in dims:
                                dims[n] = x.shape[0]
                            elif dims[n] != x.shape[0]:
                                raise RuntimeError("Found inconsistent dimension.")
                        else:
                            tbs[n].append((m, n, i, j))
                psv.append(pcomponent)
            res.append(psv)
        if n_block > len(dims):
            raise RuntimeError("There exists a component zero for all")
        for k in dims:
            if k not in tbs: continue
            for m, n, i, j in tbs[k]:
                res[m][n][i][j] = dims[k]
        if isinstance(normalization, np.ndarray):
            norm = normalization
        elif isinstance(normalization, list):
            norm_list = []
            for n, v in enumerate(normalization):
                if isinstance(v, np.ndarray):
                    if dims[n] != v.shape[0]:
                        raise RuntimeError("Found inconsistent dimension.")
                    norm_list.append(v)
                elif v == 0:
                    norm_list.append(np.full(dims[n], self(0), dtype='O'))
            norm = np.concatenate(norm_list)
        else:
            raise NotImplementedError
        if isinstance(objective, np.ndarray):
            obj = objective
        elif isinstance(objective, list):
            obj_list = []
            for n, v in enumerate(objective):
                if isinstance(v, np.ndarray):
                    if dims[n] != v.shape[0]:
                        raise RuntimeError("Found inconsistent dimension.")
                    obj_list.append(v)
                elif v == 0:
                    obj_list.append(np.full(dims[n], self(0), dtype='O'))
            obj = np.concatenate(obj_list)
        else:
            if not is_integer(objective) or Integer(objective) != 0:
                raise NotImplementedError("Got unrecognizable input for objective")
            obj = np.full(sum(dims[n] for n in dims), self(0), dtype='O')
        return self.SDP(norm, obj, [self.join(sv) for sv in res])

    def dot(self, x, y):
        # Unfortunately __numpy_ufunc__ seems to be disabled (temporarily?)
        # so I cannot override np.dot
        if isinstance(x, prefactor_numerator):
            if isinstance(y, prefactor_numerator):
                pref = x.prefactor * y.prefactor
                return prefactor_numerator(pref, np.dot(x.matrix, y.matrix), self)
            return prefactor_numerator(x.prefactor, np.dot(x.matrix, y), self)
        else:
            if isinstance(y, prefactor_numerator):
                return prefactor_numerator(y.prefactor, np.dot(x, y.matrix), self)
            return np.dot(x, y)


@cython.cfunc
@cython.returns("mpfr_t*")
@cython.locals(prec=mpfr_prec_t)
def pole_integral_c(x_power_max, base, pole_position, order_of_pole, prec):
    a = RealField(2 * prec)(pole_position)
    b = RealField(2 * prec)(base)
    if a < 0:
        incomplete_gamma = b ** a * gamma(0, a * log(b))
    elif a == 0:
        incomplete_gamma = RealField(prec)(prec)
    else:
        raise RuntimeError("A pole exists in the prefactor")
    if not incomplete_gamma.is_real():
        raise RuntimeError("Integral not real ... perhaps a mistake in pole data.")
    incomplete_gamma = RealField(prec)(incomplete_gamma)

    field = RealField(prec)
    a = field(pole_position)
    b = field(base)

    if order_of_pole == 1:
        result = simple_pole_case_c(
            cython.cast(long, x_power_max),
            cython.cast(mpfr_t, cython.cast(RealNumber, b).value),
            cython.cast(mpfr_t, cython.cast(RealNumber, a).value),
            cython.cast(mpfr_t, cython.cast(RealNumber, incomplete_gamma).value),
            prec)
    elif order_of_pole == 2:
        result = double_pole_case_c(
            cython.cast(long, x_power_max),
            cython.cast(mpfr_t, cython.cast(RealNumber, b).value),
            cython.cast(mpfr_t, cython.cast(RealNumber, a).value),
            cython.cast(mpfr_t, cython.cast(RealNumber, incomplete_gamma).value),
            prec)
    return result

@cython.ccall
@cython.locals(x_power=cython.int, n=cython.int, number_of_factors=cython.int, count=cython.int, pole_data_to_c="mpfr_t*", is_double="int*", temp1=mpfr_t, temp2=mpfr_t, temp_mpfrs="mpfr_t*")
def prefactor_integral(pole_data, base, x_power, prec, c=1):
    field = RealField(prec)
    n = len(pole_data)
    number_of_factors = sum([x[1] for x in pole_data])

    count = 0
    index_list = []
    for i, pole in enumerate(pole_data):
        if field(pole[0]) > 0:
            raise NotImplementedError("There exists a pole on the integration contour of the prefactor!")
        if pole[1] == 1:
            index_list.append([i, 1])
        elif pole[1] == 2:
            index_list.append([i, 2])
            index_list.append([i, 1])
        else:
            raise NotImplementedError
    if n == 0:
        minus_ln_b = -1 / log(RealField(prec)(base))
        result = np.ndarray(x_power + 1, dtype='O')
        result[0] = minus_ln_b * RealField(prec)(c)
        for i in range (1, x_power + 1):
            result[i] = result[i - 1] * minus_ln_b * i
        return result
    pole_data_to_c = cython.cast("mpfr_t*", malloc(cython.sizeof(mpfr_t) * len(pole_data)))
    if pole_data_to_c == NULL:
        raise NotImplementedError
    is_double = cython.cast("int*", malloc(cython.sizeof(int) * len(pole_data)))

    base_c = field(base);
    for i in range(n):
        r = field(pole_data[i][0])
        mpfr_init2(pole_data_to_c[i], prec)
        mpfr_set(pole_data_to_c[i], cython.cast(mpfr_t, cython.cast(RealNumber, r).value), MPFR_RNDN)
        if pole_data[i][1] == 2:
            is_double[i] = 1
        else:
            is_double[i] = 0
    decompose_coeffs = fast_partial_fraction_c(pole_data_to_c, is_double, n, prec)

    for i, _ in enumerate(pole_data):
        mpfr_clear(pole_data_to_c[i])

    free(pole_data_to_c)
    free(is_double)

    mpfr_init2(temp1, prec);
    mpfr_init2(temp2, prec);
    result = np.ndarray(x_power + 1, dtype='O')
    for i in range(x_power + 1):
        result[i] = cython.cast(RealNumber, cython.cast(RealField_class, field)._new())
        mpfr_init2(cython.cast(mpfr_t, cython.cast(RealNumber, result[i]).value), prec)
        mpfr_set_zero(cython.cast(mpfr_t, cython.cast(RealNumber, result[i]).value), 1)
        cython.cast(RealNumber, result[i])._parent = field

    for i in range(number_of_factors):
        temp_mpfrs = pole_integral_c(x_power, base, pole_data[index_list[i][0]][0], index_list[i][1], prec)

        for j in range(x_power + 1):
            mpfr_mul(temp1, decompose_coeffs[i], temp_mpfrs[j], MPFR_RNDN)
            mpfr_add(cython.cast(mpfr_t, cython.cast(RealNumber, result[j]).value), cython.cast(mpfr_t, cython.cast(RealNumber, result[j]).value), temp1, MPFR_RNDN)
            mpfr_clear(temp_mpfrs[j])
        free(temp_mpfrs)

    for i in range(number_of_factors):
        mpfr_clear(decompose_coeffs[i])
    free(decompose_coeffs)
    return RealField(prec)(c) * result


@cython.ccall
@cython.locals(anti_band_input="mpfr_t*", anti_band_mat="mpfr_t*", cholesky_decomposed="mpfr_t*", inversed="mpfr_t*")
def anti_band_cholesky_inverse(v, n_order_max, prec):
    field = RealField(prec)
    n_max = int(n_order_max)
    if not isinstance(n_max, int):
        raise TypeError
    if len(v) < n_max * 2 + 1:
        print("input vector is too short..")
        raise TypeError
    if n_max < 0:
        print("expected n_max to be positive integer...")
        raise TypeError

    anti_band_input = cython.cast("mpfr_t*", malloc(cython.sizeof(mpfr_t) * len(v)))
    for i, val in enumerate(v):
        r = field(val)
        mpfr_init2(anti_band_input[i], prec)
        mpfr_set(anti_band_input[i], cython.cast(mpfr_t, cython.cast(RealNumber, r).value), MPFR_RNDN)
    anti_band_mat = form_anti_band(anti_band_input, cython.cast(int, n_max + 1), int(prec))
    for i, _ in enumerate(v):
        mpfr_clear(anti_band_input[i])
    free(anti_band_input)
    cholesky_decomposed = mpfr_cholesky(anti_band_mat, cython.cast(int, n_max + 1), int(prec))
    for i in range((n_max - 1) ** 2):
        mpfr_clear(anti_band_mat[i])
    free (anti_band_mat)

    inversed = mpfr_triangular_inverse(cholesky_decomposed, cython.cast(int, n_max + 1), int(prec))
    for i in range((n_max + 1) ** 2):
        mpfr_clear(cholesky_decomposed[i])
    free(cholesky_decomposed)

    result = np.ndarray((n_max + 1, n_max + 1), dtype='O')
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            result[i][j] = cython.cast(RealNumber, cython.cast(RealField_class, field)._new())
            mpfr_init2(cython.cast(mpfr_t, cython.cast(RealNumber, result[i][j]).value), prec)
            mpfr_set(cython.cast(mpfr_t, cython.cast(RealNumber, result[i][j]).value), inversed[i * (n_max + 1) + j], MPFR_RNDN)
            cython.cast(RealNumber, result[i][j])._parent = field
            mpfr_clear(inversed[i * (n_max + 1) + j])

    free(inversed)
    return result


def max_index(v):
    # type: (Iterable[T]) -> int
    return max(enumerate(v), key=lambda x: abs(x[1]))[0]

def normalizing_component_subtract(m, normalizing_vector):
    assert isinstance(normalizing_vector, np.ndarray), "normalizing_vector must be np.ndarray, but is {0}".format(
        type(normalizing_vector))
    if len(m) != len(normalizing_vector):
        raise RuntimeError(
            "length of normalizing vector and target object must be equal.")
    __index = max_index(normalizing_vector)
    __deleted_normalizing_vector = (
        1 / normalizing_vector[__index]) * np.delete(normalizing_vector, __index)
    return np.insert(
        np.delete(m, __index, 0) - __deleted_normalizing_vector * m[__index],
        0,
        m[__index] / normalizing_vector[__index])


def recover_functional(alpha, normalizing_vector):
    __index = max_index(normalizing_vector)
    __deleted_normalizing_vector = (
        1 / normalizing_vector[__index]) * np.delete(normalizing_vector, __index)
    if len(alpha) != (len(normalizing_vector) - 1):
        raise RuntimeError(
            "length of normalizing vector and target object must be equal.")
    alpha_deleted = (
        1 / normalizing_vector[__index]) - alpha.dot(__deleted_normalizing_vector)
    return np.insert(alpha, __index, alpha_deleted)


find_y = re.compile(r'y *= *\{([^\}]+)\}')


def efm_from_sdpb_output(file_path, normalizing_vector, context):
    data_stream = open(file_path)
    data_text = data_stream.read()
    data_stream.close()
    yres_text = find_y.search(data_text).groups()[0]
    vector_text = re.split(r', ', yres_text)
    y_result = np.array([context.field(x) for x in vector_text])
    return recover_functional(y_result, normalizing_vector)


def write_real_num(file_stream, real_num, tag):
    # type: (io.TextIOWrapper, str, Union[RealNumber, int])
    file_stream.write(("<" + tag + ">"))
    file_stream.write(repr(real_num))
    file_stream.write(("</" + tag + ">\n"))


def write_vector(file_stream, name, vector):
    # type: (io.TextIOWrapper, str, Iterable[Union[RealNumber, int]])
    # write iterable object vector with tag name to file_stream.
    file_stream.write("<" + name + ">\n")
    for x in vector:
        write_real_num(file_stream, x, "elt")
    file_stream.write("</" + name + ">\n")


def write_polynomial(file_stream, polynomial):
    file_stream.write("<polynomial>\n")
    if isinstance(polynomial, RealNumber):
        __temp = [polynomial]
    else:
        __temp = polynomial.list()
        if __temp == []:
            __temp = [0]
    for x in __temp:
        write_real_num(file_stream, x, "coeff")
    file_stream.write("</polynomial>\n")


def write_polynomial_vector(file_stream, polynomialVector):
    file_stream.write("<polynomialVector>\n")
    for x in polynomialVector:
        write_polynomial(file_stream, x)
    file_stream.write("</polynomialVector>\n")


def laguerre_sample_points(n, field, rho):
    return [field(3.141592) ** 2 * (-1 + 4 * k) ** 2 / (-64 * n * log(4 * rho)) for k in range(n)]


def __map_keys(d, f):
    return {f(x): d[x] for x in d}


def format_poleinfo(poles, context=None):
    if context is None:
        def field(x): return x
    else:
        field = context.field
    if len(poles) == 0:
        return dict()
    if isinstance(poles, dict):
        return __map_keys(poles, field)
    if isinstance(poles, list):
        if not isinstance(poles[0], list):
            return __map_keys(Counter(poles), field)
        if len(poles[0]) == 2:
            return {field(x[0]): x[1] for x in poles}
    raise TypeError("unreadable initialization for poles")


@cython.cclass
class damped_rational(object):
    '''
    represents a rational function f(Delta) = pref_constant * (base ** Delta) / (polynomial of Delta),
    where (polynomial of Delta) = \\prod_{i \\in poles} (Delta - i) ** poles[i]
    '''

    @cython.locals(context=cb_universal_context)
    def __cinit__(self, poles, base, c, context):
        self.__base = context.field(base)  # type: RealNumber
        self.__pref_constant = context.field(c)  # type: RealNumber
        self.__context = context  # type: RealField_class

    @cython.locals(context=cb_universal_context)
    def __init__(self, poles, base, c, context):
        self.__poles = format_poleinfo(poles, context)  # type: Dict[RealNumber, int]

    # return new rational g(Delta), where f(Delta + shift) = g(Delta)
    def shift(self, shift):
        # type: (RealNumber) -> damped_rational
        new_poles = [[x - shift, self.__poles[x]] for x in self.__poles]
        new_const = self.__pref_constant * self.__base ** shift
        return damped_rational(new_poles, self.__base, new_const, self.__context)

    # evaluate f(x)
    def __call__(self, x):
        return (self.__base ** x) / self.denominator(x)

    # evaluate denominator of f(x)
    def denominator(self, x):
        return reduce(lambda z, w: z * w, [(x - y) ** self.__poles[y] for y in self.__poles], 1 / self.__pref_constant)

    def orthogonal_polynomial(self, order):
        passed_poles = [[x, self.__poles[x]] for x in self.__poles]
        return anti_band_cholesky_inverse(prefactor_integral(passed_poles, self.__base, order, self.__context.precision, self.__pref_constant), order // 2, self.__context.precision)

    @staticmethod
    def from_poleinfo(poles, context):
        return damped_rational(poles, context(1), context(1), context)

    @staticmethod
    def poles_max(poles_list):
        # type: (Iterable[Union[damped_rational, Dict[RealNumber, int]]]) -> Dict[RealNumber, int]
        ans = dict()
        for p in poles_list:
            if isinstance(p, damped_rational):
                p = p.__poles
            for x in p:
                if x not in ans:
                    ans[x] = p[x]
                else:
                    ans[x] = max(ans[x], p[x])
        return ans

    @staticmethod
    def poles_min(poles_list):
        # type: (Iterable[Dict[RealNumber, int]]) -> Dict[RealNumber, int]
        ans = None
        keys = None
        for p in poles_list:
            if isinstance(p, damped_rational):
                p = p.__poles
            if ans is None:
                ans = copy.copy(p)
                keys = set(ans)
            else:
                kp = set(p)
                for x in keys - kp:
                    del ans[x]
                keys &= kp
                for x in keys:
                    ans[x] = min(ans[x], p[x])
        return ans

    @staticmethod
    def poles_add(poles_list):
        # type: (Iterable[Dict[RealNumber, int]]) -> Dict[RealNumber, int]
        ans = dict()
        for p in poles_list:
            if isinstance(p, damped_rational):
                p = p.__poles
            for x in p:
                if x not in ans:
                    ans[x] = p[x]
                else:
                    ans[x] += p[x]
        return ans

    @staticmethod
    def poles_subtract(poles, poles_list):
        # type: (Dict[RealNumber, int], Iterable[Dict[RealNumber, int]]) -> Dict[RealNumber, int]
        ans = copy.copy(poles)
        for p in poles_list:
            if isinstance(p, damped_rational):
                p = p.__poles
            for x in p:
                if x not in ans:
                    raise RuntimeError("could not delete pole, {0}".format(p[x]))
                ans[x] -= p[x]
                if ans[x] < 0:
                    raise RuntimeError("could not delete pole")
                if ans[x] == 0:
                    del ans[x]
        return ans

    def __mul__(self, y):
        # type: (damped_rational) -> damped_rational
        if not isinstance(y, damped_rational):
            raise NotImplementedError
        res_poles = damped_rational.poles_add((self.__poles, y.__poles))
        new_base = self.__base * y.__base
        new_const = self.__pref_constant * y.__pref_constant
        return damped_rational(res_poles, new_base, new_const, self.__context)

    def div(self, y):
        # type: (damped_rational) -> damped_rational
        if not isinstance(y, damped_rational):
            raise NotImplementedError
        res_poles = damped_rational.poles_subtract(self.__poles, (y.__poles, ))
        new_base = self.__base / y.__base
        new_const = self.__pref_constant / y.__pref_constant
        return damped_rational(res_poles, new_base, new_const, self.__context)

    def __div__(self, y):
        return self.div(y)

    def __truediv__(self, y):
        return self.div(y)

    # this method does not change self
    def add_poles(self, location):
        # type: (Union[Dict[RealNumber, int], List[RealNumber], List[Tuple[RealNumber, int]]]) -> damped_rational
        res_poles = damped_rational.poles_add((self.__poles, format_poleinfo(location)))
        return damped_rational(res_poles, self.__base, self.__pref_constant, self.__context)

    def lcm(self, p):
        # type: (dampled_rational) -> damped_rational
        if not isinstance(p, damped_rational):
            raise TypeError("lcm supported only between damped_rationals")
        if self.__base != p.__base:
            raise RuntimeError("two damped-rational must have the same base!")
        # if p == self:
            # return self

        return damped_rational(
            damped_rational.poles_max((self.__poles, p.__poles)),
            self.__base, self.__pref_constant * p.__pref_constant, self.__context)

    def gcd(self, p):
        # type: (dampled_rational) -> damped_rational
        if not isinstance(p, damped_rational):
            raise TypeError("gcd supported only between damped_rationals")
        # if p == self:
            # return self

        return damped_rational(
            damped_rational.poles_min((self.__poles, p.__poles)),
            1, 1, self.__context)

    def __repr__(self):
        def pole_str(x):
            output = "(Delta"
            if x != 0:
                output += "{0}{1}".format("-" if x > 0 else "+", repr(abs(x)))
            output += ")"
            if self.__poles[x] != 1:
                output += "**" + repr(self.__poles[x])
            return output
        return "{0}*({1})**Delta / ({2})".format(repr(self.__pref_constant), repr(self.__base), "*".join(pole_str(x) for x in self.__poles))


@cython.cclass
class positive_matrix_with_prefactor(object):

    @cython.locals(prefactor=damped_rational, context=cb_universal_context)
    def __cinit__(self, prefactor, matrix, context):
        self.prefactor = prefactor
        self.context = context

    @cython.locals(prefactor=damped_rational)
    def __init__(self, prefactor, matrix, context):
        self.matrix = matrix

    def shift(self, x):
        return positive_matrix_with_prefactor(self.prefactor.shift(x), self.context.polynomial_vector_shift(self.matrix, x), self.context)

    def degree_max(self):
        return max((np.vectorize(lambda y: self.context.Delta_Field(y).degree())(self.matrix)).flatten())

    def normalization_subtract(self, v):
        return normalizing_component_subtract(self.matrix, v)

    def write(self, file_stream, v):
            shuffled_matrix = np.array([[normalizing_component_subtract(y, v) for y in x] for x in self.matrix])
            sample_points = laguerre_sample_points(self.degree_max() + 1, self.context.field, self.context.rho)
            sample_scalings = map(self.prefactor, sample_points)
            orthogonal_polynomial_vector = map(self.context.Delta_Field, self.prefactor.orthogonal_polynomial(self.degree_max()))

            file_stream.write("<polynomialVectorMatrix>\n")
            file_stream.write("<rows>\n")
            file_stream.write(repr(len(shuffled_matrix)))
            file_stream.write("</rows>\n")
            file_stream.write("<cols>\n")
            file_stream.write(repr(len(shuffled_matrix[0])))
            file_stream.write("</cols>\n")
            file_stream.write("<elements>\n")
            for x in shuffled_matrix:
                for y in x:
                    write_polynomial_vector(file_stream, y)
            file_stream.write("</elements>\n")
            write_vector(file_stream, "samplePoints", sample_points)
            write_vector(file_stream, "sampleScalings", sample_scalings)
            file_stream.write("<bilinearBasis>\n")
            for x in orthogonal_polynomial_vector:
                write_polynomial(file_stream, x)
            file_stream.write("</bilinearBasis>\n")
            file_stream.write("</polynomialVectorMatrix>\n")

    def reshape(self, shape=None):
        if len(self.matrix.shape) == 3 and self.matrix.shape[0] == self.matrix.shape[1] and shape is None:
            return self
        if shape is None:
            shape = (1, 1, self.matrix.shape[-1])
        new_b = self.matrix.reshape(shape)
        return prefactor_numerator(self.prefactor, new_b, self.context)


@cython.cclass
class prefactor_numerator(positive_matrix_with_prefactor):

    def add_poles(self, poles):
        new_pref = self.prefactor.add_poles(poles)
        return prefactor_numerator(new_pref, self.matrix, self.context)

    def rdot(self, M):
        return prefactor_numerator(self.prefactor, M.dot(self.matrix), self.context)

    def shift(self, x):
        return prefactor_numerator(self.prefactor.shift(x), self.context.polynomial_vector_shift(self.matrix, x), self.context)

    def multiply_factored_polynomial(self, factors, C):
        """
        multiply C * \\Prod_x  (Delta - x) ** (factors[x])
        where x in factors
        """
        # multiplication is equivalent to division by div
        div = damped_rational(factors, 1, 1 / self.context(C), self.context)
        gcd = damped_rational.gcd(self.prefactor, div)
        # reduce common factor
        new_pref = self.prefactor.div(gcd)
        div = div.div(gcd)
        # division by div is equivalent to multiplication by div.denominator
        return prefactor_numerator(new_pref, div.denominator(self.context.Delta) * self.matrix, self.context)

    def multiply_factored_rational(self, poles, factors, C):
        return self.add_poles(poles).multiply_factored_polynomial(factors, C)

    @staticmethod
    def __mul_impl(x, pn):
        return prefactor_numerator(pn.prefactor, x * pn.matrix, pn.context)

    def __mul__(self, x):
        if isinstance(self, prefactor_numerator):
            return prefactor_numerator.__mul_impl(x, self)
        if isinstance(x, prefactor_numerator):
            return prefactor_numerator.__mul_impl(self, x)
        print("type(self) = {0}, type(x) = {1}".format(type(self), type(x)))
        raise NotImplementedError

    def __pos__(self):
        return prefactor_numerator(self.prefactor, +self.matrix, self.context)

    def __neg__(self):
        return prefactor_numerator(self.prefactor, -self.matrix, self.context)

    def __div__(self, x):
        return prefactor_numerator(self.prefactor, self.matrix / self.context(x), self.context)

    def __truediv__(self, x):
        return prefactor_numerator(self.prefactor, self.matrix / self.context(x), self.context)

    def __add__(self, other):
        if not isinstance(other, prefactor_numerator):
            raise NotImplementedError
        new_pref = self.prefactor.lcm(other.prefactor)
        rem1 = new_pref.div(self.prefactor).denominator(self.context.Delta)
        rem2 = new_pref.div(other.prefactor).denominator(self.context.Delta)
        new_matrix = rem1 * self.matrix + rem2 * other.matrix
        return prefactor_numerator(new_pref, new_matrix, self.context)

    def __sub__(self, x):
        if not isinstance(x, prefactor_numerator):
            raise NotImplementedError
        return self.__add__(x.__neg__())

    def __call__(self, x):
        pref = self.prefactor(x)
        body = self.context.polynomial_vector_evaluate(self.matrix, x)
        return pref * body

    def __repr__(self):
        return repr(self.prefactor) + "\n*" + repr(self.matrix)


def find_local_minima(pol, label, field=RR, context=None):
    solpol = pol.derivative()
    solpol2 = solpol.derivative() * pol - solpol ** 2
    sols = solpol.roots()
    sols = [x[0] for x in sols if x[0] > 0]
    minsols = [[label, RR(x)] for x in sols if solpol2(x) > 0]
    return minsols


def functional_to_spectra(ef_path, problem, context, label=None):
    norm = problem.normalization
    pvm = problem.pvm
    alpha = efm_from_sdpb_output(ef_path, norm, context)
    polys = [Matrix(x.matrix.dot(alpha)).det() for x in pvm]
    if label is None:
        label = range(len(polys))
    return [find_local_minima(p, l) for p, l in zip(polys, label)]


class SDP(object):
    def __init__(
            self,
            normalization,
            objective,
            pvm,
            context=None):
        self.pvm = [x.reshape()
                    if isinstance(x, (positive_matrix_with_prefactor, prefactor_numerator))
                    else
                    context.vector_to_prefactor_numerator(x).reshape()
                    for x in pvm]
        self.normalization = normalization
        self.objective = objective
        self.context = context

    def write(self, file_path):
        with open(file_path, 'w') as file_stream:
            file_stream.write("<sdp>\n")
            write_vector(file_stream, "objective", normalizing_component_subtract(
                self.objective, self.normalization))
            file_stream.write("<polynomialVectorMatrices>\n")
            for x in self.pvm:
                x.write(file_stream, self.normalization)
            file_stream.write("</polynomialVectorMatrices>\n")
            file_stream.write("</sdp>\n")
