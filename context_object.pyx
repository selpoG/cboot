from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import copy
import re
import sys
from collections import Counter
from functools import reduce

import cython
import numpy as np

from sage.arith.misc import rising_factorial
from sage.functions.all import gamma
from sage.functions.log import log
from sage.functions.other import sqrt
from sage.matrix.constructor import matrix
from sage.rings.integer import Integer
from sage.rings.real_mpfr import RealField, RealNumber, RR

if sys.version_info.major == 2:
    from future_builtins import ascii, filter, hex, map, oct, zip


@cython.cfunc
@cython.locals(ptr="mpfr_t*", dim=cython.int, context=RealField_class)
@cython.returns(np.ndarray)
def mpfr_move_to_ndarray_1(ptr, dim, context):
    res = np.ndarray(dim, dtype='O')
    for i in range(dim):
        res[i] = context(0)
        mpfr_set(cython.cast(RealNumber, res[i]).value, ptr[i], MPFR_RNDN)
        mpfr_clear(ptr[i])
    free(ptr)
    return res


@cython.cfunc
@cython.locals(
    ptr="mpfr_t*", dim1=cython.int, dim2=cython.int,
    context=RealField_class, delete=cython.bint)
@cython.returns(np.ndarray)
def mpfr_move_to_ndarray_2(ptr, dim1, dim2, context, delete):
    res = np.ndarray((dim1, dim2), dtype='O')
    for i in range(dim1):
        for j in range(dim2):
            res[i][j] = context(0)
            mpfr_set(
                cython.cast(RealNumber, res[i][j]).value,
                ptr[i * dim2 + j], MPFR_RNDN)
            if delete:
                mpfr_clear(ptr[i * dim2 + j])
    if delete:
        free(ptr)
    return res


def is_integer(x):
    try:
        Integer(x)
    except TypeError:
        return False
    else:
        return True


def get_dimG(Lambda):
    # type: (int) -> int
    if Lambda % 2 != 0:
        return (Lambda + 1) * (Lambda + 3) // 4
    return ((Lambda + 2) ** 2) // 4


def check_set(m, k, v):
    # type: (Dict[Key, Value], Key, Value) -> bool
    if k not in m:
        m[k] = v
        return True
    return m[k] == v


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
        self.c_context = context_construct(nMax, Prec, Lambda)
        self.precision = cython.cast(mpfr_prec_t, Prec)
        self.field = RealField(Prec)
        self.Delta_Field = self.field[str('Delta')]
        self.Delta = self.Delta_Field(str('Delta'))
        self.Lambda = Lambda
        self.maxExpansionOrder = nMax
        self.rho_to_z_matrix = mpfr_move_to_ndarray_2(
            self.c_context.rho_to_z_matrix,
            Lambda + 1, Lambda + 1, self.field, False)

    def __deallocate__(self):
        clear_cb_context(self.c_context)

    @cython.locals(Lambda=cython.int, Prec=mpfr_prec_t, nMax=cython.int)
    def __init__(self, Lambda, Prec, nMax):
        # type: Function[[np.ndarray, RealNumber], np.ndarray]
        self.polynomial_vector_evaluate = np.vectorize(
            lambda x, value: self.Delta_Field(x)(value))
        # type: Function[[np.ndarray, RealNumber], np.ndarray]
        self.polynomial_vector_shift = np.vectorize(
            lambda x, shift: self.Delta_Field(x)(self.Delta + shift))
        self.rho = 3 - 2 * sqrt(self(2))

        self.zzbar_to_xy_marix = z_zbar_derivative_to_x_y_derivative_Matrix(
            self.Lambda, self.field)
        self.index_list = reduce(
            lambda x, y: x + y,
            ([np.array([i, j]) for j in range(self.Lambda + 1 - 2 * i)]
             for i in range(self.Lambda // 2 + 1)))
        self.rho_to_delta = np.ndarray(self.Lambda + 1, dtype='O')
        self.rho_to_delta[0] = self.Delta_Field(1)
        for i in range(1, self.Lambda + 1):
            self.rho_to_delta[i] = self.rho_to_delta[i - 1] * \
                (self.Delta + 1 - i) / (self.rho * i)
        self.rho_to_delta = self.rho_to_z_matrix.dot(self.rho_to_delta)
        self.null_ftype = np.full(self.dim_f(), self(0))
        self.null_htype = np.full(self.dim_h(), self(0))

    @staticmethod
    def __triangle(n):
        return int(n * (n + 1) // 2)

    # dim_f + dim_h = dim_G
    def dim_f(self):
        return cb_universal_context.__triangle((self.Lambda + 1) // 2)

    def dim_h(self):
        return cb_universal_context.__triangle((self.Lambda + 2) // 2)

    def __call__(self, x):
        """
        The default action of this class is
        to convert numbers into real numbers with the default precision.
        """
        return self.field(x)

    def __str__(self):
        return "Conformal bootstrap context with " \
            "Lambda = {0}, precision = {1}, nMax = {2}".format(
                self.Lambda, self.precision, self.maxExpansionOrder)

    def __repr__(self):
        return "cb_universal_context(Lambda={0}, Prec={1}, nMax={2})".format(
            repr(self.Lambda), repr(self.precision),
            repr(self.maxExpansionOrder))

    def identity_vector(self):
        res = np.full(get_dimG(self.Lambda), self(0))
        res[0] = self(1)
        return res

    def __v_to_d(self, d):
        """
        compute the table of derivative of v = (z z_bar) ^ d
        in the x_y basis
        """
        local_table = [d.parent(1)]
        local_res = []
        for i in range(1, self.Lambda + 1):
            local_table.append(
                local_table[-1] *
                (d.parent(-2) * (d - d.parent(len(local_table) - 1))) /
                d.parent(len(local_table)))

        for i in range(self.Lambda + 1):
            for j in range(i, self.Lambda + 1 - i):
                local_res.append(
                    (local_table[i] * local_table[j] +
                     local_table[j] * local_table[i]) / 2)
        return self.zzbar_to_xy_marix.dot(np.array(local_res))

    def __F_matrix_impl(self, d, parity):
        """
        compute a numpy matrix corresponding to
        v ^ d multiplication followed by x <-> -x (anti-)symmetrization.
        For example, the vector for
        F^{+}_{d, \\Delta, l}(x, y) is computed by
        F_minus = F_minus_matrix(d).dot(gBlock(ell, Delta, S, P))
        """
        def aligned_index(x):
            return (self.Lambda + 2 - x[0]) * x[0] + x[1]

        local_v = self.__v_to_d(d)
        return ((self(1) / 4) ** d) * np.array([
            np.array([
                local_v[aligned_index(i - m)]
                if (i - m)[0] >= 0 and (i - m)[1] >= 0
                else d.parent(0)
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
        return np.array([
            x[0:i + 1].dot(y[i::-1])
            for i in range(self.Lambda + 1)])

    def SDP(self, normalization, objective, pvm):
        return SDP(normalization, objective, pvm, context=self)

    def damped_rational(self, poles, c=1):
        return damped_rational(poles, 4 * self.rho, c, self)

    def prefactor_numerator(self, pref, array):
        return prefactor_numerator(pref, array, self)

    @cython.ccall
    @cython.locals(n="unsigned long", temp=mpfr_t)
    def pochhammer(self, x, n):
        return rising_factorial(self(x), n)

    def vector_to_prefactor_numerator(self, vector):
        """
        Convert a constant (i.e., non-polynomial) vector into prefactor_numerator.
        """
        pref = self.damped_rational([])
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
                        len_x = x.shape()[0]
                        pns.append(x)
                        pnindices.append((n, i, j))
                    else:
                        len_x = int(x)
                        bodies[(n, i, j)] = np.full(len_x, self(0))
                    if not check_set(dims, n, len_x):
                        raise RuntimeError(
                            "Input has inconsistent dimensions.")
        respref_poles = damped_rational.poles_max(pn.prefactor() for pn in pns)
        respref = self.damped_rational(respref_poles)
        for i, pn in zip(pnindices, pns):
            bodies[i] = (respref / pn.prefactor()
                         ).denom(self.Delta) * pn.matrix()
        res = np.ndarray((nrow, nrow, sum(dims[x] for x in dims)), dtype='O')
        for i in range(nrow):
            for j in range(nrow):
                v = tuple(
                    bodies[(n, i, j)].reshape((dims[n],))
                    for n in range(nBlock))
                vv = np.concatenate(v)
                res[i, j] = vv
        return prefactor_numerator(respref, res, self)

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
                # pmat: List[List[prefactor_numerator]]
                if not isinstance(component, list):
                    pmat = [[component]]
                else:
                    pmat = component
                assert len(pmat) == len(
                    pmat[0]), "Each component must be a square matrix."
                for i, row in enumerate(pmat):
                    for j, x in enumerate(row):
                        if isinstance(x, prefactor_numerator):
                            if not check_set(dims, n, x.shape()[0]):
                                raise RuntimeError(
                                    "Found inconsistent dimension.")
                        elif isinstance(x, np.ndarray):
                            pmat[i][j] = self.vector_to_prefactor_numerator(x)
                            if not check_set(dims, n, x.shape[0]):
                                raise RuntimeError(
                                    "Found inconsistent dimension.")
                        else:
                            tbs[n].append((m, n, i, j))
                psv.append(pmat)
            res.append(psv)
        if n_block > len(dims):
            raise RuntimeError("There exists a component zero for all")
        for k in dims:
            if k not in tbs:
                continue
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
                    norm_list.append(np.full(dims[n], self(0)))
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
                    obj_list.append(np.full(dims[n], self(0)))
            obj = np.concatenate(obj_list)
        else:
            if not is_integer(objective) or Integer(objective) != 0:
                raise NotImplementedError(
                    "Got unrecognizable input for objective")
            obj = np.full(sum(dims[n] for n in dims), self(0))
        return self.SDP(norm, obj, [self.join(sv) for sv in res])

    def dot(self, x, y):
        # Unfortunately __numpy_ufunc__ seems to be disabled (temporarily?)
        # so I cannot override np.dot
        if isinstance(x, prefactor_numerator):
            if isinstance(y, prefactor_numerator):
                pref = x.prefactor() * y.prefactor()
                return prefactor_numerator(
                    pref, np.dot(x.matrix(), y.matrix()), self)
            return prefactor_numerator(x.prefactor(), np.dot(x.matrix(), y), self)
        else:
            if isinstance(y, prefactor_numerator):
                return prefactor_numerator(
                    y.prefactor(), np.dot(x, y.matrix()), self)
            return np.dot(x, y)


@cython.cfunc
@cython.returns("mpfr_t*")
@cython.locals(prec=mpfr_prec_t)
def __pole_integral_c(x_power_max, base, pole_position, prec):
    field = RealField(prec)
    field2 = RealField(2 * prec)
    a = field2(pole_position)
    b = field2(base)
    if a < 0:
        igamma = b ** a * gamma(0, a * log(b))
    else:
        igamma = prec
    igamma = field(igamma)  # incomplete gamma
    a = field(pole_position)
    b = field(base)
    return simple_pole_case_c(
        cython.cast(long, x_power_max),
        cython.cast(RealNumber, b).value, cython.cast(RealNumber, a).value,
        cython.cast(RealNumber, igamma).value, prec)


@cython.ccall
@cython.locals(x_power=cython.int, n=cython.int, temp_mpfrs="mpfr_t*")
def __prefactor_integral(poles, base, x_power, prec, c):
    # type: (List[RealNumber], RealNumber, Integer, int, RealNumber) -> np.ndarray
    field = RealField(prec)
    n = len(poles)
    assert 0 < base < 1, "base must be in (0,1), but is {0}".format(base)
    result = np.ndarray(x_power + 1, dtype='O')
    if n == 0:
        assert x_power == 0, "x_power is {0} with empty poles".format(x_power)
        result[0] = -c / log(base)
        return result

    for i in range(x_power + 1):
        result[i] = field(0)

    for i, pole in enumerate(poles):
        temp_mpfrs = __pole_integral_c(x_power, base, pole, prec)
        coef = field(1)
        for j in range(n):
            if j != i:
                coef *= pole - poles[j]
        coef = c / coef
        for j in range(x_power + 1):
            mpfr_fma(
                cython.cast(RealNumber, result[j]).value,
                cython.cast(RealNumber, coef).value, temp_mpfrs[j],
                cython.cast(RealNumber, result[j]).value, MPFR_RNDN)
            mpfr_clear(temp_mpfrs[j])
        free(temp_mpfrs)

    return result


@cython.ccall
@cython.locals(
    anti_band_input="mpfr_t*", inversed="mpfr_t*", dim=cython.int)
def __anti_band_cholesky_inverse(v, n_order_max, prec):
    field = RealField(prec)
    n_max = int(n_order_max)
    if not isinstance(n_max, int):
        raise TypeError
    if len(v) < n_max * 2 + 1:
        print("input vector is too short..")
        raise ValueError
    if n_max < 0:
        print("expected n_max to be positive integer...")
        raise ValueError
    dim = cython.cast(int, n_max + 1)

    # prepare anti_band_input to call anti_band_to_inverse
    anti_band_input = cython.cast(
        "mpfr_t*", calloc(len(v), cython.sizeof(mpfr_t)))
    for i, val in enumerate(v):
        r = field(val)
        mpfr_init2(anti_band_input[i], prec)
        mpfr_set(
            anti_band_input[i],
            cython.cast(RealNumber, r).value, MPFR_RNDN)

    inversed = anti_band_to_inverse(anti_band_input, dim, int(prec))

    for i, _ in enumerate(v):
        mpfr_clear(anti_band_input[i])
    free(anti_band_input)

    return mpfr_move_to_ndarray_2(inversed, dim, dim, field, True)


def max_index(v):
    # type: (Iterable[T]) -> int
    return max(enumerate(v), key=lambda x: abs(x[1]))[0]


def normalizing_component_subtract(m, normalizing_vector):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    if len(m) != len(normalizing_vector):
        raise RuntimeError(
            "length of normalizing vector and target object must be equal.")
    ind = max_index(normalizing_vector)
    val = 1 / normalizing_vector[ind]
    deleted_normalizing_vector = val * np.delete(normalizing_vector, ind)
    return np.insert(
        np.delete(m, ind, 0) - deleted_normalizing_vector * m[ind],
        0, m[ind] * val)


def recover_functional(alpha, normalizing_vector):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    if len(alpha) != len(normalizing_vector) - 1:
        raise RuntimeError(
            "length of normalizing vector and target object must be equal.")
    ind = max_index(normalizing_vector)
    val = 1 / normalizing_vector[ind]
    deleted_normalizing_vector = val * np.delete(normalizing_vector, ind)
    alpha_deleted = val - alpha.dot(deleted_normalizing_vector)
    return np.insert(alpha, ind, alpha_deleted)


find_y = re.compile(r'y *= *\{([^\}]+)\}')


def efm_from_sdpb_output(file_path, normalizing_vector, context):
    # type: (str, np.ndarray, cb_universal_context) -> np.ndarray
    data_stream = open(file_path)
    data_text = data_stream.read()
    data_stream.close()
    yres_text = find_y.search(data_text).groups()[0]
    vector_text = re.split(r', ', yres_text)
    y_result = np.array([context(x) for x in vector_text])
    return recover_functional(y_result, normalizing_vector)


def write_real_num(file_stream, real_num, tag):
    # type: (io.TextIOWrapper, str, Union[RealNumber, int])
    file_stream.write(("<" + tag + ">"))
    file_stream.write(str(real_num))
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
        tmp = [polynomial]
    else:
        tmp = polynomial.list()
        if tmp == []:
            tmp = [0]
    for x in tmp:
        write_real_num(file_stream, x, "coeff")
    file_stream.write("</polynomial>\n")


def write_polynomial_vector(file_stream, polynomialVector):
    file_stream.write("<polynomialVector>\n")
    for x in polynomialVector:
        write_polynomial(file_stream, x)
    file_stream.write("</polynomialVector>\n")


def laguerre_sample_points(n, field, rho):
    return [field(3.141592) ** 2 * (-1 + 4 * k) ** 2 / (-64 * n * log(4 * rho))
            for k in range(n)]


def __map_keys(d, f):
    return {f(x): d[x] for x in d}


def format_poleinfo(poles, context=None):
    if context is None:
        def field(x):
            return x
    else:
        field = context
    if len(poles) == 0:
        return dict()
    if isinstance(poles, dict):
        return __map_keys(poles, field)
    if isinstance(poles, list):
        if not isinstance(poles[0], (list, tuple)):
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
        self.__base = context(base)  # type: RealNumber
        self.__pref_constant = context(c)  # type: RealNumber
        self.__context = context  # type: RealField_class

    @cython.locals(context=cb_universal_context)
    def __init__(self, poles, base, c, context):
        # type: Dict[RealNumber, int]
        self.__poles = format_poleinfo(poles, context)

    # return new rational g(Delta), where f(Delta + shift) = g(Delta)
    def shift(self, shift):
        # type: (RealNumber) -> damped_rational
        new_poles = {x - shift: self.__poles[x] for x in self.__poles}
        new_c = self.__pref_constant * self.__base ** shift
        return damped_rational(new_poles, self.__base, new_c, self.__context)

    # evaluate f(x)
    def __call__(self, x):
        return (self.__base ** x) / self.denom(x)

    # evaluate denominator of f(x)
    def denom(self, x):
        return reduce(
            lambda z, w: z * w,
            [(x - y) ** self.__poles[y] for y in self.__poles],
            1 / self.__pref_constant)

    def orthogonal_polynomial(self, order):
        deg = self.__context.Lambda + len(self.__poles) if self.__poles else 0
        assert order == deg, "max degree is {0}, while lambda = {1} and len(poles) = {2}".format(
            order, self.__context.Lambda, len(self.__poles))
        return __anti_band_cholesky_inverse(
            __prefactor_integral(
                list(self.__poles), self.__base, order,
                self.__context.precision, self.__pref_constant),
            order // 2, self.__context.precision)

    @staticmethod
    def from_poleinfo(poles, context):
        return damped_rational(poles, context(1), context(1), context)

    @staticmethod
    def poles_max(poles_list):
        # type: (Iterable[Union[damped_rational, Dict[RealNumber, int]]]) ->
        # Dict[RealNumber, int]
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
        # type: (Dict[RealNumber, int], Iterable[Dict[RealNumber, int]]) ->
        # Dict[RealNumber, int]
        ans = copy.copy(poles)
        for p in poles_list:
            if isinstance(p, damped_rational):
                p = p.__poles
            for x in p:
                if x not in ans:
                    raise RuntimeError(
                        "could not delete pole, {0}".format(p[x]))
                ans[x] -= p[x]
                if ans[x] < 0:
                    raise RuntimeError("could not delete pole")
                if ans[x] == 0:
                    del ans[x]
        return ans

    def __mul__(self, y):
        # type: (damped_rational) -> damped_rational
        if not isinstance(self, damped_rational) or not isinstance(y, damped_rational):
            raise NotImplementedError
        res_poles = damped_rational.poles_add((self.__poles, y.__poles))
        new_base = self.__base * y.__base
        new_const = self.__pref_constant * y.__pref_constant
        return damped_rational(res_poles, new_base, new_const, self.__context)

    def __div__(self, y):
        return self.__truediv__(y)

    def __truediv__(self, y):
        if not isinstance(self, damped_rational) or not isinstance(y, damped_rational):
            raise NotImplementedError
        res_poles = damped_rational.poles_subtract(self.__poles, (y.__poles, ))
        new_base = self.__base / y.__base
        new_const = self.__pref_constant / y.__pref_constant
        return damped_rational(res_poles, new_base, new_const, self.__context)

    # this method does not change self
    def add_poles(self, location):
        # type: (Union[Dict[RealNumber, int], List[RealNumber],
        # List[Tuple[RealNumber, int]]]) -> damped_rational
        res_poles = damped_rational.poles_add(
            (self.__poles, format_poleinfo(location)))
        return damped_rational(
            res_poles, self.__base, self.__pref_constant, self.__context)

    def lcm(self, p):
        # type: (dampled_rational) -> damped_rational
        if not isinstance(p, damped_rational):
            raise TypeError("lcm supported only between damped_rationals")
        if self.__base != p.__base:
            raise RuntimeError("two damped-rational must have the same base!")

        return damped_rational(
            damped_rational.poles_max((self.__poles, p.__poles)),
            self.__base,
            self.__pref_constant * p.__pref_constant,
            self.__context)

    def gcd(self, p):
        # type: (dampled_rational) -> damped_rational
        if not isinstance(p, damped_rational):
            raise TypeError("gcd supported only between damped_rationals")

        return damped_rational(
            damped_rational.poles_min((self.__poles, p.__poles)),
            1, 1, self.__context)

    def __str__(self):
        def pole_str(x):
            output = "(Delta"
            if x != 0:
                output += "{0}{1}".format("-" if x > 0 else "+", abs(x))
            output += ")"
            if self.__poles[x] != 1:
                output += "**" + str(self.__poles[x])
            return output
        return "{0}*({1})**Delta / ({2})".format(
            self.__pref_constant, self.__base,
            "*".join(pole_str(x) for x in self.__poles))

    def __repr__(self):
        return "damped_rational(poles={0}, base={1}, c={2}, context={3})".format(
            repr(self.__poles), repr(self.__base),
            repr(self.__pref_constant), repr(self.__context))


@cython.cclass
class prefactor_numerator(object):

    @cython.locals(prefactor=damped_rational, context=cb_universal_context)
    def __cinit__(self, prefactor, matrix, context):
        self.__prefactor = prefactor  # type: damped_rational
        self.__context = context

    @cython.locals(prefactor=damped_rational)
    def __init__(self, prefactor, matrix, context):
        # shape of matrix must be (n, ) or (k, k, n)
        # each element is a polynomial of Delta (sage.rings.polynomial.polynomial_real_mpfr_dense.PolynomialRealDense) or RealNumber
        self.__matrix = matrix  # type: np.ndarray
        assert len(matrix.shape) == 1 or \
            (len(matrix.shape) == 3 and matrix.shape[0] == matrix.shape[1]), \
            "unexpected shape {0}".format(matrix.shape)

    def prefactor(self):
        return self.__prefactor

    def matrix(self):
        return self.__matrix

    def shape(self):
        return self.__matrix.shape

    def shift(self, x):
        return prefactor_numerator(
            self.__prefactor.shift(x),
            self.__context.polynomial_vector_shift(self.__matrix, x), self.__context)

    def degree_max(self):
        return max(
            (np.vectorize(
                lambda y: self.__context.Delta_Field(y).degree())(
                self.__matrix)).flatten())

    def write(self, file_stream, v):
        shuffled_matrix = np.array(
            [
                [normalizing_component_subtract(y, v) for y in x]
                for x in self.__matrix])
        sample_points = laguerre_sample_points(
            self.degree_max() + 1, self.__context, self.__prefactor.__base / 4)
        sample_scalings = map(self.__prefactor, sample_points)
        orthogonal_polynomial_vector = map(
            self.__context.Delta_Field,
            self.__prefactor.orthogonal_polynomial(self.degree_max()))

        file_stream.write("<polynomialVectorMatrix>\n")
        file_stream.write("<rows>\n")
        file_stream.write(str(len(shuffled_matrix)))
        file_stream.write("</rows>\n")
        file_stream.write("<cols>\n")
        file_stream.write(str(len(shuffled_matrix[0])))
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

    def reshape(self):
        if len(
                self.__matrix.shape) == 3 and self.__matrix.shape[0] == self.__matrix.shape[1]:
            return self
        new_b = self.__matrix.reshape((1, 1, self.__matrix.size))
        new = prefactor_numerator(self.__prefactor, new_b, self.__context)
        self.__matrix = None
        self.__prefactor = None
        self.__context = None
        # if reshape happens, self is no longer available
        return new

    def __str__(self):
        return "{0}\n*{1}".format(self.__prefactor, self.__matrix)

    def __repr__(self):
        return "prefactor_numerator" \
            "(prefactor={0}, matrix={1}, context={2})".format(
                repr(self.__prefactor), repr(self.__matrix), repr(self.__context))

    def add_poles(self, poles):
        new_pref = self.__prefactor.add_poles(poles)
        return prefactor_numerator(new_pref, self.__matrix, self.__context)

    def rdot(self, M):
        return prefactor_numerator(
            self.__prefactor, M.dot(self.__matrix), self.__context)

    def multiply_factored_polynomial(self, factors, C):
        """
        multiply C * \\Prod_x  (Delta - x) ** (factors[x])
        where x in factors
        """
        # multiplication is equivalent to division by div
        div = damped_rational(
            factors, 1, 1 / self.__context(C), self.__context)
        gcd = damped_rational.gcd(self.__prefactor, div)
        # reduce common factor
        new_pref = self.__prefactor / gcd
        div = div / gcd
        # division by div is equivalent to multiplication by div.denom
        return prefactor_numerator(
            new_pref,
            div.denom(self.__context.Delta) * self.__matrix,
            self.__context)

    def multiply_factored_rational(self, poles, factors, C):
        return self.add_poles(poles).multiply_factored_polynomial(factors, C)

    def __mul__(self, x):
        if not isinstance(self, prefactor_numerator):
            return x.__mul__(self)
        return prefactor_numerator(self.__prefactor, x * self.__matrix, self.__context)

    def __pos__(self):
        return prefactor_numerator(self.__prefactor, +self.__matrix, self.__context)

    def __neg__(self):
        return prefactor_numerator(self.__prefactor, -self.__matrix, self.__context)

    def __div__(self, x):
        return self.__truediv__(x)

    def __truediv__(self, x):
        if not isinstance(self, prefactor_numerator):
            raise NotImplementedError
        return prefactor_numerator(
            self.__prefactor, self.__matrix / self.__context(x), self.__context)

    def __add__(self, other):
        if not isinstance(other, prefactor_numerator) or not isinstance(self, prefactor_numerator):
            raise NotImplementedError
        new_pref = self.__prefactor.lcm(other.__prefactor)
        rem1 = (new_pref / self.__prefactor).denom(self.__context.Delta)
        rem2 = (new_pref / other.__prefactor).denom(self.__context.Delta)
        new_matrix = rem1 * self.__matrix + rem2 * other.__matrix
        return prefactor_numerator(new_pref, new_matrix, self.__context)

    def __sub__(self, x):
        if not isinstance(x, prefactor_numerator) or not isinstance(self, prefactor_numerator):
            raise NotImplementedError
        return self.__add__(x.__neg__())

    def __call__(self, x):
        pref = self.__prefactor(x)
        body = self.__context.polynomial_vector_evaluate(self.__matrix, x)
        return pref * body


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
    polys = [matrix(x.matrix().dot(alpha)).det() for x in pvm]
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
                    if isinstance(x, prefactor_numerator)
                    else
                    context.vector_to_prefactor_numerator(x).reshape()
                    for x in pvm]
        self.normalization = normalization
        self.objective = objective
        self.context = context

    def write(self, file_path):
        with open(file_path, 'w') as file_stream:
            file_stream.write("<sdp>\n")
            write_vector(
                file_stream, "objective",
                normalizing_component_subtract(
                    self.objective, self.normalization))
            file_stream.write("<polynomialVectorMatrices>\n")
            for x in self.pvm:
                x.write(file_stream, self.normalization)
            file_stream.write("</polynomialVectorMatrices>\n")
            file_stream.write("</sdp>\n")
