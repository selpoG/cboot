from libcpp cimport bool
from sage.libs.mpfr cimport *
from sage.rings.real_mpfr cimport *
import numpy as np

from sage.all import Matrix, is_square, sqrt
cimport numpy as np
from sage.functions.gamma import gamma
from sage.rings.real_mpfr import RR
import copy
import re
from functools import reduce


def __mult_poles(poles, pref_const, context):
    return reduce(
        lambda x, y: x * y,
        [(context.Delta - z) ** poles[z] for z in poles],
        pref_const)


def z_zbar_derivative_to_x_y_derivative_Matrix(Lambda, field=RealField(400)):
    """
    z_zbar_derivative_to_x_y_derivative_Matrix(Lambda, field=RealField(400))
    returns the matrix to convert the derivatives of real function
    w.r.t. z and z_bar to those with x and y.
    Assuming the derivatives to be ordered as
    f, D_z f, D_z
    """
    q = field['x']
    if Lambda % 2 != 0:
        dimG = (Lambda + 1) * (Lambda + 3) // 4
    else:
        dimG = ((Lambda + 2) ** 2) // 4
    result = np.ndarray(dimG ** 2, dtype='O')
    result = result.reshape(dimG, dimG)
    def set_ij_elements(x, a, b, i, j):
        result[((Lambda + 2 - a) * a + b)][((Lambda + 2 - i) * i + j)] = x
    for i in range(0, (Lambda // 2 + 1)):
        for j in range(i, Lambda + 1 - i):
            if i == j:
                temp = ((q('x + 1') ** j) * (q('x - 1') ** i)).padded_list()
            else:
                temp = ((q('x + 1') ** j) * (q('x - 1') ** i) +
                        (q('x - 1') ** j) * (q('x + 1') ** i)).padded_list()
            parity = (i + j) % 2
            map(
                lambda x, y: set_ij_elements(x, (i + j - y) // 2, y, i, j - i),
                temp[parity::2], range(parity, len(temp), 2))
    return np.array(map(lambda x: 0 if x is None else x,
                        result.flatten())).reshape(dimG, dimG)


cdef class cb_universal_context:
    """
    Class to store a bunch of frequently used datum, like
    precision, cutoff parameter Lambda, and the matrix representing
    the change variables, e.g. {z, z_bar} -> (x, y) and r -> x.
    """

    def __cinit__(self, int Lambda, mpfr_prec_t Prec, long nMax, float epsilon):
        self.c_context = <cb_context>context_construct(nMax, Prec, Lambda)
        self.precision = <mpfr_prec_t>Prec
        self.field = <RealField_class>RealField(Prec)
        self.Delta_Field = self.field['Delta'] # type: sage.rings.polynomial.polynomial_ring.PolynomialRing_field_with_category
        self.Delta = self.Delta_Field('Delta') # type: sage.rings.polynomial.polynomial_real_mpfr_dense.PolynomialRealDense
        self.Lambda = Lambda
        self.maxExpansionOrder = nMax
        self.rho_to_z_matrix = np.ndarray([Lambda + 1, Lambda + 1], dtype='O')
        self.polynomial_vector_evaluate = np.vectorize(lambda x, value: self.Delta_Field(x)(value))

        for i in range(0, Lambda + 1):
            for j in range(0, Lambda + 1):
                r = <RealNumber>(<RealField_class>self.field)._new()
                r._parent = self.field
                mpfr_init2(r.value, <mpfr_prec_t>Prec)
                mpfr_set(r.value, <mpfr_t>self.c_context.rho_to_z_matrix[i * (Lambda + 1) + j], MPFR_RNDN)
                self.rho_to_z_matrix[i][j] = r

    def __init__(self, int Lambda, mpfr_prec_t Prec, long nMax):
        self.polynomial_vector_shift = np.vectorize(lambda x, shift: self.Delta_Field(x)(self.Delta + shift))
        self.rho = 3 - 2 * self.field(2).sqrt()
        self.convert_to_polynomial_vector = np.vectorize(lambda y: self.Delta_Field(y))
        self.convert_to_real_vector = np.vectorize(lambda y: self.field(y))

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

    def __deallocate__(self):
        clear_cb_context(<cb_context>(<cb_universal_context>self).c_context)

    def identity_vector(self):
        res = np.concatenate([self.null_ftype, self.null_htype])
        res[0] = self.field(1)
        return res

    def v_to_d(self, d):
        """
        compute the table of derivative of v = (z z_bar) ^ d
        in the x_y basis
        """
        local_table = [d.parent(1)]
        local_res = []
        for i in range(1, self.Lambda + 1):
            local_table.append(local_table[-1] * (d.parent(-2) * (d - d.parent(len(local_table) - 1))) / d.parent(len(local_table)))

        for i in range(0, self.Lambda + 1):
            for j in range(i, self.Lambda + 1 - i):
                local_res.append((local_table[i] * local_table[j] + local_table[j] * local_table[i]) / 2)
        return self.zzbar_to_xy_marix.dot(np.array(local_res))

    def F_minus_matrix(self, d):
        """
        compute a numpy matrix corresponding to
        v ^ d multiplication followed by x <-> -x anti-symmetrization.
        For example, the vector for
        F^{-}_{d, \\Delta, l}(x, y) is computed by
        F_minus = F_minus_matrix(d).dot(gBlock(ell, Delta, S, P))
        """
        aligned_index = lambda x: (self.Lambda + 2 - x[0]) * x[0] + x[1]
        local_v = self.v_to_d(d)
        return ((self.field(1) / 4) ** d) * np.array(map(lambda i: (np.array(map(lambda m: local_v[aligned_index(i - m)] if ((i - m)[0] >= 0 and (i - m)[1] >= 0) else d.parent(0), self.index_list)))
            , [x for x in self.index_list if x[1] % 2 != 0]))

    def F_plus_matrix(self, d):
        """
        compute a numpy matrix corresponding to
        v ^ d multiplication followed by x <-> -x symmetrization.
        For example, the vector for
        F^{+}_{d, \\Delta, l}(x, y) is computed by
        F_plus = F_plus_matrix(d).dot(gBlock(ell, Delta, S, P))
        """
        aligned_index = lambda x: (self.Lambda + 2 - x[0]) * x[0] + x[1]
        local_v = self.v_to_d(d)
        return ((self.field(1) / 4) ** d) * np.array(map(lambda i: (np.array(map(lambda m: local_v[aligned_index(i - m)] if ((i - m)[0] >= 0 and (i - m)[1] >= 0) else d.parent(0), self.index_list)))
            , [x for x in self.index_list if x[1] % 2 == 0]))

    def univariate_func_prod(self, x, y):
        return np.array(map(lambda i: x[0:i + 1].dot(y[i::-1]), range(0, self.Lambda + 1)))

    def SDP(self, normalization, objective, pvm):
        return SDP(normalization, objective, pvm, context=self)

    def damped_rational(self, poles, c):
        return damped_rational(poles, 4 * self.rho, c, self)

    def prefactor_numerator(self, pref, array):
        return prefactor_numerator(pref, array, self)

    def pochhammer(self, x, unsigned long n):
        x_c = self.field(x)
        cdef mpfr_t temp1
        mpfr_init2(temp1, self.precision)
        result = <RealNumber>(<RealField_class>self.field)._new()
        (<RealNumber>result)._parent = self.field
        mpfr_init2(<mpfr_t>(<RealNumber> result).value, self.precision)
        mpfr_set_ui(<mpfr_t>(<RealNumber> result).value, 1, MPFR_RNDN)
        for j in range(0, n):
            mpfr_add_ui(temp1, <mpfr_t>(<RealNumber>x_c).value, j, MPFR_RNDN)
            mpfr_mul(<mpfr_t>(<RealNumber>result).value, <mpfr_t>(<RealNumber>result).value, temp1, MPFR_RNDN)
        mpfr_clear(temp1)
        return result

    def vector_to_prefactor_numerator(self, vector):
        """
        Convert a constant (i.e., non-polynomial) vector into positive_matrix_with_prefactor.
        """
        pref = self.damped_rational([], 1)
        if len(vector.shape) == 1:
            return self.prefactor_numerator(pref, vector.reshape(1, 1, len(vector)))
        return self.prefactor_numerator(pref, vector)

    def lcms(self, preflist):
        # type: (List[damped_rational]) -> Tuple[damped_rational, List[Dict[RealNumber, int]]]
        res = dict()  # type: Dict[RealNumber, int]
        for pref in preflist:
            d_order = pref.poles  # type: Dict[RealNumber, int]
            for Delta in d_order:
                if Delta in res:
                    res[Delta] = max(res[Delta], d_order[Delta])
                else:
                    # `res[Delta] = d_order[Delta]` does not work... why?
                    res.update({Delta: d_order[Delta]})
        rems = []  # type: List[Dict[RealNumber, int]]
        for pref in preflist:
            d_order = pref.poles  # type: Dict[RealNumber, int]
            rem = dict()  # type: Dict[RealNumber, int]
            for Delta in res:
                mr = res[Delta]
                if Delta in d_order:
                    mr -= d_order[Delta]
                if mr > 0:
                    rem[Delta] = mr
            rems.append(rem)
        return (self.damped_rational(res, self(1)), rems)

    def join(self, l):
        dims = dict()
        pns = []
        pnindices = []
        nBlock = len(l)
        bodies = dict()
        nrow = None
        for n, mat in enumerate(l):
            if not nrow:
                nrow = len(mat)
            else:
                if nrow != len(mat):
                    raise RuntimeError("unequal dim")
            for i, row in enumerate(mat):
                if nrow != len(row):
                    raise RuntimeError("unequal dim")
                for j, x in enumerate(row):
                    if isinstance(x, prefactor_numerator):
                        len_x = x.matrix.shape[-1]
                        pns.append(x)
                        pnindices.append((n, i, j))
                    else:
                        len_x = int(x)
                        v = np.ndarray(len_x, dtype='O')
                        v[:] = self(0)
                        bodies[(n, i, j)] = v
                    if n not in dims:
                        dims[n] = len_x
                    elif dims[n] != len_x:
                        raise RuntimeError("Input has inconsistent dimensions.")

        res_pref, pref_rems = self.lcms([pn.prefactor for pn in pns])
        vecs = [(ind, __mult_poles(rem, pn.prefactor.pref_constant * pn.matrix, self))
                for ind, pn, rem in zip(pnindices, pns, pref_rems)]
        bodies.update(dict(vecs))
        res = np.ndarray((nrow, nrow, sum([dims[x] for x in dims])), dtype='O')
        for i in range(nrow):
            for j in range(nrow):
                v = (bodies[(n, i, j)].reshape((dims[n], )) for n in range(0, nBlock))
                vv = np.concatenate(tuple(v))
                res[i, j] = vv
        return prefactor_numerator(res_pref, res, self)

    def sumrule_to_SDP(self, normalization, objective, svs):
        n_block = len(svs[0])
        dims = dict()
        shapes = []
        res = []
        tbs = dict([(n, []) for n in range(n_block)])
        for m, sv in enumerate(svs):
            if len(sv) != n_block:
                raise RuntimeError("Sum rule vector has in equal dimensions!")
            psv = []
            for n, component in enumerate(sv):
                if not isinstance(component, list):
                    pcomponent = [[component]]
                else:
                    pcomponent = component
                for i, row in enumerate(pcomponent):
                    for j, x in enumerate(row):
                        if isinstance(x, prefactor_numerator):
                            try:
                                givendim = dims[n]
                                if givendim != x.matrix.shape[-1]:
                                    raise RuntimeError("Found inconsistent dimension.")
                            except KeyError:
                                dims[n] = x.matrix.shape[-1]
                        elif isinstance(x, np.ndarray):
                            pcomponent[i][j] = self.vector_to_prefactor_numerator(x)
                            try:
                                givendim = dims[n]
                                if givendim != x.shape[-1]:
                                    raise RuntimeError("Found inconsistent dimension.")
                            except KeyError:
                                dims[n] = x.shape[-1]
                        else:
                            x = int(x)
                            tbs[n].append((m, n, i, j))
                psv.append(pcomponent)
            res.append(psv)
        if n_block > len(dims):
            raise RuntimeError("There exists a component zero for all")
        for k in dims:
            try:
                mlist = tbs[k]
            except KeyError:
                mlist = []
            for m, n, i, j in mlist:
                res[m][n][i][j] = dims[k]
        if isinstance(normalization, np.ndarray):
            norm = normalization
        elif isinstance(normalization, list):
            norm_list = []
            for n, v in enumerate(normalization):
                if isinstance(v, np.ndarray):
                    norm_list.append(v)
                elif v == 0:
                    tba = np.ndarray((dims[n], ), dtype='O')
                    tba[:] = self(0)
                    norm_list.append(tba)
            norm = np.concatenate(norm_list)
        else:
            raise NotImplementedError

        if isinstance(objective, np.ndarray):
            obj = objective
        elif isinstance(objective, list):
            obj_list = []
            for n, v in enumerate(objective):
                if isinstance(v, np.ndarray):
                    obj_list.append(v)
                elif v == 0:
                    tba = np.ndarray((dims[n], ), dtype='O')
                    tba[:] = self(0)
                    obj_list.append(tba)
            obj = np.concatenate(obj_list)
        else:
            try:
                if int(objective) == 0:
                    obj = np.ndarray((sum([dims[n] for n in dims]), ), dtype='O')
                    obj[:] = self(0)
                else:
                    raise NotImplementedError("Got unrecognizable input for objective")
            except TypeError:
                raise NotImplementedError("Got unrecognizable input for objective")
        return self.SDP(norm, obj, [self.join(sv) for sv in res])

    def dot(self, x, y):
        # Unfortunately __numpy_ufunc__ seems to be disabled (temporarily?)
        # so I cannot override np.dot
        if isinstance(x, prefactor_numerator):
            if isinstance(y, prefactor_numerator):
                pref = x.prefactor * y.prefactor
                return prefactor_numerator(pref, np.dot(x.matrix, y.matrix), self)
            else:
                return prefactor_numerator(x.prefactor, np.dot(x.matrix, y), self)
        else:
            if isinstance(y, prefactor_numerator):
                return prefactor_numerator(y.prefactor, np.dot(x, y.matrix), self)
            return np.dot(x, y)


cdef mpfr_t* pole_integral_c(x_power_max, base, pole_position, order_of_pole, mpfr_prec_t prec):
    a = RealField(2 * prec)(pole_position)
    b = RealField(2 * prec)(base)
    if a < 0:
        incomplete_gamma = b ** a * gamma(0, a * (b.log()))
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

    cdef mpfr_t* result;
    if order_of_pole == 1:
        result = simple_pole_case_c(<long> x_power_max, <mpfr_t>(<RealNumber>b).value, <mpfr_t>(<RealNumber> a).value, <mpfr_t>(<RealNumber> incomplete_gamma).value, prec)
    elif order_of_pole == 2:
        result = double_pole_case_c(<long> x_power_max, <mpfr_t>(<RealNumber>b).value, <mpfr_t>(<RealNumber> a).value, <mpfr_t>(<RealNumber> incomplete_gamma).value, prec)
    return result


cpdef prefactor_integral(pole_data, base, int x_power, prec, c=1):
    field = RealField(prec)
    cdef int n = len(pole_data)
    cdef int number_of_factors = sum([x[1] for x in pole_data])

    cdef int count = 0
    index_list = []
    for i in range(0, len(pole_data)):
        if field(pole_data[i][0]) > 0:
            raise NotImplementedError("There exists a pole on the integration contour of the prefactor!")
        if pole_data[i][1] == 1:
            index_list.append([i, 1])
        elif pole_data[i][1] == 2:
            index_list.append([i, 2])
            index_list.append([i, 1])
        else:
            raise NotImplementedError
    if n == 0:
        minus_ln_b = -1 / ((RealField(prec)(base)).log())
        result = np.ndarray(x_power + 1, dtype='O')
        result[0] = minus_ln_b * RealField(prec)(c)
        for i in range (1, x_power + 1):
            result[i] = result[i - 1] * minus_ln_b * i
        return result
    cdef mpfr_t* pole_data_to_c = <mpfr_t*> malloc(sizeof(mpfr_t) * len(pole_data))
    if pole_data_to_c == NULL:
        raise NotImplementedError
    cdef int * is_double = <int*> malloc(sizeof(int) * len(pole_data))

    base_c = field(base);
    for i in range(0, n):
        r = field(pole_data[i][0])
        mpfr_init2(pole_data_to_c[i], prec)
        mpfr_set(pole_data_to_c[i], <mpfr_t>(<RealNumber>r).value, MPFR_RNDN)
        if pole_data[i][1] == 2:
            is_double[i] = 1
        else:
            is_double[i] = 0
    decompose_coeffs = fast_partial_fraction_c(pole_data_to_c, is_double, n, prec)

    for i in range(0, len(pole_data)):
        mpfr_clear(pole_data_to_c[i])

    free(pole_data_to_c)
    free(is_double)

    cdef mpfr_t temp1
    mpfr_init2(temp1, prec);
    cdef mpfr_t temp2
    mpfr_init2(temp2, prec);
    result = np.ndarray(x_power + 1, dtype='O')
    for i in range(0, x_power + 1):
        result[i] = <RealNumber>(<RealField_class>field)._new()
        mpfr_init2(<mpfr_t>(<RealNumber> result[i]).value, prec)
        mpfr_set_zero(<mpfr_t>(<RealNumber> result[i]).value, 1)
        (<RealNumber> result[i])._parent = field

    cdef mpfr_t* temp_mpfrs

    for i in range(0, number_of_factors):
        temp_mpfrs = pole_integral_c(x_power, base, pole_data[index_list[i][0]][0], index_list[i][1], prec)

        for j in range(0, x_power + 1):
            mpfr_mul(temp1, decompose_coeffs[i], temp_mpfrs[j], MPFR_RNDN)
            mpfr_add(<mpfr_t>(<RealNumber> result[j]).value, <mpfr_t>(<RealNumber> result[j]).value, temp1, MPFR_RNDN)
            mpfr_clear(temp_mpfrs[j])
        free(temp_mpfrs)

    for i in range(0, number_of_factors):
        mpfr_clear(decompose_coeffs[i])
    free(decompose_coeffs)
    return RealField(prec)(c) * result


cpdef anti_band_cholesky_inverse(v, n_order_max, prec):
    field = RealField(prec)
    n_max = int(n_order_max)
    if not isinstance(n_max, int):
        raise TypeError
    if len(v) < n_max * 2 + 1:
        print ("input vector is too short..")
        raise TypeError
    if n_max < 0:
        print ("expected n_max to be positive integer...")
        raise TypeError

    cdef mpfr_t* anti_band_input = <mpfr_t*> malloc(sizeof(mpfr_t) * len(v))
    for i in range(0, len(v)):
        r = field(v[i])
        mpfr_init2(anti_band_input[i], prec)
        mpfr_set(anti_band_input[i], <mpfr_t>(<RealNumber>r).value, MPFR_RNDN)
    cdef mpfr_t* anti_band_mat = form_anti_band(anti_band_input, <int>(n_max + 1), int(prec))
    for i in range(0, len(v)):
        mpfr_clear(anti_band_input[i])
    free(anti_band_input)
    cdef mpfr_t* cholesky_decomposed = mpfr_cholesky(anti_band_mat, <int>(n_max + 1), int(prec))
    for i in range(0, (n_max - 1) ** 2):
        mpfr_clear(anti_band_mat[i])
    free (anti_band_mat)

    cdef mpfr_t* inversed = mpfr_triangular_inverse(cholesky_decomposed, <int>(n_max + 1), int(prec))
    for i in range(0, (n_max + 1) ** 2):
        mpfr_clear(cholesky_decomposed[i])
    free(cholesky_decomposed)

    result = np.ndarray([n_max + 1, n_max + 1], dtype='O')
    for i in range(0, n_max + 1):
        for j in range(0, n_max + 1):
            result[i][j] = <RealNumber>(<RealField_class>field)._new()
            mpfr_init2(<mpfr_t>(<RealNumber>result[i][j]).value, prec)
            mpfr_set(<mpfr_t>(<RealNumber>result[i][j]).value, inversed[i * (n_max + 1) + j], MPFR_RNDN)
            (<RealNumber>result[i][j])._parent = field
            mpfr_clear(inversed[i * (n_max + 1) + j])

    free(inversed)
    return result


def max_index(_v):
    return sorted(map(lambda x, y: [x, y], _v, range(
        0, len(_v))), key=lambda x: x[0].abs(), reverse=True)[0][1]


def normalizing_component_subtract(m, normalizing_vector):
    __index = max_index(normalizing_vector)
    __deleted_normalizing_vector = (
        1 / normalizing_vector[__index]) * np.delete(normalizing_vector, __index)
    if len(m) != len(normalizing_vector):
        raise RuntimeError(
            "length of normalizing vector and target object must be equal.")
    return np.insert(
        np.delete(
            m,
            __index,
            0) -
        __deleted_normalizing_vector *
        m[__index],
        0,
        m[__index] /
        normalizing_vector[__index])


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
    file_stream.write(("<" + tag + ">"))
    file_stream.write(repr(real_num))
    file_stream.write(("</" + tag + ">\n"))


def write_vector(file_stream, name, vector):
    file_stream.write("<" + name + ">\n")
    map(lambda x: write_real_num(file_stream, x, "elt"), vector)
    file_stream.write("</" + name + ">\n")


def write_polynomial(file_stream, polynomial):
    file_stream.write("<polynomial>\n")
    try:
        __temp = polynomial.list()
    except AttributeError:
        __temp = [polynomial]
    if __temp == []:
        __temp = [0]
    map(lambda x: write_real_num(file_stream, x, "coeff"), __temp)
    file_stream.write("</polynomial>\n")


def write_polynomial_vector(file_stream, polynomialVector):
    file_stream.write("<polynomialVector>\n")
    map(lambda x: write_polynomial(file_stream, x), polynomialVector)
    file_stream.write("</polynomialVector>\n")


def laguerre_sample_points(n, field, rho):
    return map(lambda k: (field(3.141592)) ** 2 * (-1 + 4 * k)
               ** 2 / (-64 * n * (4 * rho).log()), range(0, n))


def format_poleinfo(poles, context=None):
    if context is None:
        def field(x): return x
    else:
        field = context.field
    if isinstance(poles, dict):
        res = [[field(x), poles[x]] for x in poles]
        return dict(res)
    if isinstance(poles, list):
        if poles == []:
            return dict()
        if not isinstance(poles[0], list):
            m = dict([[x, 1] for x in poles])
            for x in m:
                m[x] = poles.count(x)
            return dict([[field(x), m[x]] for x in m])
        if len(poles[0]) == 2:
            try:
                res = [[field(x[0]), x[1]] for x in poles]
                return dict(res)
            except TypeError:
                raise TypeError("unreadable initialization for poles")
        raise TypeError("unreadable initialization for poles")



cdef class damped_rational:
    def __cinit__(self, poles, base, c, cb_universal_context context):
        self.base = context.field(base)
        self.pref_constant = context.field(c)
        self.context = context

    def __init__(self, poles, base, c, cb_universal_context context):
        self.poles = format_poleinfo(poles, context)

    def shift(self, shift):
        new_poles = [[x - shift, self.poles[x]] for x in self.poles]
        new_const = self.pref_constant * self.base ** shift
        return damped_rational(new_poles, self.base, new_const, self.context)

    def __call__(self, x):
        return self.pref_constant * (self.base ** x) * (1 / reduce(lambda z, w: z * w, [(x - y) ** (self.poles[y]) for y in self.poles], 1))

    def orthogonal_polynomial(self, order):
        passed_poles = [[x, self.poles[x]] for x in self.poles]
        return anti_band_cholesky_inverse(prefactor_integral(passed_poles, self.base, order, self.context.precision, self.pref_constant), order // 2, self.context.precision)

    def __mul__(self, y):
        if isinstance(y, damped_rational):
            res_poles = copy.copy(self.poles)
            orig_keys = set(res_poles)
            for x in y.poles:
                if x in orig_keys:
                    res_poles[x] = res_poles[x] + y.poles[x]
                else:
                    res_poles[x] = y.poles[x]
            new_base = self.base * y.base
            new_const = self.pref_constant * y.pref_constant
            return damped_rational(res_poles, new_base, new_const, self.context)
        else:
            raise TypeError("damped_rational must be multiplied with itself")

    def add_poles(self, location):
        location_new = format_poleinfo(location)
        res_poles = copy.copy(self.poles)
        for x in location_new:
            if x in res_poles:
                res_poles[x] = res_poles[x] + location_new[x]
            else:
                res_poles[x] = location_new[x]
        return damped_rational(res_poles, self.base, self.pref_constant, self.context)

    def remove_poles(self, location):
        res_poles = copy.copy(self.poles)
        location_new = format_poleinfo(location)
        for x in location_new:
            if x not in res_poles:
                raise RuntimeError("could not delete pole")
            ind = res_poles[x] - location_new[x]
            if ind > 0:
                res_poles[x] = ind
            elif ind == 0:
                del res_poles[x]
            else:
                raise RuntimeError("could not delete pole")
        return damped_rational(res_poles, self.base, self.pref_constant, self.context)

    def lcm(self, p):
        if isinstance(p, damped_rational):
            if self.base != p.base:
                raise RuntimeError("two damped-rational must have the same base!")
            if p == self:
                return (self, dict(), dict())
        else:
            raise TypeError("lcm supported only between damped_rationals")

        dict1 = self.poles
        dict2 = p.poles

        def help_lcm(x):
            val1 = dict1[x]
            val2 = dict2[x]
            if val1 > val2:
                return ((x, val1), (x, 0), (x, val1 - val2))
            if val2 > val1:
                return ((x, val2), (x, val2 - val1), (x, 0))
            return ((x, val2), (x, 0), (x, 0))

        result_1, self_rem, p_rem = \
                zip(*(help_lcm(x) for x in dict2 if x in dict1))

        l1 = [(x, dict2[x]) for x in dict2 if x not in dict1]
        l2 = [(x, dict1[x]) for x in dict1 if x not in dict2]

        result_poles = dict(list(result_1) + l1 + l2)
        numerator_for_self = dict(l1 + [x for x in self_rem if x[1] != 0])
        numerator_for_p = dict(l2 + [x for x in p_rem if x[1] != 0])

        res = damped_rational(result_poles, self.base, \
                self.context(1), self.context)
        return res, numerator_for_self, numerator_for_p

    def __repr__(self):
        output = repr(self.pref_constant) + "*(" + repr(self.base) + ")**Delta /"
        for x in self.poles:
            output = output + "(Delta"
            if x > 0:
                output = output + "-" + repr(x) + ")"
            elif x == 0:
                output = output + ")"
            else:
                output = output + "+" + repr(-x) + ")"
            if self.poles[x] != 1:
                output = output + "**" + repr(self.poles[x])
            output = output + "*"
        return output[:-1]

    def __richcmp__(x, y, op):
        # 0: <, 1: <=, 2: ==, 3: !=, 4: >, 5: >=
        if op == 2:
            return x.__is_equal(y)
        if op == 3:
            return not x.__is_equal(y)
        assert False

    def __is_equal(self, x):
        if not isinstance(x, damped_rational):
            return False
        return self.base == x.base and self.pref_constant == x.pref_constant \
                and self.poles == x.poles

cdef class positive_matrix_with_prefactor:
    def __cinit__(self, damped_rational prefactor, matrix, cb_universal_context context):
        self.prefactor = prefactor
        self.context = context

    def __init__(self, damped_rational prefactor, matrix, context):
        self.matrix = (matrix)

    def shift(self, x):
        return positive_matrix_with_prefactor(self.prefactor.shift(x), self.context.polynomial_vector_shift(self.matrix, x), self.context)

    def degree_max(self):
        try:
            return max((np.vectorize(lambda y: self.context.Delta_Field(y).degree())(self.matrix)).flatten())
        except AttributeError:
            return 0

    def normalization_subtract(self, v):
        return normalizing_component_subtract(self.matrix, v)

    def write(self, file_stream, v):
            shuffled_matrix = np.array(map(lambda x: map(lambda y: normalizing_component_subtract(y, v), x), self.matrix))
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
            map(lambda x: map(lambda y: write_polynomial_vector(file_stream, y), x), shuffled_matrix)
            file_stream.write("</elements>\n")
            write_vector(file_stream, "samplePoints", sample_points)
            write_vector(file_stream, "sampleScalings", sample_scalings)
            file_stream.write("<bilinearBasis>\n")
            map(lambda x : write_polynomial(file_stream, x), orthogonal_polynomial_vector)
            file_stream.write("</bilinearBasis>\n")
            file_stream.write("</polynomialVectorMatrix>\n")

    def reshape(self, shape=None):
        if len(self.matrix.shape) == 3 and self.matrix.shape[0] == self.matrix.shape[1] and not shape:
            return self
        if not shape:
            shape = (1, 1, self.matrix.shape[-1])
        new_b = self.matrix.reshape(shape)
        return prefactor_numerator(self.prefactor, new_b, self.context)


cdef class prefactor_numerator(positive_matrix_with_prefactor):
    def add_poles(self, poles):
        new_pref = self.prefactor.add_poles(poles)
        return prefactor_numerator(new_pref, self.matrix, self.context)

    def rdot(self, M):
        newBody = M.dot(self.matrix)
        return prefactor_numerator(self.prefactor, newBody, self.context)

    def shift(self, x):
        return prefactor_numerator(self.prefactor.shift(x), self.context.polynomial_vector_shift(self.matrix, x), self.context)

    def multiply_factored_polynomial(self, factors, C):
        """
        multiply C * \\Prod_x  (Delta - x) ** (factors[x])!
        where x in factors
        """
        formated_poles = format_poleinfo(factors)
        res_poles = copy.copy(self.prefactor.poles)
        numr_new = format_poleinfo(factors)
        pole_keys = set(res_poles)
        remnant = dict()
        for x in numr_new:
            if x in pole_keys:
                val_pole = res_poles[x]
                val_numr = numr_new[x]
                if val_pole < val_numr:
                    del res_poles[x]
                    remnant[x] = val_numr - val_pole
                elif val_pole == val_numr:
                    del res_poles[x]
                else:
                    res_poles[x] = val_pole - val_numr
            else:
                remnant[x] = numr_new[x]
        remnant_poly = self.prefactor.pref_constant * reduce(lambda x, y: x * y, [(self.context.Delta - z) ** remnant[z] for z in remnant], 1)
        result_numr = C * remnant_poly * self.matrix
        result_pref = damped_rational(res_poles, self.prefactor.base, 1, self.context)
        return prefactor_numerator(result_pref, result_numr, self.context)

    def multiply_factored_rational(self, poles, factors, C):
        return self.add_poles(poles).multiply_factored_polynomial(factors, C)

    def __rmul__(self, x):
        new_mat = x * self.matrix
        return prefactor_numerator(self.prefactor, new_mat, self.context)

    def __neg__(self):
        new_mat = -self.matrix
        return prefactor_numerator(self.prefactor, new_mat, self.context)

    def __div__(self, x):
        new_mat = self.matrix / x
        return prefactor_numerator(self.prefactor, new_mat, self.context)

    def __add__(self, other):
        if not isinstance(other, prefactor_numerator):
            raise TypeError("must be added to another prefactor_numerator")
        new_pref, remnant_1, remnant_2 = self.prefactor.lcm(other.prefactor)
        res_poles = new_pref.poles
        remnant_poly1 = reduce(lambda x, y: x * y, [(self.context.Delta - z) ** remnant_1[z] for z in remnant_1], \
        self.prefactor.pref_constant)
        remnant_poly2 = reduce(lambda x, y: x * y, [(self.context.Delta - z) ** remnant_2[z] for z in remnant_2], \
        other.prefactor.pref_constant)
        new_matrix = remnant_poly1 * self.matrix \
                + remnant_poly2 * other.matrix
        return prefactor_numerator(new_pref, new_matrix, self.context)

    def new_join(self, other):
        if not isinstance(other, prefactor_numerator):
            raise TypeError("must be joined with another prefactor_numerator instance")
        new_pref, remnant_1, remnant_2 = self.prefactor.lcm(other.prefactor)
        res_poles = new_pref.poles
        remnant_poly1 = reduce(lambda x, y: x * y, [(self.context.Delta - z) ** remnant_1[z] for z in remnant_1], \
        self.prefactor.pref_constant)
        remnant_poly2 = reduce(lambda x, y: x * y, [(self.context.Delta - z) ** remnant_2[z] for z in remnant_2], \
        other.prefactor.pref_constant)
        new_matrix = np.concatenate((remnant_poly1 * self.matrix, remnant_poly2 * other.matrix))
        return prefactor_numerator(new_pref, new_matrix, self.context)

    def __sub__(self, x):
        if not isinstance(x, prefactor_numerator):
            raise TypeError("must be added to another prefactor_numerator")
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
        label = range(0, len(polys))
    efmread = map(lambda x: find_local_minima(x[0], x[1]), zip(polys, label))
    return efmread


class SDP:
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
        file_stream = open(file_path, 'w')
        file_stream.write("<sdp>\n")
        write_vector(file_stream, "objective", normalizing_component_subtract(
            self.objective, self.normalization))
        file_stream.write("<polynomialVectorMatrices>\n")
        for x in self.pvm:
            x.write(file_stream, self.normalization)
        file_stream.write("</polynomialVectorMatrices>\n")
        file_stream.write("</sdp>\n")
        file_stream.close()
