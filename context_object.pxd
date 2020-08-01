import cython
cimport numpy as np

from sage.libs.mpfr cimport (
    mpfr_add_ui, mpfr_add, mpfr_clear, mpfr_fma, mpfr_init2, mpfr_mul,
    mpfr_prec_t, MPFR_RNDN, mpfr_set_ui, mpfr_set_zero, mpfr_set, mpfr_t)
from sage.rings.real_mpfr cimport RealField_class, RealNumber

cdef extern from "stdlib.h":
    void* calloc(size_t num, size_t size)
    void free(void* ptr)

cdef extern from "sage/cboot/integral_decomp.h":
    mpfr_t* simple_pole_case_c(long pole_order_max, mpfr_t base, mpfr_t pole_position, mpfr_t incomplete_gamma_factor, mpfr_prec_t prec)

cdef extern from "sage/cboot/partial_fraction.h":
    mpfr_t* fast_partial_fraction_c(const mpfr_t* pole_locations, int expected_result_length, mpfr_prec_t prec)

cdef extern from "sage/cboot/chol_and_inverse.h":
    mpfr_t* anti_band_to_inverse(const mpfr_t* ab_vector, int dim, mpfr_prec_t prec)

cdef extern from "sage/cboot/context_variables.h":
    ctypedef struct cb_context:
        mpfr_t* rho_to_z_matrix
    cb_context context_construct(long nMax, mpfr_prec_t prec, int Lambda)
    void clear_cb_context(cb_context context)

cdef np.ndarray mpfr_move_to_ndarray_1(mpfr_t* ptr, int dim, RealField_class context)
cdef np.ndarray mpfr_move_to_ndarray_2(mpfr_t* ptr, int dim1, int dim2, RealField_class context, bint delete)
cdef mpfr_t* __pole_integral_c(x_power_max, base, pole_position, mpfr_prec_t prec)
cpdef __prefactor_integral(poles, base, int x_power, prec, c)
cpdef __anti_band_cholesky_inverse(v, n_order_max, prec)

cdef class cb_universal_context(object):
    cdef cb_context c_context
    cdef readonly mpfr_prec_t precision
    cdef readonly RealField_class field
    cdef readonly object Delta_Field
    cdef readonly object Delta
    cdef readonly int Lambda
    cdef readonly RealNumber rho
    cdef readonly int maxExpansionOrder
    cdef readonly object polynomial_vector_shift
    cdef readonly object polynomial_vector_evaluate
    cdef readonly object convert_to_polynomial_vector
    cdef readonly object convert_to_real_vector
    cdef readonly object rho_to_z_matrix
    cdef readonly object zzbar_to_xy_marix
    cdef readonly object index_list
    cdef readonly object rho_to_delta
    cdef readonly object null_ftype
    cdef readonly object null_htype
    cpdef pochhammer(self, x, unsigned long n)

cdef class damped_rational(object):
    cdef readonly object __poles
    cdef readonly RealNumber __base
    cdef readonly RealNumber __pref_constant
    cdef readonly cb_universal_context __context

cdef class prefactor_numerator(object):
    cdef readonly damped_rational __prefactor
    cdef readonly cb_universal_context __context
    cdef readonly object __matrix
