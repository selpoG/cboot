import cython

from sage.libs.mpfr cimport mpfr_t, mpfr_prec_t
from sage.rings.real_mpfr cimport RealField_class, RealNumber

cdef extern from "stdlib.h":
    void* malloc(size_t size)
    void free(void* ptr)

cdef extern from "sage/cboot/integral_decomp.h":
    mpfr_t* simple_pole_case_c(long pole_order_max, mpfr_t base, mpfr_t pole_position, mpfr_t incomplete_gamma_factor, mpfr_prec_t prec)
    mpfr_t* double_pole_case_c(long pole_order_max, mpfr_t base, mpfr_t pole_position, mpfr_t incomplete_gamma_factor, mpfr_prec_t prec)

cdef extern from "sage/cboot/partial_fraction.h":
    mpfr_t* fast_partial_fraction_c(mpfr_t* pole_locations, int* double_or_single, int expected_result_length, mpfr_prec_t prec)

cdef extern from "sage/cboot/chol_and_inverse.h":
    mpfr_t* mpfr_triangular_inverse(mpfr_t* A, int dim, mpfr_prec_t prec)
    mpfr_t* mpfr_cholesky(mpfr_t* A, int dim, mpfr_prec_t prec)
    mpfr_t* form_anti_band(mpfr_t* ab_vector, int dim, mpfr_prec_t prec)

cdef extern from "sage/cboot/context_variables.h":
    ctypedef struct cb_context:
        mpfr_t* rho_to_z_matrix
    cb_context context_construct(long nMax, mpfr_prec_t prec, int Lambda)
    void clear_cb_context(cb_context context)

@cython.locals(result=cython.pointer(mpfr_t))
cdef mpfr_t* pole_integral_c(x_power_max, base, pole_position, order_of_pole, mpfr_prec_t prec)

cdef class cb_universal_context(object):
    cdef cb_context c_context
    cdef public mpfr_prec_t precision
    cdef public RealField_class field
    cdef public object Delta_Field
    cdef public object Delta
    cdef public int Lambda
    cdef public RealNumber rho
    cdef public int maxExpansionOrder
    cdef public object polynomial_vector_shift
    cdef public object polynomial_vector_evaluate
    cdef public object convert_to_polynomial_vector
    cdef public object convert_to_real_vector
    cdef public object rho_to_z_matrix
    cdef public object zzbar_to_xy_marix
    cdef public object index_list
    cdef public object rho_to_delta
    cdef public object null_ftype
    cdef public object null_htype
    @cython.locals(temp1=mpfr_t)
    cpdef pochhammer(self, x, unsigned long n)

cdef class damped_rational(object):
    cdef public object __poles
    cdef public RealNumber __base
    cdef public RealNumber __pref_constant
    cdef public cb_universal_context __context

cdef class positive_matrix_with_prefactor(object):
    cdef public damped_rational prefactor
    cdef public cb_universal_context context
    cdef public object matrix

cdef class prefactor_numerator(positive_matrix_with_prefactor):
    pass
