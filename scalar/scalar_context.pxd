import cython
from cysignals.signals cimport sig_on, sig_off

from sage.rings.real_mpfr cimport RealField_class, RealNumber
from sage.libs.mpfr cimport (
    mpfr_clear, mpfr_init2, mpfr_prec_t,
    MPFR_RNDN, mpfr_set, mpfr_t, mpfr_zero_p)

from sage.cboot.context_object cimport (
    cb_context, cb_universal_context, mpfr_move_to_ndarray_1)

cdef extern from "stdlib.h":
    void free(void* ptr)

cdef extern from "hor_recursion.h":
    mpfr_t* h_asymptotic(mpfr_t epsilon, mpfr_t S, cb_context context)
    mpfr_t* gBlock_full(mpfr_t epsilon, mpfr_t ell, mpfr_t Delta, mpfr_t S, mpfr_t P, cb_context context)
    mpfr_t* hBlock_times_rho_n(unsigned long n, mpfr_t epsilon, mpfr_t ell, mpfr_t Delta, mpfr_t S, mpfr_t P, cb_context context)

cdef cython.bint RealNumber_is_zero(RealNumber num)

cdef class scalar_context(cb_universal_context):
    cdef readonly object epsilon
    cpdef h_times_rho_k(self, unsigned long k, ell, Delta, S, P)
    cpdef h_asymptotic_form(self, S)
    cpdef gBlock(self, ell, Delta, Delta_1_2, Delta_3_4)
