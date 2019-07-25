import cython

from sage.libs.mpfr cimport mpfr_t
from sage.cboot.context_object cimport cb_context, cb_universal_context

cdef extern from "stdlib.h":
    void free(void* ptr)

cdef extern from "hor_recursion.h":
    mpfr_t* h_asymptotic(mpfr_t epsilon, mpfr_t S, cb_context context)
    mpfr_t* gBlock_full(mpfr_t epsilon, mpfr_t ell, mpfr_t Delta, mpfr_t S, mpfr_t P, cb_context context)
    mpfr_t* hBlock_times_rho_n(unsigned long n, mpfr_t epsilon, mpfr_t ell, mpfr_t Delta, mpfr_t S, mpfr_t P, cb_context context)

cdef extern from "k_compute.h":
    mpfr_t* k_table_c(mpfr_t h, mpfr_t S, mpfr_t P, cb_context context)
    mpfr_t* chiral_h_times_rho_to_n_c(unsigned long n, mpfr_t h, mpfr_t S, mpfr_t P, cb_context context)
    mpfr_t* chiral_h_asymptotic_c(mpfr_t S, cb_context context)

cdef class scalar_cb_context_generic(cb_universal_context):
    cdef public object epsilon
    @cython.locals(array=cython.pointer(mpfr_t))
    cpdef h_times_rho_k(self, unsigned long k, ell, Delta, S, P)
    @cython.locals(_array=cython.pointer(mpfr_t))
    cpdef h_asymptotic_form(self, S)
    @cython.locals(array=cython.pointer(mpfr_t))
    cpdef gBlock(self, ell, Delta, Delta_1_2, Delta_3_4)

cdef class scalar_cb_2d_context(scalar_cb_context_generic):
    @cython.locals(_array=cython.pointer(mpfr_t))
    cpdef chiral_h_asymptotic(self, S)
    @cython.locals(_array=cython.pointer(mpfr_t))
    cpdef __chiral_h_times_rho_to_n__impl(self, long n, h, Delta_1_2, Delta_3_4)
    @cython.locals(_array=cython.pointer(mpfr_t))
    cpdef k_table(self, h, Delta_1_2, Delta_3_4)

cdef class scalar_cb_4d_context(scalar_cb_context_generic):
    cdef public scalar_cb_2d_context k_context
    cdef public object zzbar_anti_symm_to_xy_matrix
