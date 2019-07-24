from sage.libs.mpfr cimport mpfr_t
from sage.cboot.context_object cimport cb_context

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
