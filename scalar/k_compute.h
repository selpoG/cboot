#ifndef SAGE_CBOOT_SCALAR_K_COMPUTE_H
#define SAGE_CBOOT_SCALAR_K_COMPUTE_H

#include <stdio.h>
#include <stdlib.h>

#include "mpfr.h"

#include "context_variables.h"
#include "hor_recursion.h"

mpfr_t* k_table_c(mpfr_t h, mpfr_t S, mpfr_t P, cb_context context);
mpfr_t* chiral_h_times_rho_to_n_c(unsigned long n, mpfr_t h, mpfr_t S, mpfr_t P, cb_context context);
mpfr_t* chiral_h_asymptotic_c(mpfr_t S, cb_context context);

#endif
