#ifndef SAGE_CBOOT_CONTEXT_VARIABLES_H
#define SAGE_CBOOT_CONTEXT_VARIABLES_H

#include <stdlib.h>

#include "mpfr.h"

typedef struct _context {
    long n_Max;
    mpfr_prec_t prec;
    mpfr_rnd_t rnd;
    int lambda;
    mpfr_t* rho_to_z_matrix;
    mpfr_t rho;
} cb_context;

/* basic constructor for cb_context */
cb_context context_construct(long n_Max, mpfr_prec_t prec, int lambda);
void clear_cb_context(cb_context context);

#endif
