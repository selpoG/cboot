#ifndef SAGE_CBOOT_CHOL_AND_INVERSE_H
#define SAGE_CBOOT_CHOL_AND_INVERSE_H

#include "mpfr.h"

mpfr_t* anti_band_to_inverse(const mpfr_t* ab_vector, int dim, mpfr_prec_t prec);

#endif
