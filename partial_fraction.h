#ifndef SAGE_CBOOT_PARTIAL_FRACTION_H
#define SAGE_CBOOT_PARTIAL_FRACTION_H

#include <stdio.h>
#include <stdlib.h>

#include "mpfr.h"

mpfr_t* fast_partial_fraction_c(const mpfr_t* pole_locations, int expected_result_length, mpfr_prec_t prec);

#endif
