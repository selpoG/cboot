#include "partial_fraction.h"

#include <stdlib.h>

// pole_locations = {p0, p1, ...}
// partial fraction of 1 / [(x - p0) * (x - p1) * ...]
// for example, if pole_locations = {p, q, r}
// this functions returns {a, b, c} s.t.
// 1 / [(x - p) * (x - q) * (x - r)]
// = a / (x - p) + b / (x - q) + c / (x - r)
// a = 1 / ((p - q) * (p - r))
// b = 1 / ((q - p) * (q - r))
// c = 1 / ((r - p) * (r - q))
mpfr_t* fast_partial_fraction_c(const mpfr_t* pole_locations, int n_poles, mpfr_prec_t prec) {
    mpfr_t* result = calloc((unsigned int)n_poles, sizeof(mpfr_t));
    mpfr_t temp1;
    mpfr_init2(temp1, prec);
    for (int i = 0; i < n_poles; i++) {
        mpfr_init2(result[i], prec);
        mpfr_set_ui(result[i], 1, MPFR_RNDN);
        // product (pole[i] - pole[j]), j != i
        for (int j = 0; j < n_poles; j++) {
            if (i != j) {
                // temp1 = pole[i] - pole[j]
                // result[c] *= temp1
                mpfr_sub(temp1, pole_locations[i], pole_locations[j], MPFR_RNDN);
                mpfr_mul(result[i], result[i], temp1, MPFR_RNDN);
            }
        }
        mpfr_ui_div(result[i], 1, result[i], MPFR_RNDN);
    }
    mpfr_clear(temp1);
    return result;
}
