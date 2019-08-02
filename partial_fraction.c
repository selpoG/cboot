#include "partial_fraction.h"

#include <stdlib.h>

// pole_locations = {p0, p1, ...}
// partial fraction of 1 / [(x - p0) * (x - p1) * ...]
// if double_or_single[i], replace (x - pi) by (x - pi) ^ 2
// for example, if pole_locations = {p, q, r} and double_or_single = {1, 0, 1},
// this functions returns {a, b, c, d, e} s.t.
// 1 / [(x - p) ^ 2 * (x - q) * (x - r) ^ 2]
// = a / (x - p) ^ 2 + b / (x - p) + c / (x - q) + d / (x - r) ^ 2 + e / (x - r)
// a = 1 / ((p - q) * (p - r) ^ 2)
// c = 1 / ((q - p) ^ 2 * (q - r) ^ 2)
// d = 1 / ((r - p) ^ 2 * (r - q))
// 1 / [(x - q) * (x - r) ^ 2]
// = a + b * (x - p) + (x - p) ^2 * (...)
// -1 / [(x - q) ^ 2 * (x - r) ^ 2] - 2 / [(x - q) * (x - r) ^ 3]
// = b + (x - p) * (...)
// 1 / (p - q) + 2 / (p - r)
// = -b / a
mpfr_t* fast_partial_fraction_c(const mpfr_t* pole_locations, const int* double_or_single, int n_poles,
                                mpfr_prec_t prec) {
    int expected_result_length = n_poles;
    for (int i = 0; i < n_poles; i++) {
        // mpfr_printf("given pole_locations[%d]=%.16RNf\n", i, pole_locations[i]);
        // printf("given double_or_single[%d]=%d\n", i, double_or_single[i]);
        if (double_or_single[i]) {
            ++expected_result_length;
        }
    }
    mpfr_t* result = calloc((unsigned int)expected_result_length, sizeof(mpfr_t));
    mpfr_t temp1;
    mpfr_init2(temp1, prec);
    int count_result_location = 0;
    for (int i = 0; i < n_poles; i++) {
        mpfr_init2(result[count_result_location], prec);
        mpfr_set_ui(result[count_result_location], 1, MPFR_RNDN);
        // product (pole[i] - pole[j]) ^ (1 or 2), j != i
        for (int j = 0; j < n_poles; j++) {
            if (i != j) {
                // temp1 = pole[i] - pole[j]
                // result[c] *= temp1
                mpfr_sub(temp1, pole_locations[i], pole_locations[j], MPFR_RNDN);
                mpfr_mul(result[count_result_location], result[count_result_location], temp1, MPFR_RNDN);
                if (double_or_single[j]) {
                    // result[c] *= temp1
                    mpfr_mul(result[count_result_location], result[count_result_location], temp1, MPFR_RNDN);
                }
            }
        }
        mpfr_ui_div(result[count_result_location], 1, result[count_result_location], MPFR_RNDN);
        if (double_or_single[i]) {
            ++count_result_location;
            mpfr_init2(result[count_result_location], prec);
            mpfr_set_zero(result[count_result_location], 1);
            for (int j = 0; j < n_poles; j++) {
                if (i != j) {
                    // temp1 = pole[i] - pole[j]
                    // temp1 = (1 or 2) / temp1
                    mpfr_sub(temp1, pole_locations[i], pole_locations[j], MPFR_RNDN);
                    if (double_or_single[j]) {
                        mpfr_ui_div(temp1, 2, temp1, MPFR_RNDN);
                    } else {
                        mpfr_ui_div(temp1, 1, temp1, MPFR_RNDN);
                    }
                    // result[c] += temp1
                    mpfr_add(result[count_result_location], result[count_result_location], temp1, MPFR_RNDN);
                }
            }
            // result[c] = -result[c] * result[c - 1]
            mpfr_mul(result[count_result_location], result[count_result_location], result[count_result_location - 1],
                     MPFR_RNDN);
            mpfr_neg(result[count_result_location], result[count_result_location], MPFR_RNDN);
        }
        ++count_result_location;
    }
    mpfr_clear(temp1);
    return result;
}
