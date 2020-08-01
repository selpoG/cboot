#include "integral_decomp.h"

#include <stdlib.h>

#undef debug_mode
// #define debug_mode 1

mpfr_t* divergent_simple_pole(long pole_order_max, mpfr_t base, mpfr_t incomplete_gamma_factor, mpfr_prec_t prec) {
    mpfr_t* result = calloc(pole_order_max + 1, sizeof(mpfr_t));
    mpfr_init2(result[0], prec);
    // result[0] = incomplete_gamma_factor;
    mpfr_set(result[0], incomplete_gamma_factor, MPFR_RNDN);
    if (pole_order_max == 0) {
        return result;
    }
    mpfr_t ik;
    mpfr_init2(ik, prec);
    // k = log(base);
    mpfr_log(ik, base, MPFR_RNDN);
    // k = -1 / k;
    mpfr_si_div(ik, -1, ik, MPFR_RNDN);
    // k == -1 / log(base);
    mpfr_init2(result[1], prec);
    mpfr_set(result[1], ik, MPFR_RNDN);
    for (long j = 2; j <= pole_order_max; j++) {
        mpfr_init2(result[j], prec);
        // result[j] = result[j - 1] * (j - 1);
        mpfr_mul_ui(result[j], result[j - 1], j - 1, MPFR_RNDN);
        // result[j] *= ik;
        mpfr_mul(result[j], result[j], ik, MPFR_RNDN);
    }
    mpfr_clear(ik);
    return result;
}

// let pole_position = -p, base = \\exp(-k)
// calculate \\int_{0}^{\\infty} \\frac{e^{-k x} x^n}{x + p} dx, for n = 0, ..., pole_order_max
// this integral equals to
// n! p^n e^{p k} \\Gamma(-n, p k)
// = (-p)^n e^{p k} \\Gamma(0, p k)
//   + \\frac{1}{k^n}\\sum_{i = 0}^{n - 1} (n - i - 1)! (-p k)^i
// incomplete_gamma_factor = e^{p k} \\Gamma(0, p k)
mpfr_t* simple_pole_case_c(long pole_order_max, mpfr_t base, mpfr_t pole_position, mpfr_t incomplete_gamma_factor,
                           mpfr_prec_t prec) {
    if (mpfr_zero_p(pole_position)) {
        return divergent_simple_pole(pole_order_max, base, incomplete_gamma_factor, prec);
    }
    mpfr_t* result = calloc(pole_order_max + 1, sizeof(mpfr_t));

    mpfr_t temp1;
    mpfr_init2(temp1, prec);
    mpfr_t temp2;
    mpfr_init2(temp2, prec);

    mpfr_t minus_pole_position;
    mpfr_init2(minus_pole_position, prec);
    // minus_pole_position = -pole_position;
    mpfr_neg(minus_pole_position, pole_position, MPFR_RNDN);

    mpfr_t factorial;
    mpfr_init2(factorial, prec);
    // factorial = 1;
    mpfr_set_ui(factorial, 1, MPFR_RNDN);

    mpfr_t minus_log_base;
    mpfr_init2(minus_log_base, prec);
    // minus_log_base = log(base);
    mpfr_log(minus_log_base, base, prec);
    // minus_log_base = -1 / minus_log_base;
    mpfr_si_div(minus_log_base, -1, minus_log_base, MPFR_RNDN);
    // minus_log_base == -1 / log(base);

    mpfr_t log_base_power;
    mpfr_init2(log_base_power, prec);
    // log_base_power = minus_log_base;
    mpfr_set(log_base_power, minus_log_base, MPFR_RNDN);

    // temp1 = 0;
    mpfr_set_ui(temp1, 0, MPFR_RNDN);
    // temp2 = pole_position * incomplete_gamma_factor;
    mpfr_mul(temp2, pole_position, incomplete_gamma_factor, MPFR_RNDN);

    mpfr_init2(result[0], prec);
    // result[0] = incomplete_gamma_factor;
    mpfr_set(result[0], incomplete_gamma_factor, MPFR_RNDN);
    for (long j = 1; j <= pole_order_max; j++) {
        // factorial == (j - 1)!;
        // temp2 == incomplete_gamma_factor * pow(pole_position, j);
        // log_base_power == pow(minus_log_base, j);
        mpfr_init2(result[j], prec);
        // temp1 = factorial * log_base_power + temp1 * pole_position;
        mpfr_fmma(temp1, factorial, log_base_power, temp1, pole_position, MPFR_RNDN);
        // temp1 == sum((j - k - 1)! * pow(pole_position, k) * pow(minus_log_base, j - k), 0 <= k < j);
        // result[j] = temp1 + temp2
        mpfr_add(result[j], temp1, temp2, MPFR_RNDN);
        // result[j] == sum((j - k - 1)! * pow(pole_position, k) * pow(minus_log_base, j - k), 0 <= k < j)
        //              + incomplete_gamma_factor * pow(pole_position, j);

        if (j < pole_order_max) {
            // temp2 *= pole_position;
            mpfr_mul(temp2, temp2, pole_position, MPFR_RNDN);
            // log_base_power *= minus_log_base;
            mpfr_mul(log_base_power, log_base_power, minus_log_base, MPFR_RNDN);
            // factorial *= j;
            mpfr_mul_ui(factorial, factorial, j, MPFR_RNDN);
        }
    }

    mpfr_clear(temp1);
    mpfr_clear(temp2);
    mpfr_clear(minus_pole_position);
    mpfr_clear(factorial);
    mpfr_clear(minus_log_base);
    mpfr_clear(log_base_power);
    return result;
}
