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

// let pole_position = -p, base = \\exp(-k)
// calculate \\int_{0}^{\\infty} \\frac{e^{-k x} x^n}{(x + p)^2} dx, for n = 0, ..., pole_order_max
mpfr_t* double_pole_case_c(long pole_order_max, mpfr_t base, mpfr_t pole_position, mpfr_t incomplete_gamma_factor,
                           mpfr_prec_t prec) {
    mpfr_t* result = calloc(pole_order_max + 1, sizeof(mpfr_t));

    mpfr_t temp;
    mpfr_init2(temp, prec);

    mpfr_t minus_pole_position;
    mpfr_init2(minus_pole_position, prec);
    // minus_pole_position = -pole_position;
    mpfr_neg(minus_pole_position, pole_position, MPFR_RNDN);

    mpfr_t minus_log_base;
    mpfr_init2(minus_log_base, prec);
    // minus_log_base = log(base);
    mpfr_log(minus_log_base, base, prec);
    // minus_log_base = -1 / minus_log_base;
    mpfr_si_div(minus_log_base, -1, minus_log_base, MPFR_RNDN);
    // minus_log_base == -1 / log(base)

    mpfr_t log_base_power;
    mpfr_init2(log_base_power, prec);
    // log_base_power = minus_log_base;
    mpfr_set(log_base_power, minus_log_base, MPFR_RNDN);

    // temp = log(base);
    mpfr_log(temp, base, MPFR_RNDN);
    mpfr_init2(result[0], prec);
    // result[0] = incomplete_gamma_factor * temp;
    mpfr_mul(result[0], incomplete_gamma_factor, temp, MPFR_RNDN);
    // temp = 1 / minus_pole_position;
    mpfr_ui_div(temp, 1, minus_pole_position, MPFR_RNDN);
    // result[0] += temp;
    mpfr_add(result[0], result[0], temp, MPFR_RNDN);
    // result[0] == 1 / minus_pole_position + incomplete_gamma_factor * log(base);

    for (int i = 1; i <= pole_order_max; i++) {
        mpfr_init2(result[i], prec);
        // result[i] = result[i - 1] * pole_position;
        mpfr_mul(result[i], result[i - 1], pole_position, MPFR_RNDN);
    }

#ifdef debug_mode
    for (int i = 0; i <= pole_order_max; i++) {
        mpfr_printf("result[%d] = %.16RNf\n", i, result[i], MPFR_RNDN);
    }
#endif

    mpfr_t* factorial_times_power_lnb = NULL;
    mpfr_t* single_pole_coeffs = NULL;
    /*
     * x ** 1 / (x + a) ** 2 case
     * */

    if (pole_order_max >= 1) {
        single_pole_coeffs = calloc(pole_order_max, sizeof(mpfr_t));
        /*  x ^ (j + 1) = (
         *  single_pole_coeffs[0] x ^ (j - 1) +
         *  single_pole_coeffs[1] x ^ (j - 2) + ... +
         *  single_pole_coeffs[j - 1] x^0
         *  ) * (x - a) ^ 2 +
         *
         *  single_pole_coeffs[j](x - a) * ((x - a) + a)
         *
         *  + a ^ (j + 1)
         *
         *  => single_pole_coeffs[j + 1]
         *
         *  single_pole_coeffs[j + 1] = single_pole_coeffs[j] * a + a ^ j + 1
         * single_pole_coeffs[0] =
         * */
        if (pole_order_max >= 2) {
            factorial_times_power_lnb = calloc(pole_order_max - 1, sizeof(mpfr_t));
            mpfr_init2(factorial_times_power_lnb[0], prec);
            // factorial_times_power_lnb[0] = minus_log_base;
            mpfr_set(factorial_times_power_lnb[0], minus_log_base, MPFR_RNDN);
        }

        // temp = pole_position;
        mpfr_set(temp, pole_position, MPFR_RNDN);
        /* below temp is used as pole_position ^ j */

        mpfr_init2(single_pole_coeffs[0], prec);
        // single_pole_coeffs[0] = 1;
        mpfr_set_ui(single_pole_coeffs[0], 1, MPFR_RNDN);
        // result[1] += incomplete_gamma_factor;
        mpfr_add(result[1], result[1], incomplete_gamma_factor, MPFR_RNDN);

        for (int j = 1; j <= pole_order_max - 1; j++) {
            mpfr_init2(single_pole_coeffs[j], prec);
            // single_pole_coeffs[j] = single_pole_coeffs[j - 1] * pole_position + temp;
            mpfr_fma(single_pole_coeffs[j], single_pole_coeffs[j - 1], pole_position, temp, MPFR_RNDN);

            // result[j + 1] += single_pole_coeffs[j] * incomplete_gamma_factor;
            mpfr_fma(result[j + 1], single_pole_coeffs[j], incomplete_gamma_factor, result[j + 1], MPFR_RNDN);
            if (j <= pole_order_max - 2) {
                mpfr_init2(factorial_times_power_lnb[j], prec);
                // temp *= pole_position;
                mpfr_mul(temp, temp, pole_position, MPFR_RNDN);
                // factorial_times_power_lnb[j] = factorial_times_power_lnb[j - 1] * minus_log_base;
                mpfr_mul(factorial_times_power_lnb[j], factorial_times_power_lnb[j - 1], minus_log_base, MPFR_RNDN);
                // factorial_times_power_lnb[j] *= j;
                mpfr_mul_ui(factorial_times_power_lnb[j], factorial_times_power_lnb[j], j, MPFR_RNDN);
            }
        }
    }

#ifdef debug_mode
    for (int j = 0; j <= pole_order_max - 1; j++) {
        mpfr_printf("single_pole_coeffs[%d] = %.16RNf\n", j, single_pole_coeffs[j]);
    }
    for (int j = 0; j <= pole_order_max - 2; j++) {
        mpfr_printf("factorial_times_power_lnb[%d] = %.16RNf\n", j, factorial_times_power_lnb[j]);
    }
#endif

    for (int j = 0; j <= pole_order_max - 2; j++) {
        for (int k = 0; k <= j; k++) {
            // result[j + 2] += factorial_times_power_lnb[k] * single_pole_coeffs[j - k];
            mpfr_fma(result[j + 2], factorial_times_power_lnb[k], single_pole_coeffs[j - k], result[j + 2], MPFR_RNDN);
        }
    }

    for (int i = 0; i <= pole_order_max - 1; i++) {
        mpfr_clear(single_pole_coeffs[i]);
    }
    for (int i = 0; i <= pole_order_max - 2; i++) {
        mpfr_clear(factorial_times_power_lnb[i]);
    }
    if (pole_order_max > 0) {
        free(single_pole_coeffs);
    }
    if (pole_order_max > 1) {
        free(factorial_times_power_lnb);
    }
    mpfr_clear(temp);
    mpfr_clear(minus_pole_position);
    mpfr_clear(minus_log_base);
    mpfr_clear(log_base_power);
    return result;
}
