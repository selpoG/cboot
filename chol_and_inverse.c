#include "chol_and_inverse.h"

#include <stdlib.h>

// calculate the inverse matrix of lower triangular matrix A.
mpfr_t* mpfr_triangular_inverse(const mpfr_t* A, int dim, mpfr_prec_t prec) {
    mpfr_t* res = calloc(dim * dim, sizeof(mpfr_t));

    mpfr_t s;
    mpfr_init2(s, prec);
    /*
     *  A res = E_n
     *  \sum_{k = 0}^{i} A[i, k] res[k, j] = delta_{i, j}
     *
     *  i = 0 -> A[0, 0] * res[0, j] = delta_{0, j}
     *  res[0, j] = 0 for j >= 1
     *  res[0, 0] = 1 / A[0, 0]
     *
     *  i = 1 -> A[1, 0] * res[0, j] + A[1, 1] * res[1, j] = delta_{1, j}
     *  res[1, j] = 0 for j >= 2
     *  res[1, 1] = 1 / A[1, 1]
     *  res[1, 0] = -(A[1, 0] * res[0, 0]) / A[1, 1]
     *
     *  i = 2 -> A[2, 0] * res[0, j] + A[2, 1] * res[1, j] + A[2, 2] * res[2, j] = delta_{2, j}
     *  res[2, j] = 0 for j >= 3
     *  res[2, 2] = 1 / A[2, 2]
     *  res[2, 1] = -(A[2, 1] * res[1, 1]) / A[2, 2]
     *  res[2, 0] = -(A[2, 0] * res[0, 0] + A[2, 1] * res[1, 0]) / A[2, 2]
     *
     *  A[i, 0] * res[0, j] + A[i, 1] * res[1, j] + ... + A[i, i] * res[i, j] = delta_{i, j}
     *  res[i, j] = 0 for j >= i + 1
     *  res[i, i] = 1 / A[i, i]
     *  A[i, i] * res[i, j] = -(A[i, j] * res[j, j] + ... + A[i, i - 1] * res[i - 1, j])
     *                      = - \\sum_{k = j}^{i - 1} A[i, k] * res[k, j]
     * */
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            mpfr_init2(res[i * dim + j], prec);
            // res[i][j] = +0;
            mpfr_set_zero(res[i * dim + j], 1);
        }
        mpfr_init2(res[i * dim + i], prec);
        // res[i][i] = 1 / A[i][i];
        mpfr_ui_div(res[i * dim + i], 1, A[i * dim + i], MPFR_RNDN);
        for (int j = 0; j < i; j++) {
            mpfr_init2(res[i * dim + j], prec);
            // s = +0;
            mpfr_set_zero(s, 1);
            for (int k = j; k < i; k++) {
                // s = A[i][k] * res[k][j] + s;
                mpfr_fma(s, A[i * dim + k], res[k * dim + j], s, MPFR_RNDN);
            }
            // s = -s;
            mpfr_neg(s, s, MPFR_RNDN);
            // res[i][j] = s / A[i][i];
            mpfr_div(res[i * dim + j], s, A[i * dim + i], MPFR_RNDN);
        }
    }
    mpfr_clear(s);
    return res;
}

// calculate the cholesky decomposition L of positive definite matrix A,
// by Cholesky–Banachiewicz algorithm.
// A = L L^t and L is lower triangular.
mpfr_t* mpfr_cholesky(const mpfr_t* A, int dim, mpfr_prec_t prec) {
    mpfr_t* res = calloc(dim * dim, sizeof(mpfr_t));

    mpfr_t s;
    mpfr_init2(s, prec);

    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            mpfr_init2(res[i * dim + j], prec);
            // res[i][j] = +0;
            mpfr_set_zero(res[i * dim + j], 1);
        }

        for (int j = 0; j <= i; j++) {
            // s = +0;
            mpfr_set_zero(s, 1);
            mpfr_init2(res[i * dim + j], prec);
            for (int k = 0; k < j; k++) {
                // s = res[i][k] * res[j][k] + s;
                mpfr_fma(s, res[i * dim + k], res[j * dim + k], s, MPFR_RNDN);
            }
            // res[i][j] = A[i][j] - s
            mpfr_sub(res[i * dim + j], A[i * dim + j], s, MPFR_RNDN);
            if (i == j) {
                // res[i][j] = sqrt(res[i][j]);
                mpfr_sqrt(res[i * dim + j], res[i * dim + j], MPFR_RNDN);
            } else {
                // res[i][j] = res[i][j] / res[j][j];
                mpfr_div(res[i * dim + j], res[i * dim + j], res[j * dim + j], MPFR_RNDN);
            }
        }
    }
    mpfr_clear(s);
    return res;
}

// create a new matrix A from v, which element A[i][j] equals to v[i + j].
mpfr_t* form_anti_band(const mpfr_t* ab_vector, int dim, mpfr_prec_t prec) {
    mpfr_t* res = calloc(dim * dim, sizeof(mpfr_t));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            mpfr_init2(res[i * dim + j], prec);
            mpfr_set(res[i * dim + j], ab_vector[i + j], MPFR_RNDN);
        }
    }
    return res;
}

// orthogonalize from_anti_band(ab_vector).
// Let A[i][j] = ab_vector[i + j] and introduce innner product by (u, v) = u A v.
// return orthonormal vectors.
mpfr_t* anti_band_to_inverse(const mpfr_t* ab_vector, int dim, mpfr_prec_t prec) {
    mpfr_t* anti_band_mat = form_anti_band(ab_vector, dim, prec);
    mpfr_t* cholesky_decomposed = mpfr_cholesky(anti_band_mat, dim, prec);
    int len = dim * dim;
    for (int i = 0; i < len; i++) {
        mpfr_clear(anti_band_mat[i]);
    }
    mpfr_t* inversed = mpfr_triangular_inverse(cholesky_decomposed, dim, prec);
    for (int i = 0; i < len; i++) {
        mpfr_clear(cholesky_decomposed[i]);
    }
    return inversed;
}
