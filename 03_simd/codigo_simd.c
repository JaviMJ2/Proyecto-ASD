#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>   /* necesario para los intrínsecos SSE */

#define REPS 3

void init(double *A, double *B, int N) {
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
}

/* ── Versión base (igual que siempre) ── */
void matmul_base(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

/* ── Versión SIMD con SSE4.2 ──
 *
 * Para cada elemento C[i][j] acumulamos de 2 en 2.
 * En vez de un double acumulador, usamos un cajón __m128d
 * que acumula 2 resultados parciales a la vez. */
void matmul_sse(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            /* Cajón acumulador inicializado a [0.0, 0.0] */
            __m128d sum_v = _mm_setzero_pd();
            int k;

            /* Bucle principal: procesa 2 doubles por iteración */
            for (k = 0; k <= N - 2; k += 2) {

                /* Cargar A[i][k] y A[i][k+1] en un cajón */
                __m128d a_v = _mm_loadu_pd(&A[i*N + k]);

                /* B[k][j] y B[k+1][j] NO son contiguos en memoria
                 * (están separados N posiciones), los cargamos uno a uno */
                __m128d b_v = _mm_set_pd(B[(k+1)*N + j], B[k*N + j]);

                /* Multiplicar los 2 pares y acumular */
                sum_v = _mm_add_pd(sum_v, _mm_mul_pd(a_v, b_v));
            }

            /* Extraer los 2 doubles del cajón y sumarlos */
            double tmp[2];
            _mm_storeu_pd(tmp, sum_v);
            double sum = tmp[0] + tmp[1];

            /* Resto si N no es múltiplo de 2 */
            for (; k < N; k++)
                sum += A[i*N+k] * B[k*N+j];

            C[i*N+j] = sum;
        }
    }
}

double checksum(double *C, int N) {
    double s = 0.0;
    for (int i = 0; i < N * N; i++) s += C[i];
    return s;
}

int main(int argc, char *argv[]) {
    int N = argc > 1 ? atoi(argv[1]) : 512;

    double *A = malloc((size_t)N * N * sizeof(double));
    double *B = malloc((size_t)N * N * sizeof(double));
    double *C = malloc((size_t)N * N * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Error reservando memoria\n");
        free(A); free(B); free(C);
        return 1;
    }
    init(A, B, N);

    /* Medir base */
    double best_base = 1e9;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        matmul_base(A, B, C, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best_base) best_base = t1 - t0;
    }
    double chk_base = checksum(C, N);

    /* Medir SSE */
    double best_sse = 1e9;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        matmul_sse(A, B, C, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best_sse) best_sse = t1 - t0;
    }
    double chk_sse = checksum(C, N);

    /* Resultados */
    double ops = 2.0 * N * N * N;
    printf("N=%d\n", N);
    printf("%-12s  t=%.6f s  GFLOPS=%.3f  checksum=%.6f\n",
           "base", best_base, ops/best_base/1e9, chk_base);
    printf("%-12s  t=%.6f s  GFLOPS=%.3f  checksum=%.6f  speedup=%.2fx\n",
           "SSE4.2", best_sse, ops/best_sse/1e9, chk_sse,
           best_base/best_sse);

    if (chk_base - chk_sse > 1e-3 || chk_sse - chk_base > 1e-3)
        printf("AVISO: checksums distintos, revisar el codigo\n");

    free(A); free(B); free(C);
    return 0;
}
