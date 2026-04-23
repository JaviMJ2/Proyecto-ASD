#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define REPS 3

void init(double *A, double *B, int N) {
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
}

/* ── Versión base: triple bucle naive ── */
void matmul(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

/* ── Versión ILP: unrolling factor 4 en el bucle k ──
 *
 * Sin unrolling, cada iteración hace:
 *   1 multiplicación + 1 suma + 1 incremento de k + 1 salto
 *
 * Con unrolling x4, cada iteración hace 4 multiplicaciones
 * y 4 sumas INDEPENDIENTES entre sí. El procesador puede
 * solaparlas en sus unidades funcionales (ILP).
 * También reduce el overhead del bucle (menos incrementos y saltos). */
void matmul_unroll4(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            /* Cuatro acumuladores independientes.
             * Si usáramos uno solo (sum += ...) habría
             * dependencia entre iteraciones y el procesador
             * no podría solaparlas. */
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int k;
            for (k = 0; k <= N - 4; k += 4) {
                s0 += A[i*N + k+0] * B[(k+0)*N + j];
                s1 += A[i*N + k+1] * B[(k+1)*N + j];
                s2 += A[i*N + k+2] * B[(k+2)*N + j];
                s3 += A[i*N + k+3] * B[(k+3)*N + j];
            }
            /* Resto si N no es múltiplo de 4 */
            for (; k < N; k++)
                s0 += A[i*N + k] * B[k*N + j];

            C[i*N + j] = s0 + s1 + s2 + s3;
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

    /* ── Medir versión base ── */
    double best_base = 1e9;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        matmul(A, B, C, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best_base) best_base = t1 - t0;
    }
    double chk_base = checksum(C, N);

    /* ── Medir versión unroll x4 ── */
    double best_unroll = 1e9;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        matmul_unroll4(A, B, C, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best_unroll) best_unroll = t1 - t0;
    }
    double chk_unroll = checksum(C, N);

    /* ── Resultados ── */
    double ops = 2.0 * N * N * N;
    printf("N=%d\n", N);
    printf("%-15s  t=%.6f s  GFLOPS=%.3f  checksum=%.6f\n",
           "base", best_base, ops/best_base/1e9, chk_base);
    printf("%-15s  t=%.6f s  GFLOPS=%.3f  checksum=%.6f  speedup=%.2fx\n",
           "unroll x4", best_unroll, ops/best_unroll/1e9, chk_unroll,
           best_base/best_unroll);

    /* Si los checksums coinciden, los resultados son correctos */
    if (chk_base - chk_unroll > 1e-3 || chk_unroll - chk_base > 1e-3)
        printf("AVISO: checksums distintos, revisar el codigo\n");

    free(A); free(B); free(C);
    return 0;
}
