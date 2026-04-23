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

/* ── Versión base (referencia, 1 hilo) ── */
void matmul_base(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

/* ── Versión OpenMP static
 * Divide las filas del bucle i entre los hilos disponibles.
 * static: cada hilo recibe el mismo número de filas antes de empezar.
 * Es el mejor scheduler para GEMM porque la carga es uniforme. 
 * Sin race condition: cada hilo escribe en filas distintas de C. */
void matmul_static(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

/* ── Versión OpenMP dynamic
 * Cada hilo pide trabajo al terminar el anterior.
 * Peor para GEMM porque añade overhead de sincronización
 * sin necesidad (la carga ya es uniforme). */
void matmul_dynamic(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

/* ── Versión OpenMP guided
 * Bloques decrecientes: empieza con bloques grandes y los reduce.
 * Compromiso entre static y dynamic. */
void matmul_guided(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

double checksum(double *C, int N) {
    double s = 0.0;
    for (int i = 0; i < N * N; i++) s += C[i];
    return s;
}

int main(int argc, char *argv[]) {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    int max_hilos = omp_get_max_threads();

    double *A = malloc((size_t)N * N * sizeof(double));
    double *B = malloc((size_t)N * N * sizeof(double));
    double *C = malloc((size_t)N * N * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Error reservando memoria\n");
        free(A); free(B); free(C);
        return 1;
    }
    init(A, B, N);

    printf("N=%d  cores disponibles=%d\n\n", N, max_hilos);

    /* Referencia: 1 hilo sin OpenMP */
    omp_set_num_threads(1);
    double t_ref = 1e9;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        matmul_base(A, B, C, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < t_ref) t_ref = t1 - t0;
    }
    printf("Referencia (1 hilo): t=%.6f s\n\n", t_ref);

    /* Barrido de hilos con static */
    printf("%-6s  %-10s  %-10s  %-10s  %-10s\n",
           "hilos", "t(s)", "GFLOPS", "speedup", "efic(%)");

    int configs[] = {1, 2, 4};
    for (int ci = 0; ci < 3; ci++) {
        int p = configs[ci];
        if (p > max_hilos) break;
        omp_set_num_threads(p);

        double best = 1e9;
        for (int r = 0; r < REPS; r++) {
            double t0 = omp_get_wtime();
            matmul_static(A, B, C, N);
            double t1 = omp_get_wtime();
            if (t1 - t0 < best) best = t1 - t0;
        }
        double ops = 2.0 * N * N * N;
        double sp = t_ref / best;
        double ef = sp / p * 100.0;
        printf("%-6d  %-10.6f  %-10.3f  %-10.2f  %.1f%%\n",
               p, best, ops/best/1e9, sp, ef);
    }

    /* Comparativa schedulers con máximo de hilos */
    omp_set_num_threads(max_hilos);
    double ts = 1e9, td = 1e9, tg = 1e9;
    for (int r = 0; r < REPS; r++) {
        double t0, t1;
        t0=omp_get_wtime(); matmul_static (A,B,C,N); t1=omp_get_wtime(); if(t1-t0<ts)ts=t1-t0;
        t0=omp_get_wtime(); matmul_dynamic(A,B,C,N); t1=omp_get_wtime(); if(t1-t0<td)td=t1-t0;
        t0=omp_get_wtime(); matmul_guided (A,B,C,N); t1=omp_get_wtime(); if(t1-t0<tg)tg=t1-t0;
    }
    printf("\nSchedulers con %d hilos:\n", max_hilos);
    printf("  static : %.6f s  speedup=%.2fx\n", ts, t_ref/ts);
    printf("  dynamic: %.6f s  speedup=%.2fx\n", td, t_ref/td);
    printf("  guided : %.6f s  speedup=%.2fx\n", tg, t_ref/tg);

    free(A); free(B); free(C);
    return 0;
}
