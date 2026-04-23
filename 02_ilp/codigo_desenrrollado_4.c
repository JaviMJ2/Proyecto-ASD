#include <stdio.h>    // Entrada/salida (printf)
#include <stdlib.h>   // rand
#include <string.h>   // memset
#include <omp.h>      // omp_get_wtime (temporizador)

#define REPS 3        // Número de repeticiones para medir tiempo
#define N 512         // Tamaño de la matriz

// Inicializa las matrices A y B con valores aleatorios
void init(double A[N][N], double B[N][N]) {
    srand(42);  // Semilla fija para reproducibilidad
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;  // Valor aleatorio [0,1]
            B[i][j] = (double)rand() / RAND_MAX;  // Valor aleatorio [0,1]
            C[i][j] = 0.0;
        }
}

/* ── Versión base: triple bucle naive ── */
void matmul(double A[N][N], double B[N][N], double C[N][N]) {
    // Recorre filas de A
    for (int i = 0; i < N; i++)
        // Recorre columnas de B
        for (int j = 0; j < N; j++)
            // Producto escalar fila-columna
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

/*
 * Sin unrolling, cada iteración hace:
 *   1 multiplicación + 1 suma + 1 incremento de k + 1 salto
 *
 * Con unrolling x4, cada iteración hace 4 multiplicaciones
 * y 4 sumas INDEPENDIENTES entre sí. El procesador puede
 * solaparlas en sus unidades funcionales (ILP).
 * También reduce el overhead del bucle (menos incrementos y saltos). 
*/

/* ── Versión ILP correcta: unroll x4 + múltiples acumuladores ── */
void matmul_unroll4(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            int k;

            /* Unrolling x4 */
            for (k = 0; k <= N - 4; k += 4) {
                s0 += A[i][k] * B[k][j];
                s1 += A[i][k + 1] * B[k + 1][j];
                s2 += A[i][k + 2] * B[k + 2][j];
                s3 += A[i][k + 3] * B[k + 3][j];
            }

            for (; k < N; k++) s0 += A[i][k] * B[k][j]; // por si N%4 > 0
            C[i][j] = s0 + s1 + s2 + s3; // sumar los acumuladores
        }
    }
}

/* ── Checksum ── */
double checksum(double C[N][N]) {
    double s = 0.0;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) s += C[i][j];
    return s;
}

/* ── MAIN ── */
int main() {

    static double A[N][N], B[N][N], C[N][N];

    init(A, B, C);

    /* ── Base ── */
    double best_base = 1e9;
    for (int r = 0; r < REPS; r++) {
        memset(C, 0, sizeof(C)); // limpia C antes de realizar cualquier calculo

        double t0 = omp_get_wtime();
        matmul(A, B, C);
        double t1 = omp_get_wtime();

        if (t1 - t0 < best_base) best_base = t1 - t0;
    }
    double chk_base = checksum(C);

    /* ── ILP ── */
    double best_unroll = 1e9;
    for (int r = 0; r < REPS; r++) {
        memset(C, 0, sizeof(C)); // limpia C antes de realizar cualquier calculo

        double t0 = omp_get_wtime();
        matmul_unroll4(A, B, C);
        double t1 = omp_get_wtime();
        
        if (t1 - t0 < best_unroll) best_unroll = t1 - t0;
    }
    double chk_unroll = checksum(C);

    /* ── Resultados ── */
    double ops = (2.0 * N) * (N * N);

    printf("N=%d\n", N);
    printf("base       t=%.6f s  GFLOPS=%.3f  checksum=%.6f\n", best_base, ops / best_base / 1e9, chk_base);
    printf("unroll x4  t=%.6f s  GFLOPS=%.3f  checksum=%.6f  speedup=%.2fx\n", best_unroll, ops / best_unroll / 1e9, chk_unroll, best_base / best_unroll);

    return 0;
}