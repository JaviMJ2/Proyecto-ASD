#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define REPS 3
#define N 512

/* ── Inicialización ── */
void init(double A[N][N], double B[N][N]) {
    srand(42);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
}

/* ── Base (1 hilo, referencia) ── */
void matmul_base(double A[N][N], double B[N][N], double C[N][N]) {
    memset(C, 0, sizeof(double) * N * N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

/* ── Versión OpenMP static
 * Divide las filas del bucle i entre los hilos disponibles.
 * static: cada hilo recibe el mismo número de filas antes de empezar.
 * Es el mejor scheduler para GEMM porque la carga es uniforme. 
 * Sin race condition: cada hilo escribe en filas distintas de C. 
*/
void matmul_static(double A[N][N], double B[N][N], double C[N][N]) {

    memset(C, 0, sizeof(double) * N * N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

/* ── Versión OpenMP dynamic
 * Cada hilo pide trabajo al terminar el anterior.
 * Peor para GEMM porque añade overhead de sincronización
 * sin necesidad (la carga ya es uniforme). 
*/
void matmul_dynamic(double A[N][N], double B[N][N], double C[N][N]) {

    memset(C, 0, sizeof(double) * N * N);

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

/* ── Versión OpenMP guided
 * Bloques decrecientes: empieza con bloques grandes y los reduce.
 * Compromiso entre static y dynamic. 
*/
void matmul_guided(double A[N][N], double B[N][N], double C[N][N]) {

    memset(C, 0, sizeof(double) * N * N);

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

/* ── checksum ── */
double checksum(double C[N][N]) {
    double s = 0.0;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) s += C[i][j];
    return s;
}

/* ── MAIN ── */
int main() {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    static double A[N][N], B[N][N], C[N][N];

    init(A, B, C);

    printf("N=%d  cores=%d\n\n", N, omp_get_max_threads());

    /* ── Referencia 1 hilo ── */
    omp_set_num_threads(1);

    double t_ref = 1e9;

    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        matmul_base(A, B, C);
        double t1 = omp_get_wtime();
        if (t1 - t0 < t_ref) t_ref = t1 - t0;
    }

    printf("Referencia: %.6f s\n\n", t_ref);

    /* ── Escalado de hilos ── */
    int configs[] = {1, 2, 4, 8};

    printf("%-6s %-10s %-10s %-10s %-10s\n", "hilos", "t(s)", "GFLOPS", "speedup", "efic(%)");

    for (int ci = 0; ci < 4; ci++) {

        int p = configs[ci];
        if (p > omp_get_max_threads()) break;

        omp_set_num_threads(p);

        double best = 1e9;

        for (int r = 0; r < REPS; r++) {
            double t0 = omp_get_wtime();
            matmul_static(A, B, C);
            double t1 = omp_get_wtime();
            if (t1 - t0 < best) best = t1 - t0;
        }

        double ops = 2.0 * N * N * N;
        double sp = t_ref / best;
        double ef = (sp / p) * 100.0;

        printf("%-6d %-10.6f %-10.3f %-10.2f %.1f%%\n", p, best, ops / best / 1e9, sp, ef);
    }

    /* ── Comparación schedulers ── */
    omp_set_num_threads(omp_get_max_threads());

    double ts = 1e9, td = 1e9, tg = 1e9;

    for (int r = 0; r < REPS; r++) {

        double t0, t1;

        t0 = omp_get_wtime();
        matmul_static(A, B, C);
        t1 = omp_get_wtime();
        if (t1 - t0 < ts) ts = t1 - t0;

        t0 = omp_get_wtime();
        matmul_dynamic(A, B, C);
        t1 = omp_get_wtime();
        if (t1 - t0 < td) td = t1 - t0;

        t0 = omp_get_wtime();
        matmul_guided(A, B, C);
        t1 = omp_get_wtime();
        if (t1 - t0 < tg) tg = t1 - t0;
    }

    printf("\nSchedulers:\n");
    printf("static : %.6f s\n", ts);
    printf("dynamic: %.6f s\n", td);
    printf("guided : %.6f s\n", tg);

    return 0;
}