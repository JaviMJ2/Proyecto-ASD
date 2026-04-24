#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define REPS 3
#define N 512

int main(int argc, char *argv[]) {

    /* ───────── MPI INIT ───────── */
    int rank, size, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = N / size;
    int omp_hilos = omp_get_max_threads();

    /* ───────── MATRICES ───────── */
    static double A[N][N];
    static double B[N][N];
    static double C[N][N];

    double A_loc[N][N];
    double C_loc[N][N];

    /* ───────── INIT ───────── */
    if (rank == 0) {
        srand(42);

        memset(C, 0, sizeof(double) * N * N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = (double)rand() / RAND_MAX;
                B[i][j] = (double)rand() / RAND_MAX;
            }
    }

    MPI_Bcast(&B[0][0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(&A[0][0], local_rows * N, MPI_DOUBLE, &A_loc[0][0], local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double best = 1e9;

    for (int r = 0; r < REPS; r++) {
        for (int i = 0; i < local_rows; i++) for (int j = 0; j < N; j++) C_loc[i][j] = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /* ───────── KERNEL HÍBRIDO ───────── */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_rows; i++) {
            for (int k = 0; k < N; k++) {
                double a = A_loc[i][k];
                for (int j = 0; j < N; j++) C_loc[i][j] += a * B[k][j];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (t1 - t0 < best) best = t1 - t0;
    }

    MPI_Gather(&C_loc[0][0], local_rows * N, MPI_DOUBLE, &C[0][0], local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ───────── RESULTADOS ───────── */
    if (rank == 0) {
        double gflops = (2.0 * N * N * N) / best / 1e9;
        double chk = 0.0;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) chk += C[i][j];
        printf("N=%d MPI=%d OMP=%d cores=%d t=%.6f s GFLOPS=%.3f checksum=%.6f\n", N, size, omp_hilos, size * omp_hilos, best, gflops, chk);
    }

    MPI_Finalize();
    return 0;
}