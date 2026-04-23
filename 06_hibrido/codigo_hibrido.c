#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define REPS 3

int main(int argc, char *argv[]) {

    int rank, size;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = argc > 1 ? atoi(argv[1]) : 512;

    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "ERROR: N=%d no divisible entre %d procesos\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / size;
    int omp_hilos = omp_get_max_threads();

    /* === MATRICES EN FORMATO BASE === */
    static double A[N][N];
    static double B[N][N];
    static double C[N][N];

    /* buffers locales */
    double A_loc[N][N];
    double C_loc[N][N];

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = (double)rand() / RAND_MAX;
                B[i][j] = (double)rand() / RAND_MAX;
                C[i][j] = 0.0;
            }
    }

    /* Broadcast B completa */
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter de A por filas */
    MPI_Scatter(A, local_rows*N, MPI_DOUBLE, A_loc, local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double best = 1e9;
    for (int r = 0; r < REPS; r++) {
        /* inicializar salida local */
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < N; j++)
                C_loc[i][j] = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /* === KERNEL HÍBRIDO MPI + OPENMP === */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C_loc[i][j] += A_loc[i][k] * B[k][j];

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (t1 - t0 < best)
            best = t1 - t0;
    }

    /* Gather resultado global */
    MPI_Gather(C_loc, local_rows*N, MPI_DOUBLE, C, local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = (2.0 * N * N * N) / best / 1e9;

        /* checksum igual al código base */
        double chk = 0.0;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) chk += C[i][j];

        printf("N=%d MPI=%d OMP=%d cores=%d t=%.6f s GFLOPS=%.3f checksum=%.6f\n", N, size, omp_hilos, size * omp_hilos, best, gflops, chk);
    }

    MPI_Finalize();
    return 0;
}