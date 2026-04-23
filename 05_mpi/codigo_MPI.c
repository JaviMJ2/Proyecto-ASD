#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define REPS 3

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = argc > 1 ? atoi(argv[1]) : 512;

    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "ERROR: N=%d no divisible entre %d procesos\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / size;

    // Matrices en estilo del código base
    static double A[512][512];
    static double B[512][512];
    static double C[512][512];

    // buffers locales (también 2D)
    double A_loc[512][512];
    double C_loc[512][512];

    // Inicializar
    if (rank == 0) {
        srand(42);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = (double)rand() / RAND_MAX;
                B[i][j] = (double)rand() / RAND_MAX;
                C[i][j] = 0.0;
            }
    }

    // Broadcast B completa (como en el código base, acceso 2D)
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter de A por bloques de filas
    MPI_Scatter(A, local_rows*N, MPI_DOUBLE, A_loc, local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double best = 1e9;

    for (int r = 0; r < REPS; r++) {
        // limpiar C_local
        for (int i = 0; i < local_rows; i++) for (int j = 0; j < N; j++) C_loc[i][j] = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // === MULTIPLICACIÓN (MISMO ESTILO QUE BASE) ===
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C_loc[i][j] += A_loc[i][k] * B[k][j];

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (t1 - t0 < best)
            best = t1 - t0;
    }

    // Gather resultado final
    MPI_Gather(C_loc, local_rows*N, MPI_DOUBLE, C, local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = ((2.0 * N) * (N * N)) / best / 1e9;

        // checksum como en el código base
        double s = 0.0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                s += C[i][j];

        printf("N=%d procs=%d t=%.6f s GFLOPS=%.3f checksum=%.6f\n", N, size, best, gflops, s);
    }

    MPI_Finalize();
    return 0;
}