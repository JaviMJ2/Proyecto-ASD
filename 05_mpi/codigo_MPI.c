#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define REPS 3
#define N 512

int main(int argc, char *argv[]) {

    // Inicializar entorno MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = N / size;

    // Declaración de matrices
    static double A[N][N];
    static double B[N][N];
    static double C[N][N];

    // buffers locales
    double A_loc[N][N];
    double C_loc[N][N];

    // Inicializar matrices
    if (rank == 0) {
        srand(42);
        
        memset(C, 0, sizeof(double) * N * N);
        
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = (double)rand() / RAND_MAX;
                B[i][j] = (double)rand() / RAND_MAX;
            }
    }

    // Broadcast B: para copiar en todos los procesos el bloque B
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter de A: para repartirlo entre procesos
    MPI_Scatter(A, local_rows*N, MPI_DOUBLE, A_loc, local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double best = 1e9;

    for (int r = 0; r < REPS; r++) {
        // limpiar C_local
        for (int i = 0; i < local_rows; i++) for (int j = 0; j < N; j++) C_loc[i][j] = 0.0;

        // Espera para asegurar que todos los procesos comienzan al mismo tiempo
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // === MULTIPLICACIÓN DE MATRICES ===
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C_loc[i][j] += A_loc[i][k] * B[k][j];

        // Espera de los procesos y calculo de tiempo
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (t1 - t0 < best) best = t1 - t0;
    }

    // Gather resultado final: reocgemos todos los resultados y los almacenamos en C
    MPI_Gather(C_loc, local_rows*N, MPI_DOUBLE, C, local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = ((2.0 * N) * (N * N)) / best / 1e9;
        double s = 0.0;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) s += C[i][j];
        printf("N=%d procs=%d t=%.6f s GFLOPS=%.3f checksum=%.6f\n", N, size, best, gflops, s);
    }

    MPI_Finalize();
    return 0;
}