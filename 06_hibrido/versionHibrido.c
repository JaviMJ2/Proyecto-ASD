#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define REPS 3

int main(int argc, char *argv[]) {
    int rank, size;

    /* MPI_THREAD_FUNNELED: solo el hilo master de OpenMP
     * hace llamadas MPI. Necesario para combinar ambos. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = argc > 1 ? atoi(argv[1]) : 512;

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "ERROR: N=%d no divisible entre %d\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / size;
    int omp_hilos  = omp_get_max_threads();

    double *A     = (rank == 0) ? malloc(N*N*sizeof(double)) : NULL;
    double *C     = (rank == 0) ? malloc(N*N*sizeof(double)) : NULL;
    double *B     = malloc(N*N*sizeof(double));
    double *A_loc = malloc(local_rows*N*sizeof(double));
    double *C_loc = malloc(local_rows*N*sizeof(double));

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N*N; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    }

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A,     local_rows*N, MPI_DOUBLE,
                A_loc, local_rows*N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Cómputo híbrido:
     * Cada proceso MPI tiene sus filas exclusivas → sin conflictos entre procesos.
     * OpenMP divide esas filas entre sus hilos → sin race condition. */
    memset(C_loc, 0, local_rows*N*sizeof(double));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C_loc[i*N+j] += A_loc[i*N+k] * B[k*N+j];

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    MPI_Gather(C_loc, local_rows*N, MPI_DOUBLE,
               C,     local_rows*N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("N=%d  MPI=%d  OMP=%d  cores=%d  t=%.6f s  GFLOPS=%.3f\n",
               N, size, omp_hilos, size*omp_hilos,
               t1-t0, 2.0*N*N*N/(t1-t0)/1e9);

    free(B); free(A_loc); free(C_loc);
    if (rank == 0) { free(A); free(C); }
    MPI_Finalize();
    return 0;
}
