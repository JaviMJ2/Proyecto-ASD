#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    int N = 500;          // Tamaño matriz NxN
    int numThreads = 4;   // Número de hilos

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) numThreads = atoi(argv[2]);

    omp_set_num_threads(numThreads);

    int i, j, k;

    /* Reservar memoria */
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    /* Inicializar matrices */
    #pragma omp parallel for
    for (i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    double inicio = omp_get_wtime();

    /* Multiplicación paralela */
    #pragma omp parallel for private(j,k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }

    double fin = omp_get_wtime();

    printf("Tiempo: %f segundos\n", fin - inicio);
    printf("Resultado C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}

/*
COMPILACION:

gcc -fopenmp matrices.c -o matrices

EJECUCION:

./matrices
./matrices 1000
./matrices 1000 8

*/