#include <stdio.h>    // Entrada/salida (printf)
#include <stdlib.h>   // malloc, free, atoi, rand
#include <string.h>   // memset
#include <omp.h>      // omp_get_wtime (temporizador)

#define REPS 3        // Número de repeticiones para medir tiempo

// Inicializa las matrices A y B con valores aleatorios
void init(double *A, double *B, int N) {
    srand(42);  // Semilla fija para reproducibilidad
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;  // Valor aleatorio [0,1]
        B[i] = (double)rand() / RAND_MAX;  // Valor aleatorio [0,1]
    }
}

// Multiplicación de matrices (versión secuencial naive)
void matmul(double *A, double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));  // Inicializa C a 0

    // Recorre filas de A
    for (int i = 0; i < N; i++)
        // Recorre columnas de B
        for (int j = 0; j < N; j++)
            // Producto escalar fila-columna
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

// Calcula la suma de todos los elementos de C (verificación)
double checksum(double *C, int N) {
    double s = 0.0;  // Acumulador
    for (int i = 0; i < N * N; i++)
        s += C[i];   // Suma todos los elementos
    return s;
}

int main(int argc, char *argv[]) {
    // Lee tamaño de matriz desde argumento o usa 512 por defecto
    int N = argc > 1 ? atoi(argv[1]) : 512;

    // Reserva memoria para matrices A, B y C
    double *A = malloc((size_t)N * N * sizeof(double));
    double *B = malloc((size_t)N * N * sizeof(double));
    double *C = malloc((size_t)N * N * sizeof(double));

    // Comprueba que la memoria se ha reservado correctamente
    if (!A || !B || !C) {
        fprintf(stderr, "Error reservando memoria\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    init(A, B, N);  // Inicializa matrices A y B

    double best = 1e9;  // Tiempo mínimo observado

    // Ejecuta varias veces para quedarse con el mejor tiempo
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();      // Tiempo inicial
        matmul(A, B, C, N);               // Multiplicación
        double t1 = omp_get_wtime();      // Tiempo final

        if (t1 - t0 < best)               // Guarda el mínimo
            best = t1 - t0;
    }

    // Calcula rendimiento en GFLOPS (2*N^3 operaciones)
    double gflops = 2.0 * N * N * N / best / 1e9;

    double chk = checksum(C, N);  // Calcula checksum

    // Muestra resultados
    printf("N=%d  t=%.6f s  GFLOPS=%.3f  checksum=%.6f\n",
           N, best, gflops, chk);

    // Libera memoria
    free(A);
    free(B);
    free(C);

    return 0;  // Fin del programa
}
