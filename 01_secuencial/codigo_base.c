#include <stdio.h>    // Entrada/salida (printf)
#include <stdlib.h>   // rand
#include <string.h>   // memset
#include <omp.h>      // omp_get_wtime (temporizador)

#define REPS 3        // Número de repeticiones para medir tiempo
#define N 512         // Tamaño de la matriz

// Inicializa las matrices A y B con valores aleatorios
void init(double A[N][N], double B[N][N], double C[N][N]) {
    srand(42);  // Semilla fija para reproducibilidad

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;  // Valor aleatorio [0,1]
            B[i][j] = (double)rand() / RAND_MAX;  // Valor aleatorio [0,1]
        }
}

// Multiplicación de matrices
void matmul(double A[N][N], double B[N][N], double C[N][N]) {
    memset(C, 0, sizeof(double) * N * N);

    // Recorre filas de A
    for (int i = 0; i < N; i++)
        // Recorre columnas de B
        for (int j = 0; j < N; j++)
            // Producto escalar fila-columna
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// Calcula la suma de todos los elementos de C (verificación)
double checksum(double C[N][N]) {
    double s = 0.0;  // Acumulador
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) s += C[i][j];  // Suma todos los elementos
    return s;
}

int main() {
    // Matrices estáticas para evitar desbordamiento de pila
    static double A[N][N], B[N][N], C[N][N];

    init(A, B, C);  // Inicializa matrices A, B y C

    double best = 1e9;  // Tiempo mínimo observado

    // Ejecuta varias veces para quedarse con el mejor tiempo
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();      // Tiempo inicial
        matmul(A, B, C);                  // Multiplicación
        double t1 = omp_get_wtime();      // Tiempo final

        if (t1 - t0 < best)               // Guarda el mínimo
            best = t1 - t0;
    }

    // Calcula rendimiento en GFLOPS (2*N^3 operaciones)
    double gflops = (2.0 * N) * (N * N) / best / 1e9;

    double chk = checksum(C);  // Calcula checksum

    // Muestra resultados
    printf("N=%d  t=%.6f s  GFLOPS=%.3f  checksum=%.6f\n", N, best, gflops, chk);

    return 0;  // Fin del programa
}