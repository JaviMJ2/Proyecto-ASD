#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  /* Tamaño de la matriz — cámbialo o pásalo por argumento */

/* Matrices globales */
double A[N][N], B[N][N], C[N][N];

/* Función para medir tiempo en segundos */
double tiempo_actual() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main() {
    int i, j, k;

    /* Inicializar matrices con valores de ejemplo */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0.0;
        }

    /* Medir tiempo de la multiplicación */
    double t_inicio = tiempo_actual();

    /* Multiplicación de matrices ijk */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];

    double t_fin = tiempo_actual();

    /* Mostrar resultado */
    printf("Tamaño de la matriz: %d x %d\n", N, N);
    printf("Tiempo de ejecución: %.4f segundos\n", t_fin - t_inicio);

    return 0;
}
