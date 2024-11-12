//  LU DECOMPOSITION IN PLACE USING OPENMP- PARALLEL

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void lu_decomposition(double **a, int n) {
    for (int k = 0; k < n - 1; k++) {
        // Division step
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            a[i][k] = a[i][k] / a[k][k];
        }

        // Elimination step
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            }
        }
    }
}

int main() {
    int n;
    printf("Enter the size of the matrix (n x n): ");
    scanf("%d", &n);

    // Dynamically allocate a 2D array
    double **a = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        a[i] = (double *)malloc(n * sizeof(double));
    }

    // Take matrix input from the user
    printf("Enter the elements of the matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &a[i][j]);
        }
    }

    // Perform LU Decomposition
    lu_decomposition(a, n);

    // Display the result
    printf("LU Decomposition result:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }

    // Free dynamically allocated memory
    for (int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);

    return 0;
}
