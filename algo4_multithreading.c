#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void lu_decomposition(double **a, int n, int p) {
    int my_rank, bsize;
    
    #pragma omp parallel private(my_rank)
    {
        my_rank = omp_get_thread_num();
        bsize = n / p;
        
        if (my_rank == 0) {
            for (int k = 0; k < n; k++) {
                int row;
                
                for (int i = k + 1; i < n; i++) {
                    a[i][k] = a[i][k] / a[k][k];  // Division step
                    for (int j = k + 1; j < n; j++) {
                        a[i][j] -= a[i][k] * a[k][j];  // Elimination step
                    }
                }
            }
        } else {
            for (int k = (my_rank * bsize); k < (my_rank * bsize + bsize); k++) {
                for (int i = k + 1; i < n; i++) {
                    a[i][k] = a[i][k] / a[k][k];  // Division step
                    for (int j = k + 1; j < n; j++) {
                        a[i][j] -= a[i][k] * a[k][j];  // Elimination step
                    }
                }
            }
        }
    }
}

int main() {
    int n, p;
    
    printf("Enter the matrix size n (NxN): ");
    scanf("%d", &n);
    printf("Enter the number of threads (p): ");
    scanf("%d", &p);

    // Allocate memory for the matrix
    double **a = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        a[i] = (double *)malloc(n * sizeof(double));
    }
    
    // Take matrix input from user
    printf("Enter the matrix elements:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &a[i][j]);
        }
    }
    
    // Perform LU decomposition
    lu_decomposition(a, n, p);
    
    // Print the resultant matrix
    printf("Resultant matrix after LU decomposition:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);

    return 0;
}
