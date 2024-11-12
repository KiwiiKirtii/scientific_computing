import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination
    for k in range(n):
        # Division step
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] = A[i][j] - factor * A[k][j]
            b[i] = b[i] - factor * b[k]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] = x[i] / A[i][i]

    return x
def print_A(A, n):
    for i in range(0, n):
        for j in range (0, n):
            print(A[i][j], end="  ")
        print()

# Taking input from the user
n = int(input("Enter the number of variables: "))
print("Enter the coefficients matrix (A) row by row:")
A = []
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)
A = np.array(A)

print("Enter the constant terms vector (b):")
b = list(map(float, input().split()))
b = np.array(b)

# Solving the system
solution = gaussian_elimination(A, b)
print("The solution is:", solution)
print_A(A, n)
