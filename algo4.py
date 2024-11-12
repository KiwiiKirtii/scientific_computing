import numpy as np
import threading

def lu_decomposition(matrix):
    n = len(matrix)
    
    # Iterate over each row
    for k in range(n):
        # Pivoting step to avoid division by zero
        if matrix[k, k] == 0:
            raise ValueError("Matrix is singular, can't perform LU decomposition.")
        
        # Parallelize row update using multithreading
        def update_row(i):
            factor = matrix[i, k] / matrix[k, k]
            matrix[i, k] = factor  # Store L value in place
            for j in range(k + 1, n):
                matrix[i, j] -= factor * matrix[k, j]
        
        # Create and start threads for each row update
        threads = []
        for i in range(k + 1, n):
            thread = threading.Thread(target=update_row, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    
    return matrix

# Function to take matrix input from the user
def input_matrix():
    n = int(input("Enter the size of the matrix (n x n): "))
    matrix = np.zeros((n, n))
    print("Enter the matrix row by row (space-separated):")
    for i in range(n):
        row = list(map(float, input().split()))
        matrix[i, :] = row
    return matrix

# Main function
def main():
    # Take input matrix from user
    matrix = input_matrix()
    print("Original Matrix:")
    print(matrix)
    
    # Perform in-place LU decomposition
    try:
        lu_matrix = lu_decomposition(matrix)
        print("In-place LU Factorization result:")
        print(lu_matrix)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
