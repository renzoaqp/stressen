import numpy as np

def strassen_multiply(A, B):
    # Convertir las matrices a numpy arrays si no lo son
    A = np.array(A)
    B = np.array(B)
    
    # Verificar las dimensiones de las matrices
    if A.shape[1] != B.shape[0]:
        raise ValueError("El número de columnas de A debe ser igual al número de filas de B")
    
    # Tamaño original de la matriz resultante
    m = A.shape[0]
    n = B.shape[1]
    l = A.shape[1]
    
    # Encontrar el siguiente tamaño potencia de 2 que pueda contener ambas matrices
    max_size = max(m, n, l)
    size = 1
    while size < max_size:
        size *= 2
    
    # Rellenar las matrices con ceros para que sean de tamaño size x size
    A_padded = np.zeros((size, size))
    A_padded[:A.shape[0], :A.shape[1]] = A
    B_padded = np.zeros((size, size))
    B_padded[:B.shape[0], :B.shape[1]] = B
    
    # Multiplicar usando Strassen
    C_padded = strassen(A_padded, B_padded)
    
    # Extraer la parte relevante de la matriz resultante
    C = C_padded[:m, :n]
    
    return C

def strassen(A, B):
    n = A.shape[0]
    
    # Caso base: si la matriz es 1x1
    if n == 1:
        return A * B
    
    # Dividir las matrices en submatrices
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Calcular los productos de Strassen
    P1 = strassen(A11 + A22, B11 + B22)
    P2 = strassen(A21 + A22, B11)
    P3 = strassen(A11, B12 - B22)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A12, B22)
    P6 = strassen(A21 - A11, B11 + B12)
    P7 = strassen(A12 - A22, B21 + B22)
    
    # Calcular las submatrices de C
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combinar las submatrices en una matriz
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

# Ejemplo de uso
if __name__ == "__main__":
    # Matrices de ejemplo (pueden ser de cualquier tamaño)
    A = [[1, 2, 3],
         [4, 5, 6],
         [3, 7, 9]]
    
    B = [[7, 8, 10],
         [9, 10, 7],
         [11, 12, 2]]
    
    print("Matriz A:")
    print(np.array(A))
    print("\nMatriz B:")
    print(np.array(B))
    
    C = strassen_multiply(A, B)
    print("\nResultado de A x B (usando Strassen):")
    print(C)
    
    # Verificación con multiplicación tradicional
    C_trad = np.dot(np.array(A), np.array(B))
    print("\nResultado de A x B (multiplicación tradicional):")
    print(C_trad)