import numpy as np
import time

def strassen_multiply(A, B):
    A = np.array(A)
    B = np.array(B)
    
    if A.shape[1] != B.shape[0]:
        raise ValueError("El número de columnas de A debe ser igual al número de filas de B")
    
    m = A.shape[0]
    n = B.shape[1]
    l = A.shape[1]
    
    max_size = max(m, n, l)
    size = 1
    while size < max_size:
        size *= 2
    
    A_padded = np.zeros((size, size))
    A_padded[:A.shape[0], :A.shape[1]] = A
    B_padded = np.zeros((size, size))
    B_padded[:B.shape[0], :B.shape[1]] = B
    
    C_padded = strassen(A_padded, B_padded)
    C = C_padded[:m, :n]
    
    return C

def strassen(A, B):
    n = A.shape[0]
    
    if n == 1:
        return A * B
    
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    P1 = strassen(A11 + A22, B11 + B22)
    P2 = strassen(A21 + A22, B11)
    P3 = strassen(A11, B12 - B22)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A12, B22)
    P6 = strassen(A21 - A11, B11 + B12)
    P7 = strassen(A12 - A22, B21 + B22)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

if __name__ == "__main__":
    # Ejemplo con matrices pequeñas (Strassen puede ser más lento por overhead)
    A = np.random.rand(64, 64)  # Matriz 64x64
    B = np.random.rand(64, 64)  # Matriz 64x64

    print("Multiplicando matrices de 64x64...")
    
    # Medir tiempo de Strassen
    C_strassen, time_strassen = measure_time(strassen_multiply, A, B)
    print(f"Tiempo de Strassen: {time_strassen:.6f} segundos")
    
    # Medir tiempo de multiplicación tradicional (NumPy)
    C_trad, time_trad = measure_time(np.dot, A, B)
    print(f"Tiempo tradicional (NumPy): {time_trad:.6f} segundos")
    
    # Verificar que los resultados sean iguales (con tolerancia numérica)
    assert np.allclose(C_strassen, C_trad), "Los resultados no coinciden"
    print("Resultados verificados: OK")