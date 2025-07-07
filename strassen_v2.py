import numpy as np
import time
import matplotlib.pyplot as plt

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
    return execution_time

def benchmark(matrix_sizes):
    strassen_times = []
    traditional_times = []
    
    for size in matrix_sizes:
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # Medir tiempo de Strassen
        time_strassen = measure_time(strassen_multiply, A, B)
        strassen_times.append(time_strassen)
        
        # Medir tiempo tradicional (NumPy)
        time_trad = measure_time(np.dot, A, B)
        traditional_times.append(time_trad)
        
        print(f"Tamaño {size}x{size}: Strassen = {time_strassen:.6f}s, Tradicional = {time_trad:.6f}s")
    
    return strassen_times, traditional_times

def plot_results(matrix_sizes, strassen_times, traditional_times):
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, strassen_times, marker='o', label='Strassen')
    plt.plot(matrix_sizes, traditional_times, marker='s', label='Tradicional (NumPy)')
    plt.xlabel('Tamaño de la Matriz (n x n)')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.title('Comparación de Strassen vs. Multiplicación Tradicional')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    plt.show()

if __name__ == "__main__":
    # Tamaños de matrices a probar (potencias de 2 para Strassen)
    #matrix_sizes = [16, 32, 64, 128, 256, 512]  # Puedes agregar 1024 si tu PC lo soporta
    matrix_sizes = [2, 4, 16, 32, 64,128,256,512] 
    
    print("Iniciando benchmark...")
    strassen_times, traditional_times = benchmark(matrix_sizes)
    
    print("\nResultados:")
    for size, s_time, t_time in zip(matrix_sizes, strassen_times, traditional_times):
        print(f"{size}x{size}: Strassen = {s_time:.6f}s | Tradicional = {t_time:.6f}s")
    
    plot_results(matrix_sizes, strassen_times, traditional_times)