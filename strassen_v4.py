import numpy as np
import time
import matplotlib.pyplot as plt

def strassen_multiply(A, B, threshold=128):
    """Multiplicación de matrices usando Strassen con umbral para cambiar a método tradicional."""
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    
    if A.shape[1] != B.shape[0]:
        raise ValueError("El número de columnas de A debe ser igual al número de filas de B")
    
    m = A.shape[0]
    n = B.shape[1]
    l = A.shape[1]
    
    # Encontrar el siguiente tamaño potencia de 2
    max_size = max(m, n, l)
    size = 1
    while size < max_size:
        size *= 2
    
    # Añadir padding solo si es necesario
    if size != m or size != n or size != l:
        A_padded = np.zeros((size, size), dtype=np.float64)
        A_padded[:m, :l] = A
        B_padded = np.zeros((size, size), dtype=np.float64)
        B_padded[:l, :n] = B
    else:
        A_padded = A
        B_padded = B
    
    C_padded = strassen(A_padded, B_padded, threshold)
    C = C_padded[:m, :n]
    
    return C

def strassen(A, B, threshold):
    """Algoritmo de Strassen con umbral para cambiar a multiplicación tradicional."""
    n = A.shape[0]
    
    # Cambiar a multiplicación tradicional para matrices pequeñas
    if n <= threshold:
        return np.dot(A, B)
    
    mid = n // 2
    # Dividir matrices en submatrices
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Calcular los productos de Strassen
    P1 = strassen(A11 + A22, B11 + B22, threshold)
    P2 = strassen(A21 + A22, B11, threshold)
    P3 = strassen(A11, B12 - B22, threshold)
    P4 = strassen(A22, B21 - B11, threshold)
    P5 = strassen(A11 + A12, B22, threshold)
    P6 = strassen(A21 - A11, B11 + B12, threshold)
    P7 = strassen(A12 - A22, B21 + B22, threshold)
    
    # Calcular las submatrices del resultado
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combinar las submatrices
    C = np.zeros((n, n), dtype=np.float64)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

def measure_time(func, *args, **kwargs):
    """Mide el tiempo de ejecución de una función."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return execution_time, result

def benchmark(matrix_sizes, threshold=128, repetitions=3):
    """Compara el rendimiento de Strassen vs multiplicación tradicional."""
    strassen_times = []
    traditional_times = []
    
    for size in matrix_sizes:
        strassen_avg = 0
        trad_avg = 0
        
        for _ in range(repetitions):
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            time_strassen, _ = measure_time(strassen_multiply, A, B, threshold=threshold)
            strassen_avg += time_strassen
            
            time_trad, _ = measure_time(np.dot, A, B)
            trad_avg += time_trad
        
        strassen_times.append(strassen_avg / repetitions)
        traditional_times.append(trad_avg / repetitions)
        
        print(f"Tamaño {size}x{size}: Strassen = {strassen_avg/repetitions:.6f}s, Tradicional = {trad_avg/repetitions:.6f}s")
    
    return strassen_times, traditional_times

def plot_results(matrix_sizes, strassen_times, traditional_times):
    """Grafica los resultados con gráfico de líneas mejorado."""
    plt.figure(figsize=(12, 7))
    
    # Crear el gráfico de líneas
    line_strassen, = plt.plot(matrix_sizes, strassen_times, 
                             marker='o', markersize=8, 
                             linestyle='--', linewidth=2,
                             color='#1f77b4', label='Strassen')
    
    line_traditional, = plt.plot(matrix_sizes, traditional_times, 
                                marker='s', markersize=8,
                                linestyle='-', linewidth=2,
                                color='#ff7f0e', label='Tradicional (NumPy)')
    
    # Añadir etiquetas de datos
    for i, (size, s_time, t_time) in enumerate(zip(matrix_sizes, strassen_times, traditional_times)):
        plt.text(size, s_time*1.05, f'{s_time:.3f}s', 
                 ha='center', va='bottom', color='#1f77b4', fontsize=9)
        plt.text(size, t_time*0.95, f'{t_time:.3f}s', 
                 ha='center', va='top', color='#ff7f0e', fontsize=9)
    
    # Configuración de ejes
    plt.xticks(matrix_sizes, [f"{size}×{size}" for size in matrix_sizes], rotation=45)
    plt.xlabel('Tamaño de la Matriz', fontsize=12, labelpad=10)
    plt.ylabel('Tiempo Promedio (segundos)', fontsize=12)
    
    # Título y leyenda
    plt.title('Comparación de Tiempos: Strassen vs Multiplicación Tradicional\n', fontsize=14)
    plt.legend(fontsize=11, framealpha=1, shadow=True)
    
    # Escalas y grid
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Añadir área entre las curvas
    plt.fill_between(matrix_sizes, strassen_times, traditional_times, 
                    color='gray', alpha=0.1)
    
    # Anotación explicativa
    plt.annotate('El método tradicional es más eficiente\npara matrices pequeñas y medianas',
                xy=(matrix_sizes[2], traditional_times[2]),
                xytext=(matrix_sizes[2], traditional_times[-1]*3),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Tamaños de matrices a probar
    matrix_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    print("Iniciando benchmark...")
    strassen_times, traditional_times = benchmark(matrix_sizes, threshold=64)
    
    print("\nResultados finales:")
    for size, s_time, t_time in zip(matrix_sizes, strassen_times, traditional_times):
        print(f"{size}x{size}: Strassen = {s_time:.6f}s | Tradicional = {t_time:.6f}s | Ratio = {t_time/s_time:.2f}")
    
    plot_results(matrix_sizes, strassen_times, traditional_times)