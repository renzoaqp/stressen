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
    start_time = time.perf_counter()  # Más preciso que time.time()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return execution_time, result

def benchmark(matrix_sizes, threshold=128, repetitions=3):
    """Compara el rendimiento de Strassen vs multiplicación tradicional."""
    strassen_times = []
    traditional_times = []
    
    for size in matrix_sizes:
        # Promediar sobre varias ejecuciones para mayor precisión
        strassen_avg = 0
        trad_avg = 0
        
        for _ in range(repetitions):
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # Medir Strassen
            time_strassen, _ = measure_time(strassen_multiply, A, B, threshold=threshold)
            strassen_avg += time_strassen
            
            # Medir tradicional
            time_trad, _ = measure_time(np.dot, A, B)
            trad_avg += time_trad
        
        strassen_times.append(strassen_avg / repetitions)
        traditional_times.append(trad_avg / repetitions)
        
        print(f"Tamaño {size}x{size}: Strassen = {strassen_avg/repetitions:.6f}s, Tradicional = {trad_avg/repetitions:.6f}s")
    
    return strassen_times, traditional_times

def plot_results(matrix_sizes, strassen_times, traditional_times):
    """Grafica los resultados de la comparación con múltiples mejoras visuales."""
    plt.figure(figsize=(12, 7))
    
    # Crear el gráfico principal
    line_strassen, = plt.plot(matrix_sizes, strassen_times, 
                             marker='o', markersize=8, 
                             linestyle='--', linewidth=2,
                             color='#1f77b4', label='Strassen')
    
    line_traditional, = plt.plot(matrix_sizes, traditional_times, 
                                marker='s', markersize=8,
                                linestyle='-', linewidth=2,
                                color='#ff7f0e', label='Tradicional (NumPy)')
    
    # Añadir puntos de datos con valores
    for i, size in enumerate(matrix_sizes):
        plt.text(size, strassen_times[i]*1.1, f'{strassen_times[i]:.3f}s', 
                 ha='center', va='bottom', color='#1f77b4', fontsize=9)
        plt.text(size, traditional_times[i]*0.9, f'{traditional_times[i]:.3f}s', 
                 ha='center', va='top', color='#ff7f0e', fontsize=9)
    
    # Línea de relación y área de diferencia
    for i in range(len(matrix_sizes)-1):
        plt.fill_between([matrix_sizes[i], matrix_sizes[i+1]], 
                         [strassen_times[i], strassen_times[i+1]],
                         [traditional_times[i], traditional_times[i+1]],
                         color='gray', alpha=0.1)
    
    # Configuración de ejes y título
    plt.xlabel('Tamaño de la Matriz (n × n)', fontsize=12, labelpad=10)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12, labelpad=10)
    plt.title('Comparación de Eficiencia: Algoritmo de Strassen vs. Multiplicación Tradicional\n', 
              fontsize=14, pad=20)
    
    # Escalas y grid
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Leyenda mejorada
    legend = plt.legend(handles=[line_strassen, line_traditional],
                       loc='upper left', fontsize=11,
                       framealpha=1, shadow=True)
    
    # Añadir anotación explicativa
    plt.annotate('El método tradicional es más rápido para matrices pequeñas\n'
                 'Strassen puede ser competitivo para matrices muy grandes',
                 xy=(matrix_sizes[-2], traditional_times[-2]), 
                 xytext=(matrix_sizes[-3], traditional_times[-1]*10),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)
    
    # Ajustar márgenes y mostrar
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Tamaños de matrices a probar (incluyendo potencias de 2 y no potencias de 2)
    matrix_sizes = [16, 32, 64, 128, 256, 512, 1024]  # Puedes ajustar según tu hardware
    
    print("Iniciando benchmark...")
    strassen_times, traditional_times = benchmark(matrix_sizes, threshold=64)
    
    print("\nResultados finales:")
    for size, s_time, t_time in zip(matrix_sizes, strassen_times, traditional_times):
        print(f"{size}x{size}: Strassen = {s_time:.6f}s | Tradicional = {t_time:.6f}s | Ratio = {t_time/s_time:.2f}")
    
    plot_results(matrix_sizes, strassen_times, traditional_times)