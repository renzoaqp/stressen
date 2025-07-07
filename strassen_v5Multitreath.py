import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from numba import jit
import warnings
warnings.filterwarnings('ignore')

class StrassenMultiplier:
    def __init__(self, threshold=128, max_workers=None):
        """
        threshold: Tamaño mínimo para usar Strassen (menor usa método tradicional)
        max_workers: Número máximo de hilos (None = usar todos los cores)
        """
        self.threshold = threshold
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
    def multiply(self, A, B):
        """Multiplicación de matrices con Strassen optimizado"""
        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        
        if A.shape[1] != B.shape[0]:
            raise ValueError("Dimensiones incompatibles para multiplicación")
        
        m, k = A.shape
        k2, n = B.shape
        
        # Encontrar el tamaño de padding (potencia de 2)
        max_size = max(m, n, k)
        size = 1
        while size < max_size:
            size *= 2
        
        # Padding para hacer matrices cuadradas de tamaño potencia de 2
        A_padded = np.zeros((size, size), dtype=np.float64)
        B_padded = np.zeros((size, size), dtype=np.float64)
        
        A_padded[:m, :k] = A
        B_padded[:k, :n] = B
        
        # Multiplicación recursiva
        C_padded = self._strassen_recursive(A_padded, B_padded)
        
        # Extraer resultado original
        return C_padded[:m, :n]
    
    def _strassen_recursive(self, A, B):
        """Implementación recursiva de Strassen con multithreading"""
        n = A.shape[0]
        
        # Caso base: usar multiplicación tradicional para matrices pequeñas
        if n <= self.threshold:
            return self._traditional_multiply(A, B)
        
        # Dividir matrices en 4 cuadrantes
        mid = n // 2
        A11, A12, A21, A22 = self._split_matrix(A, mid)
        B11, B12, B21, B22 = self._split_matrix(B, mid)
        
        # Preparar las 7 multiplicaciones de Strassen
        operations = [
            (A11 + A22, B11 + B22),  # P1
            (A21 + A22, B11),        # P2
            (A11, B12 - B22),        # P3
            (A22, B21 - B11),        # P4
            (A11 + A12, B22),        # P5
            (A21 - A11, B11 + B12),  # P6
            (A12 - A22, B21 + B22)   # P7
        ]
        
        # Ejecutar multiplicaciones en paralelo
        if n > self.threshold * 2:  # Solo paralelizar para matrices grandes
            with ThreadPoolExecutor(max_workers=min(7, self.max_workers)) as executor:
                futures = [
                    executor.submit(self._strassen_recursive, op[0], op[1])
                    for op in operations
                ]
                products = [future.result() for future in futures]
        else:
            # Ejecutar secuencialmente para matrices medianas
            products = [
                self._strassen_recursive(op[0], op[1])
                for op in operations
            ]
        
        P1, P2, P3, P4, P5, P6, P7 = products
        
        # Combinar resultados según las fórmulas de Strassen
        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6
        
        # Ensamblar matriz resultado
        return self._combine_matrix(C11, C12, C21, C22)
    
    @staticmethod
    def _split_matrix(matrix, mid):
        """Dividir matriz en 4 cuadrantes"""
        return (
            matrix[:mid, :mid],    # Cuadrante superior izquierdo
            matrix[:mid, mid:],    # Cuadrante superior derecho
            matrix[mid:, :mid],    # Cuadrante inferior izquierdo
            matrix[mid:, mid:]     # Cuadrante inferior derecho
        )
    
    @staticmethod
    def _combine_matrix(C11, C12, C21, C22):
        """Combinar 4 cuadrantes en una matriz"""
        top = np.hstack([C11, C12])
        bottom = np.hstack([C21, C22])
        return np.vstack([top, bottom])
    
    @staticmethod
    @jit(nopython=True)
    def _traditional_multiply(A, B):
        """Multiplicación tradicional optimizada con Numba"""
        return np.dot(A, B)

def measure_execution_time(func, *args, **kwargs):
    """Medir tiempo de ejecución de una función"""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time

def comprehensive_benchmark():
    """Benchmark completo comparando diferentes implementaciones"""
    # Configurar tamaños de prueba
    matrix_sizes = [64, 128, 256, 512, 1024]  # Tamaños progresivos
    
    # Inicializar multiplicadores
    strassen_seq = StrassenMultiplier(threshold=64, max_workers=1)  # Secuencial
    strassen_par = StrassenMultiplier(threshold=64, max_workers=None)  # Paralelo
    
    results = {
        'sizes': matrix_sizes,
        'strassen_sequential': [],
        'strassen_parallel': [],
        'numpy_traditional': [],
        'speedup_parallel': [],
        'speedup_vs_numpy': []
    }
    
    print("=" * 80)
    print("BENCHMARK COMPLETO - ALGORITMO DE STRASSEN")
    print("=" * 80)
    print(f"Procesador: {multiprocessing.cpu_count()} cores disponibles")
    print(f"Threshold para Strassen: {strassen_par.threshold}")
    print()
    
    for size in matrix_sizes:
        print(f"Probando matrices {size}x{size}...")
        
        # Generar matrices aleatorias
        np.random.seed(42)  # Para reproducibilidad
        A = np.random.rand(size, size).astype(np.float64)
        B = np.random.rand(size, size).astype(np.float64)
        
        # Método tradicional (NumPy)
        _, time_numpy = measure_execution_time(np.dot, A, B)
        results['numpy_traditional'].append(time_numpy)
        
        # Strassen secuencial
        _, time_strassen_seq = measure_execution_time(strassen_seq.multiply, A, B)
        results['strassen_sequential'].append(time_strassen_seq)
        
        # Strassen paralelo
        _, time_strassen_par = measure_execution_time(strassen_par.multiply, A, B)
        results['strassen_parallel'].append(time_strassen_par)
        
        # Calcular speedups
        speedup_parallel = time_strassen_seq / time_strassen_par
        speedup_vs_numpy = time_numpy / time_strassen_par
        
        results['speedup_parallel'].append(speedup_parallel)
        results['speedup_vs_numpy'].append(speedup_vs_numpy)
        
        print(f"  NumPy:              {time_numpy:.4f}s")
        print(f"  Strassen Secuencial: {time_strassen_seq:.4f}s")
        print(f"  Strassen Paralelo:   {time_strassen_par:.4f}s")
        print(f"  Speedup Paralelo:    {speedup_parallel:.2f}x")
        print(f"  Speedup vs NumPy:    {speedup_vs_numpy:.2f}x")
        print()
    
    return results

def plot_comprehensive_results(results):
    """Crear gráficos comprehensivos de los resultados"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    sizes = results['sizes']
    
    # Gráfico 1: Tiempos de ejecución
    ax1.loglog(sizes, results['numpy_traditional'], 'b-o', label='NumPy', linewidth=2)
    ax1.loglog(sizes, results['strassen_sequential'], 'r-s', label='Strassen Secuencial', linewidth=2)
    ax1.loglog(sizes, results['strassen_parallel'], 'g-^', label='Strassen Paralelo', linewidth=2)
    ax1.set_xlabel('Tamaño de Matriz (n)')
    ax1.set_ylabel('Tiempo de Ejecución (s)')
    ax1.set_title('Comparación de Tiempos de Ejecución')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Speedup del paralelismo
    ax2.semilogx(sizes, results['speedup_parallel'], 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Sin mejora')
    ax2.set_xlabel('Tamaño de Matriz (n)')
    ax2.set_ylabel('Speedup (Secuencial/Paralelo)')
    ax2.set_title('Speedup del Paralelismo en Strassen')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Speedup vs NumPy
    ax3.semilogx(sizes, results['speedup_vs_numpy'], 'purple', marker='d', linewidth=2, markersize=8)
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Igual que NumPy')
    ax3.set_xlabel('Tamaño de Matriz (n)')
    ax3.set_ylabel('Speedup (NumPy/Strassen Paralelo)')
    ax3.set_title('Strassen Paralelo vs NumPy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Eficiencia del algoritmo
    theoretical_complexity = [n**2.807 for n in sizes]
    numpy_complexity = [n**3 for n in sizes]
    
    # Normalizar para comparación
    theoretical_complexity = np.array(theoretical_complexity) / theoretical_complexity[0]
    numpy_complexity = np.array(numpy_complexity) / numpy_complexity[0]
    strassen_normalized = np.array(results['strassen_parallel']) / results['strassen_parallel'][0]
    
    ax4.loglog(sizes, theoretical_complexity, 'r--', label='Teórico O(n^2.807)', linewidth=2)
    ax4.loglog(sizes, numpy_complexity, 'b--', label='Teórico O(n^3)', linewidth=2)
    ax4.loglog(sizes, strassen_normalized, 'g-o', label='Strassen Paralelo Real', linewidth=2)
    ax4.set_xlabel('Tamaño de Matriz (n)')
    ax4.set_ylabel('Tiempo Relativo (normalizado)')
    ax4.set_title('Complejidad Teórica vs Real')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def verification_test():
    """Verificar que los resultados sean correctos"""
    print("=" * 50)
    print("VERIFICACIÓN DE CORRECTITUD")
    print("=" * 50)
    
    strassen_mult = StrassenMultiplier(threshold=32)
    
    # Prueba con matrices pequeñas
    test_sizes = [4, 8, 16, 32]
    
    for size in test_sizes:
        np.random.seed(42)
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        result_numpy = np.dot(A, B)
        result_strassen = strassen_mult.multiply(A, B)
        
        error = np.max(np.abs(result_numpy - result_strassen))
        print(f"Matriz {size}x{size}: Error máximo = {error:.2e}")
        
        if error < 1e-10:
            print("  ✓ CORRECTO")
        else:
            print("  ✗ ERROR DETECTADO")
    
    print()

if __name__ == "__main__":
    print("ALGORITMO DE STRASSEN OPTIMIZADO CON MULTITHREADING")
    print("=" * 60)
    
    # Verificar correctitud
    verification_test()
    
    # Ejecutar benchmark completo
    results = comprehensive_benchmark()
    
    # Mostrar gráficos
    plot_comprehensive_results(results)
    
    # Resumen final
    print("=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print("El algoritmo de Strassen:")
    print("- Usa multithreading para paralelizar las 7 multiplicaciones")
    print("- Cambia automáticamente a multiplicación tradicional para matrices pequeñas")
    print("- Tiene mejor complejidad teórica O(n^2.807) vs O(n^3)")
    print("- En la práctica, el punto de cruce está en matrices >1000x1000")
    print("- NumPy sigue siendo más rápido para la mayoría de casos por sus optimizaciones")