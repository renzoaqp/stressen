import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class StrassenMultiplier:
    def __init__(self, threshold=128, max_workers=None):
        self.threshold = threshold
        self.max_workers = max_workers or multiprocessing.cpu_count()
        print(f"Inicializando Strassen con {self.max_workers} workers, threshold={self.threshold}")
        
    def multiply(self, A, B):
        """Multiplicación de matrices con Strassen multithreading"""
        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        
        if A.shape[1] != B.shape[0]:
            raise ValueError("Dimensiones incompatibles")
        
        m, k = A.shape
        k2, n = B.shape
        
        # Padding a potencia de 2
        max_size = max(m, n, k)
        size = 1
        while size < max_size:
            size *= 2
        
        A_padded = np.zeros((size, size), dtype=np.float64)
        B_padded = np.zeros((size, size), dtype=np.float64)
        
        A_padded[:m, :k] = A
        B_padded[:k, :n] = B
        
        C_padded = self._strassen_recursive(A_padded, B_padded)
        return C_padded[:m, :n]
    
    def _strassen_recursive(self, A, B):
        n = A.shape[0]
        
        # Caso base: usar NumPy para matrices pequeñas
        if n <= self.threshold:
            return np.dot(A, B)
        
        # Dividir matrices
        mid = n // 2
        A11, A12, A21, A22 = self._split_matrix(A, mid)
        B11, B12, B21, B22 = self._split_matrix(B, mid)
        
        # Las 7 multiplicaciones de Strassen
        operations = [
            (A11 + A22, B11 + B22),  # P1
            (A21 + A22, B11),        # P2
            (A11, B12 - B22),        # P3
            (A22, B21 - B11),        # P4
            (A11 + A12, B22),        # P5
            (A21 - A11, B11 + B12),  # P6
            (A12 - A22, B21 + B22)   # P7
        ]
        
        # Ejecutar en paralelo solo para matrices grandes
        if n > self.threshold * 2:
            with ThreadPoolExecutor(max_workers=min(7, self.max_workers)) as executor:
                futures = [
                    executor.submit(self._strassen_recursive, op[0], op[1])
                    for op in operations
                ]
                products = [future.result() for future in futures]
        else:
            products = [
                self._strassen_recursive(op[0], op[1])
                for op in operations
            ]
        
        P1, P2, P3, P4, P5, P6, P7 = products
        
        # Combinar resultados
        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6
        
        return self._combine_matrix(C11, C12, C21, C22)
    
    @staticmethod
    def _split_matrix(matrix, mid):
        return (
            matrix[:mid, :mid],
            matrix[:mid, mid:],
            matrix[mid:, :mid],
            matrix[mid:, mid:]
        )
    
    @staticmethod
    def _combine_matrix(C11, C12, C21, C22):
        top = np.hstack([C11, C12])
        bottom = np.hstack([C21, C22])
        return np.vstack([top, bottom])

def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def benchmark_simple():
    """Benchmark simple sin gráficos"""
    print("=" * 80)
    print("BENCHMARK STRASSEN MULTITHREADING")
    print("=" * 80)
    
    # Configurar multiplicadores
    strassen_seq = StrassenMultiplier(threshold=64, max_workers=1)
    strassen_par = StrassenMultiplier(threshold=64, max_workers=None)
    
    # Tamaños de prueba
    sizes = [128, 256, 512, 1024]
    
    results = []
    
    for size in sizes:
        print(f"\nProbando matrices {size}x{size}...")
        
        # Generar matrices
        np.random.seed(42)
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # NumPy tradicional
        _, time_numpy = measure_time(np.dot, A, B)
        
        # Strassen secuencial
        _, time_strassen_seq = measure_time(strassen_seq.multiply, A, B)
        
        # Strassen paralelo
        _, time_strassen_par = measure_time(strassen_par.multiply, A, B)
        
        # Calcular speedups
        speedup_parallel = time_strassen_seq / time_strassen_par
        speedup_vs_numpy = time_numpy / time_strassen_par
        
        results.append({
            'size': size,
            'numpy': time_numpy,
            'strassen_seq': time_strassen_seq,
            'strassen_par': time_strassen_par,
            'speedup_parallel': speedup_parallel,
            'speedup_vs_numpy': speedup_vs_numpy
        })
        
        print(f"  NumPy:              {time_numpy:.4f}s")
        print(f"  Strassen Secuencial: {time_strassen_seq:.4f}s")
        print(f"  Strassen Paralelo:   {time_strassen_par:.4f}s")
        print(f"  Speedup Paralelo:    {speedup_parallel:.2f}x")
        print(f"  Speedup vs NumPy:    {speedup_vs_numpy:.2f}x")
        
        # Indicar si es mejor o peor
        if speedup_parallel > 1.1:
            print("  ✓ PARALELISMO EFECTIVO")
        elif speedup_parallel < 0.9:
            print("  ✗ PARALELISMO CONTRAPRODUCENTE")
        else:
            print("  ~ PARALELISMO NEUTRO")
    
    return results

def print_summary(results):
    """Imprimir resumen de resultados"""
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    print(f"{'Tamaño':<10} {'NumPy':<10} {'Strassen Sec':<12} {'Strassen Par':<12} {'Speedup Par':<12} {'vs NumPy':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['size']:<10} {r['numpy']:<10.4f} {r['strassen_seq']:<12.4f} {r['strassen_par']:<12.4f} {r['speedup_parallel']:<12.2f} {r['speedup_vs_numpy']:<10.2f}")
    
    # Análisis final
    avg_speedup = sum(r['speedup_parallel'] for r in results) / len(results)
    best_speedup = max(r['speedup_parallel'] for r in results)
    
    print(f"\nSpeedup promedio del paralelismo: {avg_speedup:.2f}x")
    print(f"Mejor speedup alcanzado: {best_speedup:.2f}x")
    
    if avg_speedup > 1.2:
        print("✓ El paralelismo es EFECTIVO")
    elif avg_speedup > 0.9:
        print("~ El paralelismo es NEUTRO")
    else:
        print("✗ El paralelismo es CONTRAPRODUCENTE")

def verify_correctness():
    """Verificar que Strassen da el mismo resultado que NumPy"""
    print("=" * 50)
    print("VERIFICACIÓN DE CORRECTITUD")
    print("=" * 50)
    
    strassen = StrassenMultiplier(threshold=32)
    
    for size in [16, 32, 64]:
        np.random.seed(42)
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        result_numpy = np.dot(A, B)
        result_strassen = strassen.multiply(A, B)
        
        max_error = np.max(np.abs(result_numpy - result_strassen))
        print(f"Matriz {size}x{size}: Error máximo = {max_error:.2e}")
        
        if max_error < 1e-10:
            print("  ✓ CORRECTO")
        else:
            print("  ✗ ERROR!")
    print()

if __name__ == "__main__":
    print("ALGORITMO DE STRASSEN CON MULTITHREADING")
    print(f"Cores disponibles: {multiprocessing.cpu_count()}")
    
    # Verificar correctitud
    verify_correctness()
    
    # Ejecutar benchmark
    results = benchmark_simple()
    
    # Mostrar resumen
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("CONCLUSIONES:")
    print("- Strassen tiene mejor complejidad teórica O(n^2.807)")
    print("- NumPy usa BLAS optimizado (muy rápido)")
    print("- El paralelismo ayuda pero tiene overhead")
    print("- Para matrices >2048x2048 Strassen podría ganar")
    print("=" * 80)