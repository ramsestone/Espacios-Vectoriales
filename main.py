"""
=============================================================================
Análisis de Espacios Vectoriales
=============================================================================
Aplicación de línea de comandos para analizar propiedades de espacios
vectoriales: independencia lineal, bases, normas, ortogonalidad y
ortonormalidad. Incluye el proceso de Gram-Schmidt.

Espacios soportados:
  - Matrices (M_{n x m})  → producto interno de Frobenius
  - Espacio Euclídeo (ℝⁿ) → producto punto estándar
  - Polinomios (P_n)       → producto interno integral en [-1, 1]
  - Funciones (F)          → producto interno integral en [-1, 1]
=============================================================================
"""

import abc
import math
import sys
from typing import List, Tuple, Any

# ---------------------------------------------------------------------------
# Verificación de dependencias
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from scipy import integrate
    import questionary
    from questionary import Style
except ImportError:
    print("Error: dependencias no instaladas. Ejecute: pip install requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Estilo visual para questionary
# ---------------------------------------------------------------------------
estilo_menu = Style([
    ("qmark",       "fg:#00d4aa bold"),
    ("question",    "fg:#ffffff bold"),
    ("answer",      "fg:#00d4aa bold"),
    ("pointer",     "fg:#00d4aa bold"),
    ("highlighted", "fg:#00d4aa bold"),
    ("selected",    "fg:#00d4aa"),
    ("separator",   "fg:#444444"),
    ("instruction", "fg:#888888"),
])

# ---------------------------------------------------------------------------
# Tolerancia numérica global
# ---------------------------------------------------------------------------
TOLERANCIA = 1e-9


# ===========================================================================
# Clase abstracta base: EspacioVectorial
# ===========================================================================
class EspacioVectorial(abc.ABC):
    """
    Clase abstracta que define la interfaz de un espacio vectorial.
    Cada subclase debe implementar su propio producto interno, lo que
    determina la norma, ortogonalidad y el proceso de Gram-Schmidt.
    """

    # -----------------------------------------------------------------------
    # Métodos abstractos que cada espacio debe implementar
    # -----------------------------------------------------------------------
    @abc.abstractmethod
    def producto_interno(self, vector_u: Any, vector_v: Any) -> float:
        """Calcula el producto interno ⟨u, v⟩ propio del espacio."""

    @abc.abstractmethod
    def nombre_espacio(self) -> str:
        """Retorna el nombre simbólico del espacio (para mostrar en pantalla)."""

    @abc.abstractmethod
    def ingresar_vectores(self) -> List[Any]:
        """Guía al usuario para ingresar el conjunto de vectores B."""

    @abc.abstractmethod
    def mostrar_vector(self, vector: Any) -> str:
        """Retorna una representación legible del vector."""

    # -----------------------------------------------------------------------
    # Métodos concretos (heredados por todas las subclases)
    # -----------------------------------------------------------------------
    def calcular_norma(self, vector: Any) -> float:
        """‖v‖ = √⟨v, v⟩"""
        valor = self.producto_interno(vector, vector)
        return math.sqrt(max(valor, 0.0))

    def son_ortogonales(self, vector_u: Any, vector_v: Any) -> bool:
        """Dos vectores son ortogonales si ⟨u, v⟩ ≈ 0."""
        return abs(self.producto_interno(vector_u, vector_v)) < TOLERANCIA

    def normalizar_vector(self, vector: Any) -> Any:
        """Retorna el vector unitario v / ‖v‖."""
        norma = self.calcular_norma(vector)
        if norma < TOLERANCIA:
            raise ValueError("No se puede normalizar el vector cero.")
        return self._escalar_por(vector, 1.0 / norma)

    def verificar_independencia_lineal(self, conjunto_b: List[Any]) -> bool:
        """
        Verifica independencia lineal construyendo la matriz de Gram
        G[i,j] = ⟨b_i, b_j⟩ y comprobando que det(G) ≠ 0.
        """
        n = len(conjunto_b)
        if n == 0:
            return False
        matriz_gram = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matriz_gram[i, j] = self.producto_interno(conjunto_b[i], conjunto_b[j])
        det = np.linalg.det(matriz_gram)
        return abs(det) > TOLERANCIA

    def es_base(self, conjunto_b: List[Any], dimension_espacio: int) -> bool:
        """
        Un conjunto es base si es linealmente independiente Y
        su cardinalidad coincide con la dimensión del espacio.
        """
        return (len(conjunto_b) == dimension_espacio and
                self.verificar_independencia_lineal(conjunto_b))

    def verificar_ortogonalidad_base(self, conjunto_b: List[Any]) -> bool:
        """Verifica que todo par de vectores distintos sea ortogonal."""
        n = len(conjunto_b)
        for i in range(n):
            for j in range(i + 1, n):
                if not self.son_ortogonales(conjunto_b[i], conjunto_b[j]):
                    return False
        return True

    def verificar_ortonormalidad_base(self, conjunto_b: List[Any]) -> bool:
        """
        Una base es ortonormal si:
          - Es ortogonal.
          - Cada vector tiene norma unitaria (‖v‖ ≈ 1).
        """
        if not self.verificar_ortogonalidad_base(conjunto_b):
            return False
        for vector in conjunto_b:
            if abs(self.calcular_norma(vector) - 1.0) > TOLERANCIA:
                return False
        return True

    def proceso_gram_schmidt(self, conjunto_b: List[Any]) -> List[Any]:
        """
        Aplica el proceso de Gram-Schmidt para obtener una base ortonormal
        a partir de una base (no necesariamente ortogonal).

        Paso k:
          u_k = b_k - Σ_{j<k} ⟨b_k, e_j⟩ · e_j
          e_k = u_k / ‖u_k‖
        """
        base_ortonormal = []
        for k, vector_bk in enumerate(conjunto_b):
            # Copiar b_k como vector de trabajo
            vector_uk = self._copiar_vector(vector_bk)
            # Restar proyecciones sobre los vectores ya ortonormalizados
            for vector_ej in base_ortonormal:
                proy = self.producto_interno(vector_bk, vector_ej)
                vector_uk = self._restar_vectores(
                    vector_uk,
                    self._escalar_por(vector_ej, proy)
                )
            norma_uk = self.calcular_norma(vector_uk)
            if norma_uk < TOLERANCIA:
                raise ValueError(
                    f"El vector {k+1} es linealmente dependiente de los anteriores."
                )
            base_ortonormal.append(self._escalar_por(vector_uk, 1.0 / norma_uk))
        return base_ortonormal

    # -----------------------------------------------------------------------
    # Métodos auxiliares para aritmética de vectores (a implementar)
    # -----------------------------------------------------------------------
    @abc.abstractmethod
    def _escalar_por(self, vector: Any, escalar: float) -> Any:
        """Multiplica un vector por un escalar."""

    @abc.abstractmethod
    def _restar_vectores(self, vector_u: Any, vector_v: Any) -> Any:
        """Resta dos vectores: u - v."""

    @abc.abstractmethod
    def _copiar_vector(self, vector: Any) -> Any:
        """Retorna una copia independiente del vector."""

    @abc.abstractmethod
    def obtener_dimension(self, conjunto_b: List[Any]) -> int:
        """
        Determina la dimensión del espacio a partir del conjunto B
        (p. ej. n*m para matrices, n para ℝⁿ, grado+1 para polinomios).
        """


# ===========================================================================
# Subclase: Espacio Euclídeo ℝⁿ
# ===========================================================================
class EspacioEuclidiano(EspacioVectorial):
    """
    Espacio ℝⁿ con el producto interno estándar (producto punto):
        ⟨u, v⟩ = Σ u_i · v_i
    """

    def nombre_espacio(self) -> str:
        return "Espacio Euclídeo ℝⁿ"

    def producto_interno(self, vector_u: np.ndarray, vector_v: np.ndarray) -> float:
        return float(np.dot(vector_u.flatten(), vector_v.flatten()))

    def _escalar_por(self, vector: np.ndarray, escalar: float) -> np.ndarray:
        return vector * escalar

    def _restar_vectores(self, vector_u: np.ndarray, vector_v: np.ndarray) -> np.ndarray:
        return vector_u - vector_v

    def _copiar_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector.copy()

    def mostrar_vector(self, vector: np.ndarray) -> str:
        componentes = ", ".join(f"{x:.6g}" for x in vector.flatten())
        return f"({componentes})"

    def obtener_dimension(self, conjunto_b: List[np.ndarray]) -> int:
        if not conjunto_b:
            return 0
        return int(conjunto_b[0].size)

    def ingresar_vectores(self) -> List[np.ndarray]:
        """Solicita n y luego los vectores de ℝⁿ."""
        
        # """Opción para usar base canónica."""
        choice = questionary.select(
            "Selecciona una opción: ",
            choices= [
                "Usar base canónica.",
                "Ingresar vectores manualmente."
            ],
            style=estilo_menu
        ).ask()

        if choice == "Usar base canónica.":
            base_canonica = [np.array([1,0,0], dtype=float), np.array([0,1,0], dtype=float), np.array([0,0,1], dtype=float)]
            return base_canonica


        dimension = questionary.text(
            "¿Cuál es la dimensión n del espacio ℝⁿ?",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Ingrese un entero positivo.",
            style=estilo_menu
        ).ask()
        n = int(dimension)

        cant_str = questionary.text(
            "¿Cuántos vectores desea ingresar en el conjunto B?",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Ingrese un entero positivo.",
            style=estilo_menu
        ).ask()
        cantidad = int(cant_str)

        vectores = []
        for k in range(1, cantidad + 1):
            while True:
                entrada = questionary.text(
                    f"  Vector {k} (ingrese {n} valores separados por comas):",
                    style=estilo_menu
                ).ask()
                try:
                    componentes = [float(x.strip()) for x in entrada.split(",")]
                    if len(componentes) != n:
                        print(f"  ⚠  Se esperaban {n} componentes, se recibieron {len(componentes)}.")
                        continue
                    vectores.append(np.array(componentes, dtype=float))
                    break
                except ValueError:
                    print("  ⚠  Formato inválido. Use números separados por comas.")
        return vectores


# ===========================================================================
# Subclase: Matrices M_{n×m}
# ===========================================================================
class EspacioMatrices(EspacioVectorial):
    """
    Espacio de matrices reales M_{n×m} con el producto interno de Frobenius:
        ⟨A, B⟩ = traza(Aᵀ B) = Σ_{i,j} A_{ij} · B_{ij}
    """

    def nombre_espacio(self) -> str:
        return "Espacio de Matrices M_{n×m}"

    def producto_interno(self, matriz_a: np.ndarray, matriz_b: np.ndarray) -> float:
        return float(np.trace(matriz_a.T @ matriz_b))

    def _escalar_por(self, vector: np.ndarray, escalar: float) -> np.ndarray:
        return vector * escalar

    def _restar_vectores(self, vector_u: np.ndarray, vector_v: np.ndarray) -> np.ndarray:
        return vector_u - vector_v

    def _copiar_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector.copy()

    def mostrar_vector(self, vector: np.ndarray) -> str:
        filas = []
        for fila in vector:
            filas.append("  [" + "  ".join(f"{x:8.4f}" for x in fila) + "]")
        return "\n" + "\n".join(filas)

    def obtener_dimension(self, conjunto_b: List[np.ndarray]) -> int:
        if not conjunto_b:
            return 0
        return conjunto_b[0].size  # n * m

    def ingresar_vectores(self) -> List[np.ndarray]:
        """Solicita las dimensiones n×m y las matrices."""

        # """Opción para usar base canónica."""
        choice = questionary.select(
            "Selecciona una opción: ",
            choices= [
                "Usar base canónica.",
                "Ingresar matrices manualmente."
            ],
            style=estilo_menu
        ).ask()

        if choice == "Usar base canónica.":
            matriz_1 = np.array(
                [[1,0],
                [0,0]], dtype=float
            )
            matriz_2 = np.array(
                [[0,1],
                [0,0]], dtype=float
            )
            matriz_3 = np.array(
                [[0,0],
                [1,0]], dtype=float
            )
            matriz_4 = np.array(
                [[0,0],
                [0,1]], dtype=float
            )

            base_canonica = [matriz_1, matriz_2, matriz_3, matriz_4]
            return base_canonica
        
        n_str = questionary.text(
            "Número de filas (n):",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Entero positivo.",
            style=estilo_menu
        ).ask()
        m_str = questionary.text(
            "Número de columnas (m):",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Entero positivo.",
            style=estilo_menu
        ).ask()
        n, m = int(n_str), int(m_str)

        cant_str = questionary.text(
            "¿Cuántas matrices desea ingresar?",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Entero positivo.",
            style=estilo_menu
        ).ask()
        cantidad = int(cant_str)

        matrices = []
        for k in range(1, cantidad + 1):
            print(f"\n  Matriz {k} ({n}×{m}):")
            filas = []
            for i in range(1, n + 1):
                while True:
                    entrada = questionary.text(
                        f"    Fila {i} ({m} valores separados por comas):",
                        style=estilo_menu
                    ).ask()
                    try:
                        fila = [float(x.strip()) for x in entrada.split(",")]
                        if len(fila) != m:
                            print(f"    ⚠  Se esperaban {m} valores.")
                            continue
                        filas.append(fila)
                        break
                    except ValueError:
                        print("    ⚠  Formato inválido.")
            matrices.append(np.array(filas, dtype=float))
        return matrices


# ===========================================================================
# Subclase: Polinomios P_n
# ===========================================================================
class EspacioPolinomios(EspacioVectorial):
    """
    Espacio de polinomios P_n representados por sus coeficientes
    [a_0, a_1, ..., a_n] (de menor a mayor grado).

    Producto interno integral en [-1, 1]:
        ⟨p, q⟩ = ∫_{-1}^{1} p(t) q(t) dt
    """

    def nombre_espacio(self) -> str:
        return "Espacio de Polinomios P_n"

    def producto_interno(self, polinomio_p: np.ndarray, polinomio_q: np.ndarray) -> float:
        """Producto interno integral: ∫_{-1}^{1} p(t)·q(t) dt"""
        def integrando(t: float) -> float:
            return float(np.polyval(polinomio_p[::-1], t) * np.polyval(polinomio_q[::-1], t))
        resultado, _ = integrate.quad(integrando, -1.0, 1.0)
        return resultado

    def _escalar_por(self, vector: np.ndarray, escalar: float) -> np.ndarray:
        return vector * escalar

    def _restar_vectores(self, vector_u: np.ndarray, vector_v: np.ndarray) -> np.ndarray:
        # Igualar longitudes antes de restar
        lu, lv = len(vector_u), len(vector_v)
        if lu < lv:
            vector_u = np.pad(vector_u, (0, lv - lu))
        elif lv < lu:
            vector_v = np.pad(vector_v, (0, lu - lv))
        return vector_u - vector_v

    def _copiar_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector.copy()

    def mostrar_vector(self, vector: np.ndarray) -> str:
        terminos = []
        for i, coef in enumerate(vector):
            if abs(coef) < TOLERANCIA:
                continue
            if i == 0:
                terminos.append(f"{coef:.4g}")
            elif i == 1:
                terminos.append(f"{coef:.4g}t")
            else:
                terminos.append(f"{coef:.4g}t^{i}")
        return " + ".join(terminos) if terminos else "0"

    def obtener_dimension(self, conjunto_b: List[np.ndarray]) -> int:
        if not conjunto_b:
            return 0
        # Dimensión de P_n es n+1, donde n = grado máximo en B
        return max(len(p) for p in conjunto_b)

    def ingresar_vectores(self) -> List[np.ndarray]:
        """Solicita el grado máximo y los coeficientes de cada polinomio."""
        
        # """Opción para usar base canónica."""
        choice = questionary.select(
            "Selecciona una opción: ",
            choices= [
                "Usar base canónica.",
                "Ingresar polinomios manualmente."
            ],
            style=estilo_menu
        ).ask()

        if choice == "Usar base canónica.":
            base_canonica = [np.array([1,0,0], dtype=float), np.array([0,1,0], dtype=float), np.array([0,0,1], dtype=float)]
            return base_canonica
        
        n_str = questionary.text(
            "¿Cuál es el grado máximo n de los polinomios? (P_n tiene dimensión n+1)",
            validate=lambda t: t.isdigit() and int(t) >= 0 or "Entero no negativo.",
            style=estilo_menu
        ).ask()
        n = int(n_str)
        dim = n + 1

        cant_str = questionary.text(
            "¿Cuántos polinomios desea ingresar?",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Entero positivo.",
            style=estilo_menu
        ).ask()
        cantidad = int(cant_str)

        polinomios = []
        for k in range(1, cantidad + 1):
            while True:
                entrada = questionary.text(
                    f"  Polinomio {k} — ingrese {dim} coeficientes [a₀, a₁, …, a_{n}] separados por comas:",
                    style=estilo_menu
                ).ask()
                try:
                    coefs = [float(x.strip()) for x in entrada.split(",")]
                    if len(coefs) != dim:
                        print(f"  ⚠  Se esperaban {dim} coeficientes.")
                        continue
                    polinomios.append(np.array(coefs, dtype=float))
                    break
                except ValueError:
                    print("  ⚠  Formato inválido.")
        return polinomios


# ===========================================================================
# Subclase: Funciones F
# ===========================================================================

# Funciones predefinidas disponibles
FUNCIONES_PREDEFINIDAS: List[Tuple[str, Any]] = [
    ("1  (constante)",       lambda t: np.ones_like(np.asarray(t, dtype=float))),
    ("t",                    lambda t: np.asarray(t, dtype=float)),
    ("t²",                   lambda t: np.asarray(t, dtype=float) ** 2),
    ("t³",                   lambda t: np.asarray(t, dtype=float) ** 3),
    ("sin(πt)",              lambda t: np.sin(np.pi * np.asarray(t, dtype=float))),
    ("cos(πt)",              lambda t: np.cos(np.pi * np.asarray(t, dtype=float))),
    ("sin(2πt)",             lambda t: np.sin(2 * np.pi * np.asarray(t, dtype=float))),
    ("cos(2πt)",             lambda t: np.cos(2 * np.pi * np.asarray(t, dtype=float))),
    ("eᵗ",                   lambda t: np.exp(np.asarray(t, dtype=float))),
    ("e^(-t)",               lambda t: np.exp(-np.asarray(t, dtype=float))),
]


class EspacioFunciones(EspacioVectorial):
    """
    Espacio de funciones F con producto interno integral en [-1, 1]:
        ⟨f, g⟩ = ∫_{-1}^{1} f(t) g(t) dt

    Las funciones se seleccionan de un menú predefinido.
    """

    def nombre_espacio(self) -> str:
        return "Espacio de Funciones F (en [-1, 1])"

    def producto_interno(self, funcion_f: Any, funcion_g: Any) -> float:
        """Producto interno integral: ∫_{-1}^{1} f(t)·g(t) dt"""
        def integrando(t: float) -> float:
            return float(funcion_f(t)) * float(funcion_g(t))
        resultado, _ = integrate.quad(integrando, -1.0, 1.0)
        return resultado

    def _escalar_por(self, vector: Any, escalar: float) -> Any:
        return lambda t: escalar * vector(t)

    def _restar_vectores(self, vector_u: Any, vector_v: Any) -> Any:
        return lambda t: vector_u(t) - vector_v(t)

    def _copiar_vector(self, vector: Any) -> Any:
        # Las lambdas son inmutables; retornar la misma referencia es seguro
        return vector

    def mostrar_vector(self, vector: Any) -> str:
        # Recuperar nombre desde la tabla de funciones
        for nombre, func in FUNCIONES_PREDEFINIDAS:
            if func is vector:
                return nombre
        return "<función personalizada>"

    def obtener_dimension(self, conjunto_b: List[Any]) -> int:
        # Para el análisis de base, la dimensión se toma como el tamaño de B
        return len(conjunto_b)

    def ingresar_vectores(self) -> List[Any]:
        """Permite al usuario seleccionar funciones del catálogo."""

        # """Opción para usar base canónica."""
        choice = questionary.select(
            "Selecciona una opción: ",
            choices= [
                "Usar base canónica.",
                "Ingresar vectores manualmente."
            ],
            style=estilo_menu
        ).ask()

        if choice == "Usar base canónica.":
            sin = FUNCIONES_PREDEFINIDAS[4][1]
            cos = FUNCIONES_PREDEFINIDAS[5][1]
            base_canonica = [sin, cos]
            return base_canonica
        
        cant_str = questionary.text(
            "¿Cuántas funciones desea ingresar en el conjunto B?",
            validate=lambda t: t.isdigit() and int(t) > 0 or "Entero positivo.",
            style=estilo_menu
        ).ask()
        cantidad = int(cant_str)

        nombres_disponibles = [nombre for nombre, _ in FUNCIONES_PREDEFINIDAS]
        funciones_seleccionadas = []

        for k in range(1, cantidad + 1):
            nombre_elegido = questionary.select(
                f"  Seleccione la función {k}:",
                choices=nombres_disponibles,
                style=estilo_menu
            ).ask()
            for nombre, func in FUNCIONES_PREDEFINIDAS:
                if nombre == nombre_elegido:
                    funciones_seleccionadas.append(func)
                    break

        return funciones_seleccionadas


# ===========================================================================
# Motor de análisis
# ===========================================================================
class AnalizadorEspacioVectorial:
    """
    Orquesta el análisis completo de un conjunto de vectores B
    dentro de un EspacioVectorial dado.
    """

    def __init__(self, espacio: EspacioVectorial):
        self.espacio = espacio

    def ejecutar_analisis(self, conjunto_b: List[Any]) -> None:
        """Realiza y muestra todos los análisis sobre el conjunto B."""
        separador = "─" * 60

        print(f"\n{separador}")
        print(f"    {self.espacio.nombre_espacio()}")
        print(f"    Conjunto B con {len(conjunto_b)} vector(es)")
        print(separador)

        # ── Mostrar vectores ──────────────────────────────────────────────
        print("\n  VECTORES DEL CONJUNTO B:")
        for i, vec in enumerate(conjunto_b, 1):
            print(f"   b_{i} = {self.espacio.mostrar_vector(vec)}")

        # ── Normas ───────────────────────────────────────────────────────
        print(f"\n{separador}")
        print("  NORMAS  ‖bᵢ‖ = √⟨bᵢ, bᵢ⟩:")
        for i, vec in enumerate(conjunto_b, 1):
            norma = self.espacio.calcular_norma(vec)
            print(f"   ‖b_{i}‖ = {norma:.8f}")

        # ── Independencia lineal ──────────────────────────────────────────
        print(f"\n{separador}")
        es_independiente = self.espacio.verificar_independencia_lineal(conjunto_b)
        simbolo = "✅" if es_independiente else "❌"
        estado_indep = "LINEALMENTE INDEPENDIENTE" if es_independiente else "LINEALMENTE DEPENDIENTE"
        print(f"  INDEPENDENCIA LINEAL:  {simbolo}  {estado_indep}")

        if not es_independiente:
            print("   (Al menos un vector es combinación lineal de los demás.)")

        # ── Base ──────────────────────────────────────────────────────────
        print(f"\n{separador}")
        dimension = self.espacio.obtener_dimension(conjunto_b)
        es_base_resultado = self.espacio.es_base(conjunto_b, dimension)
        simbolo_base = "✅" if es_base_resultado else "❌"
        estado_base = "ES BASE" if es_base_resultado else "NO ES BASE"
        print(f"   BASE DEL ESPACIO:       {simbolo_base}  {estado_base}")
        print(f"   (|B| = {len(conjunto_b)}, dim = {dimension})")

        if not es_base_resultado:
            if len(conjunto_b) != dimension:
                print(f"   ⚠  El número de vectores ({len(conjunto_b)}) "
                      f"no coincide con la dimensión ({dimension}).")
            if not es_independiente:
                print("   ⚠  El conjunto no es linealmente independiente.")
            # Si no es base, no tiene sentido evaluar ortogonalidad
            print(f"\n{separador}")
            print("   ℹ  Solo se evalúa ortogonalidad y ortonormalidad para bases válidas.")
            print(separador)
            return

        # ── Ortogonalidad ─────────────────────────────────────────────────
        print(f"\n{separador}")
        es_ortogonal = self.espacio.verificar_ortogonalidad_base(conjunto_b)
        simbolo_ort = "✅" if es_ortogonal else "❌"
        estado_ort = "ORTOGONAL" if es_ortogonal else "NO ORTOGONAL"
        print(f"⊥   ORTOGONALIDAD:          {simbolo_ort}  {estado_ort}")

        # Mostrar tabla de productos internos cruzados
        n = len(conjunto_b)
        if n > 1:
            print("\n   Tabla de productos internos ⟨bᵢ, bⱼ⟩ (i≠j):")
            for i in range(n):
                for j in range(i + 1, n):
                    pi = self.espacio.producto_interno(conjunto_b[i], conjunto_b[j])
                    print(f"     ⟨b_{i+1}, b_{j+1}⟩ = {pi:.8f}")

        # ── Ortonormalidad ────────────────────────────────────────────────
        print(f"\n{separador}")
        es_ortonormal = self.espacio.verificar_ortonormalidad_base(conjunto_b)
        simbolo_on = "✅" if es_ortonormal else "❌"
        estado_on = "ORTONORMAL" if es_ortonormal else "NO ORTONORMAL"
        print(f"⊛   ORTONORMALIDAD:         {simbolo_on}  {estado_on}")

        # ── Gram-Schmidt ──────────────────────────────────────────────────
        if not es_ortonormal:
            print(f"\n{separador}")
            print("  PROCESO DE GRAM-SCHMIDT")
            print("   La base no es ortonormal → se procede a ortonormalizar:")
            try:
                base_ortonormal = self.espacio.proceso_gram_schmidt(conjunto_b)
                print("\n   Base ortonormal resultante:")
                for i, vec in enumerate(base_ortonormal, 1):
                    norma_verificacion = self.espacio.calcular_norma(vec)
                    print(f"   e_{i} = {self.espacio.mostrar_vector(vec)}")
                    print(f"         ‖e_{i}‖ = {norma_verificacion:.8f}  "
                          f"{'✅ unitario' if abs(norma_verificacion - 1.0) < 1e-6 else '⚠ revisar'}")

                # Verificar ortogonalidad de la nueva base
                print("\n   Verificación de ortogonalidad de la nueva base:")
                nueva_es_ortogonal = self.espacio.verificar_ortogonalidad_base(base_ortonormal)
                nueva_es_ortonormal = self.espacio.verificar_ortonormalidad_base(base_ortonormal)
                print(f"   Ortogonal:   {'✅' if nueva_es_ortogonal else '❌'}")
                print(f"   Ortonormal:  {'✅' if nueva_es_ortonormal else '❌'}")
            except ValueError as error:
                print(f"   ⚠  Error en Gram-Schmidt: {error}")
        else:
            print("\n   ✅  La base ya es ortonormal. Gram-Schmidt no es necesario.")

        print(f"\n{separador}\n")


# ===========================================================================
# Interfaz principal con questionary
# ===========================================================================

MAPA_ESPACIOS = {
    "Matrices  (M_{n×m})         — Producto interno de Frobenius": EspacioMatrices,
    "Espacio Euclídeo  (ℝⁿ)      — Producto punto estándar":       EspacioEuclidiano,
    "Polinomios  (P_n)           — Producto interno integral":     EspacioPolinomios,
    "Funciones  (F en [-1,1])    — Producto interno integral":     EspacioFunciones,
}

BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║         ANÁLISIS DE ESPACIOS VECTORIALES  v1.0                ║
║  Independencia · Base · Norma · Ortogonalidad · Gram-Schmidt  ║
╚═══════════════════════════════════════════════════════════════╝
"""


def ejecutar_aplicacion() -> None:
    """Punto de entrada principal de la aplicación."""
    print(BANNER)

    continuar = True
    while continuar:
        # ── Selección del espacio vectorial ───────────────────────────────
        opciones_espacio = list(MAPA_ESPACIOS.keys()) + ["[X]  Salir"]
        eleccion = questionary.select(
            "Seleccione el tipo de Espacio Vectorial:",
            choices=opciones_espacio,
            style=estilo_menu
        ).ask()

        if eleccion is None or eleccion.startswith("[X]"):
            print("\n  Saliendo. \n")
            break

        # ── Instanciar el espacio elegido ─────────────────────────────────
        clase_espacio = MAPA_ESPACIOS[eleccion]
        espacio = clase_espacio()
        print(f"\n  ▶  Espacio seleccionado: {espacio.nombre_espacio()}\n")

        # ── Ingresar vectores ─────────────────────────────────────────────
        try:
            conjunto_b = espacio.ingresar_vectores()
        except (KeyboardInterrupt, EOFError):
            print("\n  Operación cancelada.\n")
            continue

        if not conjunto_b:
            print("  ⚠  No se ingresaron vectores.\n")
            continue

        # ── Análisis ──────────────────────────────────────────────────────
        analizador = AnalizadorEspacioVectorial(espacio)
        analizador.ejecutar_analisis(conjunto_b)

        # ── ¿Continuar? ───────────────────────────────────────────────────
        continuar = questionary.confirm(
            "¿Desea analizar otro conjunto?",
            default=True,
            style=estilo_menu
        ).ask()

        if continuar is None:
            continuar = False

    print()


# ===========================================================================
# Punto de entrada
# ===========================================================================
if __name__ == "__main__":
    ejecutar_aplicacion()