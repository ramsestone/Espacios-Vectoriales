# Análisis de Espacios Vectoriales

Aplicación de línea de comandos en Python para analizar conjuntos de vectores dentro de distintos espacios vectoriales. Evalúa independencia lineal, bases, normas, ortogonalidad y ortonormalidad; y aplica el **Proceso de Gram-Schmidt** cuando la base no es ortonormal.

---

## Características

- Menú interactivo construido con [`questionary`](https://github.com/tmbo/questionary)
- Cuatro espacios vectoriales soportados, cada uno con su producto interno correcto
- Detección automática de independencia lineal vía determinante de la **Matriz de Gram**
- Proceso de Gram-Schmidt completo con verificación del resultado
- Arquitectura orientada a objetos extensible (clase abstracta `EspacioVectorial`)

---

## Espacios Vectoriales Soportados

| Espacio            | Símbolo | Producto Interno                  |
|---                 |---      |---                                |
| Espacio Euclídeo   | Rⁿ      | `⟨u, v⟩ = Σ uᵢvᵢ` (producto punto) |
| Matrices           | M_{n×m} | `⟨A, B⟩ = tr(AᵀB)` (Frobenius)     |
| Polinomios         | P_n     | `⟨p, q⟩ = ∫₋₁¹ p(t)q(t) dt`        |
| Funciones          | F       | `⟨f, g⟩ = ∫₋₁¹ f(t)g(t) dt`        |

---

## Requisitos

- Python **3.8** o superior
- Las siguientes librerías (se instalan con pip):

```
numpy
scipy
questionary
```

---

## Instalación

```bash
# 1. Clonar o descargar el archivo
git clone <url-del-repositorio>
cd analisis-espacios-vectoriales

# 2. (Opcional) Crear un entorno virtual
python -m venv .venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install numpy scipy questionary
```

---

## Uso

```bash
python analisis_espacio_vectorial.py
```

Al ejecutar el programa, aparecerá el menú principal:

```
╔══════════════════════════════════════════════════════════╗
║         ANÁLISIS DE ESPACIOS VECTORIALES  v1.0           ║
║  Independencia · Base · Norma · Ortogonalidad · Gram-Schmidt  ║
╚══════════════════════════════════════════════════════════╝

? Seleccione el tipo de Espacio Vectorial:
  ❯ Matrices  (M_{n×m})        — Producto interno de Frobenius
    Espacio Euclídeo  (ℝⁿ)     — Producto punto estándar
    Polinomios  (P_n)           — Producto interno integral
    Funciones  (F en [-1,1])    — Producto interno integral
    ❌  Salir
```

Navegue con las flechas del teclado y confirme con **Enter**.

---

## Flujo de Análisis

Para cualquier espacio y conjunto B ingresado, el programa ejecuta:

```
1. Mostrar los vectores del conjunto B
2. Calcular la norma ‖bᵢ‖ de cada vector
3. Verificar Independencia Lineal  (det de la Matriz de Gram)
4. Verificar si B es Base del espacio
5. Si es base:
   ├─ Verificar Ortogonalidad  (⟨bᵢ, bⱼ⟩ ≈ 0 para i ≠ j)
   ├─ Verificar Ortonormalidad (ortogonal + ‖bᵢ‖ ≈ 1)
   └─ Si NO es ortonormal → aplicar Gram-Schmidt y mostrar la base ortonormal
```

---

## Ejemplos de Uso

### ℝ³ — Base no ortonormal

**Entrada:**
```
Dimensión n: 3
Cantidad de vectores: 3
b₁ = 1, 1, 0
b₂ = 0, 1, 1
b₃ = 1, 0, 1
```

**Salida esperada:**
```
📋  VECTORES DEL CONJUNTO B:
   b_1 = (1, 1, 0)
   b_2 = (0, 1, 1)
   b_3 = (1, 0, 1)

📏  NORMAS:
   ‖b_1‖ = 1.41421356
   ‖b_2‖ = 1.41421356
   ‖b_3‖ = 1.41421356

🔗  INDEPENDENCIA LINEAL:  ✅  LINEALMENTE INDEPENDIENTE
🏛   BASE DEL ESPACIO:       ✅  ES BASE   (|B| = 3, dim = 3)
⊥   ORTOGONALIDAD:          ❌  NO ORTOGONAL
⊛   ORTONORMALIDAD:         ❌  NO ORTONORMAL

🔄  PROCESO DE GRAM-SCHMIDT
   e_1 = (0.7071, 0.7071, 0.0000)   ‖e_1‖ = 1.00000000  ✅ unitario
   e_2 = (-0.4082, 0.4082, 0.8165)  ‖e_2‖ = 1.00000000  ✅ unitario
   e_3 = (0.5774, -0.5774, 0.5774)  ‖e_3‖ = 1.00000000  ✅ unitario
```

---

### P₂ — Polinomios (grado ≤ 2)

Los polinomios se ingresan como listas de coeficientes `[a₀, a₁, a₂]`
correspondientes a `a₀ + a₁t + a₂t²`.

**Entrada:**
```
Grado máximo n: 2
Cantidad de polinomios: 3
p₁ = 1, 0, 0    →  1
p₂ = 0, 1, 0    →  t
p₃ = 0, 0, 1    →  t²
```

El producto interno es integral: `⟨pᵢ, pⱼ⟩ = ∫₋₁¹ pᵢ(t)pⱼ(t) dt`.
Los polinomios de Legendre `{1, t, t²}` son ortogonales en este espacio.

---

### M_{2×2} — Matrices con producto de Frobenius

Las matrices se ingresan fila por fila:

**Entrada:**
```
n = 2,  m = 2
Cantidad de matrices: 2

Matriz 1:
  Fila 1: 1, 0
  Fila 2: 0, 0

Matriz 2:
  Fila 1: 0, 1
  Fila 2: 0, 0
```

El producto de Frobenius comprueba `⟨A, B⟩ = tr(AᵀB) = Σ AᵢⱼBᵢⱼ`.

---

### Funciones predefinidas

Para el espacio de funciones, el usuario selecciona de un catálogo:

```
1  (constante)
t
t²
t³
sin(πt)
cos(πt)
sin(2πt)
cos(2πt)
eᵗ
e^(-t)
```

Por ejemplo, `{sin(πt), cos(πt)}` es una base ortogonal en `[-1, 1]`.

---

## Arquitectura del Código

```
EspacioVectorial  (clase abstracta)
│
├── producto_interno()         ← abstracto por subclase
├── calcular_norma()           ← √⟨v, v⟩
├── verificar_independencia_lineal()  ← det(Gram) ≠ 0
├── es_base()                  ← LI + |B| == dim
├── verificar_ortogonalidad_base()
├── verificar_ortonormalidad_base()
└── proceso_gram_schmidt()
│
├── EspacioEuclidiano    →  ⟨u,v⟩ = u·v
├── EspacioMatrices      →  ⟨A,B⟩ = tr(AᵀB)
├── EspacioPolinomios    →  ⟨p,q⟩ = ∫₋₁¹ p·q dt
└── EspacioFunciones     →  ⟨f,g⟩ = ∫₋₁¹ f·g dt

AnalizadorEspacioVectorial
└── ejecutar_analisis()  ← orquesta y muestra todos los resultados
```

---

## Convenciones de Nomenclatura

Todo el código usa **español** para variables, funciones y comentarios:

| Elemento | Ejemplo |
|---|---|
| Función de verificación | `verificar_independencia_lineal` |
| Función de cálculo | `calcular_norma`, `producto_interno` |
| Proceso matemático | `proceso_gram_schmidt` |
| Variables | `conjunto_b`, `base_ortonormal`, `matriz_gram` |
| Parámetros | `vector_u`, `vector_v`, `escalar` |

---

## Tolerancia Numérica

El programa utiliza una tolerancia global `TOLERANCIA = 1e-9` para todas las
comparaciones con cero (ortogonalidad, norma unitaria, determinante). Se puede
ajustar en la constante al inicio del archivo.

---

## Dependencias Externas

| Librería | Uso |
|---|---|
| `numpy` | Álgebra lineal (matrices, determinantes, producto punto) |
| `scipy.integrate` | Producto interno integral para polinomios y funciones |
| `questionary` | Menú interactivo en la terminal |

---

## Limitaciones Conocidas

- El espacio de **Funciones** sólo admite las 10 funciones del catálogo predefinido; no acepta expresiones arbitrarias del usuario.
- Para **Polinomios**, todos los vectores del conjunto B deben tener el mismo grado máximo (se rellena con ceros si es necesario).
- El programa no verifica si el espacio generado por B abarca efectivamente todo el espacio (solo comprueba cardinalidad e independencia lineal).

---

## Licencia

MIT — libre para uso, modificación y distribución.