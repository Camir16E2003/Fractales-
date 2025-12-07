import numpy as np
import matplotlib.pyplot as plt

def generar_fractal_newton(funcion_id=1, resolucion=800, max_iter=30, tol=1e-8):
    """
    Genera un fractal de Newton basado en la función seleccionada.
    
    funcion_id:
        1 -> z^3 - 1  (3 raíces, clásico)
        2 -> z^4 - 1  (4 raíces)
        3 -> z^6 + z^3 - 1 (Exótico)
    """

    x = np.linspace(-1.5, 1.5, resolucion)
    y = np.linspace(-1.5, 1.5, resolucion)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y 
    
    if funcion_id == 1:
        def f(z): return z**3 - 1
        def df(z): return 3*z**2
        titulo = "Fractal f(z) = z^3 - 1"
        
    elif funcion_id == 2:
        def f(z): return z**4 - 1
        def df(z): return 4*z**3
        titulo = "Fractal f(z) = z^4 - 1"

    elif funcion_id == 3:
        x = np.linspace(-5, 5, resolucion)
        y = np.linspace(-5, 5, resolucion)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        def f(z): return np.sin(z)
        def df(z): return np.cos(z)
        titulo = "Fractal f(z) = sin(z)"
    
    else:
        def f(z): return z**3 - 2*z + 2
        def df(z): return 3*z**2 - 2
        titulo = "Fractal f(z) = z^3 - 2z + 2"

    root_map = np.zeros(Z.shape, dtype=int)
    
    mask_divergente = np.zeros(Z.shape, dtype=bool)

    for i in range(max_iter):
        derivada = df(Z)
        zero_mask = derivada == 0
        derivada[zero_mask] = 1e-9 
        
        dz = f(Z) / derivada
        Z = Z - dz

    Z_rounded = np.round(Z, decimals=4)
    
    raices_unicas = np.unique(Z_rounded)
    
    colores = np.zeros(Z.shape)
    
    for i, raiz in enumerate(raices_unicas):
        mascara = np.abs(Z - raiz) < 0.1
        colores[mascara] = i

    plt.figure(figsize=(10, 10))
    plt.imshow(colores, cmap='inferno', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.title(titulo, fontsize=15)
    plt.xlabel("Real (Re)")
    plt.ylabel("Imaginario (Im)")
    plt.colorbar(label="Índice de la Raíz")
    plt.show()

print("Generando Fractal 1 (z^3 - 1)...")
generar_fractal_newton(funcion_id=1)

print("Generando Fractal 2 (z^4 - 1)...")
generar_fractal_newton(funcion_id=2)
