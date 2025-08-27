import numpy as np
from main import MIPCA

# Datos de juguete: 100 muestras, 5 features
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
X[:, 2] = X[:, 0] * 0.8 + rng.normal(scale=0.1, size=100)  # correlación para que PCA "vea" estructura

pca = MIPCA(n_components=2)
Z = pca.fit_transform(X)  # Z tiene forma (100, 2)

print("Media por feature:", pca.mean_)
print("Componentes (cada fila es un componente):\n", pca.components_[:pca.n_components_])
print("Varianza explicada:", pca.explained_variance_)
print("Proporción de varianza explicada:", pca.explained_variance_ratio_)
print("Suma de proporciones (<=1):", pca.explained_variance_ratio_.sum())

# Reconstrucción aproximada al espacio original
X_recon = pca.inverse_transform(Z)  # forma (100, 5)
