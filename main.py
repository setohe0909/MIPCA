import numpy as np

class MIPCA:
    """
    Implementación simple de PCA estilo sklearn.PCA con SVD.
    - n_components: None o int (número de componentes a conservar)
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (n_muestras, n_features).")

        n_samples, n_features = X.shape
        k_max = min(n_samples, n_features)

        # Validar n_components
        if self.n_components is None:
            k = k_max
        else:
            if not isinstance(self.n_components, int) or self.n_components <= 0:
                raise ValueError("n_components debe ser un entero positivo o None.")
            if self.n_components > k_max:
                raise ValueError(f"n_components ≤ {k_max} para datos de forma {X.shape}.")
            k = self.n_components

        # 1) Centrar datos
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2) SVD (Xc = U S Vt)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # 3) Componentes (direcciones principales)
        self.components_ = Vt  # forma: (n_features, n_features), tomaremos [:k] al transformar
        self.singular_values_ = S  # valores singulares (longitud = k_max)

        # 4) Varianza explicada por componente (eigenvalues de covarianza)
        #    lambda_i = S_i^2 / (n_samples - 1)
        self.n_components_ = k
        explained_var_all = (S**2) / (n_samples - 1)
        self.explained_variance_ = explained_var_all[:k]

        # 5) Proporción de varianza explicada
        total_var = explained_var_all.sum()  # equivale a var total = suma de varianzas por feature
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        """Proyecta X a las primeras n_components direcciones principales."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("Debes llamar fit() antes de transform().")
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        W = self.components_[:self.n_components_]  # (k, n_features)
        # Dos formas equivalentes: Xc @ W.T o U[:, :k] * S[:k] (usando SVD de fit)
        return Xc @ W.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruye desde el espacio reducido al original (aproximado)."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("Debes llamar fit() antes de inverse_transform().")
        W = self.components_[:self.n_components_]  # (k, n_features)
        return X_transformed @ W + self.mean_
