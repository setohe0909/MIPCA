# MIPCA (MIPCA: Multiple Imputation with PCA)

Más detalles → [Documentación](https://www.rdocumentation.org/packages/missMDA/versions/1.20/topics/MIPCA)

---

## PCA (Análisis de Componentes Principales)  
Es una forma de mirar los datos desde el mejor ángulo posible, donde se ve la mayor diferencia entre ellos.  
Así puedes resumir tus datos en menos “ángulos”, pero sin perder tanta información.

---

## Cómo funciona paso a paso

1. **Restar la media (centrar los datos)**  
   - Imagina que tienes varias reglas en la mesa con diferentes medidas 📏.  
   - Primero, mueves todas las reglas para que empiecen en el mismo lugar (como llevar todas al 0 de la regla).  
   - Eso es **restar la media**.  

2. **Buscar los ángulos donde hay más diferencia**  
   - Piensa en muchos puntos dibujados en un papel ✏️.  
   - PCA dibuja una línea en el papel que pasa por donde los puntos se estiran más (como una flecha en la dirección más larga de la nube de puntos).  
   - Esa flecha es el **primer componente principal**.  
   - Luego busca otra flecha en la siguiente dirección más importante (segunda componente).  

3. **Guardar esos ángulos**  
   - Cada flecha que encontramos se llama **componente**.  
   - Cuánto “explica” esa flecha se llama **varianza explicada** (es como decir: *“esta flecha cuenta el 70% de la historia de mis datos”*).  

4. **Transformar los datos**  
   - Ahora que tenemos las flechas (componentes), podemos proyectar nuestros puntos en esas flechas.  
   - Es como si en vez de guardar toda la foto en 3D, la dibujáramos solo en la flecha más importante en 2D.  
   - 🎯 Así resumimos los datos en menos dimensiones.  

---

## Cómo lo ponemos en Python

Escribimos una clase llamada **MIPCA**, que tiene 3 súper poderes:  
1. `fit` (ajustar) → aprende cuáles son las mejores flechas (componentes) de los datos.  
2. `transform` (transformar) → usa esas flechas para dibujar los datos en un espacio más pequeño.  
3. `explained_variance` (varianza explicada) → te dice qué tanto explica cada flecha de la historia.  

```python
import numpy as np

# Supongamos que tenemos datos de 100 niños con 5 notas diferentes
X = np.random.rand(100, 5)

# Creamos nuestro PCA casero
pca = MIPCA(n_components=2)

# Le pedimos que aprenda las mejores "flechas"
pca.fit(X)

# Transformamos los datos a 2 dimensiones
Z = pca.transform(X)

print("Varianza explicada:", pca.explained_variance_ratio_)
```

**👉 Al final nos dice:**

	-	“La primera flecha explica el 60% de la historia”
	-	“La segunda flecha explica el 30%”
	-	Total = 90% (muy bien, ya sabemos casi toda la historia con solo 2 flechas 🎉).

## Analogía final

Imagina que tienes un rompecabezas gigante con 1000 piezas 🧩.
PCA te ayuda a resumirlo en pocas piezas grandes, que aunque no sean todas las piezas originales,
sí muestran casi toda la imagen.
