# tarea2-mod2-Veru
Investigación y Desarrollo de un Modelo de Regresión Lineal Simple

Cómo se construye y evalúa un modelo de Regresión Lineal
Simple aplicando Python y librerías comunes (pandas, scikit-learn, matplotlib, seaborn,
etc.).

## 1. ¿Qué es la regresión y cuál es su propósito en el Machine Learning? ¿En qué se diferencia de otros tipos de modelos supervisados como la clasificación?

La regresión es una técnica estadística y de machine learning que busca establecer una relación entre una variable dependiente (la que queremos predecir) y una o más variables independientes (las características o features).

 Mientras que la regresión predice valores numéricos, la clasificación se enfoca en asignar etiquetas categóricas a los datos. Un ejemplo clásico es el filtrado de spam, donde un algoritmo debe decidir si un correo es «spam» o «no spam». Entre los algoritmos más utilizados en clasificación están la regresión logística (a pesar de su nombre, es un método de clasificación), los árboles de decisión, las máquinas de soporte vectorial (SVM) y las redes neuronales. Cada uno tiene sus ventajas: la regresión logística es simple y eficiente para problemas binarios, los árboles de decisión son interpretables, y las SVM funcionan bien en espacios de alta dimensionalidad.

## 2. ¿Qué es la Regresión Lineal Simple y qué representa la ecuación ŷ = β0 + β1 · X? Explica qué significan β0 (intercepto) y β1 (pendiente) y cómo interpretarlos en un contexto real.

Uno de los algoritmos más simples y conocidos es la regresión lineal, que asume una relación lineal entre las variables. Por ejemplo, si queremos predecir el precio de una vivienda, podríamos usar una ecuación del tipo y = mx + b, donde y es el precio, x es el tamaño de la casa, m es la pendiente (que indica cuánto aumenta el precio por metro cuadrado) y b es el intercepto (el valor base del precio).

 Sin embargo, no todos los problemas pueden modelarse con una simple línea recta. En casos más complejos, donde la relación entre variables no es lineal, se utilizan técnicas como la regresión polinómica, que permite ajustar curvas, o métodos más avanzados como Random Forest y Gradient Boosting, que pueden capturar patrones más intrincados en los datos. Un aspecto clave en regresión es la evaluación del modelo, donde métricas como el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²) nos indican qué tan bien se ajusta el modelo a los datos. Un valor de R² cercano a 1 significa que el modelo explica gran parte de la variabilidad de los datos, mientras que un MSE bajo indica que las predicciones son cercanas a los valores reales.

# Regresión lineal simple 

 La regresión lineal simple se usa para relacionar una variable independiente X con una variable dependiente Y. Es decir, en una regresión lineal simple solo hay dos variables (la variable explicativa X y la variable respuesta Y) y se intenta aproximar la relación que hay entre ambas variables.

 La ecuación de un modelo de regresión lineal simple es una recta, por lo que está formada por dos coeficientes: la constante de la ecuación (β0) y el coeficiente de la correlación entre las dos variables (β1). Por lo tanto, la ecuación de un modelo de regresión lineal simple es y=β0+β1x.

 y=\beta_0+\beta_1x

 Las fórmulas para calcular los coeficientes de la regresión lineal simple son las siguientes:

 \begin{array}{c}\beta_1=\cfrac{\displaystyle \sum_{i=1}^n (x_i-\overline{x})(y_i-\overline{y})}{\displaystyle \sum_{i=1}^n (x_i-\overline{x})^2}\\[12ex]\beta_0=\overline{y}-\beta_1\overline{x}\end{array}



 Donde:

 \beta_0 es la constante de la recta de regresión.
 \beta_1 es la pendiente de la recta de regresión.
 x_i es el valor de la variable independiente X del dato i.
 y_i es el valor de la variable dependiente Y del dato i.
 \overline{x} es la media de los valores de la variable independiente X.
 \overline{y} es la media de los valores de la variable dependiente Y.

# 3. ¿Cuáles son los supuestos de la Regresión Lineal? Describe los supuestos principales: linealidad, homocedasticidad, independencia de errores y normalidad de residuos. ¿Por qué es importante verificarlos?

En un modelo de regresión lineal, se deben cumplir los siguientes requisitos para que el modelo tenga validez:

Independencia: los residuos deben ser independientes entre sí. Una manera común de garantizar la independencia del modelo es añadiendo aleatoriedad en el proceso de muestreo.

Homocedasticidad: debe haber homogeneidad en las varianzas de los residuos, es decir, la variabilidad de los residuos debe ser constante.

No multicolinealidad: las variables explicativas incluidas en el modelo no pueden estar relacionadas entre sí o, al menos, su relación debe ser muy débil.

Normalidad: los residuos deben estar distribuidos normalmente, o dicho de otra forma, deben seguir una distribución normal de media 0.

Linealidad: se supone que la relación entre la variable respuesta y las variables explicativas es lineal.

## 4. ¿Qué es la función de coste (Loss Function) en regresión? Explica qué es el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE). ¿Cuál es la diferencia y cuándo preferirías usar uno u otro?

# En machine learning (ML), se utiliza una función de pérdida para medir el rendimiento del modelo calculando la desviación de las predicciones de un modelo con respecto a las predicciones correctas de "verdad fundamental". La optimización de un modelo implica ajustar sus parámetros para minimizar la obtención de alguna función de pérdida.

Una función de pérdida es un tipo de función objetivo, que en el contexto de la ciencia de datos se refiere a cualquier función cuya minimización o maximización representa el objetivo del entrenamiento del modelo. El término "función de pérdida", que suele ser sinónimo de función de coste o función de error, hace referencia específicamente a situaciones en las que la minimización es el objetivo del entrenamiento para un modelo de machine learning.

En términos simples, una función de pérdida rastrea el grado de error en los resultados de un modelo de inteligencia artificial (IA) . Lo hace cuantificando la diferencia ("pérdida") entre un valor previsto, es decir, el resultado del modelo, para una entrada determinada y el valor real o verdad fundamental. Si las predicciones de un modelo son precisas, la pérdida es pequeña. Si sus predicciones son inexactas, la pérdida es grande.

El objetivo fundamental del machine learning es entrenar modelos para generar buenas predicciones. Las funciones de pérdida nos permiten definir y perseguir ese objetivo matemáticamente. Durante el entrenamiento, los modelos "aprenden" a generar mejores predicciones ajustando los parámetros de forma que se reduzcan las pérdidas. Un modelo de machine learning se ha entrenado lo suficiente cuando la pérdida se ha minimizado por debajo de un umbral predeterminado.


# Error cuadrático medio (MSE)

 La función de pérdida de error cuadrático medio, también llamada pérdida L2 o pérdida cuadrática, suele ser la función predeterminada para la mayoría de los algoritmos de regresión. Como su nombre indica, el MSE se calcula como la media de las diferencias al cuadrado entre el valor previsto y el valor real en todos los ejemplos de entrenamiento. La fórmula para calcular el MSE a través de n puntos de datos se escribe como  1n∑i=1n(yi-yi^)2, en la que y es el valor verdadero e ŷ es el valor predicho.

 Elevar el error al cuadrado significa que el valor resultante es siempre positivo: como tal, el MSE evalúa sólo la magnitud del error y no su dirección. Elevar el error al cuadrado también da a los grandes errores un impacto desproporcionadamente fuerte en la pérdida global, lo que castiga fuertemente a los valores atípicos e incentiva al modelo a reducirlos. Por tanto, el MSE es adecuado cuando se supone que las salidas objetivo tienen una distribución normal (gaussiana).

 El MSE siempre es diferenciable, lo que lo hace práctico para optimizar los modelos de regresión mediante el descenso de gradiente.

# Error medio absoluto (MAE) 

 El error medio absoluto o pérdida L1 mide la diferencia absoluta media entre el valor previsto y el valor real. Al igual que el MSE, el MAE siempre es positivo y no distingue entre estimaciones demasiado altas o demasiado bajas. Se calcula como la suma del valor absoluto de todos los errores dividido por el tamaño de la muestra.

 Dado que no eleva al cuadrado cada valor de pérdida, el MAE es más resistente a los valores atípicos que el MSE. Por lo tanto, el MAE es ideal cuando los datos pueden contener algunos valores extremos que no deberían afectar demasiado al modelo. La pérdida de L1 también penaliza más los pequeños errores que la pérdida de L2.

 La función de pérdida del MAE no es diferenciable en los casos en los que la salida prevista coincide con la salida real. Por lo tanto, el MAE requiere más pasos intermedios durante la optimización.

# Comparación de MSE (pérdida L2) y MAE (pérdida L1)
 En pocas palabras, MSE es fácil de calcular, pero MAE es más robusto a puntos anormales. Introduzcamos las razones de la diferencia entre los dos.

 Al entrenar un modelo de aprendizaje automático, nuestro objetivo es encontrar el punto donde la función de pérdida alcanza un mínimo. Cuando el valor predicho es igual al valor real, estas dos funciones se pueden minimizar.

# Las ventajas y desventajas de MSE y MAE
 MSE eleva al cuadrado el error (sea e = valor predicho del valor verdadero), por lo que si e> 1, MSE aumentará aún más el error. Si hay puntos anormales en los datos, el valor de e será grande y e será mucho mayor que | e |. Por lo tanto, en comparación con el uso de MAE para calcular la pérdida, el modelo que usa MSE dará más peso a los puntos anormales. En el segundo ejemplo, el modelo que utiliza RMSE para calcular la pérdida se actualizará en la dirección de reducir el error de anomalías a expensas de los errores de otras muestras. Sin embargo, esto reducirá el rendimiento general del modelo. Si los datos de entrenamiento están contaminados por puntos anormales, entonces la pérdida de MAE es más útil (por ejemplo, hay una gran cantidad de ejemplos de falsos negativos y positivos en los datos de entrenamiento, pero no existe tal problema en el conjunto de prueba).

   Se puede entender intuitivamente: si minimizamos el MSE para dar solo un valor predicho para todos los puntos de muestra, entonces este valor debe ser el promedio de todos los valores objetivo. Pero si se trata de minimizar el MAE, entonces este valor será la mediana del valor objetivo de todos los puntos de muestra. Como todos sabemos, para los valores atípicos, la mediana es más robusta que la media, por lo que MAE también es más estable para los valores atípicos que el MSE.

   MAE tiene un problema grave (especialmente para redes neuronales): el gradiente actualizado es siempre el mismo, es decir, incluso para un valor de pérdida pequeño, el gradiente también es grande. Esto no favorece el aprendizaje de modelos. Para resolver esta deficiencia, podemos usar una tasa de aprendizaje variable para reducir la tasa de aprendizaje cuando la pérdida está cerca del mínimo. En este caso, MSE funciona bien y puede converger efectivamente incluso con una tasa de aprendizaje fija. El gradiente de pérdida de MSE aumenta a medida que aumenta la pérdida y disminuye cuando la pérdida tiende a cero. Esto hace que los resultados del uso del modelo MSE sean más precisos al final del entrenamiento.

# Cómo elegir la pérdida entre MSE y MAE
 Si el punto anormal representa una situación anormal que es importante en los negocios y necesita ser detectada, se debe usar la función de pérdida de MSE. Por el contrario, si solo los valores atípicos se consideran datos dañados, se debe utilizar la función de pérdida MAE. En general, cuando se trata de puntos anormales, la función de pérdida MAE es más estable, pero su derivada no es continua, por lo que la eficiencia de la solución es menor. La función de pérdida de MSE es más sensible a los puntos anormales, pero al hacer su derivada 0, se puede obtener una solución cerrada más estable.

 Cuando los dos problemas existen al mismo tiempo, debe considerar otras funciones de pérdida más adecuadas, como la pérdida de Huber (error absoluto medio suavizado), la pérdida de Log-Cosh, la pérdida de cuantiles, etc

## ¿Qué significa dividir los datos en train y test? ¿Por qué es necesario? ¿Qué
problema evitamos al no evaluar el modelo con los mismos datos con los que lo
entrenamos?