# Investigación y Desarrollo de un Modelo de Regresión Lineal Simple

Cómo se construye y evalúa un modelo de Regresión Lineal
Simple aplicando Python y librerías comunes (pandas, scikit-learn, matplotlib, seaborn,
etc.).

# 1. ¿Qué es la regresión y cuál es su propósito en el Machine Learning? ¿En qué se diferencia de otros tipos de modelos supervisados como la clasificación?

La regresión es una técnica estadística y de machine learning que busca establecer una relación entre una variable dependiente (la que queremos predecir) y una o más variables independientes (las características o features).

 Mientras que la regresión predice valores numéricos, la clasificación se enfoca en asignar etiquetas categóricas a los datos. Un ejemplo clásico es el filtrado de spam, donde un algoritmo debe decidir si un correo es «spam» o «no spam». Entre los algoritmos más utilizados en clasificación están la regresión logística (a pesar de su nombre, es un método de clasificación), los árboles de decisión, las máquinas de soporte vectorial (SVM) y las redes neuronales. Cada uno tiene sus ventajas: la regresión logística es simple y eficiente para problemas binarios, los árboles de decisión son interpretables, y las SVM funcionan bien en espacios de alta dimensionalidad.

# 2. ¿Qué es la Regresión Lineal Simple y qué representa la ecuación ŷ = β0 + β1 · X? Explica qué significan β0 (intercepto) y β1 (pendiente) y cómo interpretarlos en un contexto real.

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

# 4. ¿Qué es la función de coste (Loss Function) en regresión? Explica qué es el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE). ¿Cuál es la diferencia y cuándo preferirías usar uno u otro?

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

#  ¿Qué significa dividir los datos en train y test? ¿Por qué es necesario? ¿Qué problema evitamos al no evaluar el modelo con los mismos datos con los que lo entrenamos?

# Por qué no alcanza con “entrenar y medir”
Cuando empezamos en Machine Learning, suele parecer suficiente dividir los datos en dos partes:

una para entrenar;
otra para medir.

 Esa idea es correcta como punto de partida, pero se vuelve insuficiente cuando además queremos comparar modelos o ajustar parámetros. En ese caso, si miramos demasiadas veces el conjunto de prueba, terminamos adaptando nuestras decisiones a ese conjunto y dejamos de medir de forma objetiva.

# El rol de cada conjunto
La separación clásica en tres grupos ayuda a resolver ese problema:

Entrenamiento: se usa para que el modelo aprenda patrones.
Validación: se usa para comparar variantes del modelo y elegir configuraciones.
Prueba: se usa al final, una sola vez, para estimar cómo rendirá el modelo en datos no vistos.
La idea central es esta: el conjunto de prueba debe mantenerse “virgen” hasta el final. Si lo usamos antes, dejamos de tener una medida confiable del rendimiento real.

# Una analogía simple
Imaginemos que estás preparando un examen:

el entrenamiento sería estudiar y practicar;
la validación sería hacer simulacros para ver qué estrategia funciona mejor;
la prueba sería el examen real.
Si usaras las preguntas exactas del examen real para decidir cómo prepararte, la nota final dejaría de ser una evaluación honesta. En Machine Learning pasa algo parecido.


# 6. ¿Qué métricas se usan para evaluar un modelo de regresión? Investiga y explica con sus fórmulas.
En el mundo del Machine Learning, las métricas de evaluación son fundamentales porque permiten entender de forma objetiva qué tan bien está funcionando un modelo. A través de estas métricas es posible cuantificar la calidad de las predicciones, identificar limitaciones y estimar si el modelo se comportará correctamente cuando enfrente datos nuevos. En este artículo —dirigido a lectores con conocimientos intermedios— exploraremos las métricas más utilizadas, su significado en problemas de regresión y clasificación, y cómo elegirlas según las características del caso. También revisaremos aspectos clave de interpretación, limitaciones comunes y buenas prácticas para una evaluación confiable.

# Principales métricas de clasificación
- Exactitud (Accuracy)
 Proporción de predicciones correctas sobre el total. Útil cuando las clases están balanceadas.

- Precisión (Precision)
 De todas las predicciones positivas que hizo el modelo, ¿cuántas realmente eran positivas? Es clave cuando los falsos positivos tienen un costo alto.
- Sensibilidad / Recall (también llamada “exhaustividad”)
 De todos los casos positivos reales, ¿cuántos detectó el modelo? Importante cuando perder positivos reales es costoso.
- F1-Score
Media armónica entre precisión y recall. Es especialmente útil cuando necesitas balancear ambos —ni muchos falsos positivos, ni muchos falsos negativos.
Curva ROC y AUC (Area Under the Curve)
Para modelos que entregan probabilidades —no solo clases—, la curva ROC muestra la relación entre la tasa de verdaderos positivos y falsos positivos al variar el umbral de decisión. El AUC resume esta capacidad de discriminación en un solo valor. Cuanto más cercano a 1, mejor.
(Opcional) PR-AUC / Curva Precision–Recall
Especialmente útil cuando tienes clases desbalanceadas. En esos casos, AUC-ROC puede dar una impresión demasiado optimista. Muchas prácticas de Data Science prefieren PR-AUC en estos escenarios.

● R2 (coeficiente de determinación)
El coeficiente de determinación, cuyo símbolo es R2 (R cuadrado), es un estadístico que mide la bondad de ajuste de un modelo de regresión. El coeficiente de determinación muestra lo bien que se ajusta un modelo de regresión a un conjunto de datos, es decir, indica el porcentaje explicado por el modelo de regresión.

Por lo tanto, cuanto mayor sea el coeficiente de determinación, mejor será el modelo de regresión. Aunque no siempre se cumple esta condición, en principio nos interesa un coeficiente de determinación lo más grande posible. Más abajo veremos cómo interpretar el coeficiente de determinación.



● MSE (Mean Squared Error)
La fórmula para el error cuadrático medio es:

Error cuadrático medio = \frac{1}{n}\sum_{i = 1}^{n}(Y_i - \hat Y_i)^2

Donde:

n es el número de observaciones en el conjunto de datos.

yi es el valor real de la observación.

\hat Y_i es el valor predicho de la i-ésima observación.

● RMSE (Root Mean Squared Error)

La fórmula para RMSE se puede expresar de la siguiente manera:

[
RMSE = sqrt{left(sum(P_i – O_i)^2right) / n}
]

Dónde:

PAGi: Valor predicho
Oi: Valor observado
norte: Número total de observaciones
● MAE (Mean Absolute Error)

La fórmula para calcular el error absoluto medio es relativamente simple y se puede expresar matemáticamente de la siguiente manera:

[ MAE = frac{1}{n} suma_{i=1}^{n} |y_i – sombrero{y}_i| ]

En esta fórmula, ( n ) representa el número total de observaciones, ( y_i ) denota los valores reales y ( hat{y}_i ) significa los valores predichos. La diferencia absoluta entre cada valor real y predicho se calcula, se suma y luego se divide por el número de observaciones para obtener el promedio. Este cálculo sencillo hace que MAE sea una opción popular para muchas tareas de análisis de datos, particularmente en escenarios de regresión.

# 7. ¿Qué son los residuos y para qué sirve analizarlos? ¿Qué indica un residuo cercano a 0? ¿Qué patrón en los residuos revelaría que el modelo no es adecuado?
Un residual se define como la diferencia entre un valor observado y su valor predicho correspondiente. Si todos los puntos de datos estuvieran exactamente en la línea de mejor ajuste, entonces todos los residuos serían iguales a cero. Por otro lado, habrá un residuo distinto de cero para cualquier punto que no se encuentre en la línea.

# 8. ¿Qué es el overfitting y cómo se detecta comparando R2 en train vs. test?
El overfitting es uno de los problemas más comunes en el aprendizaje automático y ocurre cuando un modelo memoriza los datos de entrenamiento en lugar de aprender patrones generalizables. Esto se manifiesta cuando el modelo tiene un rendimiento excepcional en los datos de entrenamiento pero un desempeño pobre en datos nuevos, como los de prueba o validación. Una forma sencilla de detectar el overfitting es comparando las métricas de rendimiento en ambos conjuntos: si la precisión en entrenamiento es mucho mayor que en prueba, es probable que el modelo esté sobreajustado.

En problemas de regresión, donde predecimos valores continuos, las métricas más comunes incluyen el error cuadrático medio (MSE), el error absoluto medio (MAE) y el coeficiente de determinación (R²). El MSE penaliza más los errores grandes al elevar al cuadrado las diferencias entre predicciones y valores reales, mientras que el MAE proporciona una medida lineal del error promedio.

El R², por otro lado, indica qué proporción de la varianza en la variable dependiente es explicada por el modelo, con valores cercanos a 1 indicando un buen ajuste. Sin embargo, es importante no depender exclusivamente de estas métricas, ya que un modelo con buen R² podría estar haciendo predicciones sistemáticamente incorrectas en ciertos rangos de datos. Por ello, siempre es recomendable visualizar los residuos (diferencias entre predicciones y valores reales) para identificar patrones no capturados por el modelo.

 - Entrenar el modelo con el conjunto de entrenamiento y calcular R²_train.
 - Evaluar el modelo en el conjunto de prueba y calcular R²_test.
 - Calcular ΔR² = R²_train − R²_test.
 - ΔR² pequeño (p. ej. < 0.05): ajuste consistente.
 - ΔR² moderado (≈ 0.05–0.1): revisar según contexto.
 - ΔR² grande (> 0.1–0.2): probable overfitting (R²_train mucho mayor que R²_test).
 -Revisar valores absolutos: R²_train alto y R²_test cercano a 0 o negativo confirma mala generalización.
 -Confirmar con validación cruzada (variabilidad de R²), métricas de error (RMSE/MAE_train vs RMSE/MAE_test), curvas de aprendizaje y gráficos predicho vs real.

# 9. ¿Qué es la validación cruzada (cross-validation) y qué ventaja ofrece frente a una sola división train/test?
La validacion cruzada en machine learning es una técnica esencial en Machine Learning para evaluar la capacidad predictiva de un modelo. Su objetivo principal es estimar con mayor precisión el rendimiento del modelo en datos no vistos, reduciendo el riesgo de sobreajuste (overfitting). En este artículo aprenderás qué es la validación cruzada, porque se usa, que técnicas, cuándo deberías aplicarla, sus ventajas, desventajas y un sencillo ejemplo.

¿Cuál es el objetivo de la validacion cruzada?
 El objetivo de la validación cruzada es verificar cómo se desempeña un modelo con diferentes particiones del conjunto de datos. Así se asegura que el modelo no solo funciona bien con los datos de entrenamiento, sino también con nuevos datos que nunca ha visto antes.

 ¿Por qué usamos la validacion cruzada y no solo los datos de entrenamiento? 
 Porque si solo usamos los datos de entrenamiento para evaluar el modelo, es como corregir un examen con las respuestas que ya te sabes.

 El modelo aprende de esos datos y, naturalmente, va a tener muy buen rendimiento, pero no sabremos si realmente puede generalizar a datos nuevos, es decir, si sabe resolver ejercicios que nunca ha visto.

# Desventajas:
- Mayor coste computacional, ya que entrena múltiples modelos.
- Puede ser lento en conjuntos de datos muy grandes.
- Puede requerir configuraciones adicionales para algunos algoritmos complejos.

# Bibliografía
- https://www.iablog.info/2025/07/analisis-de-regresion-en-machine.html
- https://estudyando.com/aprendizaje-supervisado-en-ia-regresion-y-clasificacion/
- https://www.ibm.com/es-es/think/topics/loss-function
- https://programmerclick.com/article/76181561522/
- https://www.ibm.com/es-es/think/topics/loss-function
- https://www.tutorialesprogramacionya.com/cienciadedatos/machinelearningscikitlearn/tema11.html
- https://www.probabilidadyestadistica.net/coeficiente-de-determinacion-r-cuadrado/
- https://www.geeksforgeeks.org/maths/mean-squared-error/
- https://es.statisticseasily.com/glosario/%C2%BFQu%C3%A9-significa-error-absoluto-mae%3F/
- https://es.dataconomy.com/2025/04/03/que-es-el-error-cuadrado-de-raiz-rmse/
- https://estudyando.com/analisis-de-residuos-proceso-y-ejemplos/
- https://estudyando.com/evaluacion-de-modelos-de-ia-overfitting-y-metricas/
- Comparacion r2 con test IA gpt5-mini
- 