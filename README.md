# Prediccion_de_Churn
## 📝 Descripción
El **análisis de *churn***, también conocido como tasa de cancelación o de abandono, es un aspecto en auge que desempeña un papel clave en la gestión de empresas que ofrecen servicios de suscripción. El análisis de churn es un proceso que tiene como objetivo comprender las razones y factores que llevan a los clientes a rescindir su contrato con la empresa. A partir de las conclusiones obtenidas, se desarrollan medidas proactivas como campañas de retención de clientes, dirigidas a aquellos en riesgo de abandonar, con el objetivo de fidelizarlos y evitar su pérdida.
<br>
<br>
En este estudio, se analiza la tasa de abandono de una entidad financiera a través del uso de datos de cancelación de clientes con el propósito de predecir cuales son aquellos individuos que se encuentran en riesgo de abandonar. Con ese objetivo, se evalúan y comparan cuatro modelos predictivos entrenados con cuatro algoritmos distintos: ***Extreme Gradient Boosting***, ***Random Forest***, ***Regresión Logística*** y ***Redes Neuronales***. El estudio concluye que, para este problema en concreto, el mejor modelo es el entrenado mediante *Boosting*.
<br>
<br>

## 📅 Conjunto de Datos
El [conjunto de datos](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) ha sido descargado de la plataforma Kaggle.

Las variables o atributos que componen el conjunto de datos son los siguientes:<br>
&nbsp;&nbsp;&nbsp; • **RowNumber**: Representa el índice de los individuos.<br>
&nbsp;&nbsp;&nbsp; • **CustomerId**: Representa el id del cliente en el banco.<br>
&nbsp;&nbsp;&nbsp; • **Surname**: Representa el apellido del cliente.<br>
&nbsp;&nbsp;&nbsp; • **CreditScore**: Representa una medida numérica que califica la solvencia crediticia y el riesgo del cliente.<br>
&nbsp;&nbsp;&nbsp; • **Geography**: Representa el país de la sucursal a la que esta asociada el cliente.<br>
&nbsp;&nbsp;&nbsp; • **Gender**: Representa el sexo del cliente.<br>
&nbsp;&nbsp;&nbsp; • **Age**: Representa la edad del cliente.<br>
&nbsp;&nbsp;&nbsp; • **Tenure**: Representa en años la antigüedad o duración del vínculo entre el cliente y el banco.<br>
&nbsp;&nbsp;&nbsp; • **Balance**: Representa en el valor monetario total de los activos y pasivos de un cliente en el banco.<br>
&nbsp;&nbsp;&nbsp; • **NumOfProducts**: Representa el número de productos del banco utilizados por el cliente.<br>
&nbsp;&nbsp;&nbsp; • **HasCrCard**: Representa la tenencia de trajeta de crédito.<br>
&nbsp;&nbsp;&nbsp; • **IsActiveMember**: Representa si el cliente es un miembro activo del banco.<br>
&nbsp;&nbsp;&nbsp; • **EstimatedSalary**: Representa el salario anual estimado del cliente.<br>
&nbsp;&nbsp;&nbsp; • **Exited**: Representa la variable objetivo. Tendrá un valor de **0**, si el cliente sigue en el banco, o de **1**, si el cliente rescindió su contrato.
<br>
<br>

## ⚙️ Proceso
Este análisis está enfocado a obtener un modelo que prioriza el *recall* frente a la precisión. El *recall* o exhaustividad nos indica la proporción de positivos capturados, por ello, en problemas como el que se está tratando en este estudio, interesa sacrificar precisión del modelo en beneficio de tener un mejor valor de exhaustividad para detectar que individuos tienen un mayor riesgo de abandonar la entidad financiera.
<br>
<br>
Se debe sacrificar precisión ya que al tratar de aumentar el *recall* se captura una proporción mayor de casos positivos, lo que puede resultar en un mayor número de falsos positivos y, por tanto, una reducción en la precisión. En este análisis se presume que el perjuicio para la entidad financiera de capturar falsos positivos no es tan alto como el beneficio que se obtiene al tener un valor de exhaustividad alto.
<br>
<br>
Para ello, se divide el conjunto de datos en **entrenamiento** (60%), **validación** (20%) y **prueba** (20%). El subconjunto de entrenamiento, como su nombre indica, se utilizará para entrenar los modelos, el subconjunto de validación se usará para encontrar los mejores hiperparámetros para cada modelo y evitar el *overfitting*, y por último, el subconjunto de prueba se utilizará para obtener las métricas finales de rendimiento de los modelos. El objetivo es obtener en todos los modelos un valor para la exhaustividad de al menos 0.9 en el conjunto de validación, es decir, que los modelos capturen al menos un 90% de los positivos verdaderos.
<br>
<br>
## 📈 Resultados
### ✦ Boosting
Para la creación de este modelo se utiliza el algoritmo de ***XGBoost***. En el código se entrenan iterativamente quinientos modelos distintos utilizando el conjunto de entrenamiento con el objetivo de encontrar el mejor posible respecto a la *exhaustividad* y el mejor posible respecto a la *precisión*. Para ello, en cada iteración se modifican los hiperparámetros probando quinientas combinaciones diferntes entre ***nrounds*** y ***max.depth*** y se guardan los modelos con mejor rendimiento al tratar de predecir en el conjunto de validación. Los resultados para este modelo al predecir en el conjunto de prueba son:
<br>
<br>
&nbsp;**Modelo**&nbsp;&nbsp;**Accuracy**&nbsp;&nbsp;&nbsp;**Recall**&nbsp;&nbsp;**Specificity**&nbsp;&nbsp;**Precision**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**F1**&nbsp;&nbsp;&nbsp;&nbsp;<br>
Boosting&nbsp;&nbsp;&nbsp;&nbsp;0.6217&nbsp;&nbsp;&nbsp;&nbsp;0.9109&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4749&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4682&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.6185&nbsp;
<br>
<br>

### ✦ Random Forest
Para el modelo que utiliza ***Random Forest*** se procede de la misma manera que el caso anterior. En este caso, se entrenan iterativamente veinte modelos distintos modificando el hiperparámetro ***ntree***. Las métricas de rendimiento para este modelo son:
<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Modelo**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Accuracy**&nbsp;&nbsp;&nbsp;**Recall**&nbsp;&nbsp;**Specificity**&nbsp;&nbsp;**Precision**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**F1**&nbsp;&nbsp;&nbsp;&nbsp;<br>
Random Forest &nbsp;&nbsp;&nbsp;&nbsp;0.5883&nbsp;&nbsp;&nbsp;&nbsp;0.9381&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4108&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4469&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.6054&nbsp;
<br>
<br>

### ✦ Regresión Logística
En este caso, el modelo de **regresión logística**, solo es posible construir un único modelo y sus resultados son:
<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Modelo**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Accuracy**&nbsp;&nbsp;&nbsp;**Recall**&nbsp;&nbsp;**Specificity**&nbsp;&nbsp;**Precision**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**F1**&nbsp;&nbsp;&nbsp;&nbsp;<br>
Reg Logistica &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.5458&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.9132&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.3601&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4191&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.5746&nbsp;
<br>
<br>

### ✦ Redes Neuronales
La **red neuronal** entrenada consta de diez neuronas en la capa de entrada, tres neuronas en la capa oculta, y una neurona en la capa de salida. Las métricas de rendimiento para este modelo son:
<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Modelo**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Accuracy**&nbsp;&nbsp;&nbsp;**Recall**&nbsp;&nbsp;**Specificity**&nbsp;&nbsp;**Precision**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**F1**&nbsp;&nbsp;&nbsp;&nbsp;<br>
Red Neuronal &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.5625&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.9156&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.3839&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4291&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.5843&nbsp;
<br>
<br>

## 💬 Conclusiones

El modelo con mayor **exhaustividad** es el que utiliza el algoritmo *Random Forest*, captura correctamente un 93,81% de los casos positivos, mientras que el modelo con mayor **precisión** es el que hace uso de *XGBoost*, clasifica bien el 62,17% de los datos. Teniendo en cuenta que ambos cumplen el objetivo establecido de tener al menos un 90% de exhaustividad es preferible utilizar el modelo de ***Boosting*** ya que tiene mayor precisión.
<br>
<br>

## ℹ️ +Info
Para ver el proyecto completo donde se detalla más información y se explica de manera más amplia las partes más importantes del código ver el documento llamado *Informe.pdf*.

