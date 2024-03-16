#Librerías Necesarias
library(rgl)              #Visualización de Gráficos 3D
library(caret)            #Construcción de Modelos Predictivos
library(corrplot)         #Visualización de Matrices de Correlación
library(xgboost)          #Boosting
library(gbm)              #Entrenar modelos de Boosting
library(randomForest)     #Random Forest
library(neuralnet)        #Redes Neuronales
library(NeuralNetTools)   

#Se cargan los datos
data <- read.csv("datos_churn.csv")

#Se cambian los nombres de las columnas
colnames(data)=c('Índice', 'Id', 'Apellido', 'PuntuacionCredito', 'País', 'Género', 'Edad', 'Antigüedad',
                 'Balance', 'NumProductos', 'TarjetaCredito', 'MiembroActivo', 'Salario', 'target')

#Se eliminan los individuos repetidos o que tengan datos omitidos
data <- na.omit(data)  
data <- unique(data)

#Se eliminan las tres primeras variables ya que no sirven para este análisis
data <- data[,-c(1:3)]

#Se escalan las variables numérica
data_classes <- sapply(data, class)
data[,data_classes=="numeric"] <- scale(data[,data_classes=="numeric"])

#Se factorizan las variables "País" y "Género"
data$País <- as.numeric(factor(data$País))
data$Género <- as.numeric(factor(data$Género))


#Se eliminan 4000 individuos cuya variable objetivo es 0 para ajustar las proporciones
#de manera que los modelos puedan aprender patrones y predecir mejor los casos
#positivos. Pasamos de una proporción 80% - 20% a una 66% - 33%.
data1 = data[data$target==1,]
data2 = data[data$target==0,]

set.seed(100)
index <- sample.int(nrow(data2), 4000)
data2 <- data2[-index,]
data3 <- rbind(data1, data2)

set.seed(100)
data_index <- sample(nrow(data3))
data <- data3[data_index,]


#Matrices de Correlación
cor_matrix <- cor(data)
corrplot(cor_matrix, method = "circle")


#División del dataset en entrenamiento(60%), validación(20%) y prueba(20%)
set.seed(100)
train_index <- createDataPartition(data$target, p = 0.6, list = FALSE)
validation_index <- createDataPartition(data$target[-train_index], p = 0.5, list = FALSE)

#Conjuntos de entrenamiento, validación y prueba.
train_data <- data[train_index, ]
validation_data <- data[-train_index, ][validation_index, ]
test_data <- data[-train_index, ][-validation_index, ]

#Variables dependientes
train_X <- train_data[,-11]
validation_X <- validation_data[,-11]
test_X <- test_data[,-11]

#Variable objetivo sin factorizar
train_Y <- train_data[,11]
validation_Y <- validation_data[,11]
test_Y <- test_data[,11]

#Variable objetivo factorizada
train_Y_f <- as.factor(train_Y)
levels(train_Y_f) <- c("No evento", "Evento")

test_Y_f <- as.factor(test_Y)
levels(test_Y_f) <- c("No evento", "Evento")

validation_Y_f <- as.factor(validation_Y)
levels(validation_Y_f) <- c("No evento", "Evento")


#Boosting
#Creación de las variables necesarias para el algoritmo
data_train_mat <- as.matrix(train_X)
data_train_Dmat <- xgb.DMatrix(data_train_mat, label = train_Y)

data_validation_mat <- as.matrix(validation_X)
data_validation_Dmat <- xgb.DMatrix(data_validation_mat, label = validation_Y)

data_test_mat <- as.matrix(test_X)
data_test_Dmat <- xgb.DMatrix(data_test_mat, label = test_Y)

#Vectores para guardar la exhaustividad y la precisión
recall_validation <- matrix(data = 1:100, nrow = 5, ncol = 100)
accuracy_validation <- matrix(data = 1:100, nrow = 5, ncol = 100)

#Entrenamiento iterativo de los 500 modelos
recall_max <- 0
accuracy_max <- 0
for (i in 1:5) {
  for (s in 1:100) {
    xgb <- xgb.train(data = data_train_Dmat, 
                     objective = "binary:logistic", 
                     nrounds = s,
                     max.depth = i,
                     eta = 0.3, 
                     nthread = 2)
    
    #Se predicen los resultados en el conjunto de validación.
    predicciones_validation <- predict(xgb, data_validation_Dmat)
    
    predicciones_validation <- as.factor(predicciones_validation > 0.432)
    levels(predicciones_validation) <- c("No evento", "Evento") 
    
    matriz_validation <- confusionMatrix(predicciones_validation, 
                                         validation_Y_f, positive = "Evento")
    
    #Se guarda el modelo si tiene mayor "Recall" que el modelo anterior.
    if (matriz_validation$byClass["Recall"] > recall_max){
      modelo_xgb1 <-  xgb
      accuracy_xgb1 <- matriz_validation$overall["Accuracy"]
      recall_max <- matriz_validation$byClass["Recall"]
    }
    
    
    #Se guarda el modelo si tiene mayor "Accuracy" que el modelo anterior.
    if (matriz_validation$overall["Accuracy"] > accuracy_max){
      modelo_xgb2 <-  xgb
      accuracy_max <- matriz_validation$overall["Accuracy"]
    }
  }
}

#Predicciones del modelo con mayor precisión
pred_val_xgb2 <- predict(modelo_xgb2, data_validation_Dmat)    

#Se decrementa el umbral de clasificación para el segundo modelo hasta
#que su recall iguale al del primer modelo, el que tenía el recall más alto
umbral <- 0.432
recall_xgb2 <- 0
while (recall_max > recall_xgb2) {
  
  pred_val_xgb2_f <- as.factor(pred_val_xgb2 > umbral)
  levels(pred_val_xgb2_f) <- c("No evento", "Evento")
  
  matriz_validation_xgb2 <- confusionMatrix(pred_val_xgb2_f, 
                                            validation_Y_f, positive = "Evento")
  
  recall_xgb2 <- matriz_validation_xgb2$byClass["Recall"]
  accuracy_xgb2 <- matriz_validation_xgb2$overall["Accuracy"]
  umbral <- umbral-0.01
}


#Se selecciona el modelo con mayor accuracy ya que el recall ahora es el mismo   
if(accuracy_xgb1 > accuracy_xgb2 || umbral == 0.01){
  modelo_xgb <- modelo_xgb1
  umbral<-0.432
  
} else {
  modelo_xgb <- modelo_xgb2
}


#Predicciones del modelo en el conjunto de prueba
predicciones_xgb <- predict(modelo_xgb, data_test_Dmat)

predicciones_xgb_f = as.factor(predicciones_xgb > umbral)
levels(predicciones_xgb_f)=c("No evento", "Evento")

matriz_xgb <- confusionMatrix(predicciones_xgb_f, test_Y_f, positive = "Evento")

resultados_xgb <- data.frame(Modelo = "Boosting",
                             Accuracy = round(matriz_xgb$overall["Accuracy"],4),
                             Recall = round(matriz_xgb$byClass["Recall"],4),
                             Specificity = round(matriz_xgb$byClass["Specificity"],4),
                             Precision = round(matriz_xgb$byClass["Precision"],4),
                             F1 = round(matriz_xgb$byClass["F1"],4),
                             row.names = NULL)
resultados_xgb



## Random Forest
#Entrenamiento iterativo de los 20 modelos
recall_max <- 0
accuracy_max <- 0
for(i in seq(100,1000,50)){
  
  set.seed(100)
  RF <- randomForest(target ~., 
                     data = train_data, 
                     ntree=i, 
                     importance=TRUE)
  
  predicciones_RF <- predict(RF,validation_data)
  
  predicciones_RF_f = as.factor(predicciones_RF > 0.185)
  levels(predicciones_RF_f)=c("No evento", "Evento")
  
  matriz_RF <- confusionMatrix(predicciones_RF_f, validation_Y_f, positive = "Evento")
  
  #Se guarda el modelo si tiene mayor "Recall" que el modelo anterior.
  if (matriz_RF$byClass["Recall"] > recall_max){
    modelo_RF1 <-  RF
    accuracy_rf1 <- matriz_RF$overall["Accuracy"]
    recall_max <- matriz_RF$byClass["Recall"]
  }
  
  #Se guarda el modelo si tiene mayor "Accuracy" que el modelo anterior.
  if (matriz_RF$overall["Accuracy"] > accuracy_max){
    modelo_RF2 <-  RF
    accuracy_max <- matriz_RF$overall["Accuracy"]
  }
}


#Predicciones del modelo con mayor precisión
pred_val_rf2 <- predict(modelo_xgb2, data_validation_Dmat)    

#Se decrementa el umbral de clasificación para el segundo modelo hasta
#que su recall iguale al del primer modelo, el que tenía el recall más alto
umbral <- 0.185
recall_rf <- 0
while (recall_max > recall_rf) {
  
  pred_val_rf2_f = as.factor(pred_val_rf2 > umbral)
  levels(pred_val_rf2_f)=c("No evento", "Evento") 
  
  matriz_validation_rf2 <- confusionMatrix(pred_val_rf2_f, 
                                           validation_Y_f, 
                                           positive = "Evento")
  
  recall_rf <- matriz_validation_rf2$byClass["Recall"]
  accuracy_rf2 <- matriz_validation_rf2$overall["Accuracy"]
  umbral <- umbral-0.01
}

#Se selecciona el modelo con mayor accuracy ya que el recall ahora es el mismo 
if(accuracy_rf1 > accuracy_rf2 || umbral == 0.01){
  modelo_RF <- modelo_RF1
  umbral <- 0.185
} else {
  modelo_RF <- modelo_RF2
}

#Predicciones del modelo en el conjunto de prueba
predicciones_RF <- predict(modelo_RF,test_data)

predicciones_RF_f = as.factor(predicciones_RF > umbral)
levels(predicciones_RF_f)=c("No evento", "Evento")

matriz_RF <- confusionMatrix(predicciones_RF_f, test_Y_f,
                             positive = "Evento")

resultados_RF <- data.frame(Modelo = "Random Forest",
                            Accuracy = round(matriz_RF$overall["Accuracy"],4),
                            Recall = round(matriz_RF$byClass["Recall"],4),
                            Specificity = round(matriz_RF$byClass["Specificity"],4),
                            Precision = round(matriz_RF$byClass["Precision"],4),
                            F1 = round(matriz_RF$byClass["F1"],4),
                            row.names = NULL)
resultados_RF


## Regresión Logística

set.seed(100)
modeloRegLog <- glm(target~., data=train_data, family=binomial)

#Se predice el conjunto de validación y en función de los resultados se modifica
#el umbral de clasificación hasta tener un valor de al menos 0.9 de recall.
predicciones_RegLog <- predict(modeloRegLog, validation_data, type="response")
predicciones_RegLog_f <- as.factor(predicciones_RegLog>0.17)
levels(predicciones_RegLog_f) <- c("No evento", "Evento")

matriz_RegLog <- confusionMatrix(predicciones_RegLog_f, 
                                 validation_Y_f, 
                                 positive = "Evento")

resultados_RegLog <- data.frame(Modelo = "Reg Logistica",
                                Accuracy = round(matriz_RegLog$overall["Accuracy"],4),
                                Recall = round(recall_RegLog <- matriz_RegLog$byClass["Recall"],4),
                                Specificity = round(matriz_RegLog$byClass["Specificity"],4),
                                Precision = round(matriz_RegLog$byClass["Precision"],4),
                                F1 = round(matriz_RegLog$byClass["F1"],4),
                                row.names = NULL)

predicciones_RegLog <- predict(modeloRegLog, test_data, type="response")
predicciones_RegLog_f <- as.factor(predicciones_RegLog>0.17)
levels(predicciones_RegLog_f) <- c("No evento", "Evento")


matriz_RegLog <- confusionMatrix(predicciones_RegLog_f, 
                                 test_Y_f, 
                                 positive = "Evento")

resultadosRegLog <- data.frame(Modelo = "Reg Logistica",
                               Accuracy = round(matriz_RegLog$overall["Accuracy"],4),
                               Recall = round(recall_RegLog <- matriz_RegLog$byClass["Recall"],4),
                               Specificity = round(matriz_RegLog$byClass["Specificity"],4),
                               Precision = round(matriz_RegLog$byClass["Precision"],4),
                               F1 = round(matriz_RegLog$byClass["F1"],4),
                               row.names = NULL)
resultados_RegLog


## Redes Neuronales
data$target <- factor(data$target)
levels(data$target) <- c("No evento", "Evento")

set.seed(100)
nn <- neuralnet(target ~., data=train_data, hidden=3,linear.output = FALSE)

plotnet(nn)

nn_predicciones <- predict(nn, validation_data)
nn_predicciones_f <- factor(nn_predicciones > 0.15)
levels(nn_predicciones_f) <- c("No evento", "Evento")


matriz_nn <- confusionMatrix(nn_predicciones_f, validation_Y_f, 
                             positive = "Evento")

resultados_nn <- data.frame(Modelo = "Red Neuronal",
                            Accuracy = round(matriz_nn$overall["Accuracy"],4),
                            Recall = round(matriz_nn$byClass["Recall"],4),
                            Specificity = round(matriz_nn$byClass["Specificity"],4),
                            Precision = round(matriz_nn$byClass["Precision"],4),
                            F1 = round(matriz_nn$byClass["F1"],4),
                            row.names = NULL)

nn_predicciones <- predict(nn, test_data)
nn_predicciones_f <- factor(nn_predicciones > 0.15)
levels(nn_predicciones_f) <- c("No evento", "Evento")


matriz_nn <- confusionMatrix(nn_predicciones_f, test_Y_f, 
                             positive = "Evento")

resultadosnn <- data.frame(Modelo = "Red Neuronal",
                           Accuracy = round(matriz_nn$overall["Accuracy"],4),
                           Recall = round(matriz_nn$byClass["Recall"],4),
                           Specificity = round(matriz_nn$byClass["Specificity"],4),
                           Precision = round(matriz_nn$byClass["Precision"],4),
                           F1 = round(matriz_nn$byClass["F1"],4),
                           row.names = NULL)
resultados_nn


#Se sacan las métricas de rendimiento de los cuatros modelos
resultados_modelos <- as.data.frame(matrix(data = 1:6, nrow = 4, ncol = 6))
colnames(resultados_modelos) <- c("Modelo", "Accuracy", "Recall", "Specificity", "Precision","F1")

resultados_modelos[1,] <- resultados_xgb
resultados_modelos[2,] <- resultados_RF
resultados_modelos[3,] <- resultados_RegLog
resultados_modelos[4,] <- resultados_nn

resultados_modelos
