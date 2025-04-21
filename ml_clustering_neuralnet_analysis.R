
# Partition Clustering – Source Code
library(NbClust)
library(ggplot2)
library(gridExtra)
library(factoextra)
library(cluster)
library(MASS)

library(readxl)
vehicles_df <- read_excel("vehicles.xlsx")
sum(is.na(vehicles_df))

View(vehicles_df)
summary(vehicles_df)

boxplot(vehicles_df[,2:19])

detect_outlier <- function(x) {
  Q1 <- quantile(x, probs = 0.25)
  Q3 <- quantile(x, probs = 0.75)
  IQR <- Q3 - Q1
  outliers <- x > (Q3 + 1.5 * IQR) | x < (Q1 - 1.5 * IQR)
  return(outliers)
}

remove_outlier <- function(dataframe) {
  for (i in 2:ncol(dataframe)) {
    if (is.numeric(dataframe[[i]])) {
      dataframe <- dataframe[!detect_outlier(dataframe[[i]]), ]
    }
  }
  message("Outliers have been removed")
  return(dataframe)
}

clean_data <- remove_outlier(vehicles_df)
numeric_col <- clean_data[,2:19]
normalized_data <- scale(numeric_col)
boxplot(clean_data[,2:19])

set.seed(1445)
num_clusters <- NbClust(normalized_data, distance="euclidean", min.nc = 2, max.nc =8, method="kmeans", index="all")
table(num_clusters$Best.nc)

set.seed(63)
elbowmethod_plot <- fviz_nbclust(normalized_data, kmeans, method="wss") +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(subtitle = "Elbow method")
print(elbowmethod_plot)

set.seed(123)
gap_stat <- fviz_nbclust(normalized_data, kmeans, nstart = 25, method = "gap_stat", nboot = 100, iter.max=50) +
  labs(subtitle = "Gap statistic method")
print(gap_stat)

set.seed(467)
silhoutte_plot <- fviz_nbclust(normalized_data, kmeans, method="silhouette") +
  labs(subtitle = "Silhouette method")
print(silhoutte_plot)

ApplykMeans <- function(normalized_data, k) {
  Km <- kmeans(normalized_data[,-length(normalized_data)], k)
  C <- Km$centers
  S <- Km$size
  WSS <- Km$withinss
  BSS <- Km$betweenss
  TSS <- Km$betweenss / Km$totss
  I <- Km$iter
  cluster_plot <- fviz_cluster(Km, data = normalized_data,
                               palette = c("#AE3FEA", "#00AFBB", "#E7B800", "#aff0e7"),
                               geom = "point", ellipse.type = "convex", ggtheme = theme_bw(),
                               silhouette_plot <- silhouette(Km$cluster, dist(normalized_data)),
                               print(fviz_silhouette(silhouette_plot)))
  return(list(Number_of_clusters = k,
              Number_of_points_for_each_cluster = S,
              Number_of_iterations = I,
              Centers = C,
              Withinss = WSS,
              Betweenss = BSS,
              Between_to_total_ratio = TSS,
              Visual_cluster_plot = cluster_plot))
}
ApplykMeans(normalized_data, 3)

# PCA and Neural Network portions are long, continue writing in next step

# PCA and Clustering on PCA
library(fpc)
get_pca_attr <- function(pca_data) {
  eigenvalue <- pca_data$sdev^2
  eigenvector <- pca_data$rotation
  cumulative_score <- cumsum(eigenvalue / sum(eigenvalue))
  print("Eigenvalues:")
  print(eigenvalue[cumulative_score < 0.92])
  print("Eigenvectors:")
  print(eigenvector[, cumulative_score < 0.92])
  print("Cumulative scores:")
  print(cumulative_score[cumulative_score < 0.92])
  return (cumulative_score)
}
pca_great_data <- prcomp(normalized_data, center = TRUE, scale = FALSE)
summary(pca_great_data)
cumulative_score <- get_pca_attr(pca_great_data)
pcadataframe <- pca_great_data$x[, cumulative_score < 0.92]

spotClusters <- function(data_frame) {
  set.seed(1445)
  num_clusters <- NbClust(normalized_data, distance="euclidean", min.nc = 2, max.nc =8, method="kmeans", index="all")
  table(num_clusters$Best.nc)

  set.seed(63)
  elbowmethod_plot <- fviz_nbclust(normalized_data, kmeans, method="wss") +
    geom_vline(xintercept = 3, linetype = 2) +
    labs(subtitle = "Elbow method")
  print(elbowmethod_plot)

  set.seed(123)
  gap_stat <- fviz_nbclust(normalized_data, kmeans, nstart = 25, method = "gap_stat", nboot = 100, iter.max=50) +
    labs(subtitle = "Gap statistic method")
  print(gap_stat)

  set.seed(467)
  silhoutte_plot <- fviz_nbclust(normalized_data, kmeans, method="silhouette") +
    labs(subtitle = "Silhouette method")
  print(silhoutte_plot)
}
spotClusters(pcadataframe)

# MLP
library(tidyverse)
library(neuralnet)
library(dplyr)
library(readxl)
uow_load <- read_excel("uow_consumption.xlsx")
colnames(uow_load) <- c("Date", "18", "19", "20")
time_delayed <- bind_cols(
  T7 = lag(uow_load$'20', 7),
  T4 = lag(uow_load$'20', 4),
  T3 = lag(uow_load$'20', 3),
  T2 = lag(uow_load$'20', 2),
  T1 = lag(uow_load$'20', 1),
  outputprediction = lag(uow_load$'20', 0)
)
time_delayed <- na.omit(time_delayed)

t_5 <- cbind(time_delayed$T1, time_delayed$T2, time_delayed$T3, time_delayed$T4, time_delayed$T7, time_delayed$outputprediction)
colnames(t_5) <- c("Input_1", "Input_2", "Input_3", "Input_4", "Input_5", "Output")

norma <- function(x) { (x - min(x)) / (max(x) - min(x)) }
normat_5 <- norma(t_5)
Train_t_5 <- normat_5[1:380,]
Test_t_5 <- normat_5[381:nrow(normat_5),]

T_5_NN1 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4 + Input_5, data = Train_t_5, hidden = 3, linear.output = TRUE)
T_5_predic_output1 <- predict(T_5_NN1, Test_t_5)
T_5_actual_output <- Test_t_5[, "Output"]

library(Metrics)
performance_metrics <- function(actu_output, predi_output) {
  return(list(RMSE = rmse(actu_output, predi_output),
              MAE = mae(actu_output, predi_output),
              MAPE = mape(actu_output, predi_output),
              SMAPE = smape(actu_output, predi_output)))
}
T_5_NN1_performance <- performance_metrics(T_5_actual_output, T_5_predic_output1)
print(T_5_NN1_performance)
