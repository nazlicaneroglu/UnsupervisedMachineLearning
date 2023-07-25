### file path etc
#rm(list=ls())
options(scipen=6, digits=4)
##packages anf libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, tidyverse, tinytex, rmarkdown, glmnet, matlib, MASS,pdist, PMA, softImpute, dplyr,plotrix, kernlab,ranger, randomForest, )
install.packages("FactoMineR")
install.packages("vcd")
install.packages("fpc")
install.packages("factoextra")
install.packages("fastDummies")
install.packages("GGally")
install.packages("cluster")
install.packages("NbClust")
library(GGally)
library(fastDummies)
library(matlib)
library(glmnet, quietly = TRUE)
library(caTools)
library("PMA")
library("softImpute")
library(FactoMineR)
library(vcd)
library(factoextra)
library(cluster)
library(NbClust)
library(readr)

##Data & seed
set.seed(8913)
data <- read_csv("~/Desktop/USML/Loan data for finak/loan_sanction_train.csv")
loan_data<-data[,c(2,3,4,5,6,7,8,9,10,11,12,13)]
View(loan_data)

#Check whether we have omitted variables
summary(loan_data)
sum(complete.cases(loan_data))
loan_data_final<-loan_data[complete.cases(loan_data), ]
str(loan_data_final)
sum(complete.cases(loan_data_final))

#Organising and scaling
loan_data_final$Gender <- as.factor(loan_data_final$Gender)  # Convert character column to factor
loan_data_final$Married <- as.factor(loan_data_final$Married)  # Convert character column to factor
loan_data_final$Dependents <- as.factor(loan_data_final$Dependents)  # Convert character column to factor
loan_data_final$Education <- as.factor(loan_data_final$Education)  # Convert character column to factor
loan_data_final$Self_Employed <- as.factor(loan_data_final$Self_Employed)  # Convert character column to factor
loan_data_final$Property_Area <- as.factor(loan_data_final$Property_Area)  # Convert character column to factor
loan_data_final$Loan_Status <- as.factor(loan_data_final$Loan_Status)  # Convert character column to factor

loan_data_final[, 6:10] <- scale(loan_data_final[, 6:10]) #Scale numeric variables

loan_data_final$Married<-tolower(loan_data_final$Married) #this is done to prevent an error in the FAMD graphs
loan_data_final$Married<- paste("Married_", loan_data_final$Married)

#PCA data only works with numeric. So we transform categorical variables into dummies
loan_pca_data <- dummy_cols(loan_data_final)
loan_pca_data<-loan_pca_data[,c(6,7,8,9,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29)]


#Factor Analysis for Mixed Data
loan_famd <- FAMD(loan_data_final, graph=TRUE)
summary(loan_famd)
fviz_screeplot(loan_famd,choice = "eigenvalue")
fviz_famd_var(loan_famd, repel = TRUE)
fviz_contrib(loan_famd, "var", axes = 1)
fviz_contrib(loan_famd, "var", axes = 2)

#graph of qualitative variables
fviz_famd_var(loan_famd, "quanti.var", repel = TRUE,
              col.var = "black")
fviz_famd_var(loan_famd, "quanti.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE)

#graph of qualitative variables
quali.var <- get_famd_var(loan_famd, "quali.var")
quali.var 

fviz_famd_var(loan_famd, "quali.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)

ind <- get_famd_ind(loan_famd)
ind

fviz_famd_ind(loan_famd, col.ind = "cos2", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE)

fviz_mfa_ind(loan_famd, 
             habillage = "Gender", # color by groups 
             palette = c("#00AFBB", "#E7B800", "#FC4E07"),
             addEllipses = TRUE, ellipse.type = "confidence", 
             repel = TRUE # Avoid text overlapping
) 

fviz_mfa_ind(loan_famd, 
             habillage = "Married", # color by groups 
             palette = c("#00AFBB", "#E7B800", "#FC4E07"),
             addEllipses = TRUE, ellipse.type = "confidence", 
             repel = TRUE # Avoid text overlapping
)


fviz_mfa_ind(loan_famd, 
             habillage = "Dependents", # color by groups 
             palette = c("#00AFBB", "#E7B800", "#FC4E07","green"),
             addEllipses = TRUE, ellipse.type = "confidence", 
             repel = TRUE # Avoid text overlapping
)

#PCA (here we use data we have organized for PCA)
loan_pca <- prcomp(loan_pca_data)
## Scree plot of component variances
plot(loan_pca, type = "l")

## Biplot showing the first two components and the variable loadings for these
biplot(loan_pca)

## Redo but know with columns normalized
loan_pca_scaled <- prcomp(loan_pca_data, scale = TRUE)
loan_pca_scaled

## Inspect the results as before
summary(loan_pca_scaled)
plot(loan_pca_scaled, type = "l")
biplot(loan_pca_scaled)
## Result: keep 4 or 5 variables: loan amount, married, dependents, income and maybe gender


####Clustering
ggpairs(loan_data_final,          # Data frame
        columns = c(6,7,8)) # Columns
##loan amount and applicant income are very correlated: should I keep only one of them?

ggparcoord(loan_pca_data, scale = "uniminmax") #Not very readable

# we cannot simply run NbClust on data where we have categorical variables. 
#First construct dissimilarity matrix
loan_data_final$Married <- as.factor(loan_data_final$Married)  # this was added to fix an error at the next stage

loan_cluster <- daisy(loan_data_final[ ,c(1,2,3,4,5,11,12)], metric = c("gower"))
class(loan_cluster) 

# The main input for the code below is dissimilarity (distance matrix)
# After dissimilarity matrix was calculated, the further steps will be the same for all data types
#------------ DIVISIVE CLUSTERING ------------#
divisive.clust <- diana(as.matrix(loan_cluster), 
                        diss = TRUE, keep.diss = TRUE)
plot(divisive.clust, main = "Divisive")

#------------ AGGLOMERATIVE CLUSTERING ------------#
# I am looking for the most balanced approach
# Complete linkages is the approach that best fits this demand
aggl.clust.c <- hclust(loan_cluster, method = "complete")
plot(aggl.clust.c,
     main = "Agglomerative, complete linkages")

# Cluster stats comes out as list while it is more convenient to look at it as a table
# This code below will produce a dataframe with observations in columns and variables in row
# Not quite tidy data, which will require a tweak for plotting, but I prefer this view as an output here as I find it more comprehensive 
library(fpc)
cstats.table <- function(dist, tree, k) {
  clust.assess <- c("cluster.number","n","within.cluster.ss","average.within","average.between",
                    "wb.ratio","dunn2","avg.silwidth")
  clust.size <- c("cluster.size")
  stats.names <- c()
  row.clust <- c()
  output.stats <- matrix(ncol = k, nrow = length(clust.assess))
  cluster.sizes <- matrix(ncol = k, nrow = k)
  for(i in c(1:k)){
    row.clust[i] <- paste("Cluster-", i, " size")
  }
  for(i in c(2:k)){
    stats.names[i] <- paste("Test", i-1)
    
    for(j in seq_along(clust.assess)){
      output.stats[j, i] <- unlist(cluster.stats(d = dist, clustering = cutree(tree, k = i))[clust.assess])[j]
      
    }
    
    for(d in 1:k) {
      cluster.sizes[d, i] <- unlist(cluster.stats(d = dist, clustering = cutree(tree, k = i))[clust.size])[d]
      dim(cluster.sizes[d, i]) <- c(length(cluster.sizes[i]), 1)
      cluster.sizes[d, i]
      
    }
  }
  output.stats.df <- data.frame(output.stats)
  cluster.sizes <- data.frame(cluster.sizes)
  cluster.sizes[is.na(cluster.sizes)] <- 0
  rows.all <- c(clust.assess, row.clust)
  # rownames(output.stats.df) <- clust.assess
  output <- rbind(output.stats.df, cluster.sizes)[ ,-1]
  colnames(output) <- stats.names[2:k]
  rownames(output) <- rows.all
  is.num <- sapply(output, is.numeric)
  output[is.num] <- lapply(output[is.num], round, 2)
  output
}
# I am capping the maximum amout of clusters by 7
# I want to choose a reasonable number, based on which I will be able to see basic differences between customer groups as a result
stats.df.divisive <- cstats.table(loan_cluster, divisive.clust, 7)
stats.df.divisive
#for aggloerative clustering
stats.df.aggl <-cstats.table(loan_cluster, aggl.clust.c, 7) 
stats.df.aggl

# --------- Choosing the number of clusters ---------#
# Using "Elbow" and "Silhouette" methods to identify the best number of clusters
# to better picture the trend, I will go for more than 7 clusters.
library(ggplot2)
# Elbow
# Divisive clustering
ggplot(data = data.frame(t(cstats.table(loan_cluster, divisive.clust, 15))), 
       aes(x=cluster.number, y=within.cluster.ss)) + 
  geom_point()+
  geom_line()+
  ggtitle("Divisive clustering") +
  labs(x = "Num.of clusters", y = "Within clusters sum of squares (SS)") +
  theme(plot.title = element_text(hjust = 0.5))
# Agglo clustering
ggplot(data = data.frame(t(cstats.table(loan_cluster, aggl.clust.c, 15))), 
       aes(x=cluster.number, y=within.cluster.ss)) + 
  geom_point()+
  geom_line()+
  ggtitle("Divisive clustering") +
  labs(x = "Num.of clusters", y = "Within clusters sum of squares (SS)") +
  theme(plot.title = element_text(hjust = 0.5))
# Silhouette
ggplot(data = data.frame(t(cstats.table(loan_cluster, divisive.clust, 15))), 
       aes(x=cluster.number, y=avg.silwidth)) + 
  geom_point()+
  geom_line()+
  ggtitle("Divisive clustering") +
  labs(x = "Num.of clusters", y = "Average silhouette width") +
  theme(plot.title = element_text(hjust = 0.5))
#plot
ggplot(data = data.frame(t(cstats.table(loan_cluster, aggl.clust.c, 15))), 
       aes(x=cluster.number, y=avg.silwidth)) + 
  geom_point()+
  geom_line()+
  ggtitle("Agglomerative clustering") +
  labs(x = "Num.of clusters", y = "Average silhouette width") +
  theme(plot.title = element_text(hjust = 0.5))
#dendrogram
library("ggplot2")
library("reshape2")
library("purrr")
library("dplyr")
# let's start with a dendrogram
library("dendextend")
dendro <- as.dendrogram(aggl.clust.c)
dendro.col <- dendro %>%
  set("branches_k_color", k = 7, value =   c("darkslategray", "darkslategray4", "darkslategray3", "gold3", "darkcyan", "cyan3", "gold3")) %>%
  set("branches_lwd", 0.6) %>%
  set("labels_colors", 
      value = c("darkslategray")) %>% 
  set("labels_cex", 0.5)
ggd1 <- as.ggdend(dendro.col)
ggplot(ggd1, theme = theme_minimal()) +
  labs(x = "Num. observations", y = "Height", title = "Dendrogram, k = 7")
# Radial plot looks less cluttered (and cooler)
ggplot(ggd1, labels = T) + 
  scale_y_reverse(expand = c(0.2, 0)) +
  coord_polar(theta="x")

###Do the same clustering on the reduced data
reduced_data<-loan_data_final[,c(1,2,3,6,8)]

reduced_cluster <- daisy(reduced_data[ ,c(1,2,3,4,5)], metric = c("gower"))
class(reduced_cluster) 
#same divisive and agglo. clusterings
divisive.clust2 <- diana(as.matrix(reduced_cluster), 
                        diss = TRUE, keep.diss = TRUE)
plot(divisive.clust2, main = "Divisive")

aggl.clust.c2 <- hclust(reduced_cluster, method = "complete")
plot(aggl.clust.c,
     main = "Agglomerative, complete linkages")
# I am capping the maximum amout of clusters by 7
# I want to choose a reasonable number, based on which I will be able to see basic differences between customer groups as a result
stats.df.divisive2 <- cstats.table(reduced_cluster, divisive.clust2, 7)
stats.df.divisive
#for aggloerative clustering
stats.df.aggl2 <-cstats.table(reduced_cluster, aggl.clust.c2, 7) 
stats.df.aggl

# Elbow
# Divisive clustering
ggplot(data = data.frame(t(cstats.table(reduced_cluster, divisive.clust2, 15))), 
       aes(x=cluster.number, y=within.cluster.ss)) + 
  geom_point()+
  geom_line()+
  ggtitle("Divisive clustering") +
  labs(x = "Num.of clusters", y = "Within clusters sum of squares (SS)") +
  theme(plot.title = element_text(hjust = 0.5))
# Agglo clustering
ggplot(data = data.frame(t(cstats.table(reduced_cluster, aggl.clust.c2, 15))), 
       aes(x=cluster.number, y=within.cluster.ss)) + 
  geom_point()+
  geom_line()+
  ggtitle("Divisive clustering") +
  labs(x = "Num.of clusters", y = "Within clusters sum of squares (SS)") +
  theme(plot.title = element_text(hjust = 0.5))
# Silhouette
ggplot(data = data.frame(t(cstats.table(reduced_cluster, divisive.clust2, 15))), 
       aes(x=cluster.number, y=avg.silwidth)) + 
  geom_point()+
  geom_line()+
  ggtitle("Divisive clustering") +
  labs(x = "Num.of clusters", y = "Average silhouette width") +
  theme(plot.title = element_text(hjust = 0.5))
#plot
ggplot(data = data.frame(t(cstats.table(reduced_cluster, aggl.clust.c2, 15))), 
       aes(x=cluster.number, y=avg.silwidth)) + 
  geom_point()+
  geom_line()+
  ggtitle("Agglomerative clustering") +
  labs(x = "Num.of clusters", y = "Average silhouette width") +
  theme(plot.title = element_text(hjust = 0.5))
