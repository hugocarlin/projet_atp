#################### PROJET ATP 





########## Préparation du code

### Effacer les objets de l'environnement
rm(list = ls())

### Bibliothèques
library('purrr')
library('dplyr')
library('stringr')
library('nnet')
library('data.table')
library('tidyverse')
library('magrittr')
library('caret')
library('randomForest')
library('ggplot2')





########## Préparation des données

### Importation des données
# Liste des noms des fichiers à importer
lst_names <- paste0("atp_matches_", 1968:2019, ".csv")
# Chemin des données
path <- "tennis_atp/"
# On importe itérativement tous les fichiers
lst_dt <- purrr::map(.x = lst_names,
           .f = function(fichier) {
             fread(file = paste0(path, fichier),
                   header = TRUE, # La premiere ligne contient les noms des variables
                   encoding = "UTF-8", # Encodage du fichier .csv
                   data.table = TRUE,  # La fonciton retourne un data.table
                   nThread = 2)})
# On concatène tous les data.tables en un seul
rbindlist(lst_dt) -> atp


### Selection des variables et observations sur lesquelles on veut travailler
# Exclusion des matches avant US OPEN 1970 (règle du jeu décisif)
atp <- atp[-c(1:5200),]
# Suppression des variables de seed et entry
atp <- atp[,-c(9,10,17,18)]
# Suppression des variables de nationalité
atp <- atp[,-c(12,18)]
# Suppression des variables de statistique sur les matchs
atp <- atp[,-c(22:39)]
# Suppression des variables de point au classement
atp <- atp[,-c(23,25)]
# Suppression des variables d'id
atp <- atp[,-c(1,2,6:9,13,14)]


### Gestion des données manquantes et aberrantes
# Suppression les observations contenant des NA
atp <- atp %>% na.omit()		
# Suppression les observations avec le champ surface vide
atp <- subset(atp[!atp$surface=="",])
# Suppression des matchs avec abandon
atp <-atp %>% filter(str_detect(atp$score,"RET")==FALSE)
# Suppression des valeurs abérantes (score et minutes ne correspondent pas)
atp <- atp[-c(70265,73691,8756,13819,11693,8757,8100,37779,11694,8758,6558,33440,8094,9604,11692,3480,11032,19688,4668,10820,10821,13221,8098,13822,11967,34926,8108,6554,6553,6551,41096,6549,10409,23946,23947,9607,11031,6555,6558,9610,13820,8755,23944,11029,11030,23945,6557,34948,9606,8099,10824,48564,3786,252,27194,8102,9605,6550,11690,41010,28834,36925,51550,10822,33451,13111,20585,17308,23985,8105,8097,21944,6552,8104,8095,2292,17630,23168,43160),]
atp <- atp[-c(8084,6544,13770,34559),]
atp <- atp[-c(68279,71608),]
atp <- atp[-c(3636),]
# Suppression de la variable score
atp <- atp[,-c(10)]
# Nombre d'observations final
nrow(atp) # 77 545 observations une fois toutes ces manips effectuées.


### Recodage de la variable minutes (quanti -> quali)
summary(atp$minutes)
x <- c(atp$minutes)
# Découpage en 4 catégories
minutes_recod <- cut(x, breaks = c(28, 75, 96, 125, Inf),
                     labels = c("Courte durée [28min-1h15]", "Durée moyenne [1h15-1h36]", "Longue durée [1h36-2h05]", "Très longue durée [2h04-12h00]"))
# Ajout de la nouvelle variable de minutes
atp <- cbind(atp,minutes_recod)
# Suppression de l'ancienne variable minutes
atp <- atp[,-c(12)]





########## Préparation des données test et train

### Déoupage de la base atp en 70/30
n <- nrow(atp)
set.seed(200)
tirage <- sample(1:n, 0.70 * n)
atp_train <- atp[tirage, ]
atp_test <- atp[-tirage, ]
atp_test <- as.data.frame(atp_test)
atp_train <- as.data.frame(atp_train)
dim(atp_train)
dim(atp_test)
# 22 variables
# 54 281 observations dans notre base train
# 23 264 observations dans notre base test

### Vérification de la répartition
effectifs<-table(atp$minutes_recod)
prop.table(effectifs)*100





########## Classification multi-groupe en grande dimension
########## Régression logistique multinomial pénalisée

# Variable réponse : minutes_recod
# Variables explicatives (numériques) : à determiner

###### Random Forrest

### Ajustement d'une forêt aléatoire pour la prédiction du temps des matchs
set.seed(654)
randomForest(formula = minutes_recod ~ ., # Modele visé
             data = atp_train, # DOnnées utilisées
             ntree = 1000, # Nombre d'arbres
             mtry = 2, # Nombre de variables candidates à chaque coupure
             importance = TRUE # Stocker l'importance des variables dans l'objet résultat
) -> rf_results


### Résultat de la méthode RF (objet de classe randomForest)
str(rf_results)
# Parmi celles-ci (pour un arbre de régression) :
# * predicted : vecteur des prédictions de la variable cible par la forêt
# * mse : le vecteur des erreurs quadratiques des 500 arbres constituant la forêt
# * rsq : le vecteur des pseudo-r-squarred= 1-mse/Var(y) => plus c'estproche de 1, meilleur est la modélisation
# * oob.times : vectyeur indiquant pour chaque observation le nombre d'arbres pour lesquels cette observation est "out of bag"
# * importance : matrice : une ligne par variable explicative, les colonnes indiquent la contribution moyenne (sur les 500 arbres) de ces variables à la décroissance des MSE des abres. 
# * forest : le détail de la composition des 500 arbres de la forêt, sous forme d'une liste


### Représentation de la fonction qui à un nombre d'arbres associe la proportion de MSE 
plot(rf_results, ylim= c(0, .4))


### Visualisation de l'importance des variables
varImpPlot(rf_results)


### Nombre total de fois où chaque variable a été utilisé pour une coupure dans l'un des  arbres.
varUsed_rf <-varUsed(rf_results, by.tree = FALSE, count = TRUE)
# Pour ajouter les noms des variables aux composantes du résultat
names(varUsed_rf) <- attr(rf_results$importance, "dimnames")[[1]]
varUsed_rf
# On les classe par nombre décroissant d'utilisations
sort(varUsed_rf, decreasing = TRUE)


### Prédiction sur la base test
prediction <- predict(object = rf_results, newdata = atp_test)
# Nombre de predictions pour chaque composante
table(prediction)
table(atp_test$minutes_recod)
table(atp_test$minutes_recod,prediction)
# Résultats
results <- data.frame(table(prediction == atp_test$minutes_recod))
colnames(results) <- c("Valid","Nb")
# Pourcentage de bonnes prédictions
100*results[2,2]/23264


### Matrice de confusion
conf <- confusionMatrix(data = prediction, reference = atp_test$minutes_recod)
conf$table
conf
# Représentation graphique
table <- data.frame(conf$table)
plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "aquamarine1", bad = "black")) +
  theme_bw() +
  xlim(rev(levels(table$Reference))) +
  scale_y_discrete(labels=c("Courte durée", "Durée moyenne", "Longue durée","Très longue durée")) +
  scale_x_discrete(labels=c("Courte durée", "Durée moyenne", "Longue durée","Très longue durée")) +
  labs(fill = "Prediction")

