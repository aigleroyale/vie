# ANALYSE CREDIT-RISQUE-MODELISATION AVEC R

* Partie1 : MACHINE LEARNING POUR LA MODELISATION DU RISQUE DE CREDIT DANS R (CREDIT SCORING)

`Il s'agit d'un projet pour construire et évaluer un modèle de Machine Learning afin de prédire si un demandeur de crédit sera en défaut de paiement ou non`

* Partie2 : INTEGRATIONT DANS SHINYDASHBORD DU MODELE DE MACHINE LEARNING POUR LA PREDICTION DU RISQUE DE CREDIT BANCAIRE

`Construstion d'un modèle de forêt aléatoir pour modéliser la probabilité de défaut de paiement et prédire si un demandeur de prêt pourra ou non rembourser son crédit.`
`Déploiement du modèle dans une application web type tableau de bord qui affiche la probabilité de défaut de paiement d'un nouveau demandeur de crédit ainsi que son statut (Eligible ou non Eligible au crédit.)`

## Partie1 : MACHINE LEARNING POUR LA MODELISATION DU RISQUE DE CREDIT DANS R (CREDIT SCORING)

### **Compréhension de la problématique business**

```{figure} ./images/cred.jpg
---
scale: 15%
```

:::{Note}


Lorsqu'une banque prête de l'argent à une personne, elle prend le risque que cette dernière ne rembourse pas cet argent dans le délai convenu. Ce risque est appelé Risque de Crédit. Alors avant d'octroyer un crédit, les banques vérifient si le client (ou la cliente) qui demandent un prêt sera capable ou pas de le rembourser. Cette vérification se fait grâce à l'analyse de plusieurs paramètres tels que les revenus, les biens, les dépenses actuelles du client, etc. Cette analyse est encore effectuée manuellement par plusieurs banques. Ainsi, elle est très consommatrice en temps et en ressources financières.

Grâce au Machine Learning, il est possible d'automatiser cette tâche et de pouvoir prédire avec plus de précision les clients qui seront en défaut de paiement.

Dans ce projet, nous allons construire un algorithme capable de prédire si une personne sera en défaut de paiement ou pas (1 : défaut, 0 : non-défaut). Il s'agit donc d'un problème de classification car nous voulons prédire une variable discrète (binaire pour être précis).
:::


```
library(tidyverse)
library(ggthemes)
library(ROSE)
library(pROC)
```

Les données proviennent de [Kaggle](https://www.kaggle.com/laotse/credit-risk-dataset) qui est la plus célèbre plateforme de compétitions en Data Science.

L'ensemble des données compte 12 variables et 32581 observations (lignes) historiques. Chaque observation correspond à une personne ayant contracté un prêt. On a des variables qui décrivent le prêt (montant, statut, taux d'intérêt, etc.) et d'autres variables qui décrivent la personne ayant ontracté ce prêt (age, revenu, etc.). Nous allons donc utiliser ces données historiques afin de construire le modèle de *scoring* qui va prédire le statut des nouveaux candidats à un crédit.  

Il est très important de comprendre les variables de notre jeu de données :    

* ***person_age*** : variable indiquant l'âge de la personne ;           
* ***person_income*** : variable indiquant le revenu annuel (ou encore le salaire) de la personne ;             
* ***person_home_ownership*** : variable indiquant le statut de la personne par rapport à son lieu d'habitation (popriétaire, locataire, etc.) ;        
* ***person_emp_length*** : variable indiquant la durée (en mois) depuis laquelle la personne est en activité professionnelle ;           
* ***loan_intent*** : variable indiquant le motif du crédit ;          
* ***loan_grade*** : Notation de la solvabilité du client. classes de A à G avec A indiquant la classe de solvabilité la plus élevée et G la plus basse ;           
* ***loan_amnt*** : variable indiquant le montant du prêt ;                 
* ***loan_int_rate*** : variable indiquant le taux d'intérêt du crédit ;       
* ***loan_status*** : c'est la variable d'intérêt. Elle indique si la personne est en défaut de paiement (1) ou pas (0) ;       
* ***loan_percent_income*** : variable indiquant le pourcentage du crédit par rapport au revenu (ratio dette / revenu) ;          
* ***cb_person_default_on_file*** : variable indiquant si la personne a été en défaut de paiement ou pas dans le passé                  
* ***cb_person_cred_hist_length*** : variable indiquant la durée des antécédents de crédits.     

### **Analyse Exploratoire des données**

```
# Transformation de la variable cible en variable catégorielle

df$loan_status <- as.factor(df$loan_status)

# Table de fréquence de la variable cible ('loan_status')

print(prop.table(table(df$loan_status)))

# Diagramme à barre de la variable 'loan_status'

plot(df$loan_status, main = "Statut de crédit")
```

```{figure} ./images/statut_credit.png
---
scale: 100%
```

```{figure} ./images/loan.png
---
scale: 100%
```

:::{Note}
Ces résultats montrent qu'il y un déséquilibre de classe très importants dans les données. En effet, seulement environ 22% de clients sont en défaut de paiement contre un peu plus de 78% de bons clients. 

Le déséquilibre de classe est souvent observé dans les données de crédit. la majorité des demandeurs de crédit sont incités à ne pas être en défaut de paiement car plus ils remboursent le crédit dans les délais, plus leurs côtes de crédit augmentent et donc ils peuvent à nouveau emprunter pour effectuer d'autres investissements.

Si le déséquilibre observé ici est tout à fait normal, il n'en demeurre moins que cela représente un grand défi de classification pour les algorithmes de Machine Learning.
:::

Il est intéressant de visualiser la distribution de la variable indiquant le montant du crédit chez les personnes en défaut de paiement et chez les autres afin de pouvoir effectuer une comparaison.


```{figure} ./images/loan2.png
---
scale: 100%
```

Les deux histogrammes semblent être identiques mais en observant très bien on constate on remarque par exemple que les barres au niveau des montants très élevés sont plus longues chez les personnes en défaut de paiement que chez les autres. Cela veut dire qu'il y a beaucoup plus de personnes en défaut de paiement qui ont emprunté de grosses d'argent en comparaison aux personnes qui ne sont pas en défaut de paiement.


```
df %>%
    ggplot(aes(x=person_home_ownership, fill = person_home_ownership)) +
    geom_bar() +
    theme_bw()
```

```{figure} ./images/own.png
---
scale: 100%
```

### **Nettoyage de données**

#### **Outliers**

Lors de l'analyse exploratoire des données, nous avons remarqué la présence de valeurs aberrantes. Ces valeurs aberrantes peuvent affecter la qualité d'un modèle de Machine Learning. Nous allons donc traiter. Avant de traiter les valeurs aberrantes, il faut d'abord les détecter. 

Il existe plusieurs méthodes de détection des *outliers*. Selon la méthode de la gamme interquartile (*IQR*), une valeur est aberante si :  
$$valeur < Q_1−1.5 * IQR$$ ou $$valeur > Q_3 + 1.5 * IQR$$ avec $$IQR = Q_3 - Q_1$$


```
# Identification des valeurs aberantes au niveau de la variable 'person_income'

index_outlier_income <- which(df_clean$person_income < quantile(df_clean$person_income, 0.25) - 1.5 * IQR(df_clean$person_income) | df_clean$person_income > quantile(df_clean$person_income, 0.75) + 1.5 * IQR(df_clean$person_income))

# Suppression des valeurs aberantes au niveau de la variable 'person_income'

df_clean <- df_clean[-index_outlier_income, ]

# Vérification : Histogramme des revenus annuels

hist(df_clean$person_income, main = "Histogramme du revenu annuel"
```

#### **Valeurs manquantes**

Il existe deux principales techniques pour traiter les valeurs manquantes :    
* **Supression** des lignes contenant des valeurs manquantes ;                 
* **Remplacement** des valeurs manqauntes en faisant des imputations par la moyenne, la médiane, ...etc de la variable contenant ces valeurs manquantes. Il existe également d'autres méthodes d'imputation plus ou moins sophistiquées comme la méthode des k plus proches voisins (***KNN***). 

Dans la pratique, il est conseillé de choisir une méthode puis de construire et d'évaluer le modèle. Puis de changer la méthode d'imputation et reconstruit le modèle ainsi de suite afin de finalement choisir une méthode qui donne les meilleurs résultats.

### **Préparation des données pour la phase de modélisation**

#### **Normalisation des variables numériques**

```
# Création d'une fonction de normalisation

normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
}
```

#### **Données d'entraînement et de test**

Commençons par diviser les données en un ensemble pour entraîner les algorithmes et un ensemble de test pour évaluer la capacité du modèle à généraliser sur de nouvelles données (performance du modèle construit).

```
# Données d'entraînement (80%) et de test (20%) (Division aléatoire avec la fonction sample)

seed <- 131

set.seed(seed)

index_train <- sample(1:nrow(df_clean2), 0.8 * nrow(df_clean2))

train_set <- df_clean2[index_train, ]

test_set <- df_clean2[-index_train, ]
```

#### **Résolution du problème de déséquilibre de classe**

Il existe plusieurs techniques pour résoudre le problème de déséquilibre de classe dans les données. Le rééchantillonnage des données est l'une des techniques les plus utilisées. Les méthodes de rééchantillonage souvent utilisées sont : 

* de sous-échantillonement de la classe majoritaire (***Random Under Sampling*** ou [RUS](https://www.rdocumentation.org/packages/ROSE/versions/0.0-3/topics/ROSE)) : cette méthode consiste à tirer au hasard des observations de cas de non-défaut pour correspondre au nombre d'observations de cas de défaut de paiement ;

*  sur-échantillonement de la classe minoritaire (***Random Over Sampling*** ou [ROS](https://www.rdocumentation.org/packages/ROSE/versions/0.0-3/topics/ROSE)) : cette méthode consiste à effectuer des tirages aléatoires d'observations de cas de défaut et dupliquer ces observations afin de correspondre au nombre de cas de non-défaut. 

Ces deux méthodes conduisent à un équilibre parfait des cas de défaut et non-défaut de paiement mais présentent quand même des inconvénients. Avec le sous-échantillonnage, vous supprimez beaucoup d'informations. Dans la plupart des cas, jeter des données n'est pas souhaitable en apprentissage automatique. Avec le sur-échantillonnage, vous créez beaucoup de doublons d'informations ce qui peut causer des biais importants au niveau de l'entraînement des algorithmes.

* Technique de suréchantillonnnage des minorités synthétiques (***Synthetic Minority Oversampling Technique*** ou [SMOTE](https://www.rdocumentation.org/packages/smotefamily/versions/1.3.1/topics/SMOTE) en anglais) : c'est une technique sophistiquée qui ne se contente pas de juste cdupliquer des cas de défaut mais utilise les caractéristiques des plus proches voisins des cas de défaut de paiement pour créer de nouveaux cas de défaut synthétiques. Cette méthode bien qu'utilisant des algorithmes hyper-sophistiqués présente le risque que les voisins les plus proches des cas de défaut ne soient pas en réalité des cas de défaut ce qui peut entraîner des erreurs de modélisation.


*N.B : Les méthodes de rééchantillonnage des données pour résoudre le problème de déséquilibre de classe doivent être appliquées uniquement aux données d'entraînement et NON aux données de test*


### **Modélisation**
#### **Choix de la métrique d'évaluation de performance du modèle**

* Quelle métrique choisir pour évaluer la performance des modèles ?



```{figure} ./images/confusion.png
---
scale: 100%
```

La matrice de confusion est une matrice carrée qui rapporte le nombre de vrais positifs (*True Positives* ou TP), vrais négatifs (*True Négatives* ou TN), faux positifs (*False Positive* ou FP) et faux négatifs (*False Negatives* ou FN).

Dans notre cas, la classe positive est 1 : le client est en défaut de paiement et la classe négative est 0 : le client n'est pas en défaut de paiement.

* **TP** : le client est en défaut de paiement et le modèle prédit qu'il est en défaut de paiement ;

* **TN** : le client n'est pas en défaut de paiement et le modèle prédit qu'il n'est pas en défaut de paiement ;

Les 2 cas ci-dessus (TP et TN) sont les bons cas. Mais FP et FN sont les mauvais cas :

* **FP** : le client n'est pas en défaut de paiement mais le modèle prédit qu'il est en défaut de paiement ;

* **FN** : le client est en défaut de paiement mais le modèle prédit qu'il n'est pas en défaut de paiement.


A partir de la matrice de confusion, vous pouvez calculer certaines métriques pour évaluer la performance du modèle.

* La précision de la classification (***Accuracy*** en anglais) est le pourcentage d'instances correctement classifiées, c'est-à-dire la somme du nombre de vrais négatifs et de vrais positifs divisée par le nombre total des observations. Elles se calcule donc par la formule ci-dessous : 

$$Accuracy = \frac{TN + TP}{(TN + FN + FP + TP)}$$

* La sensibilité du modèle (***Sensitivity*** en anglais) se calcule par la formule ci-dessous : $$Sensitivity = \frac{TP}{(FN + TP)}$$

Dans le cas présent, la sensibilité se traduit par le pourcentage de clients en défaut de paiement (classe positive) qui ont été classifié comme tel par le modèle. Une sensibilité élevée est meilleure.

* La spécificité du modèle (***Specificity*** en anglais) se calcule par la formule ci-dessous : $$Specificity = \frac{TN}{(TN + FP)}$$

Ici, la spécificité est le pourcentage de clients qui ne sont pas en défauts de paiement (classe négative) et qui ont été classififié comme tel par le modèle.

Une spécificité élevée est meilleure. Mais il faudra un compromis entre la sensibilité du modèle et la spécificité car l'amélioration  de la sensibilité diminue la spécificité et l'amélioration de la spécificité diminue la sensibilité.

Il faut faire attention à la précision globale. Une forte précision globale ne signifie pas forcément que le modèle est performant. Le choix de la métrique pour quantifier la performance du modèle doit se faire en fonction du contexte de l'étude, c'est-à-dire de la problématique qu'on veut résoudre.

#### **Modèle de Régression logistique**

```
# Création  d'une fonction de construction d'un modèle de régression logistique

log_modeling <- function(train) {
    model <- glm(loan_status ~ ., family = 'binomial', data = train)
    return (model)
}
```

#### **Construction et évaluons un modèle de régression logistique en utilisant l'ensemble** ***train_set**

```{figure} ./images/glm.PNG
---
scale: 100%
```


```{figure} ./images/seuil1.png
---
scale: 100%
```

#### **Courbe ROC**

```{figure} ./images/roc1.png
---
scale: 100%
```

#### **Sauvegarde des modèles**

```
# Sauvegarde du meilleur modèle

saveRDS(log_model_ros, "credit_scoring_final_model.rds")
```


## Partie2 : INTEGRATIONT DANS SHINYDASHBORD DU MODELE DE MACHINE LEARNING POUR LA PREDICTION DU RISQUE DE CREDIT BANCAIRE

```
# Integration d'un modele de Machine Learning dans R Shiny : Cas de la modelisation du risque de credit

library(shiny)
library(shinydashboard)

model <- readRDS('credit_scoring_rf_model.rds')

ui <- dashboardPage(
       dashboardHeader(
              title= div(h3('Credit Scoring', style="margin: 0;"), h4('by Roland MONDJEHI', style="margin: 0;"))

       ), 
       
       dashboardSidebar(), 
       
       dashboardBody(
              
              tabItem(
                     tabName = "features",
                     fluidRow(box(valueBoxOutput("score_prediction")),
                              box(numericInput("var1", label = "Age du demandeur de credit", 
                                               value = 20, min = 18))),
                     
                     fluidRow(box(numericInput("var2", label = "Revenu annuel demandeur de credit", 
                                               value = 10000, min = 0)),
                              box(selectInput("var3", 
                                              label = "Propriété immobilière : (MORTGAGE : hypothèque, OWN : propriétaire, RENT : Locataire, OTHER : Autres cas)", 
                                              choices = c('MORTGAGE', 'OWN', 'RENT', 'OTHER')))),
                     
                     fluidRow(box(numericInput("var4", 
                                               label = "Depuis quand le demandeur est-il en activité professionnelle ? (Durée en nombre d'années)", 
                                               value = 3, min = 0)),
                              box(selectInput("var5", 
                                              label = "Motif du prêt : (DEBTCONSOLIDATION : Rachat d'un crédit, HOMEIMPROVEMENT : Travaux de rénovation immobilière, VENTURE : Business)", 
                                              choices = c('DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'))),),
                     fluidRow(box(selectInput("var6", 
                                              label = "Catégorie du crédit", 
                                              choices = c('A', 'B', 'C', 'D', 'E', 'F', 'G'))),
                              box(numericInput("var7", 
                                               label = "Montant du crédit", 
                                               value = 2000, min = 0))),
                     
                     fluidRow(box(numericInput("var8", 
                                               label = "Taux d'intéret du crédit (en %)", 
                                               value = 3.5, min = 0)),
                              box(numericInput("var9", 
                                               label = "Ratio Dette/Revenu du demandeur de crédit (valeur décimale entre 0 et 1)", 
                                               value = 0.1, min = 0, max = 1))),
                     
                     fluidRow(box(selectInput("var10", 
                                              label = "Est-ce que le demandeur de credit est à découvert bancaire ? : (Y : Oui, N : Non):", 
                                              choices = c('Y', 'N'))),
                              box(numericInput("var11", 
                                               label = "Echéance des crédits en cours (en nombre d'années)", 
                                               value = 5, min = 0)))
                     
              )
              
       )
       
)

server <- function(input, output) {
       
       prediction <- reactive({
              predict(
                     model,
                     data.frame(
                            "person_age" = input$var1,
                            "person_income" = input$var2,
                            "person_home_ownership" = input$var3,
                            "person_emp_length" = input$var4,
                            "loan_intent" = input$var5,
                            "loan_grade" = input$var6,
                            "loan_amnt" = input$var7,
                            "loan_int_rate" = input$var8,
                            "loan_percent_income" = input$var9,
                            "cb_person_default_on_file" = input$var10,
                            "cb_person_cred_hist_length" = input$var11
                     ),
                     type = 'raw'
              )
       })
       
       prediction_label <- reactive({
              ifelse(prediction() == "0", "Eligible au Crédit", "Non Eligible au Crédit")
       })
       
       prediction_prob <- reactive({
              predict(
                     model,
                     data.frame(
                            "person_age" = input$var1,
                            "person_income" = input$var2,
                            "person_home_ownership" = input$var3,
                            "person_emp_length" = input$var4,
                            "loan_intent" = input$var5,
                            "loan_grade" = input$var6,
                            "loan_amnt" = input$var7,
                            "loan_int_rate" = input$var8,
                            "loan_percent_income" = input$var9,
                            "cb_person_default_on_file" = input$var10,
                            "cb_person_cred_hist_length" = input$var11
                     ),
                     type = "prob"
              ) 
       })
       
       prediction_color <- reactive({
              ifelse(prediction() == "0", "green", "red")
       })
       
       output$score_prediction <- renderValueBox({
              
              valueBox(
                     value = paste(round(100*prediction_prob()$`1`, 0), "%"),
                     subtitle = prediction_label(),
                     color = prediction_color(),
                     icon = icon("hand-holding-usd")
              )                     
              
       })
       
}


shinyApp(ui, server)
```

```{figure} ./images/img_app1.PNG
---
scale: 50%
```

```{figure} ./images/img_app2.PNG
---
scale: 50%
```




# Conclusion

A travers ce projet, je suis capable :

* **d'analyser, de nettoyer et de préparer les données pour modéliser la probabilité de défaut de paiement** ; 

* **d'analyser la performance des différents modèles construits et de déterminer et de déterminer le seuil optimal pour la prédiction des résultats de la variable cible** ;

* **de comparer plusieurs modèles en utilisant une métrique comme l'AUC** ;

* **de bien structurer mon code R en créant des fonctions qui rendent mon flux de travail beaucoup plus clair et digeste**. 

 
La modélisation du risque de crédit par les méthodes d'apprentissage automatique est un domaine passionnant et il reste encore beaucoup de choses à apprendre.