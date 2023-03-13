Natacha GOUGEON, Sebastien ROIG, Floran DEFOSSEZ, Gatien CHOPARD

# Rapport
# Seeds : Random effect logistic regression

## Présentation de l'expérience et du jeu de données

A travers cette étude, nous modélisons la germination de plusieurs variétés de graines sur des assiettes.

L'expérience a été menée sur un total de 21 assiettes. Deux types de graines sont utilisées (seed O. aegyptiaco 75 et seed O. aegyptiaco 73) ainsi que deux racines (haricot et concombre), résultant en un total de 4 combinaisons type de graine/racine testées.

Le résultats de l'expérience sont présentés ci-dessous, avec les notations suivantes :
- n est le nombre de graines sur l'assiette.
- r est le nombre de graines ayant germé parmi ces n graines.
- r/n est la proportion de graines ayant germé.

![test](datatable.jpg)
 *Figure 1 : Tableau récapitulatif de l'expérience*

 ## Modèle utilisé

Nous allons à présent détailler le modèle associé à cette expérience.  

Tout d'abord, remarquons que toutes les graines placées sur une assiette $i$ sont du même type et ont la même racine. On peut donc supposer que les graines de cette assiette ont la même probabilité de germer, notée $p_i$.  

On suppose par ailleurs que la germination d'une graine sur une assiette est indépendante des graines qui se trouvent autour.  
Autrement dit, la germination des graines sur l'assiette $i$ consiste en $n_i$ répétitions indépendantes de la germination d'une graine avec probabilité $p_i$.  
Il s'en suit que $r_i \sim Binomiale(p_i, n_i)$.

Les $p_i$ sont modélisés par la régression logistique suivante :  
$logit(p_i) = \alpha_{0} + \alpha_{1}x_{1i} + \alpha_{2}x_{2i} + \alpha_{12}x_{1i}x_{2i} + b_i,\quad b_i \sim Normal(0,\tau)$,  
avec $x_{1i}$ le type de graine et $x_{2i}$ la racine sur l'assiette $i$. 

En effet, la probabilité $p_i$ dépend bien évidemment du type des graine et de leur racine. De plus, le terme d'interaction $\alpha_{12}x_{1i}x_{2i}$ est ajouté afin de traduire l'effet de la combinaison entre les deux sur la germination.  
Enfin, le terme $b_i$ est un effet aléatoire visant à modéliser une légère variabilité des conditions d'une assiette à l'autre (température, lumière, assiette en elle même).

N'ayant pas de connaissances sur les $\alpha$ ni sur $\tau$, nous utilisons des lois a priori non informatives :
- $\alpha_{1}$, $\alpha_{2}$, $\alpha_{12}$ $\sim Normal(0,\sigma_0^2)$ avec $\sigma_0$ grand.
- $\tau \sim Inverse Gamma(\alpha_0, \beta_0)$. On choisit une inverse gamma parce que c'est la loi conjuguée pour l'estimation de la variance d'une loi normale dont l'espérance est connue. De plus, c'est une loi à valeurs positives, tout comme $\tau$ qui est une variance.

On obtient finalement le modèle suivant :

![test](graphe_model.jpg)
 *Figure 2 : Graphe du modèle pour l'expérience seeds*

 ## Echantilloneur de Gibbs et Metropolis Hastings