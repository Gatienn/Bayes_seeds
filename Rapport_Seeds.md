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

![datatable](images\datatable.jpg)
 *Figure 1 : Tableau récapitulatif de l'expérience*

 ## Modèle utilisé

Nous allons à présent détailler le modèle associé à cette expérience.  

Tout d'abord, remarquons que toutes les graines placées sur une assiette $i$ sont du même type et ont la même racine. On peut donc supposer que les graines de cette assiette ont la même probabilité de germer, notée $p_i$.  

On suppose par ailleurs que la germination d'une graine sur une assiette est indépendante des graines qui se trouvent autour.  
Autrement dit, la germination des graines sur l'assiette $i$ consiste en $n_i$ répétitions indépendantes de la germination d'une graine avec probabilité $p_i$.  
Il s'en suit que $r_i \sim Binomiale(p_i, n_i)$.

Les $p_i$ sont modélisés par la régression logistique suivante :  
$logit(p_i) = \alpha_{0} + \alpha_{1}x_{1i} + \alpha_{2}x_{2i} + \alpha_{12}x_{1i}x_{2i} + b_i,\quad b_i \sim Normal(0,\tau)$ (tau désignera la variance et non la précision),
avec $x_{1i}$ le type de graine et $x_{2i}$ la racine sur l'assiette $i$. 

En effet, la probabilité $p_i$ dépend bien évidemment du type des graine et de leur racine. De plus, le terme d'interaction $\alpha_{12}x_{1i}x_{2i}$ est ajouté afin de traduire l'effet de la combinaison entre les deux sur la germination.  
Enfin, le terme $b_i$ est un effet aléatoire visant à modéliser une légère variabilité des conditions d'une assiette à l'autre (température, lumière, assiette en elle même).

N'ayant pas de connaissances sur les $\alpha$ ni sur $\tau$, nous utilisons des lois a priori non informatives :
- $\alpha_{0}$, $\alpha_{1}$, $\alpha_{2}$, $\alpha_{12}$ $\sim Normal(0,\sigma_0^2)$ avec $\sigma_0$ grand.
- $\tau \sim Inverse Gamma(a,b)$. On choisit une inverse gamma parce que c'est la loi conjuguée pour l'estimation de la variance d'une loi normale dont l'espérance est connue. De plus, c'est une loi à valeurs positives, tout comme $\tau$ qui est une variance.

On obtient finalement le modèle suivant :

![graphe](images\graphe_model.jpg)
 *Figure 2 : Graphe du modèle pour l'expérience seeds*

 ## Echantilloneur de Gibbs et Metropolis Hastings

 Nous utilisons un échantillonneur de Gibbs afin de simuler les lois associées aux variables du modèle. 

 Pour cela, nous devons d'abord déterminer les lois conditionnelles qui seront utilisées afin de mettre à jour chaque variable :

 - Loi des $\alpha$

 $\pi\left(\alpha_0 \mid r, \alpha_1, \alpha_2, \alpha_{12}, b, \tau\right) \propto \pi(\alpha_0) \cdot \pi\left(r\mid \alpha_0, \alpha_1, \alpha_2, \alpha_{12}, b)\right. $

$\pi\left(\alpha_0 \mid r, \alpha_1, \alpha_2, \alpha_{12}, b, \tau\right) \propto \pi(\alpha_0) \cdot \prod_{i=1}^{N} \pi\left(r_{i}\mid \alpha_0, \alpha_1, \alpha_2, \alpha_{12}, b)\right. $

$\pi\left(\alpha_0 \mid r, \alpha_1, \alpha_2, \alpha_{12}, b, \tau\right) \propto \exp \left(-\frac{\alpha_{0}^{2}}{2\sigma_0^2}\right) \prod_{i=1}^{N}\left(p_{i}\right)^{r_{i}}\left(1-p_{i}\right)^{n_{i}-r_{i}} $

où $r$ et $b$ regroupent les observations des $r_i$ et des $b_i$ respectivement.

La loi conditionnelle est la même pour $\alpha_{1}$, $\alpha_{2}$, $\alpha_{12}$, il suffit d'échanger les indices.

- Loi de tau :

$\pi\left(\tau \mid r, \alpha_0,  \alpha_1, \alpha_2, \alpha_{12}, b\right) \propto \pi(\tau) \cdot \prod_{i=1}^{N} \pi\left(b_{i}\mid \tau)\right. $

$\tau \sim Inverse gamma(a,b)$ et $(b_i\mid \tau) \sim Normal(0, \tau)$.   
La loi de $\tau$ est conjuguée et on obtient finalement $(\tau \mid r, \alpha_0,  \alpha_1, \alpha_2, \alpha_{12}, b) \sim Inverse Gamma(a^*, b^*)$, avec $a* = a + \frac{N}{2}$ et $b^* = b + \frac{1}{2} \cdot \sum_{i=1}^{N} b_i^2$

- Loi des $b_i$ :

Pour i dans 1,....,N  
 $\pi\left(b_i \mid r,\alpha_0,  \alpha_1, \alpha_2, \alpha_{12}, \tau\right) \propto \pi(b_i \mid \tau) \cdot \pi\left(r_{i}\mid \alpha_0, \alpha_1, \alpha_2, \alpha_{12}, b_i)\right. $
 $\pi\left(b_{i} \mid r,\alpha_0,  \alpha_1, \alpha_2, \alpha_{12}, \tau\right) \propto \exp \left(-\frac{b_{i}^{2}}{2 \tau}\right)\left(p_{i}\right)^{r_{i}}\left(1-p_{i}\right)^{n_{i}-r_{i}}$

 Etant donné que nous ne pouvons pas échantillonner facilement à partir des lois conditionnelles des $\alpha$ et des $b_i$, nous utiliserons Metropolis Hastings :  
 Pour ces paramètres, nous effectuons une proposition suivant une marche aléatoire $x^* = x_t + \epsilon$, où $\epsilon$ suit une loi normale centrée dont nous ajusterons la variance.  
 Puis, la proposition est validée avec probabilité valant $min(1, \frac{g(x^*)}{g(x_t)})$, g étant la densité de la loi conditionnelle.

 ## Résultats

 On commence par générer des chaînes de taille 10000 auxquelles on retire les 1000 première valeurs (burning).  
On élague ensuite les chaînes en ne conservant qu'une valeur sur 10, afin de limiter les éventuelles dépendances entre leurs états successifs (thining).

![alpha](images\resultats_alpha.jpg)
![tau_b](images\resultats_tau_b.jpg)
 *Figure 3 : Chaînes obtenues pour les alpha, tau et b0*

Les chaînes obtenues semblent bien stationnaires. De plus, nous avons modifié les variances associées aux marches aléatoires afin d'obtenir un taux d'acceptation d'environ 0.3 pour tous les paramètres, ce qui est un taux standard pour ce type d'algorithme.

Nous pouvons à présent calculer la moyenne et l'écart type pour les distributions des alpha et de tau, et les comparer aux valeurs obtenues dans le sujet pour ce même modèle :

![resultats](images\tableau_resultats.jpg)
 *Figure 4 : Tableau des résultats pour les variables du modèle*

Les écarts type obtenus sont tous légèrement supérieurs à ceux qui étaient attendus, à l'exception de celui pour tau.  
En revanche, les valeurs moyennes sont très proches, dans le sens où l'écart entre le résultat que nous avons obtenu et celui du sujet est faible devant la valeur de l'écart type, et ce pour tous les paramètres.

Enfin, nous pouvons calculer les probabilités de germination associées à chaque couple (type de graine, racine).  
Pour cela, on calcule d'abord la chaîne des $p_i, i=1,...,N$, à partir des chaînes présentées ci-dessus, et ensuite seulement nous calculons leur moyenne et écarts types respectifs. Puis, on moyenne les valeurs obtenues pour chaque couple (type de graine, racine).

![resultats_p](images\tableau_resultats_p.jpg)  
 *Figure 5 : Tableau des probabilités de germination*

Les assiettes contenant des seed 73 ont globalement moins de graines que celles avec des seed 75. Il n'est donc pas surprenant d'obtenir de plus grands écarts types pour les seed 73.  
L'information que nous pouvons tirer de ces résultats est que pour faire pousser des haricots, les deux types de graines sont similaires. Le couple haricot + seed 75 a une probabilité très légèrement inférieure de germer, mais le résultat est légèrement plus sûr en raison du plus faible écart type.  
En revanche, la différence est plus nette pour le concombre : en raison d'un écart type plus faible et surtout d'une probabilité de germination beaucoup plus grande, il vaut mieux privilégier les graines seed 75. 