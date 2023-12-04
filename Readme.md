# OT5 : Calcul parallèl

## Membres
Mario CASTILLON
Jorge KORGUT Junior

## Architecture du projet

* Dans la racine nous avons un Makefile qui compile tous les executables des differentes parties.  
>Pour compiler les sources
    ```Make```  

* Les executables sont générés dans le dossier _/Executables_  
* Pour acceder aux codes sources des parties, les dossiers _/Parti#_ sont disponibles dans la racine du projet.  
* Pour executer les scripts pythons d'analyses, veillez exécuter dans la racine du projet :  

>Pour générer un fichier stats.csv avec les données d'execution des programmes  
    ```python3 ./Analysis/Part#/evaluation.py```  

>Pour afficher un graphe avec les données de performance  
    ```python3 ./Analysis/Part#/analysis.py```  

## Introduction  

L'objectif de ce TP est d'utiliser concrètement les fonctionalités de parallelisation de la librairie OPENMP, prendre en main les outils d'analyse de performance comme Intel VTune Profiler et de se familiariser avec quelques cas classiques de parallelisation d'algorithmes.  
Il est intéréssant de remarquer que les parallelisations que nous effectuerons ici ne seront pas optimales. En effet, si nous prennons l'example de la deuxième partie, les multiplications et sommes des matrices peuvent être beaucoup plus otimisés si des librairies spécifiques faites pour ce type de calcul sont utilisés.  

Pour l'obtention des données de performance, chaque algorithme a été executé 10 fois pour chaque paramettre spécifique. Comme par example le nombre de threads, taille des matrices, nombre d'itérations.

## Partie 1  

Dans le domaine de l'informatique et de la programmation, la recherche de méthodes efficaces pour le calcul de constantes mathématiques revêt une importance cruciale. Parmi ces constantes, le nombre π occupe une place prépondérante en raison de son omniprésence dans de nombreuses disciplines scientifiques et techniques.

Pour le calculer nous avons repris le code de Tim Mattson qui a été modifié par Jonathan Rouzaud-Cornabas qui calcule une approximation de π par la résolution numérique de l'intégrale suivante :
$\int_0^1 \frac{4}{(1+x^{2})} dx$  

car :  
$\int_0^1 \frac{4}{(1+x^{2})} dx = 4*(arctg(1) - arctg(0)) = 4*(\frac{π}{4} - (0)) = π$

Il est intéressant de rémarquer que la correction de notre calcul est donc possible grâce à la connaissance en amond du résultat. Ce qui facilite le débuggage.

Dans le code fourni, il est facile à identifier le point d'optimisation. En effet, comme l'algorithme consiste seulement d'une seule boucle et nous avons un algorithme en sequence, nous pouvons nous concentrer à essayer la paralellisation de cette boucle for. 

En rajoutant le pragma suivant avant la boucle :
```c++
    #pragma omp parallel for shared(sum) num_threads(num_cores)
```
Nous spécifions à openMP que la variable _**sum**_ sera partagé entre les threads et qui nous voudrions un num de threads égal à ce que l'on souhaite.

De plus on rajoute les mots clefs :
```c++
    #pragma omp atomic
```
Pour éviter les cas des concurrences qui viennent afecter nos résultats.

![Alt text](Resources/)
_Comparaison du temps d'execution du programme en mode sequentiel et en mode parallèle atomique._ 

En regardant plus attentivement notre algorithme et le mode de fonctionnement des atomiques sur OpenMP, nous nous rendons compte qu'une bonne partie du code est passé dans la gestion des verrous pour acceder les ressources critiques. Ainsi, parce que notre cas particulier nous permet, nous pouvons diminuer ce temps d'attente, en utilisant une arbre de réduction. Par chance, juste le changement des pragmas nous permet d'atteidre cet objectif.
```c++
    #pragma omp parallel for reduction(+ : sum) num_threads(num_cores)
```

![Alt text](Resources/)
_Comparaison du temps d'execution du programme en mode parallel atomique et en mode parallèle en reduction._ 

Un autre point d'optimisation est d'augmenter le temps d'execution de chaque thread afin de diminuer la quantité de branches dans l'arbre de réduction. En effet, si le temps d'execution de chaque thread est inférieur au temps pour faire une réduction, il est intéressant de regrouper les taches par thread afin que l'arbre à la fin soit moins dense. Cependant, il reste difficil d'identifier le temps exacte passé pour une réduction, ainsi un teste empirique a été éffectué pour desmistifier le sujet.

![Alt text](Resources/)
_Comparaison du temps d'execution du programme en mode parallel en reduction et en mode parallèle en reduction fractionné._  

## Partie 2  

![Alt text](Resources/Part2_Vect_seq.png)  
_Representation graphique du temps d'execution de l'algorithme de somme et multiplication des matrices en sequence._  


Un premier pas pour identifier les points d'embouteillage de notre programme est de vérifier le ratio de temps d'execution des parties de notre code. Pour identifier cela, nous pouvons lancer l'algorithme VTune profile avec l'option hotspot, pour identifier ces parties.  
![Alt text](Resources/Sequential_vector_hotspots.PNG)  
_Analyse des hotspots de l'algorithme sequenciel avec l'outil VTune profile_  
  
Avec l'image, nous pouvons voir qu'une bonne partie du temps d'execution de notre programme est passé dans l'affectation de la matrice de départ et dans les opérations arithmetiques matricielles.  

![Alt text](Resources/Seq1Thread.PNG)
_Affichage du nombre de Threads utiliés dans l'execution de l'application_  

De plus, nous pouvons voir que juste 1 coeur est utilisé lors de l'execution du programme. Ainsi une premiere piste d'optimisation est la parallelisation de ces tâches.  
