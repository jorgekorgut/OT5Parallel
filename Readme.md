# OT5 : Calcul parallèl

## Membres
_Mario CASTILLON_
Jorge KORGUT Junior

## Architecture du projet

* Dans la racine nous avons un Makefile qui compile tous les executables des differentes parties.  
>Pour compiler les sources
    ```Make```  


* Les executables sont générés dans le dossier **/Executables** 
* Pour acceder aux codes sources des parties, les dossiers **/Parti#** sont disponibles dans la racine du projet.  
* Pour executer les scripts pythons d'analyses, veillez exécuter dans la racine du projet :  

>Pour générer un fichier stats.csv avec les données d'execution des programmes  
    ```python3 ./Analysis/Part#/evaluation.py```  

>Pour afficher un graphe avec les données de performance  
    ```python3 ./Analysis/Part#/analysis.py```  

## Introduction OpenMP 

L'objectif de ce TP est d'utiliser concrètement les fonctionalités de parallelisation de la librairie OPENMP, prendre en main les outils d'analyse de performance comme Intel VTune Profiler et de se familiariser avec quelques cas classiques de parallelisation d'algorithmes.  
Il est intéréssant de remarquer que les parallelisations que nous effectuerons ici ne seront pas optimales. En effet, si nous prennons l'example de la deuxième partie, les multiplications et sommes des matrices peuvent être beaucoup plus otimisés si des librairies spécifiques faites pour ce type de calcul sont utilisés.  

Dans les tests de performance, nous exécutons chaque paramettrage 50 fois pour le tracage des graphiques.

## Partie 1  

Pour commencer, nous avons à disposition un algorithme d'approximation de PI et notre objectif est d'utiliser les anotations OpenMP afin de paralleliser la boucle critique.

Un example de code se trouve si dessous. Si vous voulez regarder la totalité des implementations, veillez se rendre au dossier **Part1**.
```c++
#pragma omp parallel for shared(sum) num_threads(num_cores)
for (i=1;i<= num_steps; i++)
{
    x = (i-0.5)*step;
    x = 4.0/(1.0+x*x);
    #pragma omp atomic
	sum = sum + x;
}
```


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

![Alt text](Resources/Part2_Vect_parallel.png)
_Representation graphique du temps d'execution de l'algorithme de somme et multiplication des matrices en parallel._  

Au depart, nous observons une diminution du temps d'exécution jusqu'à 8 cores (nombre des coeurs de la machine de test). Après ce nombre, le déroulement du programme prend plus de temps principalement à cause de l'initialisation des instances parallèles et de l'ordonencement des tâches.

![Alt text](Resources/Compare_Sequential_Parallel.png)
_Representation graphique de la compairaison du temps d'execution de l'algorithme de somme et multiplication des matrices en parallel et en séquence._  

Une autre idée est d'implementer les optimisations du type SIMD. L'idée est de donner des directives au compilateur afin qu'il puisse utiliser le hardware spécifique et ainsi augmenter la vitesse d'execution du programme.

![Alt text](Resources/Compare_SIMD_Parallel.png)
_Representation graphique de la compairaison du temps d'execution de l'algorithme de somme et multiplication des matrices en parallel et avec le SIMD._  

Avec le graphique, nous pouvons conclure que l'implementation du SIMD n'a pas presenté d'éffets significatifs pour diminuer le temps d'exécution du programme. Cela peut avoir comme raison, la non compatibilité du compilateur ou un code pas adapté pour ce type d'optimisation.