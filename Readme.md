# OT5 : Calcul parallèl

## Membres
Mario  
Jorge KORGUT Junior

## Architecture du projet

* Dans la racine nous avons un Makefile qui compile tous les executables des differentes parties.  
>Pour compiler les sources
    ```Make```  


* Les executables sont générés dans le dossier _/Executables_  
* Pour acceder aux codes sources des parties, les dossiers _/Parti#_ sont disponibles dqns la racine du projet.  
* Pour executer les scripts pythons d'analyses, veillez exécuter dans la racine du projet :  

>Pour générer un fichier stats.csv avec les données d'execution des programmes  
    ```python3 ./Analysis/Part#/evaluation.py```  

>Pour afficher un graphe avec les données de performance  
    ```python3 ./Analysis/Part#/analysis.py```  

## Introduction  

L'objectif de ce TP est d'utiliser concrètement les fonctionalités de parallelisation de la librairie OPENMP, prendre en main les outils d'analyse de performance comme Intel VTune Profiler et de se familiariser avec quelques cas classiques de parallelisation d'algorithmes.  
Il est intéréssant de remarquer que les parallelisations que nous effectuerons ici ne seront pas optimales. En effet, si nous prennons l'example de la deuxième partie, les multiplications et sommes des matrices peuvent être beaucoup plus otimisés si des librairies spécifiques faites pour ce type de calcul sont utilisés.  

## Partie 1  

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
