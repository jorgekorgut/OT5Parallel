# OT5 : Calcul parallèl

## Membres
Mario CASTILLON  
Jorge KORGUT Junior

## Architecture du projet OpenMP

* Dans la racine nous avons un Makefile qui compile tous les executables des differentes parties.  
>Pour compiler les sources
    ```Make```  


* Les executables sont générés dans le dossier **/Executables** 
* Pour acceder aux codes sources des parties, les dossiers **/Part#** sont disponibles dans la racine du projet. Pour les sources CUDA ils se trouvent sur le dossier **/Cuda**  
* Pour executer les scripts pythons d'analyses, veillez exécuter dans la racine du projet :  

>Pour générer un fichier stats.csv avec les données d'execution des programmes  
    ```python3 ./Analysis/Part1/evaluation.py``` 
    ```python3 ./Analysis/Part2/evaluation.py``` 
    ```python3 ./Analysis/Cuda/evaluation.py```   

>Pour afficher un graphe avec les données de performance  
    ```python3 ./Analysis/Part1/analysis.py```   
    ```python3 ./Analysis/Part2/analysis.py``` 
    ```python3 ./Analysis/Cuda/analysis.py```  

## Lien d'instalation CUDA

>Install cuda SDK
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local


## Introduction OpenMP 

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
    #pragma omp parallel for num_threads(num_cores)
```
Nous spécifions à openMP que nous voudrions un certain numero de threads.

De plus on rajoute les mots clefs :
```c++
    #pragma omp atomic
```
Pour éviter les cas des concurrences qui viennent attaquer nos résultats.

![Alt text](Resources/Part1_Seq_Atomic.png)
_Comparaison du temps d'execution du programme en mode sequentiel et en mode parallèle atomique._ 

Les résultats peuvent paraître un peut surprennant. Cependant, en regardant plus attentivement notre algorithme et le mode de fonctionnement des atomiques sur OpenMP, nous nous rendons compte qu'une bonne partie du code est passé dans la gestion des verrous pour acceder les ressources critiques. Ce qui produit un resultat contre intuitive pour les programmeurs inexperimentés qui ne prennent pas en compte les intéractions des threads entre eux.

Afin d'optimiser ce processus et parce que que notre cas particulier nous permet, nous pouvons diminuer ce temps d'attente entre les verrouiage des données, en utilisant une arbre de réduction. Par chance, juste le changement des pragmas nous permet d'atteidre cet objectif.
```c++
    #pragma omp parallel for reduction(+ : sum) num_threads(num_cores)
```

![Alt text](Resources/Part1_Seq_Reduce.png)
_Comparaison du temps d'execution du programme en mode sequenciel et en mode parallèle en reduction._ 

Un autre point d'optimisation est d'augmenter le temps d'execution de chaque thread afin de diminuer la quantité de branches dans l'arbre de réduction. En effet, si le temps d'execution de chaque thread est inférieur au temps pour faire une réduction, il est intéressant de regrouper les taches par thread afin que l'arbre à la fin soit moins dense. Cependant, il reste difficil d'identifier le temps exacte passé pour une réduction, ainsi un teste empirique a été éffectué pour desmistifier le sujet.

![Alt text](Resources/Part1_Reduce_Divided.png)
_Comparaison du temps d'execution du programme en mode parallel en reduction et en mode parallèle en reduction fractionné._  

En effet, nous n'avons pas constaté une augmentation dans le temps d'execution de l'algoritme en mode reduction et en mode reduction fractionné pour nos paramèttres d'entrée.

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

## Introduction CUDA
Après avoir obtenu des notables gain de performance avec la parallelisation du processeur en utilisant OpenMP, le prochain pas est naturellement de partir vers la paralellisation en masse avec le GPU. Plusieurs marques emergent dans le marché et standards de programmation commencent à ce mettre en place. Cependant, en 2023 la petite communauté des programmeurs reste limité par des langages proprietaires si l'objectif est l'optimisation fine des tâches.

Afin de pouvoir comprendre la suite de ce rapport, quelques notions clefs doivent être clarifiés.

### Threads(Threads)
Un thread c'est une répresentation abstraite de l'execution du kernel. C'est à dire, un bout de code qui a été compilé pour être executé dans un apareil precis.  
```c++
    int unique_id = blockDim.x * blockIdx.x + threadIdx.x;
```  
 
### Bloques(Blocks)
Un bloque répresente un groupe de threads qui seront exécutés soit en serie soit en parallèle. En 2023, le bloque est limité à 1024 Threads. La particularité d'un bloque est que les threads qui tournent sur lui peuvent être à la fois synchronisé et à la fois partager une même memoire plus rapide que la memoire global.


### Grille(Grid)
Une grille est un ensemble de blocks. Le nombre de threqds étant limité, la notion de grille permet la parallelisation massive. Tout simplement ce concept, permet l'utilisation de tous multiprocesseurs et l'illusion d'une infinité de threads.


### Hierarchie en memoire
Lors de la programmation GPU, la metrise de la mémoire devient encore plus critique que dans la programmation classique. Vu la limitation du transfert des données, les gains en performance sont rapidement limités par la nécéssité du transfert des données. Afin de contourner ce problème de transfert, la technologie actuelle (2023) expose dans differents niveaux de capacité, une hiérarchie de mémoire. Celles les plus vites, mais plus chères et plus petites, plus proche de leur zone d'actuation et celles moins couteuses, plus grandes mais plus lentes, plus eloignés.

![Alt text](Resources/memory-hierarchy-in-gpus.png)  
_Illustration de la mémoire dans un GPU avec son hierarchie qui atteint jusqu'à la mémoire global. ref: https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/_  


## Cuda

Le programme qui borne la partie CUDA de ce rapport est celui utilisé auparavant de l'approximation de PI. Dans un premier temps, le code OpenMP est porté vers CUDA. Ce processus peut être facilement enoncable, cependant ne doit pas être une étapa négligeable lors d'une extimation de charge d'un projet d'optimisation.

Les méta-paramètres de la première execution sont : **1 Thread par bloque**, **N Bloques** et chaque Thread s'occupe d'une certaine plage des données.





## References

https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)