# Posture-guided image synthesis of a person (TP)
### Code réalisé par LIROULET Inès et BOUNOUA Hani (Master 2 IA)

## Dépendances

Pour installer les bibliothèques nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## Exécution

Une fois dans le répertoire **/dance_start/**, exécutez le script **DanceDemo.py**. Lors de l'exécution, vous devrez choisir parmi les quatre options proposées en entrée. Les modèles préentraînés sont déjà disponibles, et leurs poids sont sauvegardés pour une réutilisation directe. Vous aurez la possibilité de décider si vous souhaitez charger ces modèles préentraînés ou réentraîner le modèle concerné. Une fois votre choix effectué, le script se chargera de tout et le processus devrait se dérouler sans problème.

Par défaut, le script est lancé avec la vidéo "taichi2_full.mp4", les autres vidéos étant trop courtes pour bien voir les résultats et la vidéo "karate_full.mp4" présentant de nombreux problèmes d'affichages. Pour changer la vidéo, il suffit d'aller changer le path dans la ligne 131 du fichier DanceDemo.py, tout à la fin.

```bash
python DanceDemo.py
```

## Travail réalisé :

Implémentation et finalisation de trois modèles génératifs :
- **GenNearest** : 
La première méthode, implémentée dans la classe GenNearest, consiste à utiliser la technique du voisin le plus proche. Elle cherche à générer une image à partir d'un squelette en comparant ce dernier à une série de squelettes extraits d'une vidéo cible. Pour ce faire, la méthode calcule la distance entre le squelette d'entrée et ceux de la vidéo cible, puis sélectionne l'image dont le squelette est le plus proche de celui de l'entrée. En d'autres termes, elle choisit l'image qui correspond le mieux à la posture donnée.

- **GenVanillaNN** : 
La seconde méthode utilise un réseau neuronal pour générer des images à partir de squelettes. Le modèle prend un squelette en entrée, soit sous forme de vecteur réduit (26 valeurs), soit sous forme d'image avec un squelette dessiné. Il apprend à associer ces squelettes à des images correspondantes à l'aide d'un entraînement basé sur la comparaison avec des images réelles. Une fois entraîné, le modèle peut générer une image à partir d'un nouveau squelette.

- **GenGan** : 
Le GAN prend des squelettes d'une vidéo et génère des images à partir de ces squelettes. Le modèle est constitué de deux parties : le générateur qui crée des images et le discriminateur qui apprend à différencier les images réelles des images générées. L'entraînement consiste à améliorer ces deux réseaux en alternant leur mise à jour, puis on peut générer des images réalistes à partir de nouveaux squelettes. Afin d'empêcher le mode collapse (où le discriminateur devient trop bon par rapport au générateur, qui ne sait pas comment s'améliorer et finit juste par apprendre à générer une seule image "plausible" de tromper le discriminateur), nous avons intégré de la batch discrimination, où le discriminateur détecte si le générateur produit des images trop similaires au sein d'un batch. Cela permet d'empêcher le générateur de "tromper" le discriminateur avec une image "plausible". L'implémentation de la batch discrimination est inspirée par "Improved Techniques for Training GANs" (2016) de Salimans et al.