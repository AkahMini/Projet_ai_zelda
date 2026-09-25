# Zelda AI — Reinforcement Learning

![Status](https://img.shields.io/badge/status-in%20development-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)
![GPU](https://img.shields.io/badge/GPU-RTX%205070-76B900)

Projet personnel d'expérimentation en **intelligence artificielle**, ayant pour objectif d'étudier la possibilité d'entraîner progressivement un agent capable d'interagir avec **The Legend of Zelda: Ocarina of Time** grâce à l'**apprentissage par renforcement (Reinforcement Learning)**.

> **Projet expérimental et éducatif — en développement**
>
> L'objectif initial n'est pas de faire terminer immédiatement le jeu à l'IA. Le projet consiste à construire progressivement les différentes briques nécessaires : environnement de Reinforcement Learning, déplacement, combat, perception visuelle et interaction avec le jeu.

---

## Objectif

Le projet vise à étudier comment un agent d'intelligence artificielle peut apprendre à interagir avec un environnement de jeu vidéo 3D à travers ses observations, ses actions et les récompenses qu'il reçoit.

La progression prévue est :

```text
Environnement RL simple
        ↓
Déplacement
        ↓
Obstacles
        ↓
Combat
        ↓
Perception par image
        ↓
Interaction avec Ocarina of Time
        ↓
Exploration
        ↓
Objectifs plus complexes
```

À terme, l'objectif expérimental sera d'étudier jusqu'où un agent peut progresser dans *Ocarina of Time* en apprenant progressivement son comportement à partir de ses interactions avec l'environnement.

---

## Technologies utilisées

### Langage
- Python 3.12

### Intelligence artificielle
- PyTorch
- Gymnasium
- Stable-Baselines3

### Vision et traitement des données
- NumPy
- OpenCV
- Pillow
- Matplotlib

### Développement
- Visual Studio Code
- Git
- GitHub
- Jupyter

### Accélération matérielle
- NVIDIA GeForce RTX 5070
- CUDA 13.0

---

## Architecture du projet

```text
Projet_ai_zelda/
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── agent.py
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   └── zelda_env.py
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   └── screen.py
│   │
│   ├── control/
│   │   ├── __init__.py
│   │   └── controller.py
│   │
│   └── rewards/
│       ├── __init__.py
│       └── reward.py
│
├── experiments/
├── logs/
├── models/
├── notebooks/
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Rôle des principaux dossiers

| Dossier | Description |
|---|---|
| `src/agents` | Agents d'apprentissage par renforcement et logique d'entraînement |
| `src/environment` | Environnements Gymnasium utilisés pour les expériences |
| `src/vision` | Capture, traitement et analyse des images |
| `src/control` | Gestion des commandes envoyées au jeu |
| `src/rewards` | Définition et calcul des récompenses |
| `experiments` | Scripts et expérimentations ponctuelles |
| `notebooks` | Expérimentations interactives, visualisations et analyses |
| `models` | Modèles entraînés |
| `logs` | Données et journaux d'entraînement |
| `tests` | Tests automatisés |

---

## Avancement du projet

- [x] **[1]** Configuration
- [x] **[2]** Premier environnement Gymnasium
- [ ] **[3]** Agent capable d'atteindre une cible / Déplacement
- [ ] **[4]** Ajout d'obstacles
- [ ] **[5]** Ajout d'un ennemi
- [ ] **[6]** Système de combat
- [ ] **[7]** Entraînement avec PPO / Expérimentation avec DQN
- [ ] **[8]** Utilisation d'observations visuelles (perception visuelle)
- [ ] **[9]** Capture du jeu
- [ ] **[10]** Contrôle du jeu
- [ ] **[11]** Premier prototype sur *Ocarina of Time*
- [ ] **[12]** Exploration d'une zone
- [ ] **[13]** Objectifs plus complexes

Cette roadmap pourra évoluer en fonction des résultats obtenus.

**Expériences versionnées :**
- `v0.1` — Premier environnement Gymnasium

---

## Méthodologie

Le projet suit une progression incrémentale afin de comprendre et valider chaque composant avant de passer au suivant (voir [Avancement du projet](#avancement-du-projet) pour le détail des étapes).

### 1. Environnement contrôlé

Création d'un environnement 2D minimal dans lequel l'agent doit atteindre une cible.

```text
┌───────────────────────┐
│                       │
│  L                   │
│                       │
│             O        │
│                       │
└───────────────────────┘
```

L'agent doit apprendre à se déplacer jusqu'à la cible grâce aux récompenses fournies par l'environnement.

### 2. Reinforcement Learning

Introduction progressive des concepts fondamentaux :

- agent ;
- environnement ;
- observation ;
- action ;
- récompense ;
- épisode ;
- policy ;
- value function.

La boucle générale du projet est :

```text
Observation
     ↓
   Agent
     ↓
   Action
     ↓
Environment
     ↓
Reward
     ↓
Nouvelle observation
     ↓
Agent
```

### 3. Complexification de l'environnement

L'environnement sera progressivement enrichi avec :

- obstacles ;
- ennemis ;
- points de vie ;
- attaques ;
- objectifs ;
- exploration ;
- plusieurs états possibles.

L'objectif est de faire évoluer progressivement la difficulté sans changer complètement l'architecture du projet.

### 4. Perception visuelle

Dans les premières expériences, l'agent aura accès directement à des informations simplifiées sur l'environnement. Par la suite, ces informations pourront être remplacées par des observations visuelles.

```text
Capture d'écran
      ↓
Prétraitement
      ↓
CNN
      ↓
Représentation de l'état
      ↓
Agent RL
      ↓
Action
```

Les traitements envisagés comprennent :

- capture d'écran ;
- redimensionnement ;
- normalisation ;
- traitement d'images ;
- extraction de caractéristiques ;
- réseaux convolutionnels (CNN) ;
- détection d'éléments utiles à l'agent.

Cette étape permettra d'étudier l'association entre Computer Vision et Reinforcement Learning. Les expérimentations liées à ces traitements pourront être réalisées dans les notebooks présents dans `notebooks/`.

### 5. Interaction avec le jeu

Une fois les composants précédents validés, une boucle d'interaction avec *Ocarina of Time* sera mise en place :

```text
┌─────────────────────┐
│      Ocarina        │
│       of Time       │
└──────────┬──────────┘
           │
           ▼
      Capture écran
           │
           ▼
       Observation
           │
           ▼
      Agent IA / RL
           │
           ▼
         Action
           │
           ▼
       Contrôleur
           │
           └──────────────► Ocarina of Time
```

---

## Algorithmes de RL étudiés

Les premiers essais utiliseront principalement des algorithmes disponibles dans Stable-Baselines3.

### DQN

Le **Deep Q-Network** sera étudié pour des espaces d'actions discrets.

Exemple :

```text
0 = aucune action
1 = avancer
2 = reculer
3 = gauche
4 = droite
5 = attaque
```

### PPO

Le **Proximal Policy Optimization** sera également étudié et utilisé pour certains environnements.

Les performances et comportements pourront être comparés en fonction des différents environnements.

---

## Évaluation des agents

Les différents agents pourront être évalués à l'aide de plusieurs indicateurs :

- récompense moyenne ;
- taux de réussite ;
- nombre moyen d'actions ;
- durée moyenne d'un épisode ;
- nombre de morts ;
- progression au cours de l'entraînement ;
- temps d'entraînement ;
- comparaison entre algorithmes.

Des graphiques seront ajoutés progressivement afin de visualiser les performances des différents modèles.

---

## 🛠️ Installation

### Prérequis

- Windows
- Python 3.12
- Git
- GPU NVIDIA recommandé pour les expériences utilisant des réseaux neuronaux importants

### Cloner le projet

```bash
git clone <URL_DU_REPOSITORY>
cd Projet_ai_zelda
```

### Créer l'environnement virtuel

```bash
py -3.12 -m venv .venv
```

### Activer l'environnement sous PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### Vérification de PyTorch et CUDA

Le projet utilise PyTorch avec CUDA afin d'exécuter les calculs sur le GPU.

Pour vérifier la configuration :

```python
import torch

print("PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print("CUDA :", torch.version.cuda)
```

Configuration de développement actuelle :

- **GPU** : NVIDIA GeForce RTX 5070
- **PyTorch** : 2.14.0+cu130
- **CUDA** : 13.0

---

## Reproductibilité

Chaque étape importante du développement sera versionnée avec Git.

Le dépôt contient notamment :

- le code source ;
- les expériences ;
- les notebooks ;
- la configuration ;
- la liste des dépendances ;
- les résultats utiles ;
- la documentation.

Les environnements virtuels et fichiers temporaires ne sont pas versionnés.

---

## Notebooks

Les notebooks sont utilisés principalement pour :

- expérimenter rapidement avec PyTorch ;
- tester des fonctionnalités ;
- visualiser les données ;
- analyser les entraînements ;
- comparer les performances ;
- expérimenter avec la vision par ordinateur.

Les fonctionnalités principales du projet restent dans `src/`.

Exemple :

```text
notebooks/
├── 01_pytorch.ipynb
├── 02_gymnasium.ipynb
├── 03_training.ipynb
├── 04_reward_analysis.ipynb
└── 05_vision.ipynb
```

---

## Git et versionnement

Le projet utilise Git afin de versionner chaque étape importante du développement.

Exemples de messages de commit :

```text
feat: créer le premier environnement RL
feat: ajouter l'agent PPO
feat: ajouter le système de récompenses
feat: ajouter les observations visuelles
fix: corriger le calcul de la récompense
docs: mettre à jour la documentation
refactor: restructurer le système de contrôle
```

Les gros fichiers générés par l'entraînement pourront être gérés avec Git LFS lorsque cela sera nécessaire.

---

## Objectifs pédagogiques

Ce projet est principalement destiné à approfondir plusieurs domaines :

- Reinforcement Learning ;
- Deep Learning ;
- Computer Vision ;
- réseaux de neurones ;
- optimisation ;
- probabilités ;
- traitement de données ;
- conception logicielle ;
- expérimentation scientifique ;
- utilisation de GPU pour le Machine Learning.

Il constitue également un projet personnel permettant de mettre en pratique des connaissances en mathématiques et informatique.

---

## À propos de The Legend of Zelda

*The Legend of Zelda* et *Ocarina of Time* sont des propriétés intellectuelles de leurs détenteurs respectifs.

Ce dépôt constitue un projet personnel et éducatif consacré à l'expérimentation technique autour de l'intelligence artificielle.

Ce projet n'est pas affilié à Nintendo et ne prétend à aucun titre ou contenu appartenant à Nintendo.

---

## Auteur

**Yann Le Ray**

Étudiant en L3, double licence mathématiques et informatiques.
Projet personnel consacré à l'apprentissage automatique, au Reinforcement Learning et à l'expérimentation d'agents autonomes dans des environnements de jeux vidéo.

---
