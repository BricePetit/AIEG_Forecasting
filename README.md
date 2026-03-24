# AIEG Forecasting

Ce projet permet de prédire la consommation et la production électrique à court terme, puis d'exploiter ces prévisions dans une chaîne complète: entraînement, évaluation, inférence, optimisation sous contrainte métier, et monitoring.

Ce README a été écrit pour des personnes qui ne sont pas forcément expertes en machine learning. L'objectif est que vous puissiez lancer le projet sans deviner les étapes.

## Vue d'ensemble

Le projet suit un flux simple:

1. Préparer les données depuis la base de données vers un fichier H5.
2. Entraîner les modèles (actuellement orienté XGBoost par défaut selon la configuration).
3. Évaluer la qualité des modèles sur les données de test.
4. Générer des prédictions et les sauvegarder en YAML.
5. Optimiser la production pour respecter la contrainte "production <= consommation".
6. Surveiller la performance des prédictions dans le temps.

Les paramètres de ce flux sont centralisés dans les fichiers de configuration YAML dans `src/configs/global/`.

## Prérequis

Vous avez besoin de:

- Python 3.10+
- Un environnement virtuel Python
- Les dépendances du fichier `requirements.txt`
- Un accès à la source de données (si vous lancez la pipeline de données)

## Installation pas à pas

Clonez le dépôt puis placez-vous dans le dossier du projet:

```bash
git clone https://github.com/BricePetit/AIEG_Forecasting.git
cd AIEG_Forecasting
```

Créez et activez un environnement virtuel:

```bash
python -m venv .venv
source .venv/bin/activate
```

Installez les dépendances:

```bash
pip install -r requirements.txt
```

## Organisation du projet

Les éléments les plus importants pour l'exécution sont:

- `src/pipelines/`: les pipelines exécutables.
- `src/configs/global/`: la configuration (modèle, données, chemins, domaine).
- `data/`: les données brutes et traitées.
- `src/configs/features/`: les features sélectionnées sauvegardées pendant l'entraînement.
- `src/configs/predictions/`: les prédictions YAML sauvegardées par l'inférence.

## Lancement rapide (ordre recommandé)

Les commandes ci-dessous sont les commandes de référence pour lancer le projet correctement.

### 1) Pipeline de données

Cette étape télécharge les données et construit le fichier H5 utilisé ensuite par l'entraînement, l'évaluation et l'inférence.

```bash
PYTHONPATH=src python src/pipelines/data_pipeline.py
```

### 2) Pipeline d'entraînement

Cette étape entraîne les modèles pour la consommation et la production sur tous les groupes de sites définis dans la configuration.

```bash
PYTHONPATH=src python src/pipelines/training_pipeline.py
```

Les modèles XGBoost sont sauvegardés dans le dossier configuré par `saved_models_dir`.

### 3) Pipeline d'évaluation

Cette étape calcule les métriques sur le split de test (par défaut sur tous les sites et pour consommation + production).

```bash
PYTHONPATH=src python src/pipelines/evaluation_pipeline.py
```

### 4) Pipeline d'inférence

Cette étape génère les prédictions à partir des modèles entraînés et les sauvegarde en YAML.

```bash
PYTHONPATH=src python src/pipelines/inference_pipeline.py
```

Sortie attendue des prédictions:

- `src/configs/predictions/consumption/site_<id>.yaml`
- `src/configs/predictions/production/site_<id>.yaml`

### 5) Pipeline d'optimisation

Cette étape prend les prédictions de consommation et de production, puis résout un problème d'optimisation pour maximiser la production tout en respectant la contrainte métier: la production ne doit pas dépasser la consommation.

Si la contrainte est violée, la pipeline identifie les sites à désactiver pour respecter la contrainte avec un maximum de production.

```bash
PYTHONPATH=src python src/pipelines/optimization_pipeline.py
```

Le résultat global est sauvegardé dans:

- `src/configs/optimization_results.yaml`

### 6) Pipeline de monitoring

Cette étape suit la performance des prédictions dans le temps en comparant les prédictions avec les valeurs observées (lorsqu'elles sont disponibles), et remonte des alertes de dégradation.

```bash
PYTHONPATH=src python src/pipelines/monitoring_pipeline.py
```

Les rapports de monitoring sont sauvegardés dans:

- `src/configs/monitoring_reports/`

## Comprendre la différence entre évaluation, inférence et monitoring

L'évaluation sert à mesurer la qualité d'un modèle sur un jeu de test connu juste après entraînement.

L'inférence sert à produire des prédictions à utiliser par la suite (par exemple pour l'optimisation).

Le monitoring sert à vérifier en continu que la qualité reste bonne en production, une fois que les valeurs réelles deviennent disponibles.

## En cas de problème

Si une pipeline échoue, vérifiez dans cet ordre:

1. L'environnement Python est bien activé.
2. Les dépendances sont installées.
3. Le préfixe `PYTHONPATH=src` est bien présent dans la commande.
4. Les chemins de `src/configs/global/paths.yaml` sont corrects.
5. Les données H5 existent (pipeline de données lancée avant entraînement/évaluation/inférence).
6. Les modèles existent avant l'inférence et l'évaluation (pipeline d'entraînement déjà lancée).

## Exemple de session complète

Pour lancer un cycle complet depuis zéro:

```bash
PYTHONPATH=src python src/pipelines/data_pipeline.py
PYTHONPATH=src python src/pipelines/training_pipeline.py
PYTHONPATH=src python src/pipelines/evaluation_pipeline.py
PYTHONPATH=src python src/pipelines/inference_pipeline.py
PYTHONPATH=src python src/pipelines/optimization_pipeline.py
PYTHONPATH=src python src/pipelines/monitoring_pipeline.py
```

## Licence

Le projet est distribué sous licence MIT. Voir le fichier `LICENSE`.
