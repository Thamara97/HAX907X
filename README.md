# TP d'aprentissage statistique (HAX907X)

Ce dêpot regroupe les comptes rendus des TPs de l'UE HAX907X (apprentissage statistique) réalisés par **RENOIR Thamara**.

## TP n°2 : Arbres

Ce TP porte sur la création d'arbres de décisions.

Ce dossier contient les fichiers suivants :

* `tp_arbre_source.py` qui contient les codes sources de certaines des fonctions utilisées
* `tp_arbre_script.py` qui contient les codes python utilisés pour réaliser ce TP
* `tp_compte_rendu.qmd` qui contient la rédaction du TP
* `requirements.txt` qui contient les noms des packages nécessaires pour ce TP

Afin de pouvoir compiler le fichier `.qmd` au format `.pdf` vous devez avoir installé Quarto et les packages cités dans le fichier `requirements.txt` via la commande :

```sh
$ pip install -r requirements.txt
```

Vous pouvez ensuite compiler le fichier avec la commande :

```sh
$ quarto render tp_compte_rendu.qmd
```