# FF Extraction Tool

Application web pour l'analyse des Faisant Fonction (FF) PNC à partir des PDFs Indicateurs.

## Fonctionnalités

1. **Upload PDF** - Indicateurs PNC mensuel
2. **Extraction OCR** - Multi-moteur (Mistral + Tesseract + EasyOCR) avec vote
3. **Récupération Visual** - Connexion directe au portail Visual pour les données de vol
4. **Détection FF** - Comparaison grade opéré vs grade normal
5. **Matching** - Association code équipage (PDF) → trigramme (Visual)
6. **Export CSV** - Rapport complet téléchargeable

## Installation locale

```bash
# 1. Cloner/créer le répertoire
mkdir ff-extraction && cd ff-extraction

# 2. Installer les dépendances système
# macOS:
brew install poppler tesseract

# Ubuntu:
# sudo apt-get install poppler-utils tesseract-ocr

# 3. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 4. Installer les dépendances Python
pip install -r requirements.txt

# 5. Lancer l'app
streamlit run ff_app.py
```

## Déploiement sur Render

### Option 1: Web Service

1. Créer un repo GitHub avec `ff_app.py` et `requirements.txt`

2. Sur Render.com:
   - New → Web Service
   - Connect repo
   - Settings:
     - Build Command: `pip install -r requirements.txt && apt-get update && apt-get install -y poppler-utils tesseract-ocr`
     - Start Command: `streamlit run ff_app.py --server.port $PORT --server.address 0.0.0.0`
   - Environment Variables:
     - `MISTRAL_API_KEY` = votre clé

3. Deploy!

### Option 2: Docker (recommandé)

Créer un `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ff_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "ff_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Sur Render:
- New → Web Service
- Docker
- Connect repo
- Environment: `MISTRAL_API_KEY`

## Utilisation

1. **Sidebar gauche:**
   - Entrer la clé API Mistral (optionnel mais recommandé)
   - Entrer les identifiants Visual Portal
   - Uploader le fichier Crew_list.csv

2. **Zone principale:**
   - Uploader le PDF Indicateurs PNC
   - Cliquer "Extraire les données FF"
   - Une fois le mois détecté, cliquer "Récupérer les données Visual"
   - Consulter les résultats et télécharger le CSV

## Format Crew_list.csv

Le fichier CSV doit contenir au minimum:
- `trigram` ou `Trigramme`: Code 3 lettres
- `function` ou `Statut`: Grade (PU, CC, HS ou texte complet)
- `first_name` ou `Prénom`
- `last_name` ou `Nom`

Exemple:
```csv
Trigramme;Nom;Prénom;Statut
ABC;DUPONT;Marie;CHEF DE CABINE
XYZ;MARTIN;Jean;PERSONNEL NAVIGANT COMMERCIAL
```

## Sécurité

- Les identifiants Visual ne sont jamais stockés
- Utilisés uniquement pour la session de fetch en cours
- L'app ne conserve aucune donnée entre les sessions

## Coûts

- **Mistral OCR**: ~$0.003 par page (~$0.04/an pour 12 rapports)
- **Hébergement Render**: Gratuit (tier free) ou $7/mois (tier starter)

## Support

Pour toute question ou bug, contacter Arthur.
