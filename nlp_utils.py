"""
nlp_utils.py
------------
Utilitaires NLP partagés entre nettoyage.py (entraînement) et main.py (API).
Le tokenizer doit être importable depuis le même chemin pour que joblib
puisse désérialiser le TfidfVectorizer.
"""

import re
from nltk.stem.snowball import FrenchStemmer

# Stopwords français + termes du domaine redondants avec features numériques
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'l', 'de', 'du', 'des', 'd', 'un', 'une',
    'à', 'au', 'aux', 'et', 'ou', 'mais', 'donc', 'car', 'ni', 'or',
    'ce', 'cette', 'ces', 'cet', 'que', 'qui', 'quoi', 'dont', 'où',
    'son', 'sa', 'ses', 'leur', 'leurs', 'mon', 'ma', 'mes',
    'ton', 'ta', 'tes', 'notre', 'votre', 'nos', 'vos',
    'pour', 'par', 'avec', 'sans', 'sous', 'sur', 'dans', 'en', 'entre', 'vers',
    'est', 'sont', 'être', 'étant', 'a', 'ai', 'as', 'ont', 'avait', 'avaient',
    'était', 'étaient', 'sera', 'seront', 'avoir',
    'pas', 'ne', 'plus', 'moins', 'très', 'aussi', 'bien', 'peu',
    'tout', 'tous', 'toute', 'toutes',
    'on', 'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous',
    'me', 'te', 'se', 'lui',
    'comme', 'si', 'quand', 'puis', 'alors',
    # Termes de mesure (déjà dans features numériques)
    'kg', 'kgs', 'litre', 'litres', 'cm', 'mm', 'm', 'g',
    # Termes neutres très fréquents dans le domaine
    'lot', 'objet', 'site', 'général', 'générale',
}

_STEMMER = FrenchStemmer()
FRENCH_STOPWORDS_STEMMED = {_STEMMER.stem(w) for w in FRENCH_STOPWORDS}

_TOKEN_RE = re.compile(r"\b[a-zàâäéèêëîïôöùûüÿç]{2,}\b")


def french_tokenizer(text: str) -> list:
    """Tokenize, lowercase, strip stopwords, stem en français."""
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    out = []
    for t in tokens:
        if t in FRENCH_STOPWORDS:
            continue
        s = _STEMMER.stem(t)
        if s in FRENCH_STOPWORDS_STEMMED or len(s) < 2:
            continue
        out.append(s)
    return out
