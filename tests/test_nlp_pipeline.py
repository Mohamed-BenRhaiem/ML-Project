"""
Tests du pipeline NLP (tokenizer, stopwords, stemmer).
Catégorie 🔴 IA interdite — rédigés à la main selon la charte.
"""

import pytest
from nlp_utils import french_tokenizer, FRENCH_STOPWORDS


def test_tokenizer_lowercases():
    out = french_tokenizer("PLASTIQUE Rigide")
    # Tous les tokens en minuscules après stemming
    assert all(t == t.lower() for t in out)


def test_tokenizer_removes_stopwords():
    out = french_tokenizer("le plastique de la collecte")
    # 'le', 'de', 'la' doivent disparaître
    assert all('le' != t and 'de' != t and 'la' != t for t in out)


def test_tokenizer_keeps_meaningful_words():
    out = french_tokenizer("plastique rigide opaque")
    # On doit retrouver des stems courts mais reconnaissables
    assert len(out) >= 3


def test_tokenizer_handles_accents():
    out = french_tokenizer("légèrement endommagé")
    # Doit produire quelque chose, pas une liste vide
    assert len(out) > 0


def test_tokenizer_handles_empty_string():
    assert french_tokenizer("") == []


def test_tokenizer_filters_short_tokens():
    out = french_tokenizer("a b c xy plastique")
    # Tokens d'une lettre éliminés (regex {2,})
    assert all(len(t) >= 2 for t in out)


def test_tokenizer_filters_numbers():
    out = french_tokenizer("33 kg de plastique 2.5")
    # Pas de chiffres dans les tokens (regex letters-only)
    assert all(not t[0].isdigit() for t in out)


def test_french_stopwords_contains_basics():
    for w in ('le', 'la', 'les', 'de', 'du', 'et', 'à'):
        assert w in FRENCH_STOPWORDS, f"'{w}' devrait être un stopword"


def test_tokenizer_is_deterministic():
    text = "Lot de plastique récupéré, état correct."
    a = french_tokenizer(text)
    b = french_tokenizer(text)
    assert a == b
