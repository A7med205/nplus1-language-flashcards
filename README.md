# nplus1-language-flashcards

## Overview

This open project aims to provide **3,000 frequency-based vocabulary flashcards** for language learners for multiple languages, built on the **n+1 principle**:

> Each new sentence should introduce only *one* new unknown word compared to the learner’s current vocabulary.

### Features

* Covers **10 languages**:

  * English, French, Mandarin Chinese, Spanish, German, Italian, Japanese, Russian, Portuguese, Korean
* Cards contain words, definitions, example sentences, translations and audio
* Example sentences are generated via language models followed by human review [Read More](generator/README.md)
* **~3K curated cards per language**
* Repository includes corpora, frequency lists, and flashcard decks
* At a pace of **25 new cards/day**, a beginner can finish a full deck in \~4 months

---

## Roadmap

1. **Collect corpora:** Public sentence banks such as Tatoeba and OpenSubtitles are sourced for each language.
2. **Build lemma frequency lists:** The most frequent lemmas are extracted from the dataset sine Stanza (a context aware NLP model) and sorted by frequency, and word information (POS, definitions, conjugations, articles, plural form, etc) for each word are pulled from Wiktionary/standard sources.
3. **Generate sentences:** For each word sense, a sentence that contains only preveuously learned words is either sourced from the sentence bank or machine generated.
4. **Generate translations, audio and properly formatted flashcards**.
5. **Human review:** The quality/naturalness of each example sentence is reviewed by a native speaker.

| Language         | 1 | 2 | 3 | 4 | 5 |
| ---------------- | - | - | - | - | - |
| English          | ✅ | 50% | ❌ | ❌ | ❌ |
| French           | ✅ | 50% | ❌ | ❌ | ❌ |
| Mandarin Chinese | ✅ | 50% | ❌ | ❌ | ❌ |
| Spanish          | ✅ | ✅  | ✅ | ✅ | ✅ |
| German           | ✅ | 50% | ❌ | ❌ | ❌ |
| Italian          | ✅ | 50% | ❌ | ❌ | ❌ |
| Japanese         | ✅ | 50% | ❌ | ❌ | ❌ |
| Russian          | ✅ | 50% | ❌ | ❌ | ❌ |
| Portuguese       | ✅ | 50% | ❌ | ❌ | ❌ |
| Korean           | ✅ | 50% | ❌ | ❌ | ❌ |
---

## Flashcard Format

Each flashcard is **HTML-based** (Anki-compatible).

### Example: **German (Deutsch)**

**Front:**

<p align="center">
  <img src="de_front.svg" alt="das Haus example" />
</p>


**Back:**

<p align="center">
  <img src="de_back.svg" alt="das Haus example" />
</p>

---

### Example: **Japanese (日本語)**

**Front:**

<p align="center">
  <img src="ja_front.svg" alt="das Haus example" />
</p>

**Back:**

<p align="center">
  <img src="ja_back.svg" alt="das Haus example" />
</p>

---

## Repository Contents

* 📂 **anki_decks** – Finished language decks
* 📂 **generator** – Script used to generate sentences 
* 📂 **corpora** – raw sentence collections for each language
* 📂 **frequency_lists** – lemmatized, frequency-ranked vocabulary with example sentences if present

---
