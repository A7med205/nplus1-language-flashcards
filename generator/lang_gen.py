#!/usr/bin/env python3
"""
Language Learning List Generator

Implements a pipeline to:
1) Locate the next unfinished lemma (after skipping the first 300 entries) in a TSV-like frequency list file.
   The file columns are generally: lemma, part_of_speech, definition, sentence (tabs between columns).
2) Use an OpenAI model to generate 15 different 5-8 word sentences containing that lemma.
   The prompt includes the lemma's part of speech and definition. If the POS is a verb, it mentions that any
   inflection of the verb is allowed.
3) Lemmatize those sentences with Stanza, removing punctuation.
4) Select the sentence that introduces the fewest new lemmas relative to those present before the current lemma.
5) Apply results directly to the SAME input file (edit-in-place), appending the sentence to the lemma's line.
   If new lemmas are needed:
     - If they ALREADY exist later in the list, move them before the current lemma (and remove duplicates later).
     - If they are OUTSIDE the list, INSERT them immediately before the current lemma as single-column entries
       (lemma only, without POS/definition/sentence). These single-column entries are considered "no sentence needed"
       and are skipped by generation in future steps, but they are used normally when checking new words.

Notes:
- Input list files are TSV-like text files. Lines can be:
    - Single column: "lemma" (single-column entries have no POS/definition/sentence and are skipped for generation)
    - Four columns: "lemma\tpart_of_speech\tdefinition\tsentence"
- The original file IS modified in place (no new copy per step).
- This script depends on:
    - openai (>=1.0.0): pip install openai
    - stanza: pip install stanza
  Stanza will attempt to download the English models on first run if not present.

CLI:
    python lang_gen.py --steps 3 --file 1.txt --lang en --model gpt-5-mini

Public entrypoint:
    generate_language_learning_list(steps: int, filename: str, skip_count: int = 300, language: str = "en", model: str = "gpt-5-mini")

Environment:
    Requires OPENAI_API_KEY set for LLM calls. A fallback generator is provided if the API call fails.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
try:
    import stanza  # type: ignore
except Exception:  # pragma: no cover
    stanza = None  # Will handle at runtime


# --------------- Data Structures ---------------

@dataclass
class ListEntry:
    lemma: str
    pos: Optional[str]
    definition: Optional[str]
    sentence: Optional[str]  # None if not yet filled
    # True if this line should be rendered as a single-column lemma only (no POS/def/sentence).
    # Used for out-of-list insertions and for preserving original single-column lines.
    is_minimal: bool = False


@dataclass
class LemmatizedSentence:
    original: str
    cleaned: str
    tokens: List[str]      # tokens after cleaning
    lemmas: List[str]      # stanza-lemmas corresponding to tokens


# --------------- File Utilities ---------------

def parse_list_file(path: str) -> List[ListEntry]:
    """
    Parse a TSV-like list file. Each non-blank line is one entry with either:
      - single column: lemma
      - four columns: lemma, part_of_speech, definition, sentence (sentence may be empty)
    Extra columns beyond the first four are ignored.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"List file not found: {path}")

    entries: List[ListEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                lemma = parts[0].strip()
                if not lemma:
                    continue
                entries.append(
                    ListEntry(
                        lemma=lemma,
                        pos=None,
                        definition=None,
                        sentence=None,
                        is_minimal=True,
                    )
                )
                continue

            # 2+ columns: treat as 4-column shape; sentence may be empty
            lemma = parts[0].strip() if len(parts) >= 1 else ""
            pos = parts[1].strip() if len(parts) >= 2 else ""
            definition = parts[2].strip() if len(parts) >= 3 else ""
            sentence = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else None

            if not lemma:
                continue

            entries.append(
                ListEntry(
                    lemma=lemma,
                    pos=pos or None,
                    definition=definition or None,
                    sentence=sentence if sentence else None,
                    is_minimal=False,
                )
            )
    return entries


def write_list_file(entries: Sequence[ListEntry], path: str) -> None:
    """
    Write entries back to a TSV-like file.
      - If is_minimal is True and no POS/definition/sentence are present, write single-column: "lemma"
      - Otherwise, write 4 columns: lemma<TAB>part_of_speech<TAB>definition<TAB>sentence
    """
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            if e.is_minimal and not e.pos and not e.definition and not e.sentence:
                f.write(f"{e.lemma}\n")
            else:
                lemma = e.lemma if e.lemma is not None else ""
                pos = e.pos if e.pos is not None else ""
                definition = e.definition if e.definition is not None else ""
                sentence = e.sentence if e.sentence is not None else ""
                f.write(f"{lemma}\t{pos}\t{definition}\t{sentence}\n")


def get_next_filename(current_filename: str) -> str:
    """
    Increment the leading integer of the filename, preserving the extension if present.
    Examples:
        1.txt -> 2.txt
        2.text -> 3.text
        99 -> 100.txt (default to .txt if no extension)

    Note: retained for compatibility, but no longer used since we edit in place.
    """
    base = os.path.basename(current_filename)
    m = re.match(r"^(\d+)(\.[^.]+)?$", base)
    if not m:
        # Fallback: if the name doesn't match, append or increment a suffix
        name, ext = os.path.splitext(base)
        if name.isdigit():
            nxt = str(int(name) + 1)
            return nxt + (ext if ext else ".txt")
        # No leading digits at all:
        return base + ".next"
    number, ext = m.group(1), m.group(2)
    nxt = str(int(number) + 1)
    return nxt + (ext if ext else ".txt")


# --------------- Step 1: Find next unfinished lemma ---------------

def find_next_undone_lemma(entries: Sequence[ListEntry], skip_count: int = 300) -> Tuple[int, str]:
    """
    Scan entries starting after 'skip_count' lemmas. Return index and lemma of the first entry
    that still needs a generated sentence.
    - Entries marked as minimal (single-column; no POS/definition) are considered 'no sentence needed' and skipped.
    Raises ValueError if none found.
    """
    if skip_count < 0:
        skip_count = 0
    start = min(skip_count, len(entries))
    for i in range(start, len(entries)):
        e = entries[i]
        # Skip minimal entries: no generation required
        if e.is_minimal:
            continue
        if e.sentence is None:
            return i, e.lemma
    raise ValueError("No unfinished lemma found after the skip region.")


# --------------- Step 2: LLM sentence generation ---------------

def _language_label(lang: str) -> str:
    """
    Map language code or name to a human-readable label for prompts.
    """
    if not lang:
        return "English"
    code = lang.strip().lower()
    mapping = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "tr": "Turkish",
    }
    return mapping.get(code, code.title())

def call_openai_generate_sentences(
    lemma: str,
    pos: Optional[str] = None,
    definition: Optional[str] = None,
    language: str = "en",
    model: str = "gpt-5-mini",
) -> List[str]:
    """
    Call OpenAI Responses API to generate 15 different sentences containing 'lemma'.
    Each sentence must be between 5 and 8 words (inclusive).
    The prompt includes part_of_speech and definition. If the lemma is a verb, it mentions
    that any inflection of the verb is allowed.

    Returns a list of 15 strings.

    If the API call fails for any reason, a deterministic fallback generator is used.
    """
    word_count = random.randint(5, 8)

    # Construct prompt details
    pos_lower = (pos or "").strip().lower()
    verb_note = ""
    if pos_lower in {"verb", "v", "vb", "v.", "verbal"}:
        verb_note = " Any inflection of the verb is allowed."

    pos_text = f"Part of speech: {pos}." if pos else "Part of speech: (unspecified)."
    def_text = f"Definition: {definition}." if definition else "Definition: (unspecified)."

    developer_text = "You're an expert AI language assistant. Produce JSON as specified."
    user_text = (
        f"Generate 10 simple 5-word or more sentences in {_language_label(language)} that each include the lemma '{lemma}' "
        f"in one of its valid forms. {pos_text} {def_text}{verb_note} "
        "Keep sentences natural, common, and educationally useful. "
        "Return only the sentences in the JSON fields as specified."
    )

    total = 10

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": developer_text}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text}
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "fifteen_sentences",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            **{
                                f"sentence{i}": {
                                    "type": "string",
                                    "description": f"The sentence #{i}."
                                }
                                for i in range(1, total + 1)
                            }
                        },
                        "required": [f"sentence{i}" for i in range(1, total + 1)],
                        "additionalProperties": False,
                    },
                },
                "verbosity": "medium",
            },
            reasoning={"effort": "medium", "summary": "auto"},
            tools=[],
            store=True,
            include=[
                "reasoning.encrypted_content",
                "web_search_call.action.sources",
            ],
        )

        # Extract JSON text from response
        json_str: Optional[str] = None
        if hasattr(response, "output_text") and isinstance(response.output_text, str):
            json_str = response.output_text.strip()
        if not json_str:
            candidate = getattr(response, "output", None)
            if candidate and isinstance(candidate, str):
                json_str = candidate.strip()
        if not json_str:
            json_str = str(response).strip()

        json_match = re.search(r"\{.*\}", json_str, flags=re.S)
        if not json_match:
            raise ValueError("Failed to locate JSON object in LLM response.")
        data = json.loads(json_match.group(0))

        sentences = [data[f"sentence{i}"] for i in range(1, total + 1)]
        # Basic normalization and deduplication while preserving order
        normed = []
        seen = set()
        for s in sentences:
            s_clean = " ".join(str(s).strip().split())
            if s_clean and s_clean.lower() not in seen:
                normed.append(s_clean)
                seen.add(s_clean.lower())

        # Ensure exactly total items: if fewer due to deduping, pad with simple variants
        while len(normed) < total:
            normed.append(_fallback_sentence(lemma, word_count, idx=len(normed) + 1))

        return normed[:total]

    except Exception as e:
        # Fallback deterministic generator (no external calls)
        sys.stderr.write(f"[WARN] OpenAI call failed or unavailable: {e}\n")
        wc = word_count
        return [_fallback_sentence(lemma, wc, idx=i) for i in range(1, total + 1)]


def _fallback_sentence(lemma: str, word_count: int, idx: int) -> str:
    """
    Generate a simple deterministic sentence with the target lemma to serve as a fallback.
    """
    base_words = [
        "today", "people", "often", "quickly", "learn", "new", "ideas", "through", "simple", "practice",
        "we", "can", "easily", "build", "useful", "skills", "together", "by", "trying", "examples",
    ]
    rng = random.Random(idx * 7919)
    words = []
    # Guarantee the lemma appears exactly once
    insert_pos = rng.randint(0, max(0, word_count - 1))
    for i in range(word_count):
        if i == insert_pos:
            words.append(lemma)
        else:
            words.append(rng.choice(base_words))
    # Capitalize first and add period for readability
    sent = " ".join(words)
    return sent[0].upper() + sent[1:] + "."


# --------------- Step 3: Stanza Lemmatization ---------------

_STANZA_PIPELINE_CACHE: Dict[str, "stanza.Pipeline"] = {}  # type: ignore


def _ensure_stanza_pipeline(language: str = "en") -> "stanza.Pipeline":  # type: ignore
    """
    Initialize and cache a stanza pipeline for the given language.
    Downloads models if not present.
    """
    if stanza is None:
        raise RuntimeError(
            "stanza is not installed. Please run: pip install stanza"
        )
    if language in _STANZA_PIPELINE_CACHE:
        return _STANZA_PIPELINE_CACHE[language]
    try:
        # Try to build directly
        nlp = stanza.Pipeline(lang=language, processors="tokenize,mwt,pos,lemma", tokenize_pretokenized=False, verbose=False)
    except Exception:
        # Attempt download then build
        stanza.download(language, processors="tokenize,mwt,pos,lemma", verbose=False)
        nlp = stanza.Pipeline(lang=language, processors="tokenize,mwt,pos,lemma", tokenize_pretokenized=False, verbose=False)
    _STANZA_PIPELINE_CACHE[language] = nlp
    return nlp


_PUNCT_REGEX = re.compile(r"[^\w\s]")


def _remove_punctuation(text: str) -> str:
    """
    Remove punctuation from the text. Keep whitespace and word characters (letters, digits, underscore).
    """
    # Remove punctuation and normalize whitespace
    no_punct = _PUNCT_REGEX.sub(" ", text)
    return " ".join(no_punct.split())


def lemmatize_sentences_stanza(sentences: Sequence[str], language: str = "en") -> List[LemmatizedSentence]:
    """
    Remove punctuation, run stanza lemmatizer, and return tokens and lemmas for each sentence.
    """
    nlp = _ensure_stanza_pipeline(language)
    results: List[LemmatizedSentence] = []
    for s in sentences:
        cleaned = _remove_punctuation(s)
        if not cleaned.strip():
            results.append(LemmatizedSentence(original=s, cleaned="", tokens=[], lemmas=[]))
            continue
        doc = nlp(cleaned)
        toks: List[str] = []
        lems: List[str] = []
        for sent in doc.sentences:
            for w in sent.words:
                # Keep only tokens that are non-empty
                token_text = (w.text or "").strip()
                lemma_text = (w.lemma or "").strip()
                if token_text:
                    toks.append(token_text)
                    lems.append(lemma_text)
        results.append(LemmatizedSentence(original=s, cleaned=cleaned, tokens=toks, lemmas=lems))
    return results


# --------------- Step 4: Sentence selection and list update ---------------

def choose_best_sentence(
    target_lemma: str,
    lemmatized: Sequence[LemmatizedSentence],
    prev_lemmas: Sequence[str],
) -> Tuple[Optional[LemmatizedSentence], Dict[str, any]]:
    """
    From the provided lemmatized sentences:
     - Filter those whose lemmas contain the target lemma (case-insensitive)
     - Score each by the count of lemmas not present in prev_lemmas
     - Choose one with fewest new lemmas; tie-breaker: fewer tokens; then original order.

    Returns (chosen_sentence_or_None, debug_info)
    """
    prev_set = {l.lower() for l in prev_lemmas}
    target = target_lemma.lower()

    candidates: List[Tuple[LemmatizedSentence, int, int, List[str]]] = []
    for idx, ls in enumerate(lemmatized):
        lemmas_lower = [l.lower() for l in ls.lemmas if l]
        if target not in lemmas_lower:
            continue
        unknown = sorted({l for l in lemmas_lower if l not in prev_set})
        unknown_count = len(unknown)
        candidates.append((ls, unknown_count, len(ls.tokens), unknown))

    if not candidates:
        return None, {"reason": "no_sentence_contains_target_lemma"}

    # Choose by: fewest unknowns, then fewest tokens, then earliest
    candidates.sort(key=lambda t: (t[1], t[2]))
    best, unk_count, tok_count, unknown_list = candidates[0]
    debug = {
        "unknown_count": unk_count,
        "token_count": tok_count,
        "unknown_list": unknown_list,
        "candidate_count": len(candidates),
    }
    return best, debug


def apply_sentence_and_reorder(
    entries: List[ListEntry],
    current_index: int,
    chosen: LemmatizedSentence,
    unknown_lemmas: Sequence[str],
) -> Tuple[List[ListEntry], Dict[str, any]]:
    """
    Update the entries:
      - Attach chosen sentence to the current lemma at current_index.
      - If unknown_lemmas is non-empty:
           * Partition them into reorders (present later) and out_of_bound (not present in file).
           * Insert both reorders and out_of_bound right before current_index in the order they appear.
             - Reorders: preserve their existing POS/definition/sentence from their later line.
             - Out_of_bound: insert as single-column minimal entries (lemma only; no POS/definition/sentence).
           * Remove the original instances of the reorders from further down the list.

    Returns (new_entries, info_dict)
    """
    # Map lemma(lower) -> first index
    lemma_to_first_index: Dict[str, int] = {}
    for i, e in enumerate(entries):
        key = e.lemma.lower()
        if key not in lemma_to_first_index:
            lemma_to_first_index[key] = i

    # Update current lemma with chosen sentence (preserve pos/definition and minimal flag)
    updated_entries = entries.copy()
    cur_entry = entries[current_index]
    updated_entries[current_index] = ListEntry(
        lemma=cur_entry.lemma,
        pos=cur_entry.pos,
        definition=cur_entry.definition,
        sentence=chosen.original,
        is_minimal=cur_entry.is_minimal,
    )

    # Deduplicate unknown lemmas while preserving order
    unknowns_ordered: List[str] = []
    _seen_unknowns = set()
    for _l in unknown_lemmas:
        if _l not in _seen_unknowns:
            unknowns_ordered.append(_l)
            _seen_unknowns.add(_l)

    # If no unknowns: simple update
    if not unknowns_ordered:
        return updated_entries, {
            "out_of_bound": [],
            "reorders": [],
            "out_of_bound_count": 0,
            "reorders_count": 0,
        }

    current_lemma = entries[current_index].lemma
    all_head_lemmas_set = set(e.lemma.lower() for e in entries)

    # Partition unknowns: collect all later occurrences to move, preserving their fields
    current_lemma_lower = current_lemma.lower()
    reorders_map: Dict[str, List[ListEntry]] = {}
    out_of_bound: List[str] = []
    for l in unknowns_ordered:
        if l == current_lemma_lower:
            # Already present here
            continue
        # Collect all occurrences later in the list
        occs: List[ListEntry] = []
        for j in range(current_index + 1, len(entries)):
            if entries[j].lemma.lower() == l:
                occs.append(entries[j])
        if occs:
            reorders_map[l] = occs
        else:
            if l not in out_of_bound:
                out_of_bound.append(l)

    # Build items to insert (reorders + out_of_bound)
    to_insert: List[ListEntry] = []
    for l in unknowns_ordered:
        if l in reorders_map:
            for src in reorders_map[l]:
                # Preserve pos/definition/sentence and minimal flag as-is
                to_insert.append(
                    ListEntry(
                        lemma=src.lemma,
                        pos=src.pos,
                        definition=src.definition,
                        sentence=src.sentence,
                        is_minimal=src.is_minimal,
                    )
                )
        elif l in out_of_bound:
            # Insert a minimal (single-column) entry for out-of-list lemma
            to_insert.append(
                ListEntry(
                    lemma=l,
                    pos=None,
                    definition=None,
                    sentence=None,
                    is_minimal=True,
                )
            )
        else:
            # skipped (e.g., current lemma)
            pass

    # Construct final list: before + inserted + after(without duplicates of reorders)
    before = updated_entries[:current_index]
    after = updated_entries[current_index:]  # includes the current lemma at position 0 of this slice

    reorder_set = set(reorders_map.keys())
    filtered_after: List[ListEntry] = []
    seen_current = False
    for i, e in enumerate(after):
        if not seen_current:
            filtered_after.append(e)
            seen_current = True
            continue
        if e.lemma.lower() in reorder_set:
            continue
        filtered_after.append(e)

    new_entries = before + to_insert + filtered_after

    info = {
        "out_of_bound": out_of_bound,
        "reorders": list(reorders_map.keys()),
        "out_of_bound_count": len(out_of_bound),
        "reorders_count": len(reorders_map),
    }
    return new_entries, info


# --------------- Orchestration (Step loop) ---------------

def generate_language_learning_list(
    steps: int,
    filename: str,
    skip_count: int = 300,
    language: str = "en",
    model: str = "gpt-5-mini",
) -> None:
    """
    The main loop:
      - For each step:
          1) Read the current list file.
          2) Find the next unfinished lemma after skip_count, skipping any single-column (minimal) entries.
          3) Generate 15 sentences with LLM, using POS/definition in the prompt
             and noting verb inflection if applicable.
          4) Lemmatize with stanza.
          5) Choose sentence with fewest unknown lemmas vs prev list.
          6) Write updated list BACK TO THE SAME FILE (edit in place).
          7) When out-of-list lemmas are encountered, insert them before the current lemma as single-column entries
             and do not generate sentences for them in future steps.
    """
    if steps <= 0:
        print("Nothing to do: steps must be > 0.")
        return

    current_file = filename

    for step_idx in range(1, steps + 1):
        # 1) Read current list
        entries = parse_list_file(current_file)

        # 2) Find target lemma (skips single-column minimal entries)
        idx, lemma = find_next_undone_lemma(entries, skip_count=skip_count)

        # Gather POS/definition for this lemma
        target_entry = entries[idx]
        pos = target_entry.pos
        definition = target_entry.definition

        # Prepare prev lemmas (only from file order up to idx)
        prev_lemmas = [e.lemma for e in entries[:idx]]

        # Verbose step logging
        print(f"==================== Step {step_idx} ====================")
        print(f"[Step {step_idx}] Target index={idx}, lemma='{lemma}'")
        print(f"[Step {step_idx}] POS={pos or '(unspecified)'}; Definition={definition or '(unspecified)'}")
        print(f"[Step {step_idx}] Model={model}; Language={language}")

        # 3) Generate sentences
        print(f"[Step {step_idx}] Generating candidate sentences...")
        sentences = call_openai_generate_sentences(lemma, pos=pos, definition=definition, language=language, model=model)
        #for i, s in enumerate(sentences, 1):
        #    print(f"[Step {step_idx}] cand[{i:02d}]: {s}")

        # 4) Lemmatize
        lemmas_per_sentence = lemmatize_sentences_stanza(sentences, language=language)
        print(f"[Step {step_idx}] Lemmatization complete for {len(lemmas_per_sentence)} candidates.")

        # 5) Choose best sentence
        chosen, dbg = choose_best_sentence(lemma, lemmas_per_sentence, prev_lemmas)
        if chosen is not None:
            print(f"[Step {step_idx}] Chosen: '{chosen.original}'")
            print(f"[Step {step_idx}] Stats: candidates={dbg.get('candidate_count')}, unknown_count={dbg.get('unknown_count')}, token_count={dbg.get('token_count')}")
            if dbg.get("unknown_list"):
                print(f"[Step {step_idx}] Unknown lemmas (vs file prior): {', '.join(dbg.get('unknown_list'))}")

        if chosen is None:
            # No sentence included the lemma; delete the lemma line and continue to next iteration
            print(f"[Step {step_idx}] No generated sentence contained the lemma '{lemma}'. Deleting this lemma line and continuing.")
            # Remove the problematic lemma entry and write back to the same file
            del entries[idx]
            write_list_file(entries, current_file)
            print(f"[Step {step_idx}] Deleted lemma '{lemma}' at index {idx} and wrote updates to: {current_file}")
            continue

        # Build unknowns for chosen sentence relative to the FILE-ONLY prev set.
        prev_set_file_only = {l.lower() for l in prev_lemmas}
        chosen_lemmas_lower = [l.lower() for l in chosen.lemmas if l]
        unknowns = [l for l in chosen_lemmas_lower if l not in prev_set_file_only]
        print(f"[Step {step_idx}] Unknowns relative to file before index {idx}: {', '.join(unknowns) if unknowns else '(none)'}")

        # 6) Apply update and write back to the same file
        new_entries, info = apply_sentence_and_reorder(entries, idx, chosen, unknowns)

        write_list_file(new_entries, current_file)

        # 7) Print details
        print(f"[Step {step_idx}] Reorders moved before current index: {', '.join(info.get('reorders', [])) if info.get('reorders') else '(none)'}")
        print(f"[Step {step_idx}] Out-of-list insertions: {', '.join(info.get('out_of_bound', [])) if info.get('out_of_bound') else '(none)'}")
        print(f"[Step {step_idx}] Wrote updates to: {current_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Language Learning List Generator")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to run")
    parser.add_argument("--file", dest="filename", required=True, help="Path to TSV-like list file")
    parser.add_argument("--skip", dest="skip_count", type=int, default=300, help="Number of initial lemmas to skip (default: 300)")
    parser.add_argument(
        "--language", "--lang",
        dest="language",
        default="en",
        help="Target language code or name for generation and lemmatization (default: en)"
    )
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use (default: gpt-5-mini)")
    args = parser.parse_args()

    generate_language_learning_list(
        steps=args.steps,
        filename=args.filename,
        skip_count=args.skip_count,
        language=args.language,
        model=args.model,
    )

if __name__ == "__main__":
    main()
