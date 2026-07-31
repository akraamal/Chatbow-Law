# -*- coding: utf-8 -*-
"""FINAL CONSOLIDATED AUDIT — both entity schemas."""

import json, re, sys, os
from collections import Counter
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANN_DIR = r'data/annotated'

FILES = [
    'fr_BO_6718_Fr_entities.json',
    'fr_BO_6758_Fr_entities.json',
    'fr_BO_6804_Fr_entities.json',
    'fr_BO_6822_Fr_entities.json',
    'fr_BO_7132_Fr_entities.json',
    'fr_BO_7460-bis Fr_entities.json',
    'fr_BO_7480_Fr_entities.json',
    'fr_BO_7492_Fr_entities.json',
    'fr_BO_7500_Fr_entities.json',
    'fr_BO_7510_Fr_entities.json',
    'fr_BO_7522_Fr_entities.json',
]

EXEC_PHRASES = [
    'LE MINISTRE', 'LA MINISTRE', 'LE CHEF DU GOUVERNEMENT',
    'ARRETENT', 'ARRÊTENT', 'DECRETE', 'DÉCRÈTE',
    'Vu l', 'Vu le', 'Considerant', 'Considérant',
]

DOC_TITLE_RE = re.compile(
    r'(?:Arr[eê]t[eé]|D[eé]cret|Dahir|D[eé]cision|Loi(?!-)|R[eè]glement|Annexe)\b',
    re.IGNORECASE,
)

def norm(s):
    return unicodedata.normalize('NFC', str(s))

def get_entity_label(e):
    return norm(e.get('label', e.get('type', '')))

def get_entity_text(e):
    return norm(e.get('text', ''))

def get_entity_start(e):
    return e.get('start_char', e.get('start', -1))

def get_entity_end(e):
    return e.get('end_char', e.get('end', -1))

print('='*120)
print('FINAL CONSOLIDATED QUALITY AUDIT — ALL BO FILES')
print('='*120)

for fname in FILES:
    path = os.path.join(ANN_DIR, fname)
    if not os.path.exists(path):
        print(f'\n--- {fname}: NOT FOUND ---')
        continue

    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    bid = fname.replace('_entities.json', '')
    articles = d.get('articles', [])
    instruments = d.get('instruments', [])
    preamble_entities = d.get('entities', []) + d.get('preamble_entities', [])

    # Collect all entities from articles
    all_entities = list(preamble_entities)
    for a in articles:
        all_entities.extend(a.get('entities', []))

    ent_types = set()
    for e in all_entities:
        if isinstance(e, dict):
            ent_types.add(get_entity_label(e))

    print(f'\n--- {bid} ---')
    print(f'  n_articles: {d.get("n_articles","?")} | actual: {len(articles)} | instr: {len(instruments)}')
    print(f'  Total entities: {len(all_entities)} | types: {ent_types}')

    # ========= STRUCTURAL =========
    # n_articles mismatch
    if d.get('n_articles', 0) != len(articles):
        print(f'  ⚠ [1a] n_articles MISMATCH: meta={d["n_articles"]} actual={len(articles)}')

    # Bleed
    bleed = []
    for i, a in enumerate(articles):
        text = a.get('text', '')
        body = text[20:] if len(text) > 20 else ''
        found = [p for p in EXEC_PHRASES if re.search(re.escape(p), body, re.IGNORECASE)]
        if found:
            bleed.append({'idx': i, 'num': a.get('number'), 'found': found})
    if bleed:
        print(f'  {"✓" if len(bleed) < 5 else "⚠"} [1b] Bleed articles: {len(bleed)}')
        if len(bleed) >= 5:
            # Check what type of bleed: preamble-like (Vu l, Vu le, DECRETE) or just ministerial
            preamble_bleed = sum(1 for b in bleed if any(p in str(b['found']) for p in ['Vu l', 'Vu le', 'DECRETE', 'DÉCRÈTE']))
            min_bleed = len(bleed) - preamble_bleed
            print(f'       preamble-style: {preamble_bleed}, ministerial-only: {min_bleed}')
    else:
        print(f'  ✓ [1b] Bleed: 0')

    # Embedded ART markers
    emb = [i for i, a in enumerate(articles) if re.search(r'\bARTICLE\s+(?:PREMIER|\d+)', a.get('text',''), re.IGNORECASE)]
    if emb:
        print(f'  {"✓" if len(emb) < 5 else "⚠"} [1c] Embedded ART: {len(emb)}')
    else:
        print(f'  ✓ [1c] Embedded ART: 0')

    # Truncated ends
    sentence_end = {'.', '»', ')', '"', '\u201d', '\u201c', '!', '?'}
    trunc = [(i, a.get('text','')[-25:]) for i, a in enumerate(articles) if a.get('text','').strip() and a.get('text','').strip()[-1] not in sentence_end]
    if trunc:
        print(f'  {"✓" if len(trunc) < 5 else "⚠"} [1d] Truncated ends: {len(trunc)}')
    else:
        print(f'  ✓ [1d] Truncated ends: 0')

    # Garbled (2+ doc titles)
    garbled = []
    for i, a in enumerate(articles):
        text = a.get('text', '')
        matches = DOC_TITLE_RE.findall(text)
        non_annexe = [m for m in matches if m.lower() != 'annexe']
        if len(non_annexe) >= 2:
            garbled.append(i)
    if garbled:
        print(f'  {"✓" if len(garbled) < 5 else "⚠"} [1e] Garbled (2+ doc titles): {len(garbled)}')
    else:
        print(f'  ✓ [1e] Garbled: 0')

    # ========= CITATIONS =========
    # Bare-prep citations
    bare_preps = {'relative', 'du', 'des', 'aux', 'au', 'de', 'la', 'le',
                  'les', 'en', 'sur', 'portant', 'modifiant', 'complétant',
                  'fixant', 'suivant', 'susvisé', 'précité'}
    bare = []
    for e in all_entities:
        if isinstance(e, dict) and get_entity_label(e) in ('LOI', 'DECRET', 'ARRETE', 'DAHIR'):
            txt = get_entity_text(e)
            m = re.search(r'n[°\s]\s*([^\s,;]+)', txt)
            if m and m.group(1).lower() in bare_preps:
                bare.append(txt)
    if bare:
        print(f'  ⚠ [2a] Bare-prep citations: {len(bare)}')
        for c in bare[:3]:
            print(f'       {c}')
    else:
        print(f'  ✓ [2a] Bare-prep citations: 0')

    # Resolution rate
    total_c = 0
    resolved_c = 0
    for a in articles:
        for c in a.get('citations', []):
            total_c += 1
            if c.get('resolved'):
                resolved_c += 1
    if total_c:
        print(f'  ✓ [2b] Citations resolved: {resolved_c}/{total_c} ({resolved_c/total_c*100:.1f}%)')
    else:
        print(f'  - [2b] No citations')

    # Duplicate citations
    sigs = Counter()
    for a in articles:
        for c in a.get('citations', []):
            cited = c.get('cited_text', '').strip().lower()
            if cited:
                sigs[(cited, c.get('article_idx', -1))] += 1
    dups = {k: v for k, v in sigs.items() if v > 1}
    if dups:
        print(f'  ⚠ [2c] Duplicate citation sigs: {len(dups)}')
    else:
        print(f'  ✓ [2c] Duplicate citation sigs: 0')

    # ========= CLASSIFICATION =========
    # Person-org overlap per article
    po_overlap = {}
    for i, a in enumerate(articles):
        orgs = set(norm(o.get('text','').strip()) for o in a.get('organizations', []))
        persons = set(norm(p.get('text','').strip()) for p in a.get('persons', []))
        o = orgs & persons
        if o:
            po_overlap[a.get('number', i)] = list(o)
    if po_overlap:
        print(f'  ⚠ [3a] PERSON/ORG overlap: {len(po_overlap)} articles')
    else:
        print(f'  ✓ [3a] PERSON/ORG overlap: 0')

    # Company in persons
    co_in_p = []
    for a in articles:
        for p in a.get('persons', []):
            txt = norm(p.get('text','')).strip()
            if re.search(r'(?:SARL|SA\b|S\.A\.R\.L|LTD|SOCIETE|SOCIÉTÉ)', txt, re.IGNORECASE):
                co_in_p.append({'art': a.get('number'), 'text': txt})
    if co_in_p:
        print(f'  ⚠ [3b] Company in persons[]: {len(co_in_p)}')
        for c in co_in_p[:3]:
            print(f'       art {c["art"]}: {c["text"]}')
    else:
        print(f'  ✓ [3b] Company in persons[]: 0')

    # Place in persons
    place_ind = ['rue', 'avenue', 'boulevard', 'place', 'quartier', 'commune', 'ville', 'prefecture', 'préfecture']
    pl_in_p = []
    for a in articles:
        for p in a.get('persons', []):
            txt = norm(p.get('text','')).strip()
            if any(ind in txt.lower() for ind in place_ind):
                pl_in_p.append({'art': a.get('number'), 'text': txt})
    if pl_in_p:
        print(f'  ⚠ [3c] Place in persons[]: {len(pl_in_p)}')
        for c in pl_in_p[:3]:
            print(f'       art {c["art"]}: {c["text"]}')
    else:
        print(f'  ✓ [3c] Place in persons[]: 0')

    # ========= NORMALIZATION =========
    # Entity casing variants
    text_counts = Counter()
    for e in all_entities:
        if isinstance(e, dict):
            text_counts[get_entity_text(e)] += 1
    lower_map = {}
    for t in text_counts:
        lt = t.lower()
        lower_map.setdefault(lt, []).append(t)
    variants = {k: v for k, v in lower_map.items() if len(v) > 1}
    if variants:
        print(f'  ⚠ [4] Casing variants: {len(variants)} groups')
        for k, v in list(variants.items())[:3]:
            print(f'       {" | ".join(v[:3])}')
    else:
        print(f'  ✓ [4] Casing variants: 0')

    # ========= COVERAGE =========
    # Uncovered articles
    if instruments:
        covered = set()
        for inst in instruments:
            covered.update(inst.get('article_indices', []))
        uncovered = [i for i in range(len(articles)) if i not in covered]
        if uncovered:
            print(f'  ⚠ [5] Articles not in any instrument: {len(uncovered)}')
        else:
            print(f'  ✓ [5] All articles covered by instruments')
    else:
        print(f'  ⚠ [5] No instruments detected at all')

    # Non-contiguous instrument indices
    for i, inst in enumerate(instruments):
        arts = inst.get('article_indices', [])
        if arts:
            expected = list(range(arts[0], arts[-1]+1))
            if arts != expected:
                print(f'  ⚠ [5] Instrument {i}: non-contiguous indices: gap at {set(expected)-set(arts)}')

    # ========= ORPHAN ENTITIES =========
    orphan = 0
    for e in all_entities:
        if isinstance(e, dict) and (get_entity_start(e) == -1 or get_entity_end(e) == -1):
            orphan += 1
    if orphan:
        print(f'  ⚠ Orphan entities (start/end=-1): {orphan}')
    else:
        print(f'  ✓ No orphan entities')

    # Raw header schema check
    non_standard_headers = []
    for i, a in enumerate(articles):
        h = a.get('raw_header', '')
        if h and not re.match(r'^(?:ART(?:ICLE)?\.?\s*(?:PREMIER|1er|\d+(?:-\d+)?)|Annexe)', h, re.IGNORECASE):
            non_standard_headers.append((i, a.get('number'), h[:50]))
    if non_standard_headers:
        print(f'  ⚠ Non-standard raw_headers: {len(non_standard_headers)}')
    else:
        print(f'  ✓ All raw_headers standard')

