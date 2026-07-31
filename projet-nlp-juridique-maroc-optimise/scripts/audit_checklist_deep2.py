# -*- coding: utf-8 -*-
"""Deep investigation: check if 'bleed' and 'garbled' are genuine contamination or legal cross-references."""

import json, re, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANN_DIR = r'data/annotated'

# Pick BO_7492 for detailed analysis
fname = 'fr_BO_7492_Fr_entities.json'
path = os.path.join(ANN_DIR, fname)
with open(path, encoding='utf-8') as f:
    d = json.load(f)

articles = d.get('articles', [])
instruments = d.get('instruments', [])

# Show instrument boundaries
print('=== INSTRUMENTS ===')
for i, inst in enumerate(instruments):
    print(f'  [{i}] type={inst.get("instrument_type")} ref={inst.get("reference")} arts={inst.get("article_indices")}')
    if inst.get('extracted_tables'):
        print(f'      tables: {len(inst["extracted_tables"])}')

# Check which articles belong to which instrument
inst_art_sets = []
for inst in instruments:
    inst_art_sets.append(set(inst.get('article_indices', [])))
all_inst_arts = set()
for s in inst_art_sets:
    all_inst_arts.update(s)

# Articles not in any instrument
unassigned = [i for i in range(len(articles)) if i not in all_inst_arts]
if unassigned:
    print(f'\n  Articles NOT in any instrument: {unassigned}')
    for i in unassigned[:5]:
        print(f'    a#{i} ({articles[i]["number"]}): {articles[i]["text"][:80]}')

# For BO_7492: there are 2 instruments. Let me check bleed for instrument 0 vs 1
print('\n=== BLEED ANALYSIS PER INSTRUMENT ===')
for inst_idx, inst in enumerate(instruments):
    art_indices = inst.get('article_indices', [])
    bleed_in_inst = 0
    for i in art_indices:
        if i >= len(articles):
            continue
        a = articles[i]
        text = a.get('text', '')
        body = text[20:] if len(text) > 20 else ''
        # Check for LE MINISTRE etc
        phrases = ['LE MINISTRE', 'LA MINISTRE', 'Vu l', 'Vu le', 'ARRETENT', 'ARRÊTENT', 'DÉCRÈTE']
        found = [p for p in phrases if re.search(re.escape(p), body, re.IGNORECASE)]
        if found:
            bleed_in_inst += 1
    print(f'  Instrument {inst_idx} (type={inst.get("instrument_type")}): {bleed_in_inst}/{len(art_indices)} articles with bleed')

# Show detailed analysis of each "bleed" article in BO_7492
print('\n=== DETAILED BLEED ARTICLES (BO_7492) ===')
for i, a in enumerate(articles):
    text = a.get('text', '')
    body = text[20:] if len(text) > 20 else ''
    phrases = ['LE MINISTRE', 'LA MINISTRE', 'Vu l', 'Vu le', 'ARRETENT', 'ARRÊTENT', 'DÉCRÈTE']
    found = [p for p in phrases if re.search(re.escape(p), body, re.IGNORECASE)]
    if found:
        # Show first 200 chars of the article
        print(f'\n  a#{i} ({a["number"]}): bleed={found}')
        print(f'    start={text[:80]!r}')
        # Where does the first bleed phrase appear?
        for phrase in found:
            m = re.search(re.escape(phrase), body, re.IGNORECASE)
            if m:
                ctx = body[max(0, m.start()-30):m.end()+80]
                print(f'    phrase="{phrase}" at offset+20+{m.start()}: ...{ctx}...')

# Now check if these "bleed" articles are actually within their correct instrument
print('\n=== CROSS-INSTRUMENT CHECK ===')
for inst_idx, inst in enumerate(instruments):
    art_indices = set(inst.get('article_indices', []))
    # Check if article text mentions a DIFFERENT instrument's heading
    for other_idx, other_inst in enumerate(instruments):
        if other_idx == inst_idx:
            continue
        other_type = other_inst.get('instrument_type', '')
        if not other_type:
            continue
        for i in art_indices:
            if i >= len(articles):
                continue
            text = articles[i].get('text', '')
            # Look for the other instrument's type + reference as a heading
            # (not as a citation like "Vu l'arrêté...")
            other_ref = other_inst.get('reference', '')
            # Check if the other act appears as a heading in this article
            if other_ref:
                pattern = re.compile(
                    r'^' + re.escape(other_type) + r'\s+' + re.escape(other_ref),
                    re.MULTILINE | re.IGNORECASE,
                )
                if pattern.search(text):
                    print(f'  INSTRUMENT {inst_idx} art #{i} ({articles[i]["number"]}) contains HEADING of instrument {other_idx}!')
                    ctx_start = pattern.search(text).start()
                    print(f'    context: {text[max(0,ctx_start-20):ctx_start+80]!r}')

# Also check: do the "garbled" articles have actual heading content?
print('\n=== GARBLED (HEADING-LIKE) ANALYSIS ===')
heading_re = re.compile(
    r'^(?:Arr[eê]t[eé]\s+(?:conjoint\s+)?(?:du\s+|n[°]\s+)|'
    r'D[eé]cret\s+(?:n[°]\s+)?|'
    r'Dahir\s+(?:n[°]\s+)?|'
    r'Loi\s+(?:n[°]\s+)?|'
    r'D[eé]cision\s+(?:n[°]\s+)?|'
    r'R[eè]glement\s+(?:n[°]\s+)?)',
    re.MULTILINE | re.IGNORECASE,
)
for i, a in enumerate(articles):
    text = a.get('text', '')
    # Find heading-like matches (line-start with act type + reference)
    heading_matches = list(heading_re.finditer(text))
    if heading_matches:
        print(f'\n  a#{i} ({a["number"]}): {len(heading_matches)} heading-like matches')
        for m in heading_matches[:3]:
            line_start = text.rfind('\n', 0, m.start()) + 1 if '\n' in text[:m.start()] else 0
            line = text[line_start:m.start()+80].strip()
            print(f'    {line[:100]}')
