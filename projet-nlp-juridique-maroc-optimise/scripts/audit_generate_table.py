# -*- coding: utf-8 -*-
"""Generate final audit status table."""

import json, re, os, sys
from collections import Counter

ANN_DIR = r'data/annotated'

FILES = [
    ('fr_BO_6718_Fr', 'fr_BO_6718_Fr_entities.json'),
    ('fr_BO_6758_Fr', 'fr_BO_6758_Fr_entities.json'),
    ('fr_BO_6804_Fr', 'fr_BO_6804_Fr_entities.json'),
    ('fr_BO_6822_Fr', 'fr_BO_6822_Fr_entities.json'),
    ('fr_BO_7132_Fr', 'fr_BO_7132_Fr_entities.json'),
    ('fr_BO_7460-bis Fr', 'fr_BO_7460-bis Fr_entities.json'),
    ('fr_BO_7480_Fr', 'fr_BO_7480_Fr_entities.json'),
    ('fr_BO_7492_Fr', 'fr_BO_7492_Fr_entities.json'),
    ('fr_BO_7500_Fr', 'fr_BO_7500_Fr_entities.json'),
    ('fr_BO_7510_Fr', 'fr_BO_7510_Fr_entities.json'),
    ('fr_BO_7522_Fr', 'fr_BO_7522_Fr_entities.json'),
]

EXEC_PHRASES = ['LE MINISTRE','LA MINISTRE','LE CHEF DU GOUVERNEMENT',
    'ARRETENT','ARRÊTENT','DECRETE','DÉCRÈTE','Vu l','Vu le','Considerant','Considérant']
PREAMBLE_PHRASES = ['Vu l','Vu le','DECRETE','DÉCRÈTE','Considerant','Considérant']

DOC_TITLE_RE = re.compile(r'(?:Arr[eê]t[eé]|D[eé]cret|Dahir|D[eé]cision|Loi(?!-)|R[eè]glement|Annexe)\b', re.IGNORECASE)
sentence_end = {'.','»',')','"','\u201d','\u201c','!','?'}

def norm(s):
    import unicodedata
    return unicodedata.normalize('NFC', str(s))

rows = []
for bid, fname in FILES:
    path = os.path.join(ANN_DIR, fname)
    if not os.path.exists(path):
        rows.append({'BO': bid, 'Status': 'MISSING', 'Arts': '—', 'Instr': '—', 'Arts∉Inst': '—',
            'Bleed': '—', 'Preamble': '—', 'EmbART': '—', 'Trunc': '—', 'Garbl': '—',
            'BarePrep': '—', 'Cit%': '—', 'P/Ovlp': '—', 'C∈P': '—', 'P∈P': '—',
            'Casing': '—', 'Orphan': '—', 'Fails Gate': '—'})
        continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    articles = d.get('articles', [])
    instruments = d.get('instruments', [])
    all_entities = d.get('entities', []) + d.get('preamble_entities', [])
    for a in articles:
        all_entities.extend(a.get('entities', []))

    # Meta n_articles
    meta_n = d.get('n_articles', 0)
    actual_n = len(articles)
    n_mismatch = '⚠' if meta_n != actual_n else ''
    meta_str = f'{meta_n}→{actual_n}' if meta_n != actual_n else str(actual_n)

    # Instruments
    n_instr = len(instruments)
    covered = set()
    for inst in instruments:
        covered.update(inst.get('article_indices', []))
    uncovered = sum(1 for i in range(actual_n) if i not in covered) if instruments else actual_n

    # Bleed
    bleed = 0
    preamble_style = 0
    for i, a in enumerate(articles):
        body = a.get('text','')[20:]
        for p in EXEC_PHRASES:
            if re.search(re.escape(p), body, re.IGNORECASE):
                bleed += 1
                if any(re.search(re.escape(pp), body, re.IGNORECASE) for pp in PREAMBLE_PHRASES):
                    preamble_style += 1
                break

    # Embedded ART
    emb = sum(1 for a in articles if re.search(r'\bARTICLE\s+(?:PREMIER|\d+)', a.get('text',''), re.IGNORECASE))

    # Truncated
    trunc = sum(1 for a in articles if a.get('text','').strip() and a['text'].strip()[-1] not in sentence_end)

    # Garbled
    garbled = 0
    for a in articles:
        matches = [m for m in DOC_TITLE_RE.findall(a.get('text','')) if m.lower() != 'annexe']
        if len(matches) >= 2:
            garbled += 1

    # Bare-prep
    bare_preps = {'relative','du','des','aux','au','de','la','le','les','en','sur',
                  'portant','modifiant','complétant','fixant','suivant','susvisé','précité'}
    bare = 0
    for e in all_entities:
        if isinstance(e, dict):
            label = e.get('label', e.get('type', ''))
            txt = norm(e.get('text', ''))
            if label in ('LOI','DECRET','ARRETE','DAHIR'):
                m = re.search(r'n[°\s]\s*([^\s,;]+)', txt)
                if m and m.group(1).lower() in bare_preps:
                    bare += 1

    # Citation resolved
    total_c = sum(len(a.get('citations',[])) for a in articles)
    res_c = sum(1 for a in articles for c in a.get('citations',[]) if c.get('resolved'))
    cit_pct = f'{res_c/total_c*100:.0f}%' if total_c else '—'

    # Person-org overlap
    po = 0
    for a in articles:
        orgs = set(norm(o.get('text','')).strip() for o in a.get('organizations',[]))
        persons = set(norm(p.get('text','')).strip() for p in a.get('persons',[]))
        if orgs & persons:
            po += 1

    # Company in persons
    cp = sum(1 for a in articles for p in a.get('persons',[]) if re.search(r'(?:SARL|SA\b|S\.A\.R\.L|LTD|SOCIETE|SOCIÉTÉ)', norm(p.get('text','')), re.IGNORECASE))

    # Place in persons
    pi = ['rue','avenue','boulevard','place','quartier','commune','ville','prefecture','préfecture']
    pp = sum(1 for a in articles for p in a.get('persons',[]) if any(ind in norm(p.get('text','')).lower() for ind in pi))

    # Casing variants
    text_counts = Counter()
    for e in all_entities:
        if isinstance(e, dict):
            text_counts[norm(e.get('text',''))] += 1
    lower_map = {}
    for t in text_counts:
        lower_map.setdefault(t.lower(), []).append(t)
    casing = sum(1 for v in lower_map.values() if len(v) > 1)

    # Orphan entities
    orphan = 0
    for e in all_entities:
        if isinstance(e, dict):
            s = e.get('start_char', e.get('start', -1))
            if s == -1:
                orphan += 1
                break  # count per-entity, but we just want file-level
    # count all
    orphan = sum(1 for e in all_entities if isinstance(e, dict) and e.get('start_char', e.get('start', -1)) == -1)

    # Gate decision (fail if: contaminated, no instruments, n_articles mismatch, orphan entities)
    fails_gate = []
    if bleed > 0 and preamble_style > 0:
        fails_gate.append('bleed')
    if n_instr == 0:
        fails_gate.append('no-instr')
    if n_mismatch:
        fails_gate.append('mismatch')
    if orphan > 0:
        fails_gate.append('orphan')
    if uncovered == actual_n:
        fails_gate.append('uncovered')

    rows.append({
        'BO': bid,
        'Status': 'PASS' if not fails_gate else f'FAIL:{",".join(fails_gate[:3])}',
        'Arts': meta_str,
        'Instr': str(n_instr),
        'Arts∉Inst': '0' if uncovered == 0 else str(uncovered),
        'Bleed': str(bleed),
        'Preamble': str(preamble_style),
        'EmbART': str(emb),
        'Trunc': str(trunc),
        'Garbl': str(garbled),
        'BarePrep': str(bare),
        'Cit%': cit_pct,
        'P/Ovlp': str(po),
        'C∈P': str(cp),
        'P∈P': str(pp),
        'Casing': str(casing),
        'Orphan': str(orphan),
        'Fails Gate': '|'.join(fails_gate) if fails_gate else '—',
    })

# Print table
hdr = ['BO','Status','Arts','Instr','Arts∉Inst','Bleed','Preamble','EmbART','Trunc','Garbl',
       'BarePrep','Cit%','P/Ovlp','C∈P','P∈P','Casing','Orphan','Fails Gate']
col_w = {h: max(len(h), max((len(r.get(h,'')) for r in rows), default=0)) for h in hdr}

def sep_line():
    return '+'.join('-'*(col_w[h]+2) for h in hdr)

def fmt_row(row, hdr_list):
    return '| '+ ' | '.join(row.get(h,'').ljust(col_w[h]) for h in hdr_list) + ' |'

print(sep_line())
print(fmt_row({h:h for h in hdr}, hdr))
print(sep_line())
for r in rows:
    print(fmt_row(r, hdr))
print(sep_line())
