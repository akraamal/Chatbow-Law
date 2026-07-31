# -*- coding: utf-8 -*-
"""Audit all annotated JSON files against the 6-point quality checklist."""

import json, re, sys, os
from collections import Counter
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANN_DIR = r'data/annotated'

ANNOTATED_FILES = [
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

AR_FILES = [
    'ar_BO_3559_Ar_entities.json',
    'ar_BO_7132_Fr_entities.json',
    'ar_BO_7360_Ar_entities.json',
    'ar_BO_7430_Ar_entities.json',
    'ar_BO_7517_Ar_entities.json',
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
    return unicodedata.normalize('NFC', s)

def gather_all_entities(d):
    all_ents = []
    for a in d.get('articles', []):
        all_ents.extend(a.get('entities', []))
    all_ents.extend(d.get('preamble_entities', []))
    all_ents.extend(d.get('entities', []))
    return all_ents

def audit_file(fname):
    path = os.path.join(ANN_DIR, fname)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    articles = d.get('articles', [])
    instruments = d.get('instruments', [])
    entities = gather_all_entities(d)

    bid = fname.replace('_entities.json', '')

    r = {
        'file': fname, 'bo': bid,
        'n_articles_meta': d.get('n_articles', 0),
        'actual_articles': len(articles),
        'n_instruments': len(instruments),
        'bleed_articles': [],
        'embedded_art': [],
        'truncated': [],
        'garbled': [],
        'bare_prep_citations': [],
        'cit_resolved': {'total': 0, 'resolved': 0},
        'dup_citations': {},
        'person_org_overlap': {},
        'company_as_person': [],
        'place_in_person': [],
        'casing_variants': {},
    }

    # ---- CHECK 1: Structural integrity ----
    # 1a: n_articles meta mismatch
    r['n_mismatch'] = d.get('n_articles', 0) != len(articles)

    # 1b: boilerplate bleed
    for i, a in enumerate(articles):
        text = a.get('text', '')
        body = text[20:] if len(text) > 20 else ''
        found = []
        for phrase in EXEC_PHRASES:
            if re.search(re.escape(phrase), body, re.IGNORECASE):
                found.append(phrase)
        if found:
            r['bleed_articles'].append({
                'idx': i, 'num': a.get('number'),
                'found': found,
                'text_snip': text[:80],
            })

    # 1c: embedded ART marker
    for i, a in enumerate(articles):
        text = a.get('text', '')
        if re.search(r'\bARTICLE\s+(?:PREMIER|\d+)', text, re.IGNORECASE):
            r['embedded_art'].append({'idx': i, 'num': a.get('number')})

    # 1d: truncated ends
    sentence_end = {'.', '»', ')', '"', '\u201d', '\u201c', '!', '?'}
    for i, a in enumerate(articles):
        text = a.get('text', '').strip()
        if text and text[-1] not in sentence_end:
            r['truncated'].append({
                'idx': i, 'num': a.get('number'),
                'tail': text[-25:],
            })

    # 1e: garbled (multiple doc titles)
    for i, a in enumerate(articles):
        text = a.get('text', '')
        matches = DOC_TITLE_RE.findall(text)
        non_annexe = [m for m in matches if m.lower() != 'annexe']
        if len(non_annexe) >= 2:
            r['garbled'].append({
                'idx': i, 'num': a.get('number'),
                'count': len(non_annexe),
                'titles': matches,
            })

    # ---- CHECK 2: Citation integrity ----
    # 2a: bare preposition after n°
    for e in entities:
        if e.get('type') in ('LOI', 'DECRET', 'ARRETE', 'DAHIR'):
            txt = e.get('text', '')
            m = re.search(r'n[°\s]\s*([^\s,;]+)', txt)
            if m:
                num = m.group(1)
                bare_preps = {
                    'relative', 'du', 'des', 'aux', 'au', 'de', 'la', 'le',
                    'les', 'en', 'sur', 'portant', 'modifiant', 'complétant',
                    'fixant', 'suivant', 'susvisé', 'précité',
                }
                if num.lower() in bare_preps:
                    r['bare_prep_citations'].append({'text': txt, 'type': e['type']})

    # 2b: resolved rate
    total_cit = 0
    resolved_cit = 0
    for a in articles:
        for c in a.get('citations', []):
            total_cit += 1
            if c.get('resolved'):
                resolved_cit += 1
    r['cit_resolved'] = {
        'total': total_cit,
        'resolved': resolved_cit,
        'pct': round(resolved_cit / total_cit * 100, 1) if total_cit else None,
    }

    # 2c: duplicate citations
    sigs = Counter()
    for a in articles:
        for c in a.get('citations', []):
            cited = c.get('cited_text', '').strip().lower()
            idx = c.get('article_idx', -1)
            if cited:
                sigs[(cited, idx)] += 1
    dups = {str(k): v for k, v in sigs.items() if v > 1}
    r['dup_citations'] = dups

    # ---- CHECK 3: Entity classification ----
    # 3a: person-org overlap per article
    for i, a in enumerate(articles):
        orgs = set(norm(o.get('text', '')).strip() for o in a.get('organizations', []))
        persons = set(norm(p.get('text', '')).strip() for p in a.get('persons', []))
        overlap = orgs & persons
        if overlap:
            r['person_org_overlap'][a.get('number', f'a{i}')] = list(overlap)

    # 3b: company in persons
    for a in articles:
        for p in a.get('persons', []):
            txt = norm(p.get('text', '')).strip()
            if re.search(r'(?:SARL|SA\b|S\.A\.R\.L|LTD|SOCIETE|SOCIÉTÉ|societe|société)', txt, re.IGNORECASE):
                r['company_as_person'].append({'art': a.get('number'), 'text': txt})

    # 3c: place names in persons
    place_indicators = ['rue', 'avenue', 'boulevard', 'place', 'quartier',
                        'commune', 'ville', 'prefecture', 'préfecture']
    for a in articles:
        for p in a.get('persons', []):
            txt = norm(p.get('text', '')).strip()
            if any(ind in txt.lower() for ind in place_indicators):
                r['place_in_person'].append({'art': a.get('number'), 'text': txt})

    # ---- CHECK 4: Casing variants ----
    text_counts = Counter()
    for e in entities:
        text_counts[norm(e.get('text', '')).strip()] += 1
    lower_map = {}
    for t in text_counts:
        lt = t.lower()
        lower_map.setdefault(lt, []).append(t)
    variants = {k: v for k, v in lower_map.items() if len(v) > 1}
    r['casing_variants'] = variants

    return r


def print_report(results):
    print('=' * 120)
    print('QUALITY AUDIT REPORT — ALL BO FILES')
    print('=' * 120)

    for bid, r in sorted(results.items()):
        f = r['file']
        print(f'\n--- {bid} ({f}) ---')
        print(f'  Articles: meta={r["n_articles_meta"]} actual={r["actual_articles"]} mismatch={r["n_mismatch"]}')
        print(f'  Instruments: {r["n_instruments"]}')

        print(f'  [1] STRUCTURAL:')
        if r['bleed_articles']:
            print(f'    BLEED: {len(r["bleed_articles"])} articles')
            for b in r['bleed_articles'][:3]:
                print(f'      a#{b["idx"]} ({b["num"]}): {b["found"]}')
        else:
            print(f'    BLEED: 0')

        if r['embedded_art']:
            print(f'    EMBEDDED ART: {len(r["embedded_art"])} articles')
        else:
            print(f'    EMBEDDED ART: 0')

        if r['truncated']:
            print(f'    TRUNCATED END: {len(r["truncated"])} articles')
            for b in r['truncated'][:5]:
                print(f'      a#{b["idx"]} ({b["num"]}): ...{b["tail"]!r}')
        else:
            print(f'    TRUNCATED END: 0')

        if r['garbled']:
            print(f'    GARBLED (2+ doc titles): {len(r["garbled"])} articles')
            for b in r['garbled'][:3]:
                print(f'      a#{b["idx"]} ({b["num"]}): {b["titles"][:5]}')
        else:
            print(f'    GARBLED (2+ doc titles): 0')

        print(f'  [2] CITATIONS:')
        if r['bare_prep_citations']:
            print(f'    BARE PREP: {len(r["bare_prep_citations"])}')
            for c in r['bare_prep_citations'][:3]:
                print(f'      {c["text"]}  ({c["type"]})')
        else:
            print(f'    BARE PREP: 0')

        cr = r['cit_resolved']
        if cr['total']:
            print(f'    RESOLVED: {cr["resolved"]}/{cr["total"]} ({cr["pct"]}%)')
        else:
            print(f'    RESOLVED: N/A')

        if r['dup_citations']:
            print(f'    DUPLICATE SIGS: {len(r["dup_citations"])}')
        else:
            print(f'    DUPLICATE SIGS: 0')

        print(f'  [3] CLASSIFICATION:')
        if r['person_org_overlap']:
            print(f'    PERSON/ORG OVERLAP: {len(r["person_org_overlap"])} arts')
            for art, lst in list(r['person_org_overlap'].items())[:3]:
                print(f'      {art}: {lst}')
        else:
            print(f'    PERSON/ORG OVERLAP: 0')

        if r['company_as_person']:
            print(f'    COMPANY IN persons[]: {len(r["company_as_person"])}')
            for c in r['company_as_person'][:3]:
                print(f'      {c["art"]}: {c["text"]}')
        else:
            print(f'    COMPANY IN persons[]: 0')

        if r['place_in_person']:
            print(f'    PLACE IN persons[]: {len(r["place_in_person"])}')
            for c in r['place_in_person'][:3]:
                print(f'      {c["art"]}: {c["text"]}')
        else:
            print(f'    PLACE IN persons[]: 0')

        print(f'  [4] NORMALIZATION:')
        if r['casing_variants']:
            print(f'    CASING VARIANTS: {len(r["casing_variants"])} groups')
            for k, v in list(r['casing_variants'].items())[:5]:
                print(f'      {" | ".join(v)}')
        else:
            print(f'    CASING VARIANTS: 0')

    # ---- Summary ----
    print('\n' + '=' * 120)
    print('AGGREGATE SUMMARY')
    print('=' * 120)
    checks = [
        ('bleed_articles', 'Boilerplate bleed'),
        ('embedded_art', 'Embedded ART markers'),
        ('truncated', 'Truncated article ends'),
        ('garbled', 'Garbled (2+ doc titles)'),
        ('bare_prep_citations', 'Bare-prep citations'),
        ('company_as_person', 'Company in persons[]'),
        ('place_in_person', 'Place in persons[]'),
    ]
    for key, label in checks:
        affected = sum(1 for r in results.values() if r[key])
        total_occ = sum(len(r[key]) for r in results.values())
        print(f'  {label}: {affected}/{len(results)} files{", " + str(total_occ) + " occurrences" if total_occ else ""}')

    print(f'  Has n_articles mismatch: {sum(1 for r in results.values() if r["n_mismatch"])}/{len(results)} files')
    print(f'  Has PERSON/ORG overlap: {sum(1 for r in results.values() if r["person_org_overlap"])}/{len(results)} files')
    print(f'  Has duplicate citation sigs: {sum(1 for r in results.values() if r["dup_citations"])}/{len(results)} files')

    # Resolved rates
    rates = [r['cit_resolved']['pct'] for r in results.values() if r['cit_resolved']['pct'] is not None]
    if rates:
        print(f'  Citation resolution rate: min={min(rates):.1f}% max={max(rates):.1f}%')


if __name__ == '__main__':
    results = {}
    for fname in ANNOTATED_FILES:
        path = os.path.join(ANN_DIR, fname)
        if not os.path.exists(path):
            print(f'SKIP (not found): {fname}')
            continue
        bid = fname.replace('_entities.json', '')
        print(f'  Auditing {fname} ...')
        results[bid] = audit_file(fname)

    print_report(results)
