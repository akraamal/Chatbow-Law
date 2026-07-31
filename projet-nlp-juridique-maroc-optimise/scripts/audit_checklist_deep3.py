# -*- coding: utf-8 -*-
"""Deep-dive: coverage, structural consistency, empty files, n_articles mismatch."""

import json, re, sys, os

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

print('='*120)
print('COVERAGE & CONSISTENCY AUDIT')
print('='*120)

for fname in ANNOTATED_FILES:
    path = os.path.join(ANN_DIR, fname)
    if not os.path.exists(path):
        print(f'\nMISSING: {fname}')
        continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    
    bid = fname.replace('_entities.json', '')
    articles = d.get('articles', [])
    instruments = d.get('instruments', [])
    
    print(f'\n--- {bid} ---')
    
    # 1. n_articles mismatch
    meta_n = d.get('n_articles', 0)
    actual_n = len(articles)
    if meta_n != actual_n:
        print(f'  ⚠ n_articles MISMATCH: meta={meta_n} actual={actual_n}')
    else:
        print(f'  ✓ n_articles: {meta_n}')
    
    # 2. Empty file check
    if actual_n == 0:
        print(f'  ⚠ EMPTY: 0 articles — is this a non-legal BO (report/summary only)?')
        print(f'  Total entities: {len(d.get("entities", [])) + len(d.get("preamble_entities", []))}')
        continue
    
    # 3. Instrument coverage
    if instruments:
        all_covered = set()
        overlaps = []
        gaps = []
        for inst in instruments:
            indices = inst.get('article_indices', [])
            all_covered.update(indices)
        uncovered = [i for i in range(actual_n) if i not in all_covered]
        if uncovered:
            print(f'  ⚠ Articles NOT in any instrument: {len(uncovered)} (indices: {uncovered[:10]})')
        else:
            print(f'  ✓ All articles covered by instruments')
        
        n_inst = len(instruments)
        print(f'  Instruments: {n_inst} (covers {len(all_covered)}/{actual_n} articles)')
    else:
        print(f'  ⚠ NO instruments detected (0 instruments)')
    
    # 4. Article raw_header consistency
    non_art_headers = []
    for i, a in enumerate(articles):
        h = a.get('raw_header', '')
        if not re.match(r'^ART(?:ICLE)?\.?\s*(?:PREMIER|1er|\d+(?:-\d+)?)', h, re.IGNORECASE):
            non_art_headers.append({'idx': i, 'num': a.get('number'), 'header': h[:40]})
    if non_art_headers:
        print(f'  ⚠ Non-standard raw_headers: {len(non_art_headers)}')
        for h in non_art_headers[:5]:
            print(f'    a#{h["idx"]} (num={h["num"]}): header={h["header"]!r}')
    else:
        print(f'  ✓ All raw_headers follow standard pattern')
    
    # 5. Article number uniqueness per-instrument
    inst_nums_dup = []
    for inst in instruments:
        seen = {}
        for idx in inst.get('article_indices', []):
            if idx >= len(articles):
                continue
            num = articles[idx].get('number', '')
            if num in seen:
                inst_nums_dup.append(f'art{idx}={num} (also art{seen[num]})')
            seen[num] = idx
    if inst_nums_dup:
        print(f'  ⚠ Duplicate article numbers within same instrument: {len(inst_nums_dup)}')
        for d in inst_nums_dup[:5]:
            print(f'    {d}')
    else:
        print(f'  ✓ Article numbers unique within instruments')
    
    # 6. Article text non-emptiness
    empty = [(i, a.get('number', '')) for i, a in enumerate(articles) if not a.get('text', '').strip()]
    if empty:
        print(f'  ⚠ Empty article texts: {len(empty)}')
        for i, n in empty[:5]:
            print(f'    a#{i} (num={n})')
    else:
        print(f'  ✓ No empty articles')
    
    # 7. Article text length profile
    lengths = [len(a.get('text', '')) for a in articles]
    if lengths:
        print(f'  Text length: min={min(lengths)} max={max(lengths)} median={sorted(lengths)[len(lengths)//2]}')
    
    # 8. Instrument structural consistency
    for i, inst in enumerate(instruments):
        inst_type = inst.get('instrument_type', '?')
        ref = inst.get('reference', '?')
        n_arts_inst = inst.get('n_articles', 0)
        art_indices = inst.get('article_indices', [])
        actual_n_arts = len(art_indices)
        if n_arts_inst != actual_n_arts:
            print(f'  ⚠ Instrument {i}: n_articles mismatch: meta={n_arts_inst} actual={actual_n_arts}')
        
        # Check article indices are contiguous
        if art_indices and art_indices != list(range(art_indices[0], art_indices[-1]+1)):
            print(f'  ⚠ Instrument {i}: non-contiguous article indices: {art_indices[:10]}...')
    
    # 9. Citation articles referenced exist
    for a in articles:
        for c in a.get('citations', []):
            tgt = c.get('article_idx', -1)
            if tgt >= 0 and tgt >= actual_n:
                print(f'  ⚠ Citation target art #{tgt} out of range (only {actual_n} articles)')
    
    # 10. Person/org extraction quality
    all_orgs = set()
    all_persons = set()
    for a in articles:
        for o in a.get('organizations', []):
            all_orgs.add(o.get('text', '').strip().lower())
        for p in a.get('persons', []):
            all_persons.add(p.get('text', '').strip().lower())
    
    # Check: common legal entities in persons
    legal_in_person = [p for p in all_persons if any(t in p for t in ['sarl', 'sa', 'societe', 'société', 'ltd', 's.a.'])]
    if legal_in_person:
        print(f'  ⚠ Legal entities in persons[]: {legal_in_person[:5]}')
    
    # 11. Check for entity `start=-1` or `end=-1` (orphaned entities)
    orphan_ents = 0
    for e in d.get('entities', []):
        if isinstance(e, dict) and (e.get('start', 0) == -1 or e.get('end', 0) == -1):
            orphan_ents += 1
    for a in articles:
        for e in a.get('entities', []):
            if isinstance(e, dict) and (e.get('start', 0) == -1 or e.get('end', 0) == -1):
                orphan_ents += 1
    if orphan_ents:
        print(f'  ⚠ Orphan entities (start/end=-1): {orphan_ents}')
    else:
        print(f'  ✓ No orphan entities')
