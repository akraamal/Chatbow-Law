# -*- coding: utf-8 -*-
"""Deep-dive audit: distinguish cross-references from true contamination."""

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

DOC_TITLE_RE = re.compile(
    r'(?:Arr[eê]t[eé]|D[eé]cret|Dahir|D[eé]cision|Loi(?!-)|R[eè]glement|Annexe)\b',
    re.IGNORECASE,
)
# Patterns that suggest a title is a citation header, not a document boundary
CITATION_CONTEXT_RE = re.compile(
    r'(?:Vu\s+|vis[eé]|notamment|conform[eé]ment|application|vertu|conformit[eé]|en ex[eé]cution|pr[eé]vu|fix[eé])',
    re.IGNORECASE,
)
# Novel doc-title patterns (heading-like)
HEADING_RE = re.compile(
    r'^(?:Arr[eê]t[eé]\s+(?:conjoint\s+)?(?:du\s+|n[°]\s+)|'
    r'D[eé]cret\s+(?:n[°]\s+)?|'
    r'Dahir\s+(?:n[°]\s+)?|'
    r'Loi\s+(?:n[°]\s+)?|'
    r'D[eé]cision\s+(?:n[°]\s+)?|'
    r'R[eè]glement\s+(?:n[°]\s+)?)',
    re.MULTILINE | re.IGNORECASE,
)

def gather_all_entities(d):
    all_ents = []
    for a in d.get('articles', []):
        all_ents.extend(a.get('entities', []))
    all_ents.extend(d.get('preamble_entities', []))
    all_ents.extend(d.get('entities', []))
    return all_ents

def classify_doc_title_match(text, pos, match_str):
    """Classify a doc title match as 'citation' (legal ref) or 'heading' (new act)."""
    # Check preceding text for citation context
    before = text[max(0, pos-60):pos]
    if CITATION_CONTEXT_RE.search(before):
        return 'citation_ref'
    # Check if it's a heading pattern
    line_start = text.rfind('\n', 0, pos) + 1 if '\n' in text[:pos] else 0
    line = text[line_start:pos+len(match_str)+30]
    if HEADING_RE.match(line, line_start - line_start):
        return 'heading_new_act'
    # Check if it's an annexe reference
    if match_str.lower() == 'annexe' and 'annexe' in before.lower():
        return 'annexe_ref'
    # Default: could be either
    return 'ambiguous'

for fname in ANNOTATED_FILES:
    path = os.path.join(ANN_DIR, fname)
    if not os.path.exists(path):
        continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    articles = d.get('articles', [])
    bid = fname.replace('_entities.json', '')
    text = None
    txt_path = d.get('source', '')
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, encoding='utf-8') as f:
            text = f.read()
    
    print(f'\n=== {bid} ===')
    
    # --- CHECK 6 DEEP: Classify every 'garbled' article ---
    true_contamination = 0
    citation_only = 0
    ambiguous = 0
    examples = []
    
    for i, a in enumerate(articles):
        txt = a.get('text', '')
        matches = list(DOC_TITLE_RE.finditer(txt))
        # Find all doc title occurrences in the ENTIRE document at their original position
        # to better classify them
        doc_matches = []
        for m in DOC_TITLE_RE.finditer(txt):
            m_str = m.group()
            m_pos = m.start()
            before = txt[max(0, m_pos-80):m_pos]
            after = txt[m.end():m.end()+60].strip()[:40]
            
            # If preceded by "Vu", "dudit", "ledit", number, etc → likely citation
            if re.search(r'(?:Vu|dudit|ledit|dud\.|n[°]|article\s+\d+|annexe\s+|du\s+(?:pr[eé]sent|\d+|m[eê]me))', before[-40:], re.IGNORECASE):
                cls = 'citation_ref'
            elif m_str.lower() == 'annexe' and re.search(r"(?:l'annexe|annexe\s+\d+|en\s+annexe)", before[-40:], re.IGNORECASE):
                cls = 'annexe_ref'
            elif re.match(r'(?:Arr[eê]t[eé]\s+(?:conjoint\s+)?(?:du\s+|n[°]\s+)|D[eé]cret\s+(?:n[°]\s+)?|Dahir\s+(?:n[°]\s+)?)', after, re.IGNORECASE):
                cls = 'heading_new_act'
            else:
                cls = 'uncertain'
            doc_matches.append({'text': m_str, 'pos': m_pos, 'cls': cls, 'before': before[-40:].strip(), 'after': after})
        
        non_annexe = [m for m in doc_matches if m['text'].lower() != 'annexe']
        if len(non_annexe) >= 2:
            headings = [m for m in non_annexe if m['cls'] == 'heading_new_act']
            citations = [m for m in non_annexe if m['cls'] == 'citation_ref']
            uncertain = [m for m in non_annexe if m['cls'] == 'uncertain']
            
            if headings:
                true_contamination += 1
                if len(examples) < 5:
                    examples.append({
                        'art': a.get('number'),
                        'idx': i,
                        'headings': [(m['text'], m['after'][:40]) for m in headings[:3]],
                        'citations': len(citations),
                        'uncertain': len(uncertain),
                    })
            elif uncertain:
                ambiguous += 1
            else:
                citation_only += 1
    
    print(f'  True contamination (heading in body): {true_contamination}')
    print(f'  Citation-only cross-refs: {citation_only}')
    print(f'  Uncertain: {ambiguous}')
    for ex in examples:
        print(f'    a#{ex["idx"]} ({ex["art"]}): headings={ex["headings"]} cites={ex["citations"]} uncer={ex["uncertain"]}')
    
    # --- CHECK 5: Coverage — extract SOMMAIRE from raw text ---
    if text:
        # Try to find the sommaire: usually starts with "SOMMAIRE" or "TABLE DES MATI\u00c8RES"
        sommaire_match = re.search(
            r'(?:SOMMAIRE|TABLE\s+DES\s+MATI[EÈ]RES)\s*\n(.*?)(?:\n(?:BULLETIN|\d+\s+(?:JANVIER|F[EÉ]VRIER|MARS|AVRIL|MAI|JUIN|JUILLET|AO[UÛ]T|SEPTEMBRE|OCTOBRE|NOVEMBRE|D[EÉ]CEMBRE)|[A-Z][A-Z\s]{10,}\n|$))',
            text[:30000], re.DOTALL | re.IGNORECASE,
        )
        if sommaire_match:
            sommaire_text = sommaire_match.group(1).strip()
            # Count line items in sommaire
            sommaire_lines = [l.strip() for l in sommaire_text.split('\n') if l.strip() and not re.match(r'^[\d\s\-–.]+$', l.strip())]
            print(f'  SOMMAIRE found: ~{len(sommaire_lines)} items (text: {sommaire_text[:100]}...)')
        else:
            # Try hyphen-line sommaire
            sommaire_match = re.search(
                r'^[ \t]*[-–—]{3,}\s*$.*?(?=^[ \t]*[-–—]{3,}\s*$)',
                text[:30000], re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
            if sommaire_match:
                print(f'  SOMMAIRE-like block found')
            else:
                print(f'  SOMMAIRE not found (or not in first 30K chars)')
