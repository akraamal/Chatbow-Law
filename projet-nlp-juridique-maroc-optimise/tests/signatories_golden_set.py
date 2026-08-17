"""
tests/signatories_golden_set.py
-------------------------------
Jeu doré de régression des signataires (priorité 4 du guide v5).

Chaque entrée fixe la sortie attendue du parseur de signatures pour un
instrument donné (fichier annoté + instrument_id) : liste ordonnée de
blocs {role, name, type}.

Vérité terrain : chaque zone de signature a été relue manuellement sur le
texte source (couche native du PDF pour les BO à couche texte — 7488-bis,
7480, 7496 — et OCR pour BO_6804 ; édition arabe vérifiée sur les zones
« االمضاء : ... » / « وقعه بالعطف : »).

Couverture (19 cas) :
  - 5+ signataires  : 2-19-615 (BO_6804, 5 sigs), 2.26.23 (BO_7506 AR, 5 sigs)
  - 4 signataires   : 2-05-1577, 2-10-552, 2-91-609 (BO_7488-bis)
  - 2-3 signataires : 2-19-592, 2-19-575, 2-04-554, 2-09-494, 2-11-696,
                      2-24-927, 2-26-183, 2-20-716, 2-25-1133, 2-25-565 (AR),
                      2-25-885 (AR), 2-25-365 (AR), 181-26 (BO_7510,
                      colonnes de hauteurs différentes : rôles entrelacés,
                      noms en ordre inversé — vérifié sur la géométrie du PDF)
  - signataire unique (aucun contreseing) : 2-25-473 (BO_7421 AR)
  - noms imprimés en Title-case (éditions 2004-2011) : 2-04-554,
    2-05-1577, 2-09-494 (Driss Jettou / Abbas El Fassi)
  - rôle émetteur non déductible (préambule tronqué, role=null) : 2-11-696,
    2-24-927

Utilisation :
    from tests.signatories_golden_set import GOLDEN_SIGNATORIES
    # diff contre data/annotated/<file> -> instruments[<instrument_id>].signatories
"""

GOLDEN_SIGNATORIES = [
    {
        "file": "fr_BO_6804_Fr_692ac82f_entities.json",
        "instrument_id": "instr_2_19_615",
        "expected": [
            {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI", "type": "issuer"},
            {"role": "Le ministre de l'économie et des finances", "name": "MOHAMED BENCHAABOUN", "type": "contreseing"},
            {"role": "Le ministre de l'agriculture, de la pêche maritime, du développement rural et des eaux et forêts", "name": "AZIZ AKHANNOUCH", "type": "contreseing"},
            {"role": "Le ministre de l'industrie, de l'investissement, du commerce et de l'économie numérique", "name": "MLY HAFID ELALAMY", "type": "contreseing"},
            {"role": "Le ministre du tourisme, du transport aérien, de l'artisanat et de l'économie sociale", "name": "MOHAMED SAJID", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_6804_Fr_692ac82f_entities.json",
        "instrument_id": "instr_2_19_592",
        "expected": [
            {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI", "type": "issuer"},
            {"role": "Le ministre de l'économie et des finances", "name": "MOHAMED BENCHAABOUN", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_6804_Fr_692ac82f_entities.json",
        "instrument_id": "instr_2_19_575",
        "expected": [
            {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI", "type": "issuer"},
            {"role": "Le ministre de l'économie et des finances", "name": "MOHAMED BENCHAABOUN", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_04_554",
        "expected": [
            {"role": "Premier Ministre", "name": "Driss Jettou", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "AHMED TOUFIQ", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_05_1577",
        "expected": [
            {"role": "Premier Ministre", "name": "Driss Jettou", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "AHMED TOUFIQ", "type": "contreseing"},
            {"role": "Le ministre des finances et de la privatisation", "name": "FATHALLAH OUALALOU", "type": "contreseing"},
            {"role": "Le ministre chargé de la modernisation des secteurs publics", "name": "MOHAMED BOUSSAID", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_09_494",
        "expected": [
            {"role": "Premier Ministre", "name": "Abbas El Fassi", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "AHMED TOUFIQ", "type": "contreseing"},
            {"role": "Le ministre de l'économie et des finances", "name": "SALAHEDDINE MEZOUAR", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_10_552",
        "expected": [
            {"role": "Premier Ministre", "name": "ABBAS EL FASSI", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "AHMED TOUFIQ", "type": "contreseing"},
            {"role": "Le ministre de l'économie et des finances", "name": "SALAHEDDINE MEZOUAR", "type": "contreseing"},
            {"role": "Le ministre délégué auprès du premier ministre, chargé de la modernisation des secteurs publics", "name": "MOHAMED SAAD ALAMI", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_91_609",
        "expected": [
            {"role": "Premier Ministre", "name": "MOHAMED KARIM AMRANI", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "ABDELKEBIR M'DAGHRI ALAOUI", "type": "contreseing"},
            {"role": "Le ministre des finances", "name": "MOHAMED BERRADA", "type": "contreseing"},
            {"role": "Le ministre délégué auprès du premier ministre, chargé des affaires administratives", "name": "AZIZ HASBI", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_11_696",
        "expected": [
            {"role": None, "name": "ABBAS EL FASSI", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "AHMED TOUFIQ", "type": "contreseing"},
            {"role": "Le ministre de la culture", "name": "BENSALEM HIMMICH", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7488-bis_Fr_entities.json",
        "instrument_id": "instr_2_24_927",
        "expected": [
            {"role": None, "name": "AZIZ AKHANNOUCH", "type": "issuer"},
            {"role": "Le ministre des Habous et des affaires islamiques", "name": "AHMED TOUFIQ", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7496_Fr_entities.json",
        "instrument_id": "instr_2_26_183",
        "expected": [
            {"role": "Chef du Gouvernement", "name": "AZIZ AKHANNOUCH", "type": "issuer"},
            {"role": "La ministre de l'économie et des finances", "name": "NADIA FETTAH", "type": "contreseing"},
            {"role": "La ministre de la transition énergétique et du développement durable", "name": "LEILA BENALI", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7480_Fr_entities.json",
        "instrument_id": "instr_2_20_716",
        "expected": [
            {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI", "type": "issuer"},
            {"role": "Le ministre de l'énergie, des mines et de l'environnement", "name": "AZIZ RABBAH", "type": "contreseing"},
            {"role": "Le ministre de l'industrie, du commerce, de l'économie verte et numérique", "name": "MOULAY HAFID EL ALAMI", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7480_Fr_entities.json",
        "instrument_id": "instr_2_25_1133",
        "expected": [
            {"role": "Chef du Gouvernement", "name": "AZIZ AKHANNOUCH", "type": "issuer"},
            {"role": "Le ministre délégué auprès de la ministre de l'économie et des finances, chargé du budget", "name": "FOUZI LEKJAA", "type": "contreseing"},
        ],
    },
    {
        "file": "ar_BO_7421_Ar_a7e9d588_entities.json",
        "instrument_id": "instr_2_25_473",
        "expected": [
            {"role": "رئيس الحكومة", "name": "عزيز  اخنوش", "type": "issuer"},
        ],
    },
    {
        "file": "ar_BO_7421_Ar_a7e9d588_entities.json",
        "instrument_id": "instr_2_25_565",
        "expected": [
            {"role": "رئيس الحكومة", "name": "عزيز اخنوش", "type": "issuer"},
            {"role": "الوزيرة المنتدبة لدى رئيس الحكومة المكلفة باالنتقال الرقمي وااصلح االدارة", "name": "امل الافلح", "type": "contreseing"},
        ],
    },
    {
        "file": "ar_BO_7506_Ar_entities.json",
        "instrument_id": "instr_2_26_23",
        "expected": [
            {"role": "رئيس الحكومة", "name": "عزيز اخنوش", "type": "issuer"},
            {"role": "وزير الافلحة والصيد البحري والتنمي ة القروية والمياه والغابات", "name": "احمد البواري", "type": "contreseing"},
            {"role": "وزير التعليم العالي والبحث العلمي واالبتكار", "name": "عز الدين المداوي", "type": "contreseing"},
            {"role": "الوزير المنتدب لدى وزيرة االقتصا د والمالية المكلف بالميزانية", "name": "فوزي لقجع", "type": "contreseing"},
            {"role": "الوزيرة المنتدبة لدى رئيس الحكوم ة المكلفة باالنتقال الرقمي وااصلح االدارة", "name": "امل الافلح", "type": "contreseing"},
        ],
    },
    {
        "file": "ar_BO_7506_Ar_entities.json",
        "instrument_id": "instr_2_25_885",
        "expected": [
            {"role": "رئيس الحكومة", "name": "عزيز اخنوش", "type": "issuer"},
            {"role": "وزير العدل", "name": "عبد اللطيف وهبي", "type": "contreseing"},
            {"role": "الوزير المنتدب لدى وزيرة االقتصاد والمالي ة المكلف بالميزانية", "name": "فوزي لقجع", "type": "contreseing"},
        ],
    },
    {
        "file": "ar_BO_7415_Ar_entities.json",
        "instrument_id": "instr_2_25_365",
        "expected": [
            {"role": "رئيس الحكومة", "name": "عزيز اخنوش", "type": "issuer"},
            {"role": "وزير الشباب والثقافة والتواصل", "name": "محمد المهدي بنسعيد", "type": "contreseing"},
            {"role": "الوزير المنتدب لدى وزيرة االقتصاد والمالية المكلف بالميزانية", "name": "فوزي لقجع", "type": "contreseing"},
        ],
    },
    {
        "file": "fr_BO_7510_Fr_entities.json",
        "instrument_id": "instr_181_26",
        "expected": [
            {"role": "Le ministre délégué auprès de la ministre de l'économie et des finances, chargé du budget", "name": "FOUZI LEKJAA", "type": "contreseing"},
            {"role": "La secrétaire d'Etat auprès du ministre de l'agriculture, de la pêche maritime, du développement rural et des eaux et forêts, chargée de la pêche maritime", "name": "ZAKIA DRIOUICH", "type": "contreseing"},
        ],
    },
]
