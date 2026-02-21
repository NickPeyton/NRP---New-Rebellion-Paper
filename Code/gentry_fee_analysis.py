
import pandas as pd
import numpy as np
import geopandas as gp
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import phonetics as ph
import spacy
from unidecode import unidecode
from difflib import SequenceMatcher
from tqdm import tqdm

# Set directory
os.chdir('C:/PhD/DissolutionProgramming/REB---Rebellion-Paper')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load("en_core_web_sm")

# --- Model Definition ---

def encode_surname(surname, max_len=24):
    CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0'
    CHARSET_DICT = {char: i + 1 for i, char in enumerate(CHARSET)}
    PAD = 0
    surname = unidecode(str(surname)).upper()
    surname = ''.join([char for char in surname if char in CHARSET])
    try:
        metaphone = ph.metaphone(surname)
    except:
        metaphone = ""
    encoded = [CHARSET_DICT[char] for char in surname]
    if len(encoded) < max_len: encoded += [PAD] * (max_len - len(encoded))
    else: encoded = encoded[:max_len]
    encoded = torch.tensor(encoded).long()
    encoded_metaphone = [CHARSET_DICT[char] for char in metaphone]
    if len(encoded_metaphone) < max_len: encoded_metaphone += [PAD] * (max_len - len(encoded_metaphone))
    else: encoded_metaphone = encoded_metaphone[:max_len]
    encoded_metaphone = torch.tensor(encoded_metaphone).long()
    return encoded, encoded_metaphone, metaphone

class CrossEncoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, fc_dim=32):
        super(CrossEncoder, self).__init__()
        self.name_embedding = nn.Embedding(28, embed_dim)
        self.metaphone_embedding = nn.Embedding(28, embed_dim)
        self.name_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.metaphone_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(4*2 * hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
    def name_encode(self, x):
        name_embedded = self.name_embedding(x)
        _, (name_hidden, _) = self.name_lstm(name_embedded)
        return torch.cat((name_hidden[0], name_hidden[1]), dim=1)
    def metaphone_encode(self, x):
        metaphone_embedded = self.metaphone_embedding(x)
        _, (metaphone_hidden, _) = self.metaphone_lstm(metaphone_embedded)
        return torch.cat((metaphone_hidden[0], metaphone_hidden[1]), dim=1)
    def forward(self, name1, metaphone1, name2, metaphone2):
        n1 = self.name_encode(name1)
        m1 = self.metaphone_encode(metaphone1)
        n2 = self.name_encode(name2)
        m2 = self.metaphone_encode(metaphone2)
        combined = torch.cat((n1, m1, n2, m2), dim=1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(combined))))

# --- Setup ---

model = CrossEncoder()
model.load_state_dict(torch.load('Code/ml_models/name_matcher/cross_encoder_1.pth', map_location=device))
model.to(device)
model.eval()

def extract_surname(name_str):
    if pd.isna(name_str) or not str(name_str).strip():
        return ""
    
    s_name = str(name_str)
    # Handle comma-separated names (often SURNAME, FIRSTNAME)
    if ',' in s_name:
        return s_name.split(',')[0].strip().lower()
        
    # Cleaning for the query
    s_name = re.sub(r'\b(lord|sir|master|steward|bailiff|auditor|of|the|care|court|military|senescal|captain|receiver|general|bailiff|instructor|poor|boys|rectory|count|king|commissioner|there|chief|sheriff|nottinghamshire|rutland|yorkshire|cumberland|westmorland|durham|lancashire|lincolnshire)\b', '', s_name, flags=re.IGNORECASE).strip()
    
    if not s_name: return ""
    
    doc = nlp(s_name)
    
    # Strategy:
    # 1. Look for the last PROPN in the whole string.
    # 2. If no PROPN, but there's a PERSON entity, take the last word of the PERSON entity.
    # 3. Fallback to the last word of the whole cleaned string.
    
    propns = [tok.text for tok in doc if tok.pos_ == "PROPN"]
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if propns:
        target = propns[-1]
    elif persons:
        target = persons[-1].split()[-1]
    else:
        target = s_name.split()[-1]

    return target.strip().lower()

gentry_gdf = gp.read_file('Data/Raw/GIS/BNG Projections/gentlemenInvolved.shp')
# Use the improved extraction for gentry list too
gentry_surnames = []
for name in gentry_gdf['gentleman'].dropna().unique():
    extracted = extract_surname(name)
    if extracted:
        gentry_surnames.append(extracted)
gentry_surnames = sorted(list(set(gentry_surnames)))

gentry_encoded = []
for s in gentry_surnames:
    n, m, meta = encode_surname(s)
    gentry_encoded.append({'surname': s, 'n': n.to(device), 'm': m.to(device), 'metaphone': meta})

north_list = ['NORTHUMBERLAND', 'CUMBERLAND', 'DURHAM', 'WESTMORLAND', 'YORKSHIRE, NORTH RIDING', 'LANCASHIRE', 'YORKSHIRE, WEST RIDING', 'YORKSHIRE, EAST RIDING', 'LINCOLNSHIRE']
line_items = pd.read_csv('Data/Raw/CSV/ValorLineItems.csv')
natl_archives = pd.read_csv('Data/Raw/CSV/NationalArchivesData.csv')
df = pd.merge(line_items, natl_archives[['name', 'county']], on='name', how='left')
north_fees = df[(df['county'].isin(north_list)) & (df['fee'] == 1)].copy()
north_fees['total_abs'] = pd.to_numeric(north_fees['total'], errors='coerce').abs()
north_fees = north_fees.dropna(subset=['total_abs', 'counterParty'])

# --- Matching Logic ---

def check_match_ml(row):
    cp = str(row['counterParty']).lower()
    for g_sur in gentry_surnames:
        if re.search(rf'\b{re.escape(g_sur)}\b', cp):
            return True, g_sur, 1.0, "Exact"
            
    extracted = extract_surname(cp)
    if not extracted or len(extracted) < 3:
        return False, None, 0.0, None
        
    n_q, m_q, meta_q = encode_surname(extracted)
    n_q = n_q.unsqueeze(0).to(device)
    m_q = m_q.unsqueeze(0).to(device)
    
    candidates = []
    with torch.no_grad():
        for g in gentry_encoded:
            # Pre-filter: Either phonetic similarity OR sequence similarity must be present
            # to even bother with the ML model, to avoid random high-scoring pairs.
            sm_ratio = SequenceMatcher(None, extracted, g['surname']).ratio()
            meta_ratio = SequenceMatcher(None, meta_q, g['metaphone']).ratio()
            
            if sm_ratio > 0.6 or meta_ratio > 0.8:
                score = model(n_q, m_q, g['n'].unsqueeze(0), g['m'].unsqueeze(0)).item()
                if score > 0.98: # Raised threshold
                    candidates.append((score, g['surname']))
    
    if candidates:
        best_score, best_surname = max(candidates)
        return True, best_surname, best_score, "ML"
        
    return False, None, 0.0, None

print("--- Performing ML-Enhanced Matching (with Pre-filters) ---")
results = [check_match_ml(row) for _, row in tqdm(north_fees.iterrows(), total=len(north_fees))]
north_fees['is_match'] = [r[0] for r in results]
north_fees['matched_surname'] = [r[1] for r in results]
north_fees['match_score'] = [r[2] for r in results]
north_fees['match_type'] = [r[3] for r in results]

# --- Results ---
total_val = north_fees['total_abs'].sum()
gen_val = north_fees[north_fees['is_match']]['total_abs'].sum()
print(f"\nSHARE OF VALUE: {(gen_val/total_val)*100:.2f}%")
print(north_fees[north_fees['is_match']]['match_type'].value_counts())

print("\n--- Sample of ML Matches ---")
ml_samples = north_fees[north_fees['match_type'] == 'ML'].sample(min(15, len(north_fees[north_fees['match_type'] == 'ML'])))
for _, row in ml_samples.iterrows():
    print(f"[{row['match_score']:.3f}] {row['counterParty']} -> {row['matched_surname']}")

print("\n--- Top Matches ---")
top = north_fees[north_fees['is_match']].sort_values('total_abs', ascending=False).head(10)
for _, row in top.iterrows():
    print(f"{row['total_abs']:>8.0f}d | {row['counterParty']:<40} | {row['matched_surname']} ({row['match_type']})")
