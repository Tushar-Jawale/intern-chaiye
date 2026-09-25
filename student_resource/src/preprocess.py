"""
Preprocessing Module for Business Entity Resolution
Handles: text cleaning, normalization, transliteration, legal suffix standardization
"""
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Optional


# ── Legal suffix standardization maps ──────────────────────────
LEGAL_SUFFIXES = {
    # US
    'inc': 'incorporated', 'incorp': 'incorporated', 'incorporated': 'incorporated',
    'corp': 'corporation', 'corporation': 'corporation',
    'llc': 'llc', 'l.l.c': 'llc', 'l.l.c.': 'llc',
    'llp': 'llp', 'l.l.p': 'llp', 'l.l.p.': 'llp',
    'ltd': 'limited', 'limited': 'limited',
    'co': 'company', 'company': 'company',
    'lp': 'lp', 'l.p': 'lp', 'l.p.': 'lp',
    'pllc': 'pllc',
    'pa': 'pa', 'p.a': 'pa', 'p.a.': 'pa',
    'pc': 'pc', 'p.c': 'pc', 'p.c.': 'pc',
    'na': 'na', 'n.a': 'na', 'n.a.': 'na',
    # India
    'pvt': 'private', 'private': 'private', 'pvt.': 'private',
    'priv': 'private',
    # France
    'sarl': 'sarl', 's.a.r.l': 'sarl', 's.a.r.l.': 'sarl',
    'sas': 'sas', 's.a.s': 'sas', 's.a.s.': 'sas',
    'sasu': 'sasu',
    'sa': 'sa', 's.a': 'sa', 's.a.': 'sa',
    'sci': 'sci', 's.c.i': 'sci', 's.c.i.': 'sci',
    'eurl': 'eurl',
    'snc': 'snc',
    'scp': 'scp',
    'groupe': 'groupe',
    'cie': 'company',
    'et': 'and',
    'fils': 'fils',
    # Generic
    'assoc': 'associates', 'associates': 'associates',
    'intl': 'international', 'international': 'international',
    'natl': 'national', 'national': 'national',
    'grp': 'group', 'group': 'group',
    'svcs': 'services', 'svc': 'services', 'services': 'services',
    'mfg': 'manufacturing', 'manufacturing': 'manufacturing',
    'enterprises': 'enterprises', 'enterprise': 'enterprises',
    'foundation': 'foundation', 'fdn': 'foundation',
    'holdings': 'holdings', 'holding': 'holdings',
    'solutions': 'solutions', 'soln': 'solutions',
    'technologies': 'technologies', 'tech': 'technologies',
    'partners': 'partners',
    'industries': 'industries',
    'consultants': 'consultants', 'consulting': 'consulting',
    'investments': 'investments',
    'properties': 'properties',
    'ventures': 'ventures',
    'dba': 'dba',
}

# ── Address abbreviation expansion ─────────────────────────────
ADDRESS_ABBREVS = {
    'st': 'street', 'str': 'street',
    'rd': 'road',
    'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'cir': 'circle',
    'pkwy': 'parkway',
    'hwy': 'highway',
    'ter': 'terrace', 'terr': 'terrace',
    'trl': 'trail',
    'sq': 'square',
    'apt': 'apartment',
    'ste': 'suite',
    'fl': 'floor',
    'bldg': 'building',
    'dept': 'department',
    'no': 'number',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    # French address abbreviations
    'r': 'rue', 'r.': 'rue',
    'bd': 'boulevard',
    'all': 'allee',
    'imp': 'impasse',
    'ch': 'chemin',
    'rte': 'route',
    'pl': 'place',
    # Indian
    'nagar': 'nagar',
    'marg': 'marg',
    'chowk': 'chowk',
    'gali': 'gali',
    'kh': 'khasra',
}

# ── Indian state names: Devanagari → English ───────────────────
INDIAN_STATES_TRANSLITERATION = {
    'उत्तर प्रदेश': 'uttar pradesh',
    'महाराष्ट्र': 'maharashtra',
    'कर्नाटक': 'karnataka',
    'ಕರ್ನಾಟಕ': 'karnataka',  # Kannada script
    'तमिल नाडु': 'tamil nadu',
    'தமிழ் நாடு': 'tamil nadu',  # Tamil script
    'राजस्थान': 'rajasthan',
    'गुजरात': 'gujarat',
    'मध्य प्रदेश': 'madhya pradesh',
    'बिहार': 'bihar',
    'पश्चिम बंगाल': 'west bengal',
    'तेलंगाना': 'telangana',
    'आंध्र प्रदेश': 'andhra pradesh',
    'केरल': 'kerala',
    'छत्तीसगढ़': 'chhattisgarh',
    'हरियाणा': 'haryana',
    'दिल्ली': 'delhi',
    'पंजाब': 'punjab',
    'झारखण्ड': 'jharkhand',
    'उत्तराखंड': 'uttarakhand',
    'ओडिशा': 'odisha',
    'असम': 'assam',
    'गोवा': 'goa',
    'हिमाचल प्रदेश': 'himachal pradesh',
    'जम्मू और कश्मीर': 'jammu and kashmir',
    'त्रिपुरा': 'tripura',
    'मेघालय': 'meghalaya',
    'मणिपुर': 'manipur',
    'नागालैंड': 'nagaland',
    'मिजोरम': 'mizoram',
    'अरुणाचल प्रदेश': 'arunachal pradesh',
    'सिक्किम': 'sikkim',
    'चंडीगढ़': 'chandigarh',
}


def normalize_unicode(text: str) -> str:
    """NFKD normalize and strip combining marks for accent-insensitive matching."""
    if not text:
        return ''
    # NFKD decomposition
    nfkd = unicodedata.normalize('NFKD', text)
    # Remove combining diacritical marks (accents) for Latin text
    # But preserve Devanagari/Tamil/Kannada combining marks
    result = []
    for ch in nfkd:
        cat = unicodedata.category(ch)
        if cat == 'Mn':  # Combining mark
            # Check if it's a Latin combining mark (accent) vs Indic
            cp = ord(ch)
            if cp < 0x0300 or (cp >= 0x0300 and cp <= 0x036F):
                # Latin combining marks - remove accents
                continue
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def transliterate_indic(text: str) -> str:
    """
    Transliterate Devanagari/Tamil/Kannada text to Latin script.
    Uses a practical approach: try indic_transliteration library,
    fall back to unidecode.
    """
    if not text:
        return ''
    
    # Check if text contains Indic characters
    has_indic = False
    for ch in text:
        cp = ord(ch)
        # Devanagari: 0900-097F, Tamil: 0B80-0BFF, Kannada: 0C80-0CFF
        # Telugu: 0C00-0C7F, Bengali: 0980-09FF, Gujarati: 0A80-0AFF
        if (0x0900 <= cp <= 0x097F or 0x0980 <= cp <= 0x09FF or
            0x0A00 <= cp <= 0x0A7F or 0x0A80 <= cp <= 0x0AFF or
            0x0B00 <= cp <= 0x0B7F or 0x0B80 <= cp <= 0x0BFF or
            0x0C00 <= cp <= 0x0CFF):
            has_indic = True
            break
    
    if not has_indic:
        return text
    
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as itrans
        
        # Detect script and transliterate
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                return itrans(text, sanscript.DEVANAGARI, sanscript.IAST)
            elif 0x0B80 <= cp <= 0x0BFF:
                return itrans(text, sanscript.TAMIL, sanscript.IAST)
            elif 0x0C80 <= cp <= 0x0CFF:
                return itrans(text, sanscript.KANNADA, sanscript.IAST)
            elif 0x0C00 <= cp <= 0x0C7F:
                return itrans(text, sanscript.TELUGU, sanscript.IAST)
            elif 0x0980 <= cp <= 0x09FF:
                return itrans(text, sanscript.BENGALI, sanscript.IAST)
            elif 0x0A80 <= cp <= 0x0AFF:
                return itrans(text, sanscript.GUJARATI, sanscript.IAST)
    except ImportError:
        pass
    
    # Fallback: unidecode
    try:
        from unidecode import unidecode
        return unidecode(text)
    except ImportError:
        pass
    
    return text


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, normalize whitespace, remove special chars."""
    if not isinstance(text, str) or text == 'nan' or text == 'null':
        return ''
    
    text = text.lower().strip()
    # Replace & with 'and'
    text = text.replace('&', ' and ')
    text = text.replace('<<', '').replace('>>', '')
    # Normalize attached punctuation: 'obsidian,-llc' → 'obsidian llc'
    text = re.sub(r'[,;:\-_/]+', ' ', text)
    # Remove stray punctuation except periods (keep abbreviations like C.I.T.)
    text = re.sub(r'[()\[\]{}"\']', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' .')
    return text


def standardize_legal_suffix(name: str) -> str:
    """Standardize legal entity suffixes in business names."""
    if not name:
        return ''
    
    tokens = name.split()
    result = []
    for token in tokens:
        clean_token = token.strip('.,;:()[]')
        lookup = clean_token.replace('.', '')
        if lookup in LEGAL_SUFFIXES:
            result.append(LEGAL_SUFFIXES[lookup])
        else:
            result.append(token)
    
    return ' '.join(result)


def extract_name_core(name: str) -> str:
    """
    Extract the core business name by removing legal suffixes and stopwords.
    Used for blocking keys.
    """
    if not name:
        return ''
    
    stopwords = {'the', 'of', 'and', 'for', 'in', 'at', 'by', 'a', 'an', 'to', 'on',
                 'dba', 'llc', 'llp', 'incorporated', 'corporation', 'limited', 
                 'private', 'company', 'sarl', 'sas', 'sasu', 'sa', 'sci', 'eurl',
                 'snc', 'groupe', 'fils', 'et'}
    
    tokens = name.split()
    # Strip all punctuation from each token before checking stopwords
    core = []
    for t in tokens:
        cleaned = re.sub(r'[^a-z0-9\u0900-\u0DFF]', '', t)  # Keep alphanumeric + Indic
        if cleaned and cleaned not in stopwords and len(cleaned) > 1:
            core.append(cleaned)
    return ' '.join(core) if core else name


def replace_indian_states(text: str) -> str:
    """Replace Indic script state names with English equivalents."""
    if not text:
        return text
    for indic, english in INDIAN_STATES_TRANSLITERATION.items():
        if indic in text:
            text = text.replace(indic, english)
    return text


def extract_numbers(text: str) -> str:
    """Extract all numeric tokens from text (for address number matching)."""
    if not text:
        return ''
    return ' '.join(re.findall(r'\d+', text))


def preprocess_name(name: str) -> dict:
    """
    Full preprocessing pipeline for business_name.
    Returns dict with multiple representations for different matching stages.
    """
    if not isinstance(name, str) or name == 'nan' or name == 'null' or pd.isna(name):
        return {
            'name_clean': '',
            'name_transliterated': '',
            'name_core': '',
            'name_normalized': '',
        }
    
    # Step 1: Basic cleaning
    cleaned = clean_text(name)
    
    # Step 2: Replace Indian state names in Indic scripts
    cleaned = replace_indian_states(cleaned)
    
    # Step 3: Transliterate Indic scripts
    transliterated = transliterate_indic(cleaned)
    transliterated = clean_text(transliterated)  # re-clean after transliteration
    
    # Step 4: Unicode normalization (remove accents)
    normalized = normalize_unicode(transliterated)
    normalized = normalized.lower()
    
    # Step 5: Standardize legal suffixes
    standardized = standardize_legal_suffix(normalized)
    
    # Step 6: Extract core name (without suffixes/stopwords)
    core = extract_name_core(standardized)
    
    return {
        'name_clean': cleaned,
        'name_transliterated': transliterated,
        'name_core': core,
        'name_normalized': standardized,
    }


def preprocess_address(address: str) -> dict:
    """
    Full preprocessing pipeline for business_address.
    Returns dict with multiple representations.
    """
    if not isinstance(address, str) or address == 'nan' or address == 'null' or pd.isna(address):
        return {
            'addr_clean': '',
            'addr_transliterated': '',
            'addr_normalized': '',
            'addr_numbers': '',
        }
    
    # Step 1: Basic cleaning
    cleaned = clean_text(address)
    
    # Step 2: Replace Indian state names
    cleaned = replace_indian_states(cleaned)
    
    # Step 3: Transliterate Indic scripts  
    transliterated = transliterate_indic(cleaned)
    transliterated = clean_text(transliterated)
    
    # Step 4: Unicode normalization
    normalized = normalize_unicode(transliterated)
    normalized = normalized.lower()
    
    # Step 5: Extract numbers
    numbers = extract_numbers(normalized)
    
    return {
        'addr_clean': cleaned,
        'addr_transliterated': transliterated,
        'addr_normalized': normalized,
        'addr_numbers': numbers,
    }


def preprocess_dataframe(df: pd.DataFrame, batch_size: int = 50000) -> pd.DataFrame:
    """
    Apply full preprocessing to a source dataframe.
    Processes in batches to show progress on large datasets.
    """
    import sys
    
    total = len(df)
    print(f"  Preprocessing {total:,} records...")
    
    # Pre-allocate lists
    name_clean = []
    name_transliterated = []
    name_core = []
    name_normalized = []
    addr_clean = []
    addr_transliterated = []
    addr_normalized = []
    addr_numbers = []
    
    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            # Process name
            n = preprocess_name(row.get('business_name', ''))
            name_clean.append(n['name_clean'])
            name_transliterated.append(n['name_transliterated'])
            name_core.append(n['name_core'])
            name_normalized.append(n['name_normalized'])
            
            # Process address
            a = preprocess_address(row.get('business_address', ''))
            addr_clean.append(a['addr_clean'])
            addr_transliterated.append(a['addr_transliterated'])
            addr_normalized.append(a['addr_normalized'])
            addr_numbers.append(a['addr_numbers'])
        
        processed = min(i + batch_size, total)
        pct = processed / total * 100
        print(f"    [{processed:>10,}/{total:,}] {pct:.1f}%", flush=True)
    
    # Add new columns
    result = df.copy()
    result['name_clean'] = name_clean
    result['name_transliterated'] = name_transliterated
    result['name_core'] = name_core
    result['name_normalized'] = name_normalized
    result['addr_clean'] = addr_clean
    result['addr_transliterated'] = addr_transliterated
    result['addr_normalized'] = addr_normalized
    result['addr_numbers'] = addr_numbers
    
    return result


if __name__ == '__main__':
    """Test preprocessing on a small sample to verify correctness."""
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Test cases from EDA
    test_cases = [
        # (name, address, country)
        ("Raj Investments LLP", "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu", "India"),
        ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu", "India"),
        ("एसएस फूड प्राइवेट लिमिटेड", "AF-0684, NANDGRAM, GHAZIABAD, उत्तर प्रदेश", "India"),
        ("Ss Food Private Limited", "Af-684, Nandgram, Ghaziabad, Uttar Pradesh", "India"),
        ("Payne Enterprises", "3315 Fremont Street, Peoria, IL", "US"),
        ("Payne Énterprises", "3315 FREMONT ST, PEORIA, IL", "US"),
        ("Obsidian, LLC", "3907 Hamilton Road, Deer Park, WA", "US"),
        ("Obsidian,-LLC", "3907 HAMILTON RD, DEER PARK CIYT, WA", "US"),
        ("ZNB Club SARL", "Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac", "France"),
        ("Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France", "France"),
    ]
    
    print("=" * 80)
    print("PREPROCESSING TEST")
    print("=" * 80)
    
    for name, addr, country in test_cases:
        n = preprocess_name(name)
        a = preprocess_address(addr)
        print(f"\nOriginal:      name='{name}'")
        print(f"               addr='{addr}' | country={country}")
        print(f"  name_clean:         {n['name_clean']}")
        print(f"  name_transliterated: {n['name_transliterated']}")
        print(f"  name_normalized:    {n['name_normalized']}")
        print(f"  name_core:          {n['name_core']}")
        print(f"  addr_normalized:    {a['addr_normalized']}")
        print(f"  addr_numbers:       {a['addr_numbers']}")
        print(f"  ---")
    
    print("\n✅ Preprocessing test complete!")
