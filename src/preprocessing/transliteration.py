from unidecode import unidecode
import pandas as pd

# Empirical mapping of Devanagari loanwords to standard English spellings
PHONETIC_MAPPING = {
    'limittedd': 'limited',
    'limittett': 'limited',
    'limirrrrdd': 'limited',
    'praaivett': 'private',
    'praiveett': 'private',
    'praaibhett': 'private',
    'piraiveett': 'private',
    'praivrrrr': 'private',
    'praa': 'private',
    'li': 'limited',
    'elelpii': 'llp',
    'ttek': 'tech',
    'inttrneshnl': 'international',
    'globl': 'global',
    'inphraa': 'infra',
    'proddktts': 'products',
    'bilddrs': 'builders',
    'inphraasttrkcr': 'infrastructure',
    'knsltting': 'consulting',
    'ddevlprs': 'developers',
    'investtmentt': 'investment',
    'sonlyuushns': 'solutions',
    'knsttrkshns': 'constructions',
    'phuudds': 'foods',
    'phuudd': 'food',
    'injiiniyring': 'engineering',
    'ttredding': 'trading',
    'inddsttriij': 'industries',
    'enttrpraaijej': 'enterprises',
    'tteknolonjiij': 'technologies',
    'tteknolonjii': 'technology',
    'lkssmii': 'lakshmi',
    'shrii': 'shri',
    'vencrs': 'ventures',
    'esttett': 'estate',
    'sonphttveyr': 'software',
    'eksportts': 'exports',
    'bijnes': 'business',
    'phaaunddeshn': 'foundation',
    'sisttms': 'systems',
    'inphottek': 'infotech',
    'impeks': 'impex',
    'phaainens': 'finance',
    'aaiittii': 'it',
    'mainejmentt': 'management',
    'maarketting': 'marketing',
    'enrjii': 'energy',
    'proddyuusr': 'producer',
    'projektts': 'projects',
    'srvisej': 'services',
    'egro': 'agro',
    'pronprttiij': 'properties',
    'knslttensii': 'consultancy',
    'paavr': 'power',
    'honspittailittii': 'hospitality',
    'aaiinnsii': 'inc',
    'kaNpnii': 'company',
    'kampnii': 'company',
    'kompnii': 'company',
    'kaarporeshn': 'corporation',
}

def transliterate_text(text: str) -> str:
    """
    Transliterates non-Latin scripts (Devanagari, Kannada, etc.) to Latin characters.
    Uses unidecode which handles almost all Unicode characters.
    Applies phonetic mapping to correct common English loanwords.
    """
    if pd.isna(text):
        return ""
    
    # Fast path: check if it only has ascii
    try:
        text.encode('ascii')
        return text
    except UnicodeEncodeError:
        trans = unidecode(str(text))
        # Apply word-by-word mapping for loanwords
        words = trans.split()
        mapped = [PHONETIC_MAPPING.get(w.lower(), w) for w in words]
        return " ".join(mapped)

def transliterate_series(series: pd.Series) -> pd.Series:
    """Applies transliteration to a pandas Series efficiently."""
    return series.apply(transliterate_text)
