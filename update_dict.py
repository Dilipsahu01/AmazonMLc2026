import re

new_entries = {
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
    'kaarporeshn': 'corporation'
}

filepath = "src/preprocessing/transliteration.py"
with open(filepath, "r") as f:
    content = f.read()

dict_str = "PHONETIC_MAPPING = {\n"
for k, v in new_entries.items():
    dict_str += f"    '{k}': '{v}',\n"
dict_str += "}"

content = re.sub(r'PHONETIC_MAPPING = \{[^}]+\}', dict_str, content)

with open(filepath, "w") as f:
    f.write(content)
