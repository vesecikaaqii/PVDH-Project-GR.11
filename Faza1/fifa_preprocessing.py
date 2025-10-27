import pandas as pd

ds = pd.read_csv("../fifa21 raw data v2.csv", low_memory=False)

# Quick check per kolona
#print(ds.shape)
#print(ds.columns)
ds.head()
ds.info()

# Missing vlaues count
ds.isnull().sum()

def convert_value(x):
    if isinstance(x, str):
        x = x.replace('€','').replace(',','')  # Remove € and commas
        if 'M' in x:
            return float(x.replace('M','')) * 1_000_000
        elif 'K' in x:
            return float(x.replace('K','')) * 1_000
        else:
            try:
                return float(x)
            except:
                return 0  
    return x

for col in ['Value','Wage','Release Clause','Hits']:
    ds[col] = ds[col].apply(convert_value)
    

# Konvertimi i height in to cm
def height_to_cm(x):
    if isinstance(x, str):
        x = x.strip()
        if 'cm' in x:  
            return int(x.replace('cm',''))
        elif "'" in x:  # in to cm
            try:
                feet, inches = x.split("'")
                inches = inches.replace('"','')
                return int(feet)*30.48 + int(inches)*2.54
            except:
                return None
        else:
            return None
    return x

ds['Height'] = ds['Height'].apply(height_to_cm).astype(int)


# Konvertimi i weight
def weight_to_kg(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if 'kg' in x:
            return int(x.replace('kg',''))
        elif 'lb' in x:  #lb to kg
            try:
                lbs = float(x.replace('lbs','').replace('lb',''))
                return round(lbs * 0.453592)  # 1 lb = 0.453592 kg
            except:
                return None
        else:
            return None
    return x

ds['Weight'] = ds['Weight'].apply(weight_to_kg).astype(int)

#star rating columns, removing stars/symbols
star_cols = ['W/F', 'SM', 'IR']
for col in star_cols:
    if col in ds.columns:
        ds[col] = ds[col].astype(str).str.extract(r'(\d+)').astype(int)

clean_file_path = "fifa21_cleaned.csv"


# Kontroll i vlerave unike për disa kolona kryesore
for col in ['Nationality', 'Club', 'Preferred Foot']:
    if col in ds.columns:
        print(f"\nKolona: {col}")
        print("Numri i vlerave unike:", ds[col].nunique())




# dataset i ri
ds.to_csv("fifa21_cleaned.csv", index=False)
#ds.to_excel("fifa21_cleaned.xlsx", index=False, engine='openpyxl') #saved as an excel file
ds.head()
ds.info()
 

 