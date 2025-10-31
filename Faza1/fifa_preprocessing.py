import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ds = pd.read_csv("../fifa21 raw data v2.csv", low_memory=False)

# Quick check per kolona
#print(ds.shape)
#print(ds.columns)
ds.head()
ds.info()

# Missing vlaues count
ds.isnull().sum()
#print("Duplicate rows:", ds.duplicated().sum()) #checking for duplicates

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

#PCA
cols_to_use = [ 'Height', 'Weight', 'W/F', 'SM', 'IR']
cols_to_use = [c for c in cols_to_use if c in ds.columns]
ds_pca = ds[cols_to_use].copy()
ds_pca = ds_pca.dropna(thresh=len(ds_pca.columns)//2)
for col in ds_pca.columns:
    ds_pca[col] = pd.to_numeric(ds_pca[col].replace('[^0-9.-]', '', regex=True), errors='coerce')

ds_pca = ds_pca.fillna(ds_pca.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(ds_pca)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
explained_variance = pca.explained_variance_ratio_.cumsum()

pca_df.shape, explained_variance[-1]
#star rating columns, removing stars/symbols
star_cols = ['W/F', 'SM', 'IR']
for col in star_cols:
    if col in ds.columns:
        ds[col] = ds[col].astype(str).str.extract(r'(\d+)').astype(int)

# Aggregation
club_avg_value = ds.groupby('Club')['Value'].mean().reset_index()
club_avg_value_sorted = club_avg_value.sort_values(by='Value', ascending=False)

nationality_wage_total = ds.groupby('Nationality')['Wage'].sum().reset_index()
nationality_wage_total_sorted = nationality_wage_total.sort_values(by='Wage', ascending=False)

players_per_club = ds.groupby('Club').size().reset_index(name='Number of Players')
players_per_club_sorted = players_per_club.sort_values(by='Number of Players', ascending=False)

club_median_age = ds.groupby('Club')['Age'].median().reset_index()
club_median_age_sorted = club_median_age.sort_values(by='Age', ascending=False)

club_ova_pot_agg = ds.groupby('Club').agg(
    AverageOVA=('↓OVA', 'mean'),
    MaxOVA=('↓OVA', 'max'),
    MinOVA=('↓OVA', 'min'),
    AveragePOT=('POT', 'mean'),
    MaxPOT=('POT', 'max'),
    MinPOT=('POT', 'min')
).reset_index()

def simplify_position(pos):
    if isinstance(pos, str):
        pos = pos.upper()
        if pos in ['GK']:
            return 'Goalkeeper'
        elif pos in ['CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB']:
            return 'Defender'
        elif pos in ['CDM', 'CM', 'LCM', 'RCM', 'CAM', 'LM', 'RM']:
            return 'Midfielder'
        elif pos in ['ST', 'CF', 'LW', 'RW', 'LF', 'RF']:
            return 'Forward'
        else:
            return 'Other'
    return 'Unknown'

ds['Best Position'] = ds['Best Position'].apply(simplify_position) #added per tabele
club_ova_pot_agg_sorted = club_ova_pot_agg.sort_values(by='AverageOVA', ascending=False)

nationality_ova_pot_agg = ds.groupby('Nationality').agg(
    AverageOVA=('↓OVA', 'mean'),
    MaxOVA=('↓OVA', 'max'),
    MinOVA=('↓OVA', 'min'),
    AveragePOT=('POT', 'mean'),
    MaxPOT=('POT', 'max'),
    MinPOT=('POT', 'min')
).reset_index()

nationality_ova_pot_agg_sorted = nationality_ova_pot_agg.sort_values(by='AverageOVA', ascending=False)

# Aggregation by Position Category for OVA and POT (average, max, min)
position_ova_pot_agg = ds.groupby(ds['Best Position'].apply(simplify_position)).agg(
    AverageOVA=('↓OVA', 'mean'),
    MaxOVA=('↓OVA', 'max'),
    MinOVA=('↓OVA', 'min'),
    AveragePOT=('POT', 'mean'),
    MaxPOT=('POT', 'max'),
    MinPOT=('POT', 'min')
).reset_index()

# Sorting by average OVA
position_ova_pot_agg_sorted = position_ova_pot_agg.sort_values(by='AverageOVA', ascending=False)
club_ova_pot_agg_sorted.head(), nationality_ova_pot_agg_sorted.head(), position_ova_pot_agg_sorted.head()

position_avg_ova_pot_direct = ds.groupby(ds['Best Position'].apply(simplify_position))[['↓OVA', 'POT']].mean().reset_index()
position_avg_ova_pot_direct_sorted = position_avg_ova_pot_direct.sort_values(by='↓OVA', ascending=False)
position_avg_ova_pot_direct_sorted.head()
club_value_summary = ds.groupby('Club')['Value'].agg(
    TotalValue='sum',
    AverageValue='mean',
    MaxValue='max',
    MinValue='min',
    NumPlayers='size'
).reset_index()

print(club_avg_value_sorted.head())
print(nationality_wage_total_sorted.head())
print(players_per_club_sorted.head())
print(club_median_age_sorted.head())
print(club_value_summary.head())

#clean_file_path = "fifa21_cleaned.csv"

if 'Loan Date End' in ds.columns:
    ds = ds.drop(columns=['Loan Date End'])

if 'Hits' in ds.columns:
    club_avg_hits = ds.groupby('Club')['Hits'].transform('mean')
    ds['Hits'] = ds['Hits'].fillna(club_avg_hits)
    ds['Hits'] = ds['Hits'].fillna(ds['Hits'].mean())

drop_cols = [col for col in ds.columns if any(word in col.lower() for word in ['photo', 'flag', 'logo'])]
ds = ds.drop(columns=drop_cols, errors='ignore')

# Kontroll i vlerave unike për disa kolona kryesore
for col in ['Nationality', 'Club', 'Preferred Foot']:
    if col in ds.columns:
        print(f"\nColumns: {col}")
        print("Number of unique values:", ds[col].nunique())

#Trajtimi i kolonave me datat e kontrates
ds['Joined'] = pd.to_datetime(ds['Joined'], errors='coerce')

# ds['Loan_Date_Orig'] = ds['Loan Date End']
# ds['Loan Date End'] = pd.to_datetime(ds['Loan Date End'], errors='coerce')
# # Shfaq 10 rreshtat e pare dhe bej krahasimin para-pas   
# #NAN -> NaT
# print(ds[['Name', 'Loan_Date_Orig', 'Loan Date End']].head(10))

#krijimi i vetive te reja prej vetive ekzistuese
ds['SKILL'] = ds[['Dribbling', 'Curve', 'FK Accuracy', 'Long Passing', 'Ball Control']].mean(axis=1)
ds['MOVEMENT'] = ds[['Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance']].mean(axis=1)
ds['POWER'] = ds[['Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots']].mean(axis=1)
ds['MENTALITY'] = ds[['Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure']].mean(axis=1)
ds['DEFENDING'] = ds[['Marking', 'Standing Tackle', 'Sliding Tackle']].mean(axis=1)
ds['GOALKEEPING'] = ds[['GK Diving', 'GK Handling', 'GK Kicking', 'GK Positioning', 'GK Reflexes']].mean(axis=1)

# quick preview of the new features
# print(ds[['Name', 'SKILL', 'MOVEMENT', 'POWER', 'MENTALITY', 'DEFENDING', 'GOALKEEPING']].head())

#Diskretizimi ne baze te moshes (binning)
if 'Age' in ds.columns:
    ds['Age_Group'] = pd.cut(
        ds['Age'],
        bins=[15, 22, 30, 40, 50],  # define boundaries
        labels=['Young', 'Prime', 'Veteran', 'Retired'],  # assign names
        include_lowest=True
    )


#print(ds.iloc[35:101][['Name', 'Age_Group']])
print(ds[['Name','Age_Group']].head())

# Transformimi me normalizim
ds['Value'] = pd.to_numeric(ds['Value'], errors='coerce').fillna(0)
# Z-Score
#z_scaler = StandardScaler()
#ds['Value_Zscore'] = z_scaler.fit_transform(ds[['Value']])
# Min-Max
#mm_scaler = MinMaxScaler()
#ds['Value_MinMax'] = mm_scaler.fit_transform(ds[['Value']])
# Robust 
r_scaler = RobustScaler()
ds['Value_Robust'] = r_scaler.fit_transform(ds[['Value']]) #our choice
#print(ds[['Name', 'Value', 'Value_Zscore', 'Value_MinMax', 'Value_Robust']].head(10))

# Binarizim i Preferred Foot (po krijohet kolone e re)
ds['Preferred_Foot_Binary'] = ds['Preferred Foot'].apply(lambda x: 1 if x=='Right' else 0)
# Kontroll
print(ds[['Preferred Foot', 'Preferred_Foot_Binary']].head())
#print(ds['Preferred_Foot_Binary'].value_counts())

# dataset i ri
#ds.to_csv("fifa21_cleaned.csv", index=False)
#ds.to_excel("fifa21_cleaned.xlsx", index=False, engine='openpyxl') #saved as an excel file

selected_features = [
    'Name', 'Nationality', 'Club', 'Wage','Joined','Age_Group','Best Position','Preferred_Foot_Binary',
    'SKILL', 'MOVEMENT', 'POWER', 'MENTALITY', 'DEFENDING', 'GOALKEEPING','Value_Robust'
] 
#reducted height+weight needs to be added
#reduced W/F, SM, IR+... needs to be added

subset_ds = ds[selected_features].copy()
subset_ds.to_csv("fifa21_cleaned.csv", index=False)

