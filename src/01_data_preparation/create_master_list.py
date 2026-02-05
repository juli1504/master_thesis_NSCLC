import pandas as pd
import os
from pathlib import Path

# --- KONFIGURATION ---
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent 

# Pfade definieren
DIR_RAW = PROJECT_ROOT / "data" / "raw"
PATH_CLINICAL = DIR_RAW / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
PATH_METADATA = DIR_RAW / "metadata.csv"
DIR_XML = DIR_RAW / "xml"
DIR_DICOM_ROOT = DIR_RAW / "dicom" 

# Output Pfad
PATH_OUTPUT = PROJECT_ROOT / "data" / "processed" / "patient_overview.csv"

def main():
    print(f"--- Start Data Mapping (VS Code Mode) ---")
    print(f"Projekt-Root erkannt als: {PROJECT_ROOT}")

    # 1. KLINISCHE DATEN LADEN
    if not PATH_CLINICAL.exists():
        raise FileNotFoundError(f"Klinische CSV nicht gefunden: {PATH_CLINICAL}")
    
    print(f"Lade Clinical Data...")
    df_clinical = pd.read_csv(PATH_CLINICAL)
    df_clinical.columns = df_clinical.columns.str.strip() 
    
    df_labels = df_clinical[['Case ID', 'Histology']].copy()
    df_labels = df_labels.rename(columns={'Case ID': 'Subject ID'})
    print(f"-> {len(df_labels)} Patienten in Clinical CSV.")

    # 2. XML DATEN SCANNEN
    print(f"Scanne XML Ordner...")
    if not DIR_XML.exists():
        print(f"ACHTUNG: XML Ordner nicht gefunden unter {DIR_XML}")
        xml_ids = []
    else:
        xml_files = [f.name for f in DIR_XML.glob("*.xml")]
        xml_ids = [f.replace('.xml', '') for f in xml_files]
    
    df_xml = pd.DataFrame({'Subject ID': xml_ids, 'Has_XML': True})
    print(f"-> {len(df_xml)} XML-Dateien gefunden.")

    # 3. DICOM DATEN SCANNEN (VIA METADATA)
    print(f"Lade Metadata & prüfe DICOM Pfade...")
    if not PATH_METADATA.exists():
        raise FileNotFoundError(f"Metadata CSV nicht gefunden: {PATH_METADATA}")

    df_meta = pd.read_csv(PATH_METADATA)
    
    # Nur CTs filtern
    df_ct = df_meta[df_meta['Modality'] == 'CT'].copy()
    
    def check_path_exists(path_str):
        clean_path = path_str.replace('.\\', '').replace('\\', os.sep)
        full_path = DIR_DICOM_ROOT / clean_path
        return full_path.exists()

    df_ct['Path_Exists'] = df_ct['File Location'].apply(check_path_exists)
    
    available_ct_ids = df_ct[df_ct['Path_Exists'] == True]['Subject ID'].unique()
    df_dicom = pd.DataFrame({'Subject ID': available_ct_ids, 'Has_DICOM': True})
    
    print(f"-> {len(df_dicom)} Patienten haben auffindbare CT-Ordner.")

    # 4. MERGE
    all_ids = sorted(list(set(df_labels['Subject ID']) | set(xml_ids) | set(available_ct_ids)))
    df_master = pd.DataFrame({'Subject ID': all_ids})
    
    df_master = df_master.merge(df_labels, on='Subject ID', how='left')
    df_master = df_master.merge(df_xml, on='Subject ID', how='left')
    df_master = df_master.merge(df_dicom, on='Subject ID', how='left')
    
    df_master['Has_XML'] = df_master['Has_XML'].fillna(False)
    df_master['Has_DICOM'] = df_master['Has_DICOM'].fillna(False)

    # 5. ERGEBNIS PRÜFEN
    df_ready = df_master[
        (df_master['Histology'].notna()) & 
        (df_master['Has_XML'] == True) & 
        (df_master['Has_DICOM'] == True)
    ]

    print("-" * 30)
    print(f"GESAMT-STATUS:")
    print(f"Patienten in Liste gesamt: {len(df_master)}")
    print(f"VOLLSTÄNDIGE DATENSÄTZE (Ready for ML): {len(df_ready)}")
    print("-" * 30)

    PATH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(PATH_OUTPUT, index=False)
    print(f"Datei gespeichert: {PATH_OUTPUT}")

if __name__ == "__main__":
    main()