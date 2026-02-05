import pandas as pd
import os
from pathlib import Path

# --- KONFIGURATION ---
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent 

DIR_RAW = PROJECT_ROOT / "data" / "raw"
PATH_CLINICAL = DIR_RAW / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
PATH_METADATA = DIR_RAW / "metadata.csv"
DIR_XML = DIR_RAW / "xml"
DIR_DICOM_ROOT = DIR_RAW / "dicom" 

PATH_OUTPUT = PROJECT_ROOT / "data" / "processed" / "patient_overview.csv"

def main():
    print(f"--- Start Data Mapping (Robust Mode) ---")
    
    # 1. KLINISCHE DATEN
    df_clinical = pd.read_csv(PATH_CLINICAL)
    df_clinical.columns = df_clinical.columns.str.strip()
    df_labels = df_clinical[['Case ID', 'Histology']].copy().rename(columns={'Case ID': 'Subject ID'})
    print(f"-> {len(df_labels)} Patienten in Clinical CSV.")

    # 2. XML DATEN
    xml_ids = [f.stem for f in DIR_XML.glob("*.xml")]
    df_xml = pd.DataFrame({'Subject ID': xml_ids, 'Has_XML': True})
    print(f"-> {len(df_xml)} XML-Dateien gefunden.")

    # 3. DICOM DATEN (Verbesserte Logik)
    df_meta = pd.read_csv(PATH_METADATA)
    df_meta.columns = df_meta.columns.str.strip()
    
    # Filter für CT (Sucht nach 'CT' im Text, egal ob 'CT' oder 'CT Image Storage')
    df_ct = df_meta[df_meta['Modality'].str.contains('CT', na=False)].copy()
    
    def check_path_exists(path_str):
        if pd.isna(path_str): return False
        # Bereinige den Pfad aus der CSV
        clean_path_str = path_str.lstrip('./').lstrip('.\\').replace('\\', os.sep).replace('/', os.sep)
        full_path = DIR_DICOM_ROOT / clean_path_str
        return full_path.exists()

    # Kleiner Debug-Print für den ersten Patienten
    sample_row = df_ct.iloc[0]
    sample_path_csv = sample_row['File Location']
    clean_sample = sample_path_csv.lstrip('./').lstrip('.\\').replace('\\', os.sep)
    print(f"\nDEBUG: Prüfe Pfad für {sample_row['Subject ID']}:")
    print(f"  Erwartet: {DIR_DICOM_ROOT / clean_sample}")
    print(f"  Existiert dieser Ordner? {'JA' if (DIR_DICOM_ROOT / clean_sample).exists() else 'NEIN'}\n")

    df_ct['Path_Exists'] = df_ct['File Location'].apply(check_path_exists)
    available_ct_ids = df_ct[df_ct['Path_Exists'] == True]['Subject ID'].unique()
    df_dicom = pd.DataFrame({'Subject ID': available_ct_ids, 'Has_DICOM': True})
    print(f"-> {len(df_dicom)} Patienten mit validen CT-Pfaden gefunden.")

    # 4. MERGE
    all_ids = sorted(list(set(df_labels['Subject ID']) | set(xml_ids) | set(available_ct_ids)))
    df_master = pd.DataFrame({'Subject ID': all_ids})
    df_master = df_master.merge(df_labels, on='Subject ID', how='left')
    df_master = df_master.merge(df_xml, on='Subject ID', how='left')
    df_master = df_master.merge(df_dicom, on='Subject ID', how='left')
    
    df_master['Has_XML'] = df_master['Has_XML'].fillna(False)
    df_master['Has_DICOM'] = df_master['Has_DICOM'].fillna(False)

    # 5. ERGEBNIS
    df_ready = df_master[(df_master['Histology'].notna()) & (df_master['Has_XML']) & (df_master['Has_DICOM'])]
    print("-" * 30)
    print(f"VOLLSTÄNDIGE DATENSÄTZE (Ready for ML): {len(df_ready)}")
    print("-" * 30)

    PATH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(PATH_OUTPUT, index=False)

if __name__ == "__main__":
    main()