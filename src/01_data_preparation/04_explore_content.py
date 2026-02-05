import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import os

# --- KONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
PATH_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

# Neuer Ordner für die gespeicherten Bilder
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

def print_xml_features(xml_path):
    """Liest die semantischen Merkmale aus der XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        print("\n--- XML INHALT (Auszug) ---")
        features_found = 0
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag == 'ImagingObservationCharacteristic':
                label = None
                value = None
                for child in elem:
                    c_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if c_tag == 'label':
                        label = child.attrib.get('value')
                    elif c_tag == 'typeCode':
                        value = child.attrib.get('code') or child.attrib.get('codeSystem')
                
                if label:
                    print(f"  • {label}: {value}")
                    features_found += 1
                    
        if features_found == 0:
            print("  (Keine strukturierten Merkmale gefunden)")
            
    except Exception as e:
        print(f"Fehler beim XML lesen: {e}")

def main():
    print("### SCHRITT 2: ERSTE DATENEXPLORATION ###\n")
    
    # 1. LABEL VERTEILUNG (CSV)
    print("(1) LABEL VERTEILUNG (CSV)")
    df_clinical = pd.read_csv(PATH_CLINICAL)
    df_clinical.columns = df_clinical.columns.str.strip()
    
    counts = df_clinical['Histology'].value_counts()
    print(counts)
    
    # Plot erstellen
    plt.figure(figsize=(10, 6)) # Breiteres Bild, damit Text Platz hat
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title('Distribution of diagnoses (histology)')
    plt.ylabel('Number of Patients')
    plt.xlabel('Diagnosis')
    
    # Hier ist der Fix für die horizontale Schrift:
    plt.xticks(rotation=0) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Verhindert, dass Text abgeschnitten wird
    
    # Erst speichern, dann anzeigen!
    save_path_chart = DIR_FIGURES / "distribution_histology.png"
    plt.savefig(save_path_chart, dpi=300)
    print(f"\n>> Diagramm gespeichert unter: {save_path_chart}")
    
    plt.show() # Pop-up Fenster
    
    # 2. BEISPIEL LADEN
    print("\n(2) BEISPIEL-PATIENT ANALYSIEREN")
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['File Location'].notna()].iloc[0]
    subject_id = match['Subject ID']
    
    print(f"Patient: {subject_id}")
    print_xml_features(DIR_XML / match['XML_File'])
    
    # 3. DICOM BILD
    print(f"\n(3) DICOM BILD (CT) FÜR {subject_id}")
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    dicom_path = DIR_DICOM / clean_path
    
    target_file = None
    if dicom_path.is_dir():
        files = sorted(list(dicom_path.glob("*.dcm")))
        if files:
            mid_slice = len(files) // 2
            target_file = files[mid_slice]
            print(f"  • Lade Slice {mid_slice+1} von {len(files)}: {target_file.name}")
    elif dicom_path.exists():
        target_file = dicom_path

    if target_file:
        ds = pydicom.dcmread(target_file)
        
        plt.figure(figsize=(8,8))
        plt.imshow(ds.pixel_array, cmap='gray')
        plt.title(f"Patient: {subject_id} | {ds.Modality} | Slice Location: {ds.get('SliceLocation', 'N/A')}")
        plt.axis('off')
        
        # Speichern & Anzeigen
        save_path_img = DIR_FIGURES / f"example_ct_{subject_id}.png"
        plt.savefig(save_path_img, dpi=300)
        print(f">> CT-Bild gespeichert unter: {save_path_img}")
        
        plt.show() # Pop-up Fenster
        print("\nERFOLG! Datenpipeline steht. 🚀")

if __name__ == "__main__":
    main()