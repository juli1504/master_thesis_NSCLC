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

def print_xml_features(xml_path):
    """Liest die semantischen Merkmale aus der XML (wie im Plan gefordert)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        print("\n--- XML INHALT (Auszug) ---")
        # Wir suchen nach den 'ImagingObservationCharateristic' (den Merkmalen)
        features_found = 0
        for elem in root.iter():
            # Namespace entfernen
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag == 'ImagingObservationCharacteristic':
                # Suche Label und Value
                label = None
                value = None
                
                for child in elem:
                    c_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if c_tag == 'label':
                        label = child.attrib.get('value')
                    elif c_tag == 'typeCode':
                        # Oft steht der Wert im 'codeSystemName' oder 'code'
                        value = child.attrib.get('code') or child.attrib.get('codeSystem')
                
                if label:
                    print(f"  • {label}: {value}")
                    features_found += 1
                    
        if features_found == 0:
            print("  (Keine strukturierten Merkmale gefunden, XML ist evtl. anders aufgebaut)")
            
    except Exception as e:
        print(f"Fehler beim XML lesen: {e}")

def main():
    print("### SCHRITT 2: ERSTE DATENEXPLORATION ###\n")
    
    # 1. CSV LABELS ANALYSIEREN
    print("(1) LABEL VERTEILUNG (CSV)")
    df_clinical = pd.read_csv(PATH_CLINICAL)
    # Spalten bereinigen
    df_clinical.columns = df_clinical.columns.str.strip()
    
    # Verteilung zählen
    counts = df_clinical['Histology'].value_counts()
    print(counts)
    
    # Plot der Verteilung
    counts.plot(kind='bar', title='Verteilung der Diagnosen')
    plt.show() # Zeigt kurz das Diagramm
    
    # 2. EIN BEISPIEL LADEN (DICOM + XML)
    print("\n(2) BEISPIEL-PATIENT ANALYSIEREN")
    
    # Wir nehmen den ersten Match aus unserer Mapping-Liste
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['File Location'].notna()].iloc[0]
    
    subject_id = match['Subject ID']
    xml_file = match['XML_File']
    
    print(f"Patient: {subject_id}")
    
    # A) XML Merkmale anzeigen
    xml_path = DIR_XML / xml_file
    print_xml_features(xml_path)
    
    # B) DICOM BILD ANZEIGEN
    print(f"\n(3) DICOM BILD (CT) FÜR {subject_id}")
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    dicom_path = DIR_DICOM / clean_path
    
    # Check: Ist es ein Ordner?
    target_file = None
    
    if dicom_path.is_dir():
        # Suche die erste .dcm Datei im Ordner
        # Wir sortieren, damit wir nicht zufällig irgendeine nehmen, sondern z.B. die erste
        files = sorted(list(dicom_path.glob("*.dcm")))
        if files:
            # Wir nehmen mal die mittlere Slice, da sieht man meist mehr Lunge als ganz oben
            mid_slice = len(files) // 2
            target_file = files[mid_slice]
            print(f"  • Ordner gefunden. Lade Slice {mid_slice+1} von {len(files)}: {target_file.name}")
        else:
            print("  • Ordner existiert, ist aber leer (keine .dcm Dateien).")
    elif dicom_path.exists():
        # Es ist direkt eine Datei
        target_file = dicom_path
    else:
        print(f"  • Pfad nicht gefunden: {dicom_path}")

    # Laden & Plotten
    if target_file:
        try:
            ds = pydicom.dcmread(target_file)
            print(f"  • Modality: {ds.Modality}")
            print(f"  • Bildgröße: {ds.Rows} x {ds.Columns}")
            
            plt.figure(figsize=(6,6))
            plt.imshow(ds.pixel_array, cmap='gray')
            plt.title(f"Patient: {subject_id} | {ds.Modality}")
            plt.axis('off')
            plt.show()
            print("\nERFOLG! Datenpipeline steht. 🚀")
            
        except Exception as e:
            print(f"  • Fehler beim Lesen der Datei: {e}")

if __name__ == "__main__":
    main()