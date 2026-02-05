import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys

# Pfade
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"

def extract_series_uid_robust(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            tag_clean = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_clean == 'imageSeries':
                for child in elem:
                    child_tag_clean = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag_clean == 'instanceUid':
                        return child.attrib.get('root')
    except Exception as e:
        print(f"Fehler bei {xml_path.name}: {e}")
        return None
    return None

def main():
    print("Starte robusten XML-Deep-Scan (AIM v4)...")
    
    xml_files = list(DIR_XML.glob("*.xml"))
    results = []
    for xml_file in xml_files:
        patient_id = xml_file.stem 
        series_uid = extract_series_uid_robust(xml_file)
        results.append({
            'Subject ID': patient_id,
            'XML_File': xml_file.name,
            'Linked_Series_UID': series_uid
        })
    
    df_xml_mapping = pd.DataFrame(results)
    df_xml_mapping['Linked_Series_UID'] = df_xml_mapping['Linked_Series_UID'].astype(str).str.strip()
    
    print(f"XML-Scan fertig. UIDs gefunden: {df_xml_mapping['Linked_Series_UID'].nunique()}")
    
    print("Gleiche mit metadata.csv ab...")
    if not PATH_METADATA.exists():
        print("FEHLER: Metadata CSV nicht gefunden.")
        return

    # --- DER FIX ---
    # Wir lesen die CSV normal ein (Pandas macht die UID fälschlicherweise zum Index)
    df_meta = pd.read_csv(PATH_METADATA)
    
    # Wir holen uns die UID aus dem Index zurück in eine echte Spalte!
    df_meta['Series UID Fix'] = df_meta.index.astype(str).str.strip()
    
    df_meta['File Location'] = df_meta['File Location'].astype(str).str.strip()

    print(f"Metadata geladen. Erste UID (Fix): {df_meta['Series UID Fix'].iloc[0]}")

    # Merge auf der neuen, gefixten Spalte
    df_final_mapping = df_xml_mapping.merge(
        df_meta[['Series UID Fix', 'File Location']], 
        left_on='Linked_Series_UID', 
        right_on='Series UID Fix', 
        how='left'
    )
    
    matched = df_final_mapping['File Location'].notna().sum()
    
    print("-" * 30)
    print(f"MATCHING ERGEBNIS:")
    print(f"XMLs gesamt: {len(df_xml_mapping)}")
    print(f"Davon erfolgreich mit DICOM verknüpft: {matched}")
    print("-" * 30)
    
    if matched > 0:
        out_path = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final_mapping.to_csv(out_path, index=False)
        print(f"SUCCESS! Mapping gespeichert: {out_path}")
    else:
        print("Immer noch 0 Matches. Prüfe die UIDs im Debug-Print oben.")

if __name__ == "__main__":
    main()