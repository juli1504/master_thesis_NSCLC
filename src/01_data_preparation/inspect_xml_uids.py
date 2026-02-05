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
    
    if not DIR_XML.exists():
        print(f"FEHLER: XML Ordner nicht gefunden: {DIR_XML}")
        return

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
    
    # --- WICHTIG: Cleaning ---
    df_xml_mapping['Linked_Series_UID'] = df_xml_mapping['Linked_Series_UID'].astype(str).str.strip()
    
    print(f"XML-Scan fertig. UIDs gefunden: {df_xml_mapping['Linked_Series_UID'].nunique()}")
    
    print("Gleiche mit metadata.csv ab...")
    if not PATH_METADATA.exists():
        print("FEHLER: Metadata CSV nicht gefunden.")
        return

    df_meta = pd.read_csv(PATH_METADATA)
    df_meta.columns = df_meta.columns.str.strip()
    
    # --- WICHTIG: Cleaning auch hier ---
    df_meta['Series UID'] = df_meta['Series UID'].astype(str).str.strip()

    df_final_mapping = df_xml_mapping.merge(
        df_meta[['Series UID', 'File Location', 'Series Description']], 
        left_on='Linked_Series_UID', 
        right_on='Series UID', 
        how='left'
    )
    
    matched = df_final_mapping['File Location'].notna().sum()
    
    print("-" * 30)
    print(f"MATCHING ERGEBNIS:")
    print(f"XMLs gesamt: {len(df_xml_mapping)}")
    print(f"Davon erfolgreich mit DICOM verknüpft: {matched}")
    
    if matched == 0:
        print("\nDEBUG: Vergleich der IDs (Erste 3 Zeilen):")
        print("XML ID:", df_xml_mapping['Linked_Series_UID'].iloc[0])
        print("Meta ID:", df_meta['Series UID'].iloc[0])
        print("(Prüfe ob sie gleich aussehen!)")
    
    print("-" * 30)
    
    out_path = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final_mapping.to_csv(out_path, index=False)
    print(f"Gespeichert: {out_path}")

if __name__ == "__main__":
    main()