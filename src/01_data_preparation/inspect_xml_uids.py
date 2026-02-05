import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys

# Pfade
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"

def extract_series_uid_robust(xml_path):
    """
    Durchsucht AIM v4 XML Dateien nach der Series UID.
    Ignoriert Namespaces, um robuster zu sein.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Strategie: Wir iterieren über ALLE Elemente
        for elem in root.iter():
            # Wir entfernen den Namespace-Prefix (alles vor dem '}')
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
    print(f"Scanne {len(xml_files)} Dateien...")
    
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
    
    found_uids = df_xml_mapping['Linked_Series_UID'].notna().sum()
    print(f"XML-Scan fertig. UIDs in XML gefunden: {found_uids} von {len(xml_files)}")
    
    print("Gleiche mit metadata.csv ab...")
    if not PATH_METADATA.exists():
        print("FEHLER: Metadata CSV nicht gefunden.")
        return

    df_meta = pd.read_csv(PATH_METADATA)
    df_meta.columns = df_meta.columns.str.strip()
    
    df_final_mapping = df_xml_mapping.merge(
        df_meta[['Series UID', 'File Location', 'Series Description']], 
        left_on='Linked_Series_UID', 
        right_on='Series UID', 
        how='left'
    )
    
    matched = df_final_mapping['File Location'].notna().sum()
    print("-" * 30)
    print(f"MATCHING ERGEBNIS:")
    print(f"XMLs mit gültiger Series UID: {found_uids}")
    print(f"Davon auch in Metadata (DICOM) gefunden: {matched}")
    print("-" * 30)
    
    out_path = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final_mapping.to_csv(out_path, index=False)
    print(f"Gespeichert: {out_path}")
    
    if matched > 0:
        print("\nErfolgreiche Matches (Beispiele):")
        print(df_final_mapping[df_final_mapping['File Location'].notna()]
              [['Subject ID', 'Series Description']].head())

if __name__ == "__main__":
    main()