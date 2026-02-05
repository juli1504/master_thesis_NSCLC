import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"

def extract_series_uid(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for uid_tag in root.iter():
            if 'imageSeriesUid' in uid_tag.tag:
                return uid_tag.attrib.get('root') or uid_tag.text
    except Exception as e:
        return None
    return None

def main():
    print("Starte XML-Deep-Scan...")
    xml_files = list(DIR_XML.glob("*.xml"))
    
    results = []
    for xml_file in xml_files:
        patient_id = xml_file.stem 
        series_uid = extract_series_uid(xml_file)
        results.append({
            'Subject ID': patient_id,
            'XML_File': xml_file.name,
            'Linked_Series_UID': series_uid
        })
    
    df_xml_mapping = pd.DataFrame(results)
    print(f"Fertig! {len(df_xml_mapping)} XMLs gescannt.")
    
    print("Gleiche mit metadata.csv ab...")
    df_meta = pd.read_csv(PATH_METADATA)
    df_meta.columns = df_meta.columns.str.strip()
    
    df_final_mapping = df_xml_mapping.merge(
        df_meta[['Series UID', 'File Location', 'Series Description']], 
        left_on='Linked_Series_UID', 
        right_on='Series UID', 
        how='left'
    )
    
    found = df_final_mapping['File Location'].notna().sum()
    print(f"Ergebnis: Für {found} von {len(df_xml_mapping)} XMLs wurde der exakte Bild-Ordner gefunden.")
    
    out_path = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
    df_final_mapping.to_csv(out_path, index=False)
    print(f"Mapping gespeichert unter: {out_path}")

    print("\nBeispiel-Matching:")
    print(df_final_mapping[['Subject ID', 'Series Description']].head())

if __name__ == "__main__":
    main()