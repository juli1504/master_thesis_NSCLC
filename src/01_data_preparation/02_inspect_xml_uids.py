import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys

# Paths
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
        print(f"Error processing {xml_path.name}: {e}")
        return None
    return None

def main():
    print("Starting robust XML deep scan (AIM v4)...")
    
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
    
    print(f"XML scan complete. Unique UIDs found: {df_xml_mapping['Linked_Series_UID'].nunique()}")
    
    print("Comparing with metadata.csv...")
    if not PATH_METADATA.exists():
        print("ERROR: Metadata CSV not found.")
        return

    # --- THE FIX ---
    # Read the CSV normally (Pandas incorrectly sets the UID as the index)
    df_meta = pd.read_csv(PATH_METADATA)
    
    # Retrieve the UID from the index back into a real column
    df_meta['Series UID Fix'] = df_meta.index.astype(str).str.strip()
    
    df_meta['File Location'] = df_meta['File Location'].astype(str).str.strip()

    print(f"Metadata loaded. First UID (Fix): {df_meta['Series UID Fix'].iloc[0]}")

    # Merge on the new, fixed column
    df_final_mapping = df_xml_mapping.merge(
        df_meta[['Series UID Fix', 'File Location']], 
        left_on='Linked_Series_UID', 
        right_on='Series UID Fix', 
        how='left'
    )
    
    matched = df_final_mapping['File Location'].notna().sum()
    
    print("-" * 30)
    print("MATCHING RESULT:")
    print(f"Total XMLs: {len(df_xml_mapping)}")
    print(f"Successfully linked with DICOM: {matched}")
    print("-" * 30)
    
    if matched > 0:
        out_path = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final_mapping.to_csv(out_path, index=False)
        print(f"Mapping saved to: {out_path}.")
    else:
        print("Still 0 matches. Check the UIDs in the debug print above.")

if __name__ == "__main__":
    main()