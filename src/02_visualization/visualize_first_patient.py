import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# --- KONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"

def parse_aim_xml(xml_path):
    """
    Sucht nach geometrischen Formen (Polylines) in der AIM XML
    und gibt die Koordinaten + die verknüpfte Bild-ID (SOP UID) zurück.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Namespace-Blindheit (wie vorher)
        roi_data = []
        
        # Wir suchen nach 'ImageAnnotation'
        for annotation in root.iter():
            tag_clean = annotation.tag.split('}')[-1] if '}' in annotation.tag else annotation.tag
            
            if tag_clean == 'ImageAnnotation':
                # Jetzt suchen wir die Markup-Collection (die Zeichnungen)
                for markup in annotation.iter():
                    m_tag = markup.tag.split('}')[-1] if '}' in markup.tag else markup.tag
                    
                    # Wir suchen nach 'TwoDimensionPolyline' (das ist meist der Tumor-Umriss)
                    if m_tag in ['TwoDimensionPolyline', 'Circle', 'Ellipse']:
                        shape_type = m_tag
                        
                        # Zu welchem Bild gehört das? (imageReferenceUid)
                        # Suche das Attribut im Tag oder in Kindern
                        sop_uid = None
                        
                        # Oft steht die UID in einem <imageReferenceUid> Kind-Tag
                        for child in markup:
                            c_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                            if c_tag == 'imageReferenceUid':
                                sop_uid = child.attrib.get('root')
                        
                        # Koordinaten sammeln
                        points = []
                        for coord in markup.iter():
                            co_tag = coord.tag.split('}')[-1] if '}' in coord.tag else coord.tag
                            if co_tag == 'TwoDimensionSpatialCoordinate':
                                x = float(coord.find("coordinateIndex[@value='0']").attrib['value'])
                                y = float(coord.find("coordinateIndex[@value='1']").attrib['value'])
                                points.append((x, y))
                            # AIM v4 kann Koordinaten auch anders speichern, wir prüfen mal diesen Standardweg
                                
                        if sop_uid and points:
                            roi_data.append({
                                'type': shape_type,
                                'sop_uid': sop_uid,
                                'points': points
                            })
                            
        return roi_data

    except Exception as e:
        print(f"Fehler beim Parsen von {xml_path.name}: {e}")
        return []

def main():
    print("--- Starte Tumor-Visualisierung (Scanner Mode) ---")
    
    # 1. Liste laden
    df_map = pd.read_csv(PATH_MAPPING)
    valid_patients = df_map[df_map['File Location'].notna()]
    
    print(f"Durchsuche {len(valid_patients)} Patienten nach Zeichnungen...")
    
    found_any = False
    
    # Wir iterieren durch die Patienten, bis wir einen mit ROIs finden
    for idx, row in valid_patients.iterrows():
        subject_id = row['Subject ID']
        xml_filename = row['XML_File']
        xml_path = DIR_XML / xml_filename
        
        # Testen ob ROIs da sind
        rois = parse_aim_xml(xml_path)
        
        if not rois:
            # Kleiner Status-Print, damit man sieht, dass was passiert
            print(f"Skipping {subject_id}: Keine Zeichnungen.")
            continue
            
        # --- TREFFER! ---
        print("\n" + "="*40)
        print(f"TREFFER BEI PATIENT: {subject_id}")
        print("="*40)
        
        first_roi = rois[0]
        target_sop_uid = first_roi['sop_uid']
        print(f"Zeichnungstyp: {first_roi['type']}")
        print(f"Gehört zu Bild-UID: {target_sop_uid}")
        
        # Metadata Suche (wie vorher)
        df_meta = pd.read_csv(PATH_METADATA)
        
        # Robuste Suche nach der UID in der ganzen Zeile
        mask = df_meta.apply(lambda r: r.astype(str).str.contains(target_sop_uid).any(), axis=1)
        
        if mask.sum() == 0:
            print(f"Schade: Zeichnung da, aber Bild-UID nicht in Metadata gefunden. Suche weiter...")
            continue
            
        meta_row = df_meta[mask].iloc[0]
        
        # Pfad finden
        found_path = None
        for val in meta_row.values:
            if isinstance(val, str) and ("NSCLC Radiogenomics" in val or "dicom" in val):
                found_path = val
                break
        
        if not found_path:
            continue

        clean_path = found_path.lstrip('./').lstrip('.\\').replace('\\', '/')
        full_dicom_path = PROJECT_ROOT / "data" / "raw" / "dicom" / clean_path
        
        print(f"Lade DICOM: {full_dicom_path}")
        
        if not full_dicom_path.exists():
            print("Datei physisch nicht gefunden. Nächster...")
            continue
            
        # Plotten
        try:
            ds = pydicom.dcmread(full_dicom_path)
            img_array = ds.pixel_array
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img_array, cmap='gray')
            plt.title(f"Patient: {subject_id} | {first_roi['type']}")
            
            # Alle ROIs dieses Patienten einzeichnen
            for roi in rois:
                points = roi['points']
                # Polygon schließen (erster Punkt = letzter Punkt)
                if points:
                    points.append(points[0])
                    x_vals, y_vals = zip(*points)
                    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Tumor')
            
            plt.legend()
            plt.show()
            
            found_any = True
            break # Wir haben unser Bild, wir hören auf zu suchen!
            
        except Exception as e:
            print(f"Fehler beim Plotten: {e}")
            continue

    if not found_any:
        print("\nFAZIT: Keine anzeigbaren Zeichnungen in den gescannten Patienten gefunden.")

if __name__ == "__main__":
    main()