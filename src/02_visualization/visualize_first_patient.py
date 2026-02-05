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
    print("--- Starte Tumor-Visualisierung ---")
    
    # 1. Einen Patienten laden
    df_map = pd.read_csv(PATH_MAPPING)
    # Wir nehmen einen Patienten, der ein Match hat (File Location ist nicht leer)
    valid_patients = df_map[df_map['File Location'].notna()]
    
    if len(valid_patients) == 0:
        print("Keine verknüpften Patienten gefunden! Bitte Mapping prüfen.")
        return

    # Nimm den ersten Patienten (oder ändere den Index, z.B. .iloc[5] für einen anderen)
    patient_row = valid_patients.iloc[0]
    subject_id = patient_row['Subject ID']
    xml_filename = patient_row['XML_File']
    
    print(f"Patient ausgewählt: {subject_id}")
    print(f"XML Datei: {xml_filename}")
    
    # 2. ROI aus XML holen
    xml_path = DIR_XML / xml_filename
    rois = parse_aim_xml(xml_path)
    
    if not rois:
        print("KEINE Region-of-Interest (Zeichnungen) in dieser XML gefunden.")
        print("Das ist normal bei manchen XMLs (z.B. nur Text-Befunde). Versuche einen anderen Patienten.")
        return
        
    print(f"Gefundene ROIs: {len(rois)}")
    first_roi = rois[0]
    target_sop_uid = first_roi['sop_uid']
    print(f"Zeichnung gehört zu Bild-Slice (SOP UID): {target_sop_uid}")
    
    # 3. Den Dateipfad zu dieser SOP UID finden (via Metadata)
    print("Suche DICOM-Datei in Metadata...")
    df_meta = pd.read_csv(PATH_METADATA)
    
    # ACHTUNG: Hier müssen wir wieder schauen, ob wir den Spalten-Fix brauchen
    # Wir machen es dynamisch: Suche in ALLEN Spalten nach der UID
    found_row = None
    file_path_col = 'File Location' # Standardannahme
    
    # Wir scannen die ganze CSV nach der UID (etwas langsam, aber sicher)
    # Da die Spalten verschoben sein können, suchen wir einfach den Wert.
    mask = df_meta.apply(lambda row: row.astype(str).str.contains(target_sop_uid).any(), axis=1)
    
    if mask.sum() == 0:
        print(f"FEHLER: Konnte SOP UID {target_sop_uid} nicht in Metadata finden.")
        return
        
    meta_row = df_meta[mask].iloc[0]
    
    # Jetzt den Pfad extrahieren. Wir wissen, er enthält "NSCLC Radiogenomics"
    # Wir suchen in der Zeile nach einem String, der wie ein Pfad aussieht
    found_path = None
    for val in meta_row.values:
        if isinstance(val, str) and ("NSCLC Radiogenomics" in val or "dicom" in val):
            found_path = val
            break
            
    if not found_path:
        print("Konnte keinen Dateipfad in der Metadata-Zeile finden.")
        return

    # Pfad bereinigen
    clean_path = found_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    full_dicom_path = PROJECT_ROOT / "data" / "raw" / "dicom" / clean_path
    
    print(f"DICOM Pfad gefunden: {full_dicom_path}")
    
    if not full_dicom_path.exists():
        print("DATEI EXISTIERT NICHT AUF FESTPLATTE!")
        # Fallback: Manchmal ist der Dateiname im Pfad anders kodiert
        # Wir suchen im Ordner nach der Datei
        folder = full_dicom_path.parent
        print(f"Suche im Ordner: {folder}")
        if folder.exists():
            files = list(folder.glob("*.dcm"))
            print(f"Dateien im Ordner: {len(files)}")
            # Wir laden einfach mal die erste, falls wir die exakte nicht finden (nur zum Test)
            # Aber besser: Wir versuchen die Datei zu finden
            # (Dieser Teil ist tricky, da Dateinamen oft UIDs sind)
    
    # 4. Plotten
    try:
        ds = pydicom.dcmread(full_dicom_path)
        img_array = ds.pixel_array
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_array, cmap='gray')
        plt.title(f"Patient: {subject_id} | ROI: {first_roi['type']}")
        
        # Polygon zeichnen
        points = first_roi['points']
        poly = patches.Polygon(points, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(poly)
        
        plt.show()
        print("Visualisierung erfolgreich!")
        
    except Exception as e:
        print(f"Konnte Bild nicht laden/anzeigen: {e}")

if __name__ == "__main__":
    main()