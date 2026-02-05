from pathlib import Path

# Pfad anpassen
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"

# Nimm die erste Datei, die du findest
xml_files = list(DIR_XML.glob("*.xml"))

if xml_files:
    sample_file = xml_files[0]
    print(f"--- Untersuche Datei: {sample_file.name} ---")
    
    # Lies die Datei als reinen Text
    with open(sample_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Drucke die ersten 50 Zeilen
    for i, line in enumerate(lines[:50]):
        print(f"{i+1}: {line.strip()}")
else:
    print("Keine XML Dateien gefunden!")