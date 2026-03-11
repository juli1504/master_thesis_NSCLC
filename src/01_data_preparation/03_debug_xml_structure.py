from pathlib import Path

# Adjust path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"

# Get the first file found
xml_files = list(DIR_XML.glob("*.xml"))

if xml_files:
    sample_file = xml_files[0]
    print(f"--- Inspecting file: {sample_file.name} ---")
    
    # Read the file as raw text
    with open(sample_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Print the first 50 lines
    for i, line in enumerate(lines[:50]):
        print(f"{i+1}: {line.strip()}")
else:
    print("No XML files found.")