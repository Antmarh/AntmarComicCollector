# antmar/metadata.py
import xml.etree.ElementTree as ET
from xml.dom import minidom

def generate_comicinfo_xml(metadata):
    """Genera el contenido XML de ComicInfo a partir de un diccionario metadata."""
    clean_metadata = {
        k: v for k, v in metadata.items()
        if k != 'CoverURL' and v is not None
    }
    root = ET.Element(
        'ComicInfo',
        {
            'xmlns:xsd': 'http://www.w3.org/2001/XMLSchema',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
    )
    for key, value in clean_metadata.items():
        if value:
            ET.SubElement(root, key).text = str(value)
    dom = minidom.parseString(ET.tostring(root, 'utf-8'))
    return dom.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8')

def parse_comicinfo_xml(xml_str:str)->dict:
    if not xml_str: return {}
    try:
        root = ET.fromstring(xml_str)
        return {child.tag: (child.text or "").strip() for child in root}
    except Exception:
        return {}