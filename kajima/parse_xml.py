"""Parse XML files into generic dict structures with Japanese tag names."""

import xml.etree.ElementTree as ET
from pathlib import Path


def parse_xml(xml_path: str | Path) -> dict:
    """Parse a UTF-8 encoded XML file and return a nested dict.

    Tag names are preserved as-is (Japanese).
    Repeated sibling tags with the same name become lists.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = _elem_to_dict(root)
    if not isinstance(result, dict):
        msg = f"Expected root element to have children: {xml_path}"
        raise ValueError(msg)
    return result


def _elem_to_dict(elem: ET.Element) -> dict | str:
    """Recursively convert an XML element to a dict.

    - Leaf element (no children): returns its text (str).
    - Parent element: returns a dict of tag -> value.
    - When multiple children share the same tag, they become a list.
    - XML attributes and mixed content are ignored.
    """
    children = list(elem)
    if not children:
        return (elem.text or "").strip()

    result: dict = {}
    for child in children:
        tag = child.tag
        value = _elem_to_dict(child)

        if tag in result:
            existing = result[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[tag] = [existing, value]
        else:
            result[tag] = value

    return result


def build_json_schema(data: dict) -> dict:
    """Build a JSON Schema (Draft-07) from a parsed XML dict."""
    from genson import SchemaBuilder

    builder = SchemaBuilder(schema_uri="http://json-schema.org/draft-07/schema#")
    builder.add_object(data)
    return builder.to_schema()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Parse XML file into dict or generate JSON Schema"
    )
    parser.add_argument("xml_file", help="Path to XML file")
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Output JSON Schema instead of parsed data",
    )
    args = parser.parse_args()

    data = parse_xml(args.xml_file)
    if args.schema:
        output = build_json_schema(data)
    else:
        output = data
    print(json.dumps(output, ensure_ascii=False, indent=2))
