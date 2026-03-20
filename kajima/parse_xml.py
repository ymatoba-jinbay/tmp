"""Parse XML files to extract boring information into Pydantic models."""

import xml.etree.ElementTree as ET
from pathlib import Path

from kajima.schema import (
    BoringBasicInfo,
    BoringInfo,
    ColorTone,
    CoordinateInfo,
    CoreInfo,
    GroundwaterLevel,
    HeaderInfo,
    OrderingOrganization,
    SampleCollection,
    SoilRockClassification,
    StandardPenetrationTest,
    SurveyBasicInfo,
    SurveyCompany,
    SurveyPeriod,
)


def _text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return (elem.text or "").strip()


def _find_text(parent: ET.Element | None, tag: str) -> str:
    if parent is None:
        return ""
    return _text(parent.find(tag))


def parse_xml(xml_path: str | Path) -> BoringInfo:
    """Parse a Shift_JIS encoded XML file and return a BoringInfo model."""
    xml_path = Path(xml_path)
    with open(xml_path, "rb") as f:
        raw = f.read()

    content = raw.decode("shift_jis")
    content = content.replace('encoding="shift_jis"', 'encoding="utf-8"')
    lines = content.split("\n")
    lines = [
        line for line in lines if not line.strip().startswith("<!DOCTYPE")
    ]
    content = "\n".join(lines)

    root = ET.fromstring(content.encode("utf-8"))

    return BoringInfo(
        header_info=_parse_header(root),
        core_info=_parse_core(root),
    )


def _parse_header(root: ET.Element) -> HeaderInfo:
    header = root.find("標題情報")
    if header is None:
        return HeaderInfo()

    survey = header.find("調査基本情報")
    latlon = header.find("経度緯度情報")
    org = header.find("発注機関")
    period = header.find("調査期間")
    company = header.find("調査会社")
    basic = header.find("ボーリング基本情報")

    return HeaderInfo(
        survey_basic_info=SurveyBasicInfo(
            project_name=_find_text(survey, "事業工事名"),
            survey_name=_find_text(survey, "調査名"),
            boring_name=_find_text(survey, "ボーリング名"),
            boring_total_count=_find_text(survey, "ボーリング総数"),
            boring_serial_number=_find_text(survey, "ボーリング連番"),
        ),
        coordinate_info=CoordinateInfo(
            longitude_degrees=_find_text(latlon, "経度_度"),
            longitude_minutes=_find_text(latlon, "経度_分"),
            longitude_seconds=_find_text(latlon, "経度_秒"),
            latitude_degrees=_find_text(latlon, "緯度_度"),
            latitude_minutes=_find_text(latlon, "緯度_分"),
            latitude_seconds=_find_text(latlon, "緯度_秒"),
        ),
        ordering_organization=OrderingOrganization(
            organization_name=_find_text(org, "発注機関名称"),
        ),
        survey_period=SurveyPeriod(
            start_date=_find_text(period, "調査期間_開始年月日"),
            end_date=_find_text(period, "調査期間_終了年月日"),
        ),
        survey_company=SurveyCompany(
            company_name=_find_text(company, "調査会社_名称"),
            chief_engineer=_find_text(company, "調査会社_主任技師"),
            site_representative=_find_text(
                company, "調査会社_現場代理人"
            ),
            core_appraiser=_find_text(
                company, "調査会社_コア鑑定者"
            ),
            boring_supervisor=_find_text(
                company, "調査会社_ボーリング責任者"
            ),
        ),
        boring_basic_info=BoringBasicInfo(
            ground_elevation=_find_text(basic, "孔口標高"),
            total_drilling_length=_find_text(basic, "総掘進長"),
            drilling_angle=_find_text(basic, "掘進角度"),
            drilling_direction=_find_text(basic, "掘進方向"),
        ),
    )


def _parse_core(root: ET.Element) -> CoreInfo:
    core = root.find("コア情報")
    if core is None:
        return CoreInfo()

    rocks = []
    for elem in core.findall("岩石土区分"):
        rocks.append(SoilRockClassification(
            bottom_depth=_find_text(elem, "岩石土区分_下端深度"),
            soil_rock_name=_find_text(elem, "岩石土区分_岩石土名"),
            soil_rock_symbol=_find_text(elem, "岩石土区分_岩石土記号"),
        ))

    colors = []
    for elem in core.findall("色調"):
        colors.append(ColorTone(
            bottom_depth=_find_text(elem, "色調_下端深度"),
            color_name=_find_text(elem, "色調_色調名"),
        ))

    spts = []
    for elem in core.findall("標準貫入試験"):
        spts.append(StandardPenetrationTest(
            start_depth=_find_text(elem, "標準貫入試験_開始深度"),
            total_blow_count=_find_text(
                elem, "標準貫入試験_合計打撃回数"
            ),
            total_penetration=_find_text(
                elem, "標準貫入試験_合計貫入量"
            ),
        ))

    water_levels = []
    for elem in core.findall("孔内水位"):
        water_levels.append(GroundwaterLevel(
            water_level=_find_text(elem, "孔内水位_孔内水位"),
        ))

    samples = []
    for elem in core.findall("試料採取"):
        samples.append(SampleCollection(
            top_depth=_find_text(elem, "試料採取_上端深度"),
            bottom_depth=_find_text(elem, "試料採取_下端深度"),
            sample_number=_find_text(elem, "試料採取_試料番号"),
            test_name=_find_text(elem, "試料採取_試験名"),
        ))

    return CoreInfo(
        soil_rock_classifications=rocks,
        color_tones=colors,
        standard_penetration_tests=spts,
        groundwater_levels=water_levels,
        sample_collections=samples,
    )


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python -m kajima.parse_xml <xml_file>")
        sys.exit(1)

    result = parse_xml(sys.argv[1])
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
