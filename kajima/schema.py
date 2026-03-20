"""Pydantic schema for boring column chart data."""

from pydantic import BaseModel, Field


class SurveyBasicInfo(BaseModel):
    project_name: str = ""
    survey_name: str = ""
    boring_name: str = ""
    boring_total_count: str = ""
    boring_serial_number: str = ""


class CoordinateInfo(BaseModel):
    longitude_degrees: str = ""
    longitude_minutes: str = ""
    longitude_seconds: str = ""
    latitude_degrees: str = ""
    latitude_minutes: str = ""
    latitude_seconds: str = ""


class OrderingOrganization(BaseModel):
    organization_name: str = ""


class SurveyPeriod(BaseModel):
    start_date: str = ""
    end_date: str = ""


class SurveyCompany(BaseModel):
    company_name: str = ""
    chief_engineer: str = ""
    site_representative: str = ""
    core_appraiser: str = ""
    boring_supervisor: str = ""


class BoringBasicInfo(BaseModel):
    ground_elevation: str = ""
    total_drilling_length: str = ""
    drilling_angle: str = ""
    drilling_direction: str = ""


class HeaderInfo(BaseModel):
    survey_basic_info: SurveyBasicInfo = Field(
        default_factory=SurveyBasicInfo
    )
    coordinate_info: CoordinateInfo = Field(
        default_factory=CoordinateInfo
    )
    ordering_organization: OrderingOrganization = Field(
        default_factory=OrderingOrganization
    )
    survey_period: SurveyPeriod = Field(
        default_factory=SurveyPeriod
    )
    survey_company: SurveyCompany = Field(
        default_factory=SurveyCompany
    )
    boring_basic_info: BoringBasicInfo = Field(
        default_factory=BoringBasicInfo
    )


class SoilRockClassification(BaseModel):
    bottom_depth: str = ""
    soil_rock_name: str = ""
    soil_rock_symbol: str = ""


class ColorTone(BaseModel):
    bottom_depth: str = ""
    color_name: str = ""


class StandardPenetrationTest(BaseModel):
    start_depth: str = ""
    total_blow_count: str = ""
    total_penetration: str = ""


class GroundwaterLevel(BaseModel):
    water_level: str = ""


class SampleCollection(BaseModel):
    top_depth: str = ""
    bottom_depth: str = ""
    sample_number: str = ""
    test_name: str = ""


class CoreInfo(BaseModel):
    soil_rock_classifications: list[SoilRockClassification] = Field(
        default_factory=list
    )
    color_tones: list[ColorTone] = Field(default_factory=list)
    standard_penetration_tests: list[StandardPenetrationTest] = Field(
        default_factory=list
    )
    groundwater_levels: list[GroundwaterLevel] = Field(
        default_factory=list
    )
    sample_collections: list[SampleCollection] = Field(
        default_factory=list
    )


class BoringInfo(BaseModel):
    header_info: HeaderInfo = Field(default_factory=HeaderInfo)
    core_info: CoreInfo = Field(default_factory=CoreInfo)
