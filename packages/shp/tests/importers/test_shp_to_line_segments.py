#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import shapefile
from evo.data_converters.shp.importer.shp_to_evo import convert_shp
from evo_schemas.objects import LineSegments_V2_1_0


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    return tmp_path / "parquet"


def _convert(shp_path: Path, parquet_path: Path):
    return convert_shp(
        filepath=shp_path,
        filepath_shx=shp_path.with_suffix(".shx"),
        filepath_dbf=shp_path.with_suffix(".dbf"),
        upload_path=parquet_path,
        publish_objects=False,
        overwrite_existing_objects=True,
    )


def _read_column_values(parquet_path: Path, data_name: str, column_name: str = "values"):
    return pq.read_table(parquet_path / data_name).column(column_name).to_pylist()


@pytest.mark.parametrize(
    "shape_type, method_name, geometry",
    [
        (shapefile.POLYLINE, "line", [[[0, 0], [1, 0], [1, 1]]]),
        (shapefile.POLYLINEZ, "linez", [[[0, 0, 0, 1], [1, 0, 2, 2], [1, 1, 3, 3]]]),
        (shapefile.POLYLINEM, "linem", [[[0, 0, 1], [1, 0, 2], [1, 1, 3]]]),
        (shapefile.POLYGON, "poly", [[[0, 0], [1, 0], [1, 1], [0, 1]]]),
        (shapefile.POLYGONZ, "polyz", [[[0, 0, 0, 1], [1, 0, 1, 2], [1, 1, 2, 3], [0, 1, 3, 4]]]),
        (shapefile.POLYGONM, "polym", [[[0, 0, 1], [1, 0, 2], [1, 1, 3], [0, 1, 4]]]),
    ],
)
def test_convert_polyline_polygon_variants(
    tmp_path: Path,
    parquet_path: Path,
    shape_type: int,
    method_name: str,
    geometry: list[list[list[float]]],
):
    shp_path = tmp_path / f"variant_{shape_type}"
    with shapefile.Writer(shp_path, shapeType=shape_type) as writer:
        writer.field("NAME", "C")
        getattr(writer, method_name)(geometry)
        writer.record(NAME="shape-1")

    result = _convert(shp_path, parquet_path)

    assert len(result) == 1
    assert isinstance(result[0], LineSegments_V2_1_0)
    assert result[0].schema == "/objects/line-segments/2.1.0/line-segments.schema.json"


def test_polyline_m_parts_indices_and_attributes(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "polyline_m"
    with shapefile.Writer(shp_path, shapeType=shapefile.POLYLINEM) as writer:
        writer.field("NAME", "C")

        writer.linem(
            [
                [[0, 0, 1.0], [1, 0, 2.0], [1, 1, 3.0]],
                [[2, 0, 4.0], [2, 1, 5.0]],
            ]
        )
        writer.record(NAME="A")

        writer.linem(
            [
                [[10, 0, 6.0], [11, 0, 7.0], [11, 1, 8.0], [12, 1, 9.0]],
            ]
        )
        writer.record(NAME="B")

    result = _convert(shp_path, parquet_path)
    line_segments: LineSegments_V2_1_0 = result[0]

    assert line_segments.parts is not None
    assert line_segments.parts.chunks.length == 3

    chunks_table = pq.read_table(parquet_path / line_segments.parts.chunks.data)
    assert chunks_table.column("offset").to_pylist() == [0, 2, 3]
    assert chunks_table.column("count").to_pylist() == [2, 1, 3]

    indices_table = pq.read_table(parquet_path / line_segments.segments.indices.data)
    assert indices_table.column("n0").to_pylist() == [0, 1, 3, 5, 6, 7]
    assert indices_table.column("n1").to_pylist() == [1, 2, 4, 6, 7, 8]

    part_attributes_by_name = {attribute.name: attribute for attribute in line_segments.parts.attributes}
    assert set(part_attributes_by_name.keys()) >= {"NAME", "shp_part_index", "shp_record_index"}

    name_values = _read_column_values(parquet_path, part_attributes_by_name["NAME"].values.data)
    assert name_values == ["A", "A", "B"]

    part_index_values = _read_column_values(parquet_path, part_attributes_by_name["shp_part_index"].values.data)
    assert part_index_values == [0, 1, 0]

    record_index_values = _read_column_values(parquet_path, part_attributes_by_name["shp_record_index"].values.data)
    assert record_index_values == [0, 0, 1]

    vertex_attributes = line_segments.segments.vertices.attributes
    assert vertex_attributes is not None
    assert len(vertex_attributes) == 1
    assert vertex_attributes[0].name == "M"
    assert _read_column_values(parquet_path, vertex_attributes[0].values.data) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_polygon_converted_to_closed_segments(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "polygon"
    with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as writer:
        writer.field("NAME", "C")

        writer.poly(
            [
                [[0, 0], [2, 0], [2, 2], [0, 2]],
                [[10, 10], [11, 10], [11, 11]],
            ]
        )
        writer.record(NAME="poly")

    result = _convert(shp_path, parquet_path)
    line_segments: LineSegments_V2_1_0 = result[0]

    assert line_segments.parts is not None
    chunks_table = pq.read_table(parquet_path / line_segments.parts.chunks.data)

    # Each polygon part is expected to be closed, so a part with N vertices yields N segments.
    assert chunks_table.column("count").to_pylist() == [4, 3]


def test_polyline_numeric_decimal_field_is_continuous(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "polyline_numeric_decimal"
    with shapefile.Writer(shp_path, shapeType=shapefile.POLYLINE) as writer:
        writer.field("SCORE", "N", decimal=2)
        writer.line([[[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]])
        writer.record(SCORE=12.34)

    result = _convert(shp_path, parquet_path)
    line_segments: LineSegments_V2_1_0 = result[0]

    assert line_segments.parts is not None
    part_attributes_by_name = {attribute.name: attribute for attribute in line_segments.parts.attributes}
    score_attr = part_attributes_by_name["SCORE"]
    assert score_attr.attribute_type == "scalar"
    assert _read_column_values(parquet_path, score_attr.values.data) == [12.34]


def test_line_segments_crs_override(tmp_path: Path, parquet_path: Path):
    """Test that crs_override works for line segments."""
    shp_path = tmp_path / "polyline_crs"
    with shapefile.Writer(shp_path, shapeType=shapefile.POLYLINE) as writer:
        writer.field("NAME", "C")
        writer.line([[[0, 0], [1, 0], [1, 1]]])
        writer.record(NAME="test-line")

    result = convert_shp(
        filepath=shp_path,
        filepath_shx=shp_path.with_suffix(".shx"),
        filepath_dbf=shp_path.with_suffix(".dbf"),
        crs_override={"epsg_code": 4326},
        upload_path=parquet_path,
        publish_objects=False,
        overwrite_existing_objects=True,
    )

    assert len(result) == 1
    line_segments: LineSegments_V2_1_0 = result[0]

    crs = line_segments.coordinate_reference_system
    assert hasattr(crs, "epsg_code")
    assert crs.epsg_code == 4326
