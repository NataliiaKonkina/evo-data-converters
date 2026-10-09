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
from evo.data_converters.shp.importer.exceptions import InvalidSHPError
from evo.data_converters.shp.importer.shp_to_evo import convert_shp
from evo_schemas.objects.pointset import Pointset_V1_3_0


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


def _get_attr(pointset: Pointset_V1_3_0, name: str):
    attributes = pointset.locations.attributes or []
    return next((attribute for attribute in attributes if attribute.name == name), None)


@pytest.mark.parametrize(
    "shape_type, geometry_writer",
    [
        (shapefile.POINT, lambda writer: writer.point(0.0, 1.0)),
        (shapefile.POINTM, lambda writer: writer.pointm(0.0, 1.0, 2.0)),
        (shapefile.POINTZ, lambda writer: writer.pointz(0.0, 1.0, 3.0, 2.0)),
        (shapefile.MULTIPOINT, lambda writer: writer.multipoint([[0.0, 1.0], [2.0, 3.0]])),
        (shapefile.MULTIPOINTM, lambda writer: writer.multipointm([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])),
        (
            shapefile.MULTIPOINTZ,
            lambda writer: writer.multipointz([[0.0, 1.0, 5.0, 2.0], [2.0, 3.0, 6.0, 4.0]]),
        ),
    ],
)
def test_convert_point_and_multipoint_variants(
    tmp_path: Path,
    parquet_path: Path,
    shape_type: int,
    geometry_writer,
):
    shp_path = tmp_path / f"variant_{shape_type}"
    with shapefile.Writer(shp_path, shapeType=shape_type) as writer:
        writer.field("NAME", "C")
        geometry_writer(writer)
        writer.record(NAME="shape-1")

    result = _convert(shp_path, parquet_path)

    assert len(result) == 1
    assert isinstance(result[0], Pointset_V1_3_0)
    assert result[0].schema == "/objects/pointset/1.3.0/pointset.schema.json"


def test_pointz_records_to_pointset_with_measurement_and_dbf(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "pointz"
    with shapefile.Writer(shp_path, shapeType=shapefile.POINTZ) as writer:
        writer.field("NAME", "C")

        writer.pointz(1.0, 2.0, 3.0, 10.0)
        writer.record(NAME="A")

        writer.pointz(4.0, 5.0, 6.0, 20.0)
        writer.record(NAME="B")

    result = _convert(shp_path, parquet_path)
    pointset: Pointset_V1_3_0 = result[0]

    assert pointset.locations.coordinates.length == 2

    coordinates_table = pq.read_table(parquet_path / pointset.locations.coordinates.data)
    assert coordinates_table.column("x").to_pylist() == [1.0, 4.0]
    assert coordinates_table.column("y").to_pylist() == [2.0, 5.0]
    assert coordinates_table.column("z").to_pylist() == [3.0, 6.0]

    measurement_attr = _get_attr(pointset, "M")
    assert measurement_attr is not None
    assert _read_column_values(parquet_path, measurement_attr.values.data) == [10.0, 20.0]

    name_attr = _get_attr(pointset, "NAME")
    assert name_attr is not None
    assert _read_column_values(parquet_path, name_attr.values.data) == ["A", "B"]


def test_multipointm_repeats_record_attributes_per_point(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "multipointm"
    with shapefile.Writer(shp_path, shapeType=shapefile.MULTIPOINTM) as writer:
        writer.field("NAME", "C")

        writer.multipointm([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
        writer.record(NAME="first")

        writer.multipointm([[2.0, 2.0, 3.0], [3.0, 3.0, 4.0]])
        writer.record(NAME="second")

    result = _convert(shp_path, parquet_path)
    pointset: Pointset_V1_3_0 = result[0]

    assert pointset.locations.coordinates.length == 4

    measurement_attr = _get_attr(pointset, "M")
    assert measurement_attr is not None
    assert _read_column_values(parquet_path, measurement_attr.values.data) == [1.0, 2.0, 3.0, 4.0]

    name_attr = _get_attr(pointset, "NAME")
    assert name_attr is not None
    assert _read_column_values(parquet_path, name_attr.values.data) == ["first", "first", "second", "second"]


def test_multipointz_preserves_z_and_measurements(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "multipointz"
    with shapefile.Writer(shp_path, shapeType=shapefile.MULTIPOINTZ) as writer:
        writer.field("NAME", "C")

        writer.multipointz([[0.0, 0.0, 10.0, 100.0], [1.0, 1.0, 20.0, 200.0]])
        writer.record(NAME="record-1")

    result = _convert(shp_path, parquet_path)
    pointset: Pointset_V1_3_0 = result[0]

    coordinates_table = pq.read_table(parquet_path / pointset.locations.coordinates.data)
    assert coordinates_table.column("z").to_pylist() == [10.0, 20.0]

    measurement_attr = _get_attr(pointset, "M")
    assert measurement_attr is not None
    assert _read_column_values(parquet_path, measurement_attr.values.data) == [100.0, 200.0]


def test_empty_cpg_raises_invalid_shp_error(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "point_with_empty_cpg"
    with shapefile.Writer(shp_path, shapeType=shapefile.POINT) as writer:
        writer.field("NAME", "C")
        writer.point(1.0, 2.0)
        writer.record(NAME="A")

    cpg_path = shp_path.with_suffix(".cpg")
    cpg_path.write_text("", encoding="utf-8")

    with pytest.raises(InvalidSHPError, match="is empty"):
        convert_shp(
            filepath=shp_path,
            filepath_shx=shp_path.with_suffix(".shx"),
            filepath_dbf=shp_path.with_suffix(".dbf"),
            filepath_cpg=cpg_path,
            upload_path=parquet_path,
            publish_objects=False,
            overwrite_existing_objects=True,
        )


def test_dbf_header_terminator_fallback_repairs_and_converts(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "point_with_bad_dbf_terminator"
    with shapefile.Writer(shp_path, shapeType=shapefile.POINTZ) as writer:
        writer.field("ID", "N")
        writer.pointz(1.0, 2.0, 3.0, 10.0)
        writer.record(ID=7)

    dbf_path = shp_path.with_suffix(".dbf")
    dbf_bytes = bytearray(dbf_path.read_bytes())
    header_length = dbf_bytes[8] + (dbf_bytes[9] << 8)
    dbf_bytes[header_length - 1] = 0x0A
    dbf_path.write_bytes(dbf_bytes)

    result = convert_shp(
        filepath=shp_path,
        filepath_shx=shp_path.with_suffix(".shx"),
        filepath_dbf=dbf_path,
        upload_path=parquet_path,
        publish_objects=False,
        overwrite_existing_objects=True,
    )

    assert len(result) == 1
    pointset: Pointset_V1_3_0 = result[0]

    measurement_attr = _get_attr(pointset, "M")
    assert measurement_attr is not None
    assert _read_column_values(parquet_path, measurement_attr.values.data) == [10.0]

    id_attr = _get_attr(pointset, "ID")
    assert id_attr is not None
    assert _read_column_values(parquet_path, id_attr.values.data) == [7]


def test_non_publish_does_not_create_evo_clients(tmp_path: Path, parquet_path: Path, monkeypatch: pytest.MonkeyPatch):
    shp_path = tmp_path / "point_non_publish"
    with shapefile.Writer(shp_path, shapeType=shapefile.POINT) as writer:
        writer.field("NAME", "C")
        writer.point(5.0, 6.0)
        writer.record(NAME="A")

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("Connection factory must not be called when publish_objects=False")

    monkeypatch.setattr(
        "evo.data_converters.shp.importer.shp_to_evo.create_evo_object_service_and_data_client",
        _raise_if_called,
    )

    result = convert_shp(
        filepath=shp_path,
        filepath_shx=shp_path.with_suffix(".shx"),
        filepath_dbf=shp_path.with_suffix(".dbf"),
        evo_workspace_metadata=object(),
        upload_path=parquet_path,
        publish_objects=False,
        overwrite_existing_objects=True,
    )

    assert len(result) == 1
    assert isinstance(result[0], Pointset_V1_3_0)


def test_point_numeric_decimal_field_is_continuous(tmp_path: Path, parquet_path: Path):
    shp_path = tmp_path / "point_numeric_decimal"
    with shapefile.Writer(shp_path, shapeType=shapefile.POINT) as writer:
        writer.field("SCORE", "N", decimal=3)
        writer.point(1.0, 2.0)
        writer.record(SCORE=1.25)

    result = _convert(shp_path, parquet_path)
    pointset: Pointset_V1_3_0 = result[0]

    score_attr = _get_attr(pointset, "SCORE")
    assert score_attr is not None
    assert score_attr.attribute_type == "scalar"
    assert _read_column_values(parquet_path, score_attr.values.data) == [1.25]


def test_publish_omits_none_uuid_without_mutating_model(monkeypatch: pytest.MonkeyPatch):
    class _DummyObject:
        def __init__(self):
            self.uuid = None
            self.name = "dummy"

        def as_dict(self):
            return {"uuid": self.uuid, "name": self.name}

    dummy_object = _DummyObject()
    captured: dict[str, object] = {}

    class _DummyParser:
        def __init__(self, **_kwargs):
            pass

        def parse_shp(self):
            return dummy_object

    def _fake_create_clients(**_kwargs):
        return object(), object()

    def _fake_publish(models, *_args, **_kwargs):
        captured["model"] = models[0]
        captured["payload"] = models[0].as_dict()
        return ["published"]

    monkeypatch.setattr("evo.data_converters.shp.importer.shp_to_evo.ShpParser", _DummyParser)
    monkeypatch.setattr("evo.data_converters.shp.importer.shp_to_evo.prj_to_crs", lambda _path: "unspecified")
    monkeypatch.setattr(
        "evo.data_converters.shp.importer.shp_to_evo.create_evo_object_service_and_data_client",
        _fake_create_clients,
    )
    monkeypatch.setattr("evo.data_converters.shp.importer.shp_to_evo.publish_geoscience_objects_sync", _fake_publish)

    result = convert_shp(
        filepath="dummy.shp",
        filepath_shx="dummy.shx",
        filepath_dbf="dummy.dbf",
        publish_objects=True,
    )

    assert result == ["published"]
    assert captured["payload"] == {"name": "dummy"}
    assert dummy_object.as_dict() == {"uuid": None, "name": "dummy"}
    assert "as_dict" not in dummy_object.__dict__


def test_pointset_crs_override(tmp_path: Path, parquet_path: Path):
    """Test that crs_override works for pointset."""
    shp_path = tmp_path / "point_crs"
    with shapefile.Writer(shp_path, shapeType=shapefile.POINT) as writer:
        writer.field("NAME", "C")
        writer.point(0.0, 1.0)
        writer.record(NAME="test-point")

    result = convert_shp(
        filepath=shp_path,
        filepath_shx=shp_path.with_suffix(".shx"),
        filepath_dbf=shp_path.with_suffix(".dbf"),
        crs_override={"epsg_code": 3857},
        upload_path=parquet_path,
        publish_objects=False,
        overwrite_existing_objects=True,
    )

    assert len(result) == 1
    pointset: Pointset_V1_3_0 = result[0]

    crs = pointset.coordinate_reference_system
    assert hasattr(crs, "epsg_code")
    assert crs.epsg_code == 3857
