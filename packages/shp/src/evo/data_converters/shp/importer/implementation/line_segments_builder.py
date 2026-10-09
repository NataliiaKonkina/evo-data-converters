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

from collections.abc import Iterable

import pyarrow as pa
import shapefile
from evo.data_converters.common.utils import vertices_bounding_box
from evo.data_converters.shp.importer.implementation.utils import date_to_evo_timestamp
from evo.objects.utils.data import ObjectDataClient
from evo_schemas.components import (
    BoolAttribute_V1_1_0,
    ContinuousAttribute_V1_1_0,
    DateTimeAttribute_V1_1_0,
    IntegerAttribute_V1_1_0,
    NanCategorical_V1_0_1,
    NanContinuous_V1_0_1,
    OneOfAttribute_V1_2_0_Item,
    Segments_V1_2_0,
    Segments_V1_2_0_Indices,
    Segments_V1_2_0_Vertices,
    StringAttribute_V1_1_0,
)
from evo_schemas.elements import (
    BoolArray1_V1_0_1,
    DateTimeArray_V1_0_1,
    FloatArray1_V1_0_1,
    IndexArray2_V1_0_1,
    IntegerArray1_V1_0_1,
    StringArray_V1_0_1,
)
from evo_schemas.objects import LineSegments_V2_1_0_Parts
from numpy import iinfo


class LineSegmentsBuilder:
    """Uses pyshp ShapeRecords to build line-segment geometry, parts, and attributes."""

    class _Part_Data:
        """Represents one set of attributes for line-segment parts."""

        def __init__(self, nan_def, data_type: shapefile.FieldType, numeric_is_float: bool = False):
            self.nan_def = nan_def
            self.data_type = data_type
            self.numeric_is_float = numeric_is_float
            self.data = []

    data_client: ObjectDataClient
    is_polygon: bool
    include_measurement: bool
    vertices: list[tuple[float, float, float]]
    measurements: list[float]
    segment_indices: list[tuple[int, int]]
    parts: list[tuple[int, int]]
    part_attributes: dict[str, _Part_Data]
    part_index_attr_name: str
    record_index_attr_name: str

    def __init__(self, data_client: ObjectDataClient, fields: Iterable[shapefile.Field], shape_type: int):
        """
        Initialize the LineSegmentsBuilder.

        :param data_client: Object data client for uploading parquet files (real or stub).
        :param fields: A list of shapefile fields and their data types.
        :param shape_type: Shapefile shape type used for this file.
        """
        self.data_client = data_client
        self.is_polygon = shape_type in {shapefile.POLYGON, shapefile.POLYGONM, shapefile.POLYGONZ}
        self.include_measurement = shape_type in {
            shapefile.POLYLINEM,
            shapefile.POLYLINEZ,
            shapefile.POLYGONM,
            shapefile.POLYGONZ,
        }
        self.vertices = []
        self.measurements = []
        self.segment_indices = []
        self.parts = []
        self.part_attributes = {}

        for field in fields:
            numeric_is_float = field.field_type == shapefile.FieldType.N and field.decimal > 0
            self.part_attributes[field.name] = self._Part_Data(
                self._get_nan_def(field.field_type, numeric_is_float), field.field_type, numeric_is_float
            )

        # Keep part metadata explicit so downstream users can map record/part provenance.
        self.part_index_attr_name = "shp_part_index"
        self.record_index_attr_name = "shp_record_index"
        self.part_attributes[self.part_index_attr_name] = self._Part_Data(iinfo("int64").min, shapefile.FieldType.N)
        self.part_attributes[self.record_index_attr_name] = self._Part_Data(iinfo("int64").min, shapefile.FieldType.N)

    def add_shape_record(self, shape_record: shapefile.ShapeRecord, record_index: int):
        """
        Adds all segments and attributes in the provided ShapeRecord.

        :param shape_record: The ShapeRecord to add.
        :param record_index: Zero-based index of this ShapeRecord within the shapefile.
        """
        shape = shape_record.shape
        record = shape_record.record

        # Null shapes may carry record data but no geometry.
        if shape.shapeType == shapefile.NULL:
            return

        points_xyz = self._shape_points_to_xyz(shape)
        points_m = self._shape_points_to_m(shape, len(points_xyz))

        shape_parts = list(shape.parts) + [len(points_xyz)]
        record_dict = record.as_dict()

        for part_index in range(len(shape_parts) - 1):
            start = shape_parts[part_index]
            end = shape_parts[part_index + 1]
            part_points = points_xyz[start:end]
            part_m = points_m[start:end]

            if len(part_points) < 2:
                continue

            vertex_start = len(self.vertices)
            self.vertices.extend(part_points)
            self.measurements.extend(part_m)

            segment_start = len(self.segment_indices)

            for i in range(len(part_points) - 1):
                self.segment_indices.append((vertex_start + i, vertex_start + i + 1))

            # Polygon parts are closed rings. Add explicit close segment only when needed.
            if self.is_polygon and part_points[-1] != part_points[0]:
                self.segment_indices.append((vertex_start + len(part_points) - 1, vertex_start))

            segment_count = len(self.segment_indices) - segment_start
            if segment_count == 0:
                continue

            self.parts.append((segment_start, segment_count))

            for key, attr in self.part_attributes.items():
                if key == self.part_index_attr_name:
                    attr.data.append(part_index)
                    continue
                if key == self.record_index_attr_name:
                    attr.data.append(record_index)
                    continue

                value = record_dict.get(key)
                if attr.data_type == shapefile.FieldType.D:
                    value = date_to_evo_timestamp(value)
                elif attr.data_type == shapefile.FieldType.N and value is not None:
                    value = self._normalize_numeric_value(value, attr)

                default = attr.nan_def
                attr.data.append(value if value is not None else default)

    def build(self) -> tuple[Segments_V1_2_0, LineSegments_V2_1_0_Parts | None]:
        """
        Builds line-segment geometry and parts from ShapeRecords added so far.
        """
        vertices_schema = pa.schema([("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())])
        vertices_table = pa.table(
            {
                "x": [p[0] for p in self.vertices],
                "y": [p[1] for p in self.vertices],
                "z": [p[2] for p in self.vertices],
            },
            schema=vertices_schema,
        )
        vertices_data = self.data_client.save_table(vertices_table)

        vertex_attributes = []
        if self.include_measurement:
            measurement_name = "M"
            nan_value = -1.0e38
            measurement_values = [value if value is not None else nan_value for value in self.measurements]
            schema = pa.schema([("values", pa.float64())])
            table = pa.table({"values": measurement_values}, schema=schema)
            data_hash = self._create_parquet_file(table)
            values = FloatArray1_V1_0_1(data=data_hash, length=len(measurement_values), data_type="float64")
            vertex_attributes.append(
                ContinuousAttribute_V1_1_0(
                    name=measurement_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=NanContinuous_V1_0_1(values=[nan_value]),
                    values=values,
                )
            )

        vertices_go = Segments_V1_2_0_Vertices(**vertices_data, attributes=vertex_attributes or None)

        indices_schema = pa.schema([("n0", pa.uint64()), ("n1", pa.uint64())])
        indices_table = pa.table(
            {
                "n0": [index[0] for index in self.segment_indices],
                "n1": [index[1] for index in self.segment_indices],
            },
            schema=indices_schema,
        )
        indices_go = Segments_V1_2_0_Indices(**self.data_client.save_table(indices_table), attributes=None)

        segments = Segments_V1_2_0(vertices=vertices_go, indices=indices_go)

        if not self.parts:
            return segments, None

        attributes = []
        for name, part in self.part_attributes.items():
            attribute = self._attribute_from_part(name, part)
            if attribute is not None:
                attributes.append(attribute)

        offset, count = zip(*self.parts)
        chunks_schema = pa.schema([("offset", pa.uint64()), ("count", pa.uint64())])
        chunks_table = pa.table({"offset": offset, "count": count}, schema=chunks_schema)
        chunks_hash = self._create_parquet_file(chunks_table)
        chunks = IndexArray2_V1_0_1(data=chunks_hash, length=len(self.parts))

        parts = LineSegments_V2_1_0_Parts(attributes=attributes, chunks=chunks)

        return segments, parts

    def bounding_box(self):
        """Returns a bounding box for all vertices added so far."""
        import numpy as np

        return vertices_bounding_box(np.asarray(self.vertices, dtype=float))

    def _shape_points_to_xyz(self, shape: shapefile.Shape) -> list[tuple[float, float, float]]:
        """
        Converts pyshp points/z arrays to explicit XYZ tuples.
        """
        points = list(shape.points)
        z_values = list(shape.z) if getattr(shape, "z", None) is not None else []

        if len(z_values) == len(points):
            return [points[i] + (z_values[i],) for i in range(len(points))]

        return [point + (0.0,) for point in points]

    def _shape_points_to_m(self, shape: shapefile.Shape, point_count: int) -> list[float | None]:
        """
        Converts pyshp measurement array to a per-point list.
        """
        if not self.include_measurement:
            return [None] * point_count

        m_values = list(shape.m) if getattr(shape, "m", None) is not None else []
        if len(m_values) != point_count:
            return [None] * point_count

        nodata = -1.0e38
        return [value if value is not None and value > nodata else None for value in m_values]

    def _get_nan_def(self, field_type: shapefile.FieldType, numeric_is_float: bool = False):
        """
        Get an appropriate default NaN definition for the given pyshp FieldType.
        """
        match field_type:
            case shapefile.FieldType.N:
                if numeric_is_float:
                    return -1.0e38
                return iinfo("int64").min
            case shapefile.FieldType.F:
                return -1.0e38  # Standard sentinel value for missing float data
            case shapefile.FieldType.D:
                return iinfo("int64").min
            case _:
                return None

    def _normalize_numeric_value(self, value, part: _Part_Data) -> int | float:
        """Normalize DBF numeric values based on field precision metadata."""
        try:
            if part.numeric_is_float:
                return float(value)
            return int(value)
        except (TypeError, ValueError):
            return part.nan_def

    def _attribute_from_part(self, field_name: str, part: _Part_Data) -> OneOfAttribute_V1_2_0_Item | None:
        """
        Converts one part data column to the corresponding Evo attribute.
        """
        match part.data_type:
            case shapefile.FieldType.N:
                if part.numeric_is_float:
                    nan_description = NanContinuous_V1_0_1(values=[part.nan_def])
                    schema = pa.schema([("values", pa.float64())])
                    table = pa.table({"values": part.data}, schema=schema)
                    data_hash = self._create_parquet_file(table)
                    values = FloatArray1_V1_0_1(data=data_hash, length=len(part.data), data_type="float64")
                    return ContinuousAttribute_V1_1_0(
                        name=field_name,
                        key=data_hash,
                        attribute_description=None,
                        nan_description=nan_description,
                        values=values,
                    )

                nan_description = NanCategorical_V1_0_1(values=[part.nan_def])
                schema = pa.schema([("values", pa.int64())])
                table = pa.table({"values": part.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = IntegerArray1_V1_0_1(data=data_hash, length=len(part.data), data_type="int64")
                return IntegerAttribute_V1_1_0(
                    name=field_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=nan_description,
                    values=values,
                )
            case shapefile.FieldType.F:
                nan_description = NanContinuous_V1_0_1(values=[part.nan_def])
                schema = pa.schema([("values", pa.float64())])
                table = pa.table({"values": part.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = FloatArray1_V1_0_1(data=data_hash, length=len(part.data), data_type="float64")
                return ContinuousAttribute_V1_1_0(
                    name=field_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=nan_description,
                    values=values,
                )
            case shapefile.FieldType.C | shapefile.FieldType.M:
                schema = pa.schema([("values", pa.string())])
                table = pa.table({"values": part.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = StringArray_V1_0_1(data=data_hash, length=len(part.data))
                return StringAttribute_V1_1_0(name=field_name, key=data_hash, attribute_description=None, values=values)
            case shapefile.FieldType.L:
                schema = pa.schema([("values", pa.bool_())])
                table = pa.table({"values": part.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = BoolArray1_V1_0_1(data=data_hash, length=len(part.data))
                return BoolAttribute_V1_1_0(name=field_name, key=data_hash, attribute_description=None, values=values)
            case shapefile.FieldType.D:
                schema = pa.schema([("values", pa.timestamp("us", tz="UTC"))])
                table = pa.table({"values": part.data}, schema=schema)
                nan_description = NanCategorical_V1_0_1(values=[part.nan_def])
                data_hash = self._create_parquet_file(table)
                values = DateTimeArray_V1_0_1(data=data_hash, length=len(part.data))
                return DateTimeAttribute_V1_1_0(
                    name=field_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=nan_description,
                    values=values,
                )
            case _:
                return None

    def _create_parquet_file(self, table: pa.Table) -> str:
        """
        Writes a parquet file from the given table using the data client.

        :param table: The table to write.

        :return: The file hash of the written table.
        """
        saved_table_info = self.data_client.save_table(table)
        return saved_table_info["data"]
