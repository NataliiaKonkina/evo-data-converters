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
    StringAttribute_V1_1_0,
)
from evo_schemas.elements import (
    BoolArray1_V1_0_1,
    DateTimeArray_V1_0_1,
    FloatArray1_V1_0_1,
    FloatArray3_V1_0_1,
    IntegerArray1_V1_0_1,
    StringArray_V1_0_1,
)
from evo_schemas.objects.pointset import Pointset_V1_3_0_Locations
from numpy import iinfo


class PointsetBuilder:
    """Uses pyshp ShapeRecords to build pointset coordinates and point attributes."""

    class _Point_Data:
        """Represents one attribute column for points."""

        def __init__(self, nan_def, data_type: shapefile.FieldType, numeric_is_float: bool = False):
            self.nan_def = nan_def
            self.data_type = data_type
            self.numeric_is_float = numeric_is_float
            self.data = []

    data_client: ObjectDataClient
    include_measurement: bool
    coordinates: list[tuple[float, float, float]]
    measurements: list[float | None]
    point_attributes: dict[str, _Point_Data]

    def __init__(self, data_client: ObjectDataClient, fields: Iterable[shapefile.Field], shape_type: int):
        """
        Initialize the PointsetBuilder.

        :param data_client: Object data client for uploading parquet files (real or stub).
        :param fields: A list of shapefile fields and their data types.
        :param shape_type: Shapefile shape type used for this file.
        """
        self.data_client = data_client
        self.include_measurement = shape_type in {
            shapefile.POINTM,
            shapefile.POINTZ,
            shapefile.MULTIPOINTM,
            shapefile.MULTIPOINTZ,
        }
        self.coordinates = []
        self.measurements = []
        self.point_attributes = {}

        for field in fields:
            numeric_is_float = field.field_type == shapefile.FieldType.N and field.decimal > 0
            self.point_attributes[field.name] = self._Point_Data(
                self._get_nan_def(field.field_type, numeric_is_float),
                field.field_type,
                numeric_is_float,
            )

    def add_shape_record(self, shape_record: shapefile.ShapeRecord):
        """
        Add all points and per-point attributes from the provided ShapeRecord.

        For point shapes this adds one point. For multipoint shapes this adds all points in the record.
        DBF record values are repeated for all points in the same shape record.

        :param shape_record: The ShapeRecord to add.
        """
        shape = shape_record.shape
        record = shape_record.record

        # Null shapes may carry record data but no geometry.
        if shape.shapeType == shapefile.NULL:
            return

        points_xyz = self._shape_points_to_xyz(shape)
        if not points_xyz:
            return

        points_m = self._shape_points_to_m(shape, len(points_xyz))
        self.coordinates.extend(points_xyz)
        self.measurements.extend(points_m)

        record_dict = record.as_dict()
        repeat_count = len(points_xyz)

        for key, attr in self.point_attributes.items():
            value = record_dict.get(key)
            if attr.data_type == shapefile.FieldType.D:
                value = date_to_evo_timestamp(value)
            elif attr.data_type == shapefile.FieldType.N and value is not None:
                value = self._normalize_numeric_value(value, attr)

            default = attr.nan_def
            point_value = value if value is not None else default
            attr.data.extend([point_value] * repeat_count)

    def build(self) -> Pointset_V1_3_0_Locations:
        """
        Build point coordinates and point attributes from ShapeRecords added so far.
        """
        coordinates_schema = pa.schema([("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())])
        coordinates_table = pa.table(
            {
                "x": [p[0] for p in self.coordinates],
                "y": [p[1] for p in self.coordinates],
                "z": [p[2] for p in self.coordinates],
            },
            schema=coordinates_schema,
        )
        coordinates_data = self.data_client.save_table(coordinates_table)
        coordinates = FloatArray3_V1_0_1(**coordinates_data)

        attributes = []

        if self.include_measurement:
            measurement_name = "M"
            nan_value = -1.0e38
            measurement_values = [value if value is not None else nan_value for value in self.measurements]
            schema = pa.schema([("values", pa.float64())])
            table = pa.table({"values": measurement_values}, schema=schema)
            data_hash = self._create_parquet_file(table)
            values = FloatArray1_V1_0_1(data=data_hash, length=len(measurement_values), data_type="float64")
            attributes.append(
                ContinuousAttribute_V1_1_0(
                    name=measurement_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=NanContinuous_V1_0_1(values=[nan_value]),
                    values=values,
                )
            )

        for name, point_attr in self.point_attributes.items():
            attribute = self._attribute_from_point(name, point_attr)
            if attribute is not None:
                attributes.append(attribute)

        return Pointset_V1_3_0_Locations(coordinates=coordinates, attributes=attributes or None)

    def _shape_points_to_xyz(self, shape: shapefile.Shape) -> list[tuple[float, float, float]]:
        """
        Convert pyshp point arrays to explicit XYZ tuples.
        """
        points = list(shape.points)
        z_values = list(shape.z) if getattr(shape, "z", None) is not None else []

        if len(z_values) == len(points):
            return [points[i] + (z_values[i],) for i in range(len(points))]

        return [point + (0.0,) for point in points]

    def _shape_points_to_m(self, shape: shapefile.Shape, point_count: int) -> list[float | None]:
        """
        Convert pyshp measurement array to a per-point list.
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
                return -1.0e38
            case shapefile.FieldType.D:
                return iinfo("int64").min
            case _:
                return None

    def _normalize_numeric_value(self, value, point_attr: _Point_Data) -> int | float:
        """Normalize DBF numeric values based on field precision metadata."""
        try:
            if point_attr.numeric_is_float:
                return float(value)
            return int(value)
        except (TypeError, ValueError):
            return point_attr.nan_def

    def _attribute_from_point(self, field_name: str, point_attr: _Point_Data) -> OneOfAttribute_V1_2_0_Item | None:
        """
        Convert one point-attribute column to the corresponding Evo attribute.
        """
        match point_attr.data_type:
            case shapefile.FieldType.N:
                if point_attr.numeric_is_float:
                    nan_description = NanContinuous_V1_0_1(values=[point_attr.nan_def])
                    schema = pa.schema([("values", pa.float64())])
                    table = pa.table({"values": point_attr.data}, schema=schema)
                    data_hash = self._create_parquet_file(table)
                    values = FloatArray1_V1_0_1(data=data_hash, length=len(point_attr.data), data_type="float64")
                    return ContinuousAttribute_V1_1_0(
                        name=field_name,
                        key=data_hash,
                        attribute_description=None,
                        nan_description=nan_description,
                        values=values,
                    )

                nan_description = NanCategorical_V1_0_1(values=[point_attr.nan_def])
                schema = pa.schema([("values", pa.int64())])
                table = pa.table({"values": point_attr.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = IntegerArray1_V1_0_1(data=data_hash, length=len(point_attr.data), data_type="int64")
                return IntegerAttribute_V1_1_0(
                    name=field_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=nan_description,
                    values=values,
                )
            case shapefile.FieldType.F:
                nan_description = NanContinuous_V1_0_1(values=[point_attr.nan_def])
                schema = pa.schema([("values", pa.float64())])
                table = pa.table({"values": point_attr.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = FloatArray1_V1_0_1(data=data_hash, length=len(point_attr.data), data_type="float64")
                return ContinuousAttribute_V1_1_0(
                    name=field_name,
                    key=data_hash,
                    attribute_description=None,
                    nan_description=nan_description,
                    values=values,
                )
            case shapefile.FieldType.C | shapefile.FieldType.M:
                schema = pa.schema([("values", pa.string())])
                table = pa.table({"values": point_attr.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = StringArray_V1_0_1(data=data_hash, length=len(point_attr.data))
                return StringAttribute_V1_1_0(name=field_name, key=data_hash, attribute_description=None, values=values)
            case shapefile.FieldType.L:
                schema = pa.schema([("values", pa.bool_())])
                table = pa.table({"values": point_attr.data}, schema=schema)
                data_hash = self._create_parquet_file(table)
                values = BoolArray1_V1_0_1(data=data_hash, length=len(point_attr.data))
                return BoolAttribute_V1_1_0(name=field_name, key=data_hash, attribute_description=None, values=values)
            case shapefile.FieldType.D:
                schema = pa.schema([("values", pa.timestamp("us", tz="UTC"))])
                table = pa.table({"values": point_attr.data}, schema=schema)
                nan_description = NanCategorical_V1_0_1(values=[point_attr.nan_def])
                data_hash = self._create_parquet_file(table)
                values = DateTimeArray_V1_0_1(data=data_hash, length=len(point_attr.data))
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
        Write a parquet file from the given table using the data client.

        :param table: The table to write.

        :return: The file hash of the written table.
        """
        saved_table_info = self.data_client.save_table(table)
        return saved_table_info["data"]
