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

from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile

import shapefile
from evo.data_converters.shp.importer.exceptions import InvalidSHPError
from evo.data_converters.shp.importer.implementation.line_segments_builder import LineSegmentsBuilder
from evo.data_converters.shp.importer.implementation.mesh_builder import MeshBuilder
from evo.data_converters.shp.importer.implementation.pointset_builder import PointsetBuilder
from evo.objects.utils.data import ObjectDataClient
from evo_schemas.components import BoundingBox_V1_0_1, Crs_V1_0_1
from evo_schemas.objects import LineSegments_V2_1_0
from evo_schemas.objects.pointset import Pointset_V1_3_0
from evo_schemas.objects.triangle_mesh import TriangleMesh_V2_2_0


class ShpParser:
    """
    Converts shapefiles (consisting of at least .shp, .shx, and .dbf files) to Evo geoscience objects.
    Supported shape types:
    - Multipatch -> Triangle Mesh
    - PolyLine/PolyLineZ/PolyLineM -> Line Segments
    - Polygon/PolygonZ/PolygonM -> Line Segments
    - Point/PointZ/PointM -> Pointset
    - MultiPoint/MultiPointZ/MultiPointM -> Pointset
    """

    _LINEAR_SHAPE_TYPES = {
        shapefile.POLYLINE,
        shapefile.POLYLINEZ,
        shapefile.POLYLINEM,
        shapefile.POLYGON,
        shapefile.POLYGONZ,
        shapefile.POLYGONM,
    }

    _POINT_SHAPE_TYPES = {
        shapefile.POINT,
        shapefile.POINTM,
        shapefile.POINTZ,
        shapefile.MULTIPOINT,
        shapefile.MULTIPOINTM,
        shapefile.MULTIPOINTZ,
    }

    _DBF_TERMINATOR_ERROR_SUBSTRING = "dbf header lacks expected terminator"

    def __init__(
        self,
        path: str,
        shx_path: str,
        dbf_path: str,
        data_client: ObjectDataClient,
        crs: Crs_V1_0_1,
        tags: dict[str, str] = None,
        cpg_path: str | None = None,
        sbn_path: str | None = None,
        sbx_path: str | None = None,
        xml_path: str | None = None,
    ):
        """
        Initialize the shapefile to triangle mesh converter.

        :param path: The path to the .shp geometry file.
        :param shx_path: The path to the .shx spatial index file.
        :param dbf_path: The path to the .dbf attribute file.
        :param data_client: Object data client for uploading parquet files (real or stub).
        :param crs: Coordiante reference system to use for the file. Cannot be None, but can be "unspecified".
        :param tags: (Optional) Dict of tags to add to the Geoscience Object(s).
        :param cpg_path: (Optional) Path to the .cpg code page file. If provided, the encoding
        specified in the file will be used when reading the .dbf attribute data.
        :param sbn_path: (Optional) Path to the .sbn spatial index file. Accepted but not currently used.
        :param sbx_path: (Optional) Path to the .sbx spatial index file. Accepted but not currently used.
        :param xml_path: (Optional) Path to the .shp.xml metadata file. Accepted but not currently used.
        """
        self.path = path
        self.shx_path = shx_path
        self.dbf_path = dbf_path
        self.data_client = data_client
        self.crs = crs
        self.tags = tags
        self.cpg_path = cpg_path
        self.sbn_path = sbn_path
        self.sbx_path = sbx_path
        self.xml_path = xml_path

    def parse_shp(self) -> TriangleMesh_V2_2_0 | LineSegments_V2_1_0 | Pointset_V1_3_0:
        """
        Open, validate, and parse the shapefile, converting and returning it as a Geoscience Object.

        :return: Triangle Mesh or Line Segments object representing the input shapefile.

        :raise InvalidSHPError: If the input shapefile is invalid or cannot be parsed
        """
        reader_kwargs: dict = {"shp": self.path, "shx": self.shx_path, "dbf": self.dbf_path}
        if self.cpg_path is not None:
            reader_kwargs["encoding"] = self._read_cpg_encoding()

        try:
            with ExitStack() as exit_stack:
                sf = self._open_reader_with_dbf_fallback(reader_kwargs, exit_stack)
                return self._parse_reader(sf)
        except InvalidSHPError:
            raise
        except (UnicodeError, OSError, ValueError, shapefile.ShapefileException) as exc:
            if self._is_dbf_terminator_error(exc):
                raise InvalidSHPError(
                    f"Failed to parse shapefile DBF '{self.dbf_path}': DBF header terminator is invalid. "
                    "Expected 0x0D at the end of the DBF header. "
                    "Try re-exporting the shapefile from GIS software or repairing the DBF header."
                ) from exc
            raise InvalidSHPError(f"Failed to parse shapefile: {exc}") from exc

    def get_output_schema_type(self) -> type[TriangleMesh_V2_2_0 | LineSegments_V2_1_0 | Pointset_V1_3_0]:
        """
        Get the output schema type based on the input shapefile's shape type.

        :return: The schema type of the output geoscience object.
        :raise InvalidSHPError: If the input shapefile is invalid or cannot be parsed.
        """
        reader_kwargs: dict = {"shp": self.path, "shx": self.shx_path, "dbf": self.dbf_path}
        if self.cpg_path is not None:
            reader_kwargs["encoding"] = self._read_cpg_encoding()

        try:
            with ExitStack() as exit_stack:
                sf = self._open_reader_with_dbf_fallback(reader_kwargs, exit_stack)
                if sf.shapeType == shapefile.MULTIPATCH:
                    return TriangleMesh_V2_2_0
                if sf.shapeType in self._LINEAR_SHAPE_TYPES:
                    return LineSegments_V2_1_0
                if sf.shapeType in self._POINT_SHAPE_TYPES:
                    return Pointset_V1_3_0
                raise InvalidSHPError(
                    "Provided shapefile is not supported. Supported shape types are Multipatch, PolyLine[Z/M], Polygon[Z/M], Point[Z/M], and MultiPoint[Z/M]."
                )
        except InvalidSHPError:
            raise
        except (UnicodeError, OSError, ValueError, shapefile.ShapefileException) as exc:
            if self._is_dbf_terminator_error(exc):
                raise InvalidSHPError(
                    f"Failed to parse shapefile DBF '{self.dbf_path}': DBF header terminator is invalid. "
                    "Expected 0x0D at the end of the DBF header. "
                    "Try re-exporting the shapefile from GIS software or repairing the DBF header."
                ) from exc
            raise InvalidSHPError(f"Failed to parse shapefile: {exc}") from exc

    def _parse_reader(self, sf: shapefile.Reader) -> TriangleMesh_V2_2_0 | LineSegments_V2_1_0 | Pointset_V1_3_0:
        if sf.shapeType == shapefile.MULTIPATCH:
            mesh_builder = MeshBuilder(self.data_client, sf.fields[1:])

            bounding_box = self._build_bounding_box(sf)

            for shape_record in sf.iterShapeRecords():
                mesh_builder.add_shape_record(shape_record)

            embedded_mesh = mesh_builder.build()

            return TriangleMesh_V2_2_0(
                triangles=embedded_mesh.triangles,
                parts=embedded_mesh.parts,
                name=Path(self.path).stem,
                uuid=None,
                tags=self.tags,
                bounding_box=bounding_box,
                coordinate_reference_system=self.crs,
            )

        if sf.shapeType in self._LINEAR_SHAPE_TYPES:
            line_segments_builder = LineSegmentsBuilder(self.data_client, sf.fields[1:], sf.shapeType)

            bounding_box = self._build_bounding_box(sf)

            for i, shape_record in enumerate(sf.iterShapeRecords()):
                line_segments_builder.add_shape_record(shape_record, i)

            segments, parts = line_segments_builder.build()

            return LineSegments_V2_1_0(
                name=Path(self.path).stem,
                uuid=None,
                tags=self.tags,
                bounding_box=bounding_box,
                coordinate_reference_system=self.crs,
                segments=segments,
                parts=parts,
            )

        if sf.shapeType in self._POINT_SHAPE_TYPES:
            pointset_builder = PointsetBuilder(self.data_client, sf.fields[1:], sf.shapeType)

            bounding_box = self._build_bounding_box(sf)

            for shape_record in sf.iterShapeRecords():
                pointset_builder.add_shape_record(shape_record)

            locations = pointset_builder.build()

            return Pointset_V1_3_0(
                name=Path(self.path).stem,
                uuid=None,
                description=None,
                tags=self.tags,
                bounding_box=bounding_box,
                coordinate_reference_system=self.crs,
                locations=locations,
            )

        raise InvalidSHPError(
            "Provided shapefile is not supported. Supported shape types are Multipatch, PolyLine[Z/M], Polygon[Z/M], Point[Z/M], and MultiPoint[Z/M]."
        )

    def _open_reader_with_dbf_fallback(self, reader_kwargs: dict, exit_stack: ExitStack) -> shapefile.Reader:
        try:
            reader = shapefile.Reader(**reader_kwargs)
        except shapefile.ShapefileException as exc:
            if not self._is_dbf_terminator_error(exc):
                raise

            repaired_dbf = self._create_repaired_dbf_copy(exit_stack)
            fallback_kwargs = {**reader_kwargs, "dbf": repaired_dbf}

            try:
                reader = shapefile.Reader(**fallback_kwargs)
            except (OSError, ValueError, shapefile.ShapefileException) as fallback_exc:
                raise InvalidSHPError(
                    f"Failed to parse shapefile DBF '{self.dbf_path}': detected invalid DBF header terminator and "
                    f"automatic repair fallback failed: {fallback_exc}"
                ) from fallback_exc

        return exit_stack.enter_context(reader)

    @classmethod
    def _is_dbf_terminator_error(cls, exc: BaseException) -> bool:
        return cls._DBF_TERMINATOR_ERROR_SUBSTRING in str(exc).lower()

    def _create_repaired_dbf_copy(self, exit_stack: ExitStack) -> str:
        source_dbf_path = Path(self.dbf_path)
        temp_dir = Path(exit_stack.enter_context(tempfile.TemporaryDirectory()))
        repaired_dbf_path = temp_dir / source_dbf_path.name
        shutil.copyfile(source_dbf_path, repaired_dbf_path)

        dbf_bytes = bytearray(repaired_dbf_path.read_bytes())
        if len(dbf_bytes) < 10:
            raise InvalidSHPError(f"DBF file '{self.dbf_path}' is too short to contain a valid header.")

        header_length = dbf_bytes[8] + (dbf_bytes[9] << 8)
        if header_length <= 0 or header_length > len(dbf_bytes):
            raise InvalidSHPError(f"DBF file '{self.dbf_path}' has an invalid header length value ({header_length}).")

        dbf_bytes[header_length - 1] = 0x0D
        repaired_dbf_path.write_bytes(dbf_bytes)
        return str(repaired_dbf_path)

    def _read_cpg_encoding(self) -> str:
        """Read code page text from .cpg file, allowing UTF-8 BOM."""
        try:
            with open(self.cpg_path, encoding="utf-8-sig") as cpg_file:
                encoding = cpg_file.read().strip()
        except OSError as exc:
            raise InvalidSHPError(f"Could not read CPG file '{self.cpg_path}': {exc}") from exc

        if not encoding:
            raise InvalidSHPError(f"CPG file '{self.cpg_path}' is empty.")

        return encoding

    @staticmethod
    def _build_bounding_box(reader: shapefile.Reader) -> BoundingBox_V1_0_1:
        """Build bounding box from shapefile metadata."""
        return BoundingBox_V1_0_1(
            min_x=reader.bbox[0],
            min_y=reader.bbox[1],
            min_z=reader.zbox[0] or 0.0,
            max_x=reader.bbox[2],
            max_y=reader.bbox[3],
            max_z=reader.zbox[1] or 0.0,
        )
