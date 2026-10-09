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

from evo.data_converters.shp.importer.shp_to_evo import Pointset_V1_3_0, get_output_schema
from evo_schemas.objects.triangle_mesh import TriangleMesh_V2_2_0
from evo_schemas.objects.line_segments import LineSegments_V2_1_0

data_dir = Path(__file__).parent.parent / "data"


def test_resolve_schema_test_shape_is_triangle_mesh():
    shp_path = data_dir / "test_shape.shp"

    schema_type = get_output_schema(
        filepath=str(shp_path),
        filepath_shx=str(shp_path.with_suffix(".shx")),
        filepath_dbf=str(shp_path.with_suffix(".dbf")),
        filepath_prj=str(shp_path.with_suffix(".prj")),
        filepath_cpg=str(shp_path.with_suffix(".cpg")),
    )

    assert schema_type is TriangleMesh_V2_2_0


def test_resolve_schema_test_shape_is_line_segments():
    shp_path = data_dir / "ne_10m_admin_0_countries_bra.shp"

    schema_type = get_output_schema(
        filepath=str(shp_path),
        filepath_shx=str(shp_path.with_suffix(".shx")),
        filepath_dbf=str(shp_path.with_suffix(".dbf")),
        filepath_prj=str(shp_path.with_suffix(".prj")),
    )

    assert schema_type is LineSegments_V2_1_0


def test_resolve_schema_test_shape_is_pointset():
    shp_path = data_dir / "ne_10m_airports.shp"

    schema_type = get_output_schema(
        filepath=str(shp_path),
        filepath_shx=str(shp_path.with_suffix(".shx")),
        filepath_dbf=str(shp_path.with_suffix(".dbf")),
        filepath_prj=str(shp_path.with_suffix(".prj")),
        filepath_cpg=str(shp_path.with_suffix(".cpg")),
    )

    assert schema_type is Pointset_V1_3_0
