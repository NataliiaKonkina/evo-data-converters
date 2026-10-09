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
import os
from typing import TYPE_CHECKING, Optional

from evo.data_converters.common import (
    EvoWorkspaceMetadata,
    crs_from_any,
    create_evo_object_service_and_data_client,
    publish_geoscience_objects_sync,
)
from evo.data_converters.shp.importer.implementation.local_data import LocalDataClient
from evo.data_converters.shp.importer.implementation.prj_parser import prj_to_crs
from evo.data_converters.shp.importer.implementation.shp_parser import ShpParser
from evo.objects.data import ObjectMetadata
from evo_schemas.objects import LineSegments_V2_1_0
from evo_schemas.objects.pointset import Pointset_V1_3_0
from evo_schemas.objects.triangle_mesh import TriangleMesh_V2_2_0

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


class _PublishObjectAdapter:
    """Wraps a geoscience object and omits `uuid` from payload when it is None."""

    def __init__(self, obj):
        self._obj = obj

    @property
    def uuid(self):
        return getattr(self._obj, "uuid", None)

    @uuid.setter
    def uuid(self, value):
        setattr(self._obj, "uuid", value)

    def as_dict(self):
        payload = self._obj.as_dict()
        if getattr(self._obj, "uuid", None) is None:
            payload.pop("uuid", None)
        return payload

    def __getattr__(self, item):
        return getattr(self._obj, item)


def convert_shp(
    filepath: str,
    filepath_shx: str,
    filepath_dbf: str,
    filepath_prj: Optional[str] = None,
    filepath_cpg: Optional[str] = None,
    filepath_sbn: Optional[str] = None,
    filepath_sbx: Optional[str] = None,
    filepath_xml: Optional[str] = None,
    crs_override: Optional[dict] = None,
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
    tags: Optional[dict[str, str]] = None,
    upload_path: str = "",
    publish_objects: bool = True,
    overwrite_existing_objects: bool = False,
) -> list[TriangleMesh_V2_2_0 | LineSegments_V2_1_0 | Pointset_V1_3_0] | list[ObjectMetadata]:
    """
    Convert an ESRI shapefile (.shp, .shx, and .dbf) to a geoscience object.

    :param filepath: Path to the .shp geometry file.
    :param filepath_shx: Path to the .shx spatial index file.
    :param filepath_dbf: Path to the .dbf attribute file.
    :param filepath_prj: (Optional) Path to the .prj projection file. If not provided,
     the CRS will be set to "unspecified".
    :param filepath_cpg: (Optional) Path to the .cpg code page file. If provided, the encoding
     specified in the file will be used when reading the .dbf attribute data.
    :param filepath_sbn: (Optional) Path to the .sbn spatial index file. Accepted but not currently used.
    :param filepath_sbx: (Optional) Path to the .sbx spatial index file. Accepted but not currently used.
    :param filepath_xml: (Optional) Path to the .shp.xml metadata file. Accepted but not currently used.
    :param crs_override: (Optional) Override for CRS, e.g. ``{"epsg_code": 4326}``. When provided,
     this takes precedence over the .prj file.
    :param evo_workspace_metadata: (Optional) Evo workspace metadata.
    :param service_manager_widget: (Optional) Service Manager Widget for use in jupyter notebooks.
    :param tags: (Optional) Dict of tags to add to the Geoscience Object(s).
    :param upload_path: (Optional) Path objects will be published under.
    :param publish_objects: (Optional) Set False to prevent publishing and instead return Geoscience models.
    :param overwrite_existing_objects: (Optional) Set True to overwrite any existing object at the destiation path.

    :return: list[ObjectMetadata] if publish_objects is true, otherwise list of converted geoscience objects.

    :raise MissingConnectionDetailsError: If no connections details could be derived.
    :raise ConflictingConnectionDetailsError: If both evo_workspace_metadata and service_manager_widget present.
    :raise InvalidOBJError: If the input shapefile is invalid or cannot be parsed.
    :raise InvalidCRSError: If the input CRS information is invalid.

    One of evo_workspace_metadata or service_manager_widget is required for publishing.
    """
    geoscience_objects = []

    if publish_objects:
        object_service_client, data_client = create_evo_object_service_and_data_client(
            evo_workspace_metadata=evo_workspace_metadata, service_manager_widget=service_manager_widget
        )
    else:
        data_client = LocalDataClient(upload_path)

    full_tags = {
        "Source": f"{os.path.basename(filepath)} (via Evo Data Converters)",
        "Stage": "Experimental",
        "InputType": "SHP",
        **(tags or {}),
    }

    # Resolve CRS: override takes precedence, then .prj file, then unspecified
    if crs_override:
        crs = crs_from_any(crs_override.get("epsg_code")) if "epsg_code" in crs_override else crs_from_any(None)
    else:
        crs = prj_to_crs(filepath_prj)

    parser = ShpParser(
        path=filepath,
        shx_path=filepath_shx,
        dbf_path=filepath_dbf,
        data_client=data_client,
        crs=crs,
        tags=full_tags,
        cpg_path=filepath_cpg,
        sbn_path=filepath_sbn,
        sbx_path=filepath_sbx,
        xml_path=filepath_xml,
    )
    geoscience_object = parser.parse_shp()

    geoscience_objects.append(geoscience_object)

    objects_metadata = None
    if publish_objects:
        print("Publishing Shapefile")

        # Preserve original models and sanitize publish payload via adapters.
        publish_models = [_PublishObjectAdapter(obj) for obj in geoscience_objects]

        objects_metadata = publish_geoscience_objects_sync(
            publish_models, object_service_client, data_client, upload_path, overwrite_existing_objects
        )

    return objects_metadata if objects_metadata else geoscience_objects


def get_output_schema(
    filepath: str,
    filepath_shx: str,
    filepath_dbf: str,
    filepath_prj: Optional[str] = None,
    filepath_cpg: Optional[str] = None,
    filepath_sbn: Optional[str] = None,
    filepath_sbx: Optional[str] = None,
    filepath_xml: Optional[str] = None,
) -> type[TriangleMesh_V2_2_0 | LineSegments_V2_1_0 | Pointset_V1_3_0]:
    parser = ShpParser(
        path=filepath,
        shx_path=filepath_shx,
        dbf_path=filepath_dbf,
        data_client=None,
        crs=None,
        tags=None,
        cpg_path=filepath_cpg,
        sbn_path=filepath_sbn,
        sbx_path=filepath_sbx,
        xml_path=filepath_xml,
    )
    return parser.get_output_schema_type()
