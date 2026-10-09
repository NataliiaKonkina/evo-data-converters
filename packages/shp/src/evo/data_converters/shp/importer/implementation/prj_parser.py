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

from evo.data_converters.common import crs_from_any
from evo_schemas.components import Crs_V1_0_1
from pyproj import CRS
from pyproj.exceptions import CRSError

try:
    from evo.data_converters.common import InvalidCRSError
except ImportError:  # pragma: no cover - compatibility with older common package versions
    InvalidCRSError = None


def prj_to_crs(filepath: str | None) -> Crs_V1_0_1:
    invalid_crs_errors = (CRSError,) if InvalidCRSError is None else (InvalidCRSError, CRSError)

    if filepath is None:
        return crs_from_any(filepath)

    with open(filepath) as reader:
        crs_str = reader.read()

    try:
        # Prefer EPSG when the PRJ can be resolved to an authority code.
        # This improves interoperability with consumers that do not fully
        # resolve arbitrary OGC/ESRI WKT strings.
        parsed = CRS.from_user_input(crs_str)
        authority = parsed.to_authority()
        if authority and authority[0] == "EPSG":
            return crs_from_any(f"EPSG:{authority[1]}")

        # Some pyproj/proj builds do not return to_authority() for ESRI WKT
        # even when an EPSG match exists. Fall back to EPSG inference.
        epsg_code = parsed.to_epsg()
        if epsg_code is None:
            epsg_code = parsed.to_epsg(min_confidence=0)
        if epsg_code is not None:
            return crs_from_any(f"EPSG:{epsg_code}")

        return crs_from_any(parsed.to_wkt(version="WKT2_2019"))
    except invalid_crs_errors:
        # Keep backward-compatible fallback behavior for unreadable PRJ content.
        try:
            return crs_from_any(crs_str)
        except invalid_crs_errors:
            return crs_from_any(None)
