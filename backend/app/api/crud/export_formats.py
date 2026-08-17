"""
Pure serializers for the export endpoints.

No database calls, no request context. Given already-assembled row data,
each function produces bytes (or str for CSV/TSV) in the requested format.
Keeping these pure makes them trivially unit-testable without spinning up
a session and keeps the DB layer in ``crud/export.py`` focused on queries.

Most of the binary serializers (Shapefile ZIP, GeoPackage) are ports of
the equivalent code in AddaxAI Connect's ``services/api/routers/export.py``
so the outputs match byte-for-byte where possible.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openpyxl import Workbook

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

WGS84_SRS_ID = 4326

WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)

WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)


def slugify(text: str) -> str:
    """Lowercase, strip non-word characters, collapse whitespace/underscores to hyphens."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text or "project"


# ---------------------------------------------------------------------------
# Text serializers (CSV / TSV / XLSX)
# ---------------------------------------------------------------------------

# Excel's hard cap on rows in one sheet, header row included. This is a
# limit of the file format, not of openpyxl, and there is no way to write
# more of them into a valid workbook.
XLSX_MAX_ROWS = 1_048_576


class XlsxRowLimitError(ValueError):
    """A sheet holds more rows than the XLSX format can carry.

    Raised before anything is written. openpyxl does not check this
    itself: in write-only mode it accepts any number of rows and saves a
    file whose row indexes run past the cap (verified against openpyxl
    3.1.5, which happily wrote index 1048600), and Excel then refuses to
    open it. So without this guard the user gets a corrupt download and
    no explanation, which is the silent failure the export is supposed
    to be incapable of.

    The message is written for the end user: every caller surfaces it
    verbatim, the export endpoints as the 422 detail and the folder-run
    Save step as a module error.
    """


def _check_xlsx_row_limit(
    sheets: list[tuple[str, list[str], list[list[Any]]]],
) -> None:
    """Refuse a workbook that cannot be written as valid XLSX.

    Checked up front for every sheet: the rows are already in memory by
    this point, so counting them is free and nothing is written before we
    know the whole workbook fits.
    """
    for title, _headers, rows in sheets:
        total = len(rows) + 1  # the header occupies a row too
        if total > XLSX_MAX_ROWS:
            raise XlsxRowLimitError(
                f"The {title.lower()} table has {len(rows):,} rows. An Excel "
                f"file can hold {XLSX_MAX_ROWS:,}, so this export cannot be "
                f"saved as XLSX. Export as CSV instead, it has no row limit."
            )


def serialize_csv(headers: list[str], rows: list[list[Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def serialize_tsv(headers: list[str], rows: list[list[Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def serialize_xlsx(
    headers: list[str], rows: list[list[Any]], sheet_title: str = "Sheet1"
) -> bytes:
    return serialize_xlsx_multi([(sheet_title, headers, rows)])


def serialize_xlsx_multi(
    sheets: list[tuple[str, list[str], list[list[Any]]]],
) -> bytes:
    """One workbook with several sheets, as bytes (for HTTP responses)."""
    buf = io.BytesIO()
    _build_xlsx_workbook(sheets).save(buf)
    return buf.getvalue()


def write_xlsx_multi(
    sheets: list[tuple[str, list[str], list[list[Any]]]],
    path: Path,
) -> None:
    """One workbook with several sheets, saved straight to ``path``.

    For disk targets, so the zipped workbook never has to exist as one
    in-memory bytes blob the way ``serialize_xlsx_multi`` requires.
    """
    _build_xlsx_workbook(sheets).save(str(path))


def _build_xlsx_workbook(
    sheets: list[tuple[str, list[str], list[list[Any]]]],
) -> Workbook:
    """Each entry is ``(sheet_title, headers, rows)``; sheets are added
    in order.

    Uses openpyxl's ``write_only`` mode: rows are streamed straight into the
    sheet instead of building an in-memory cell graph, so memory stays flat
    for large exports (the whole point of the Detections / Files grains).
    A write-only workbook has no default sheet, so every sheet is created
    explicitly.

    Raises ``XlsxRowLimitError`` when any sheet is too tall for the
    format. Every XLSX writer in the app funnels through here, so the
    single-table endpoints, the combined project spreadsheet and the
    folder-run workbook all inherit the guard.
    """
    from openpyxl import Workbook

    _check_xlsx_row_limit(sheets)

    wb = Workbook(write_only=True)
    for title, headers, rows in sheets:
        ws = wb.create_sheet(title=title)
        ws.append(headers)
        for row in rows:
            ws.append(row)
    return wb


# ---------------------------------------------------------------------------
# Spatial serializers (GeoJSON / Shapefile / GeoPackage)
#
# All three consume the same ``layers`` shape:
#   {
#     "<layer_name>": [
#       {"lon": float, "lat": float, "properties": {<col>: <value>, ...}},
#       ...
#     ],
#     ...
#   }
# ---------------------------------------------------------------------------


def serialize_geojson(layers: dict[str, list[dict[str, Any]]]) -> bytes:
    """Single FeatureCollection; every feature carries a ``layer`` property."""
    features: list[dict[str, Any]] = []
    for layer_name, layer_features in layers.items():
        for feat in layer_features:
            props = dict(feat["properties"])
            props["layer"] = layer_name
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [feat["lon"], feat["lat"]],
                    },
                    "properties": props,
                }
            )
    payload = {"type": "FeatureCollection", "features": features}
    return json.dumps(payload, indent=2).encode("utf-8")


# Field definitions per layer for the Shapefile writer.
# Each entry: (short_name, field_type, size, decimal).
# Shapefile DBF fields have an 11-char name limit, so names are abbreviated.
_SHP_LAYER_FIELDS: dict[str, list[tuple[str, str, int, int]]] = {
    "deployments": [
        ("site_name", "C", 80, 0),
        ("deploy_id", "C", 36, 0),
        ("start_date", "C", 10, 0),
        ("end_date", "C", 10, 0),
        ("trap_days", "N", 10, 0),
        ("det_count", "N", 10, 0),
        ("det_rate", "N", 10, 2),
    ],
    "species_summary": [
        ("site_name", "C", 80, 0),
        ("common", "C", 80, 0),
        ("sci_name", "C", 100, 0),
        ("tax_class", "C", 80, 0),
        ("tax_order", "C", 80, 0),
        ("tax_family", "C", 80, 0),
        ("tax_genus", "C", 80, 0),
        ("tax_specs", "C", 80, 0),
        ("total_cnt", "N", 10, 0),
        ("det_rate", "N", 10, 2),
    ],
}

# Property keys to pull from each feature, in the same order as the DBF fields above.
_SHP_PROPERTY_KEYS: dict[str, list[str]] = {
    "deployments": [
        "site_name",
        "deployment_id",
        "start_date",
        "end_date",
        "trap_days",
        "detection_count",
        "detection_rate_per_100",
    ],
    "species_summary": [
        "site_name",
        "classification_label",
        "scientific_name",
        "taxon_class",
        "taxon_order",
        "taxon_family",
        "taxon_genus",
        "taxon_species",
        "total_count",
        "detection_rate_per_100",
    ],
}


def serialize_shapefile_zip(layers: dict[str, list[dict[str, Any]]]) -> bytes:
    """
    ZIP containing one shapefile quartet per layer (`.shp`/`.shx`/`.dbf`/`.prj`).

    Flat at ZIP root. QGIS, ArcGIS, and Windows Explorer all handle this
    layout correctly.
    """
    import shapefile  # pyshp

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for layer_name, features in layers.items():
            shp_buf = io.BytesIO()
            shx_buf = io.BytesIO()
            dbf_buf = io.BytesIO()
            writer = shapefile.Writer(shp=shp_buf, shx=shx_buf, dbf=dbf_buf)
            writer.shapeType = shapefile.POINT

            field_defs = _SHP_LAYER_FIELDS[layer_name]
            for fname, ftype, fsize, fdecimal in field_defs:
                writer.field(fname, ftype, size=fsize, decimal=fdecimal)

            keys = _SHP_PROPERTY_KEYS[layer_name]
            for feat in features:
                writer.point(feat["lon"], feat["lat"])
                record = []
                for key, (_, ftype, _, _) in zip(keys, field_defs, strict=True):
                    val = feat["properties"].get(key, "")
                    if ftype == "N" and (val == "" or val is None):
                        record.append(0)
                    else:
                        record.append(val)
                writer.record(*record)

            writer.close()

            zf.writestr(f"{layer_name}.shp", shp_buf.getvalue())
            zf.writestr(f"{layer_name}.shx", shx_buf.getvalue())
            zf.writestr(f"{layer_name}.dbf", dbf_buf.getvalue())
            zf.writestr(f"{layer_name}.prj", WGS84_PRJ)

    return zip_buf.getvalue()


def make_gpkg_point_blob(lon: float, lat: float) -> bytes:
    """
    GeoPackage binary geometry for a Point (EPSG:4326).

    Layout (29 bytes total):
      - 'GP' magic (2 bytes)
      - version 0 (1 byte)
      - flags 1 = little-endian, no envelope (1 byte)
      - SRID 4326 (4 bytes, int32 LE)
      - WKB Point: byte-order 1, type 1, X=lon, Y=lat (25 bytes)
    """
    header = b"GP" + struct.pack("<BBi", 0, 1, WGS84_SRS_ID)
    wkb = struct.pack("<BI2d", 1, 1, lon, lat)
    return header + wkb


# GeoPackage column definitions per layer: (col_name, sql_type).
_GPKG_LAYER_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "deployments": [
        ("site_name", "TEXT"),
        ("deployment_id", "TEXT"),
        ("start_date", "TEXT"),
        ("end_date", "TEXT"),
        ("trap_days", "INTEGER"),
        ("detection_count", "INTEGER"),
        ("detection_rate_per_100", "REAL"),
    ],
    "species_summary": [
        ("site_name", "TEXT"),
        ("classification_label", "TEXT"),
        ("scientific_name", "TEXT"),
        ("taxon_class", "TEXT"),
        ("taxon_order", "TEXT"),
        ("taxon_family", "TEXT"),
        ("taxon_genus", "TEXT"),
        ("taxon_species", "TEXT"),
        ("total_count", "INTEGER"),
        ("detection_rate_per_100", "REAL"),
    ],
}


def serialize_geopackage(layers: dict[str, list[dict[str, Any]]]) -> bytes:
    """
    Serialize spatial layers as a GeoPackage (.gpkg) file.

    Uses a temporary SQLite DB because ``sqlite3`` cannot serialize an
    in-memory database back to bytes. The temp file is always deleted.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".gpkg")
    os.close(tmp_fd)
    try:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA application_id = 1196444487")  # 'GPKG'
            conn.execute("PRAGMA user_version = 10200")  # GeoPackage 1.2

            conn.execute(
                """
                CREATE TABLE gpkg_spatial_ref_sys (
                    srs_name TEXT NOT NULL,
                    srs_id INTEGER NOT NULL PRIMARY KEY,
                    organization TEXT NOT NULL,
                    organization_coordsys_id INTEGER NOT NULL,
                    definition TEXT NOT NULL,
                    description TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
                ("Undefined Cartesian", -1, "NONE", -1, "undefined", None),
            )
            conn.execute(
                "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
                ("Undefined Geographic", 0, "NONE", 0, "undefined", None),
            )
            conn.execute(
                "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
                ("WGS 84", WGS84_SRS_ID, "EPSG", WGS84_SRS_ID, WGS84_WKT, "WGS 84"),
            )

            conn.execute(
                """
                CREATE TABLE gpkg_contents (
                    table_name TEXT NOT NULL PRIMARY KEY,
                    data_type TEXT NOT NULL DEFAULT 'features',
                    identifier TEXT UNIQUE,
                    description TEXT DEFAULT '',
                    last_change DATETIME NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                    srs_id INTEGER,
                    CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
                        REFERENCES gpkg_spatial_ref_sys(srs_id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE gpkg_geometry_columns (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    geometry_type_name TEXT NOT NULL,
                    srs_id INTEGER NOT NULL,
                    z TINYINT NOT NULL,
                    m TINYINT NOT NULL,
                    CONSTRAINT pk_gc PRIMARY KEY (table_name, column_name),
                    CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)
                        REFERENCES gpkg_contents(table_name),
                    CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id)
                        REFERENCES gpkg_spatial_ref_sys(srs_id)
                )
                """
            )

            for layer_name, features in layers.items():
                columns = _GPKG_LAYER_COLUMNS[layer_name]
                col_defs = ", ".join(f"{name} {typ}" for name, typ in columns)
                conn.execute(
                    f"""
                    CREATE TABLE "{layer_name}" (
                        fid INTEGER PRIMARY KEY AUTOINCREMENT,
                        geom BLOB,
                        {col_defs}
                    )
                    """
                )

                if features:
                    min_x = min(f["lon"] for f in features)
                    max_x = max(f["lon"] for f in features)
                    min_y = min(f["lat"] for f in features)
                    max_y = max(f["lat"] for f in features)
                else:
                    min_x = max_x = min_y = max_y = 0.0

                conn.execute(
                    "INSERT INTO gpkg_contents (table_name, data_type, identifier, "
                    "description, min_x, min_y, max_x, max_y, srs_id) "
                    "VALUES (?, 'features', ?, '', ?, ?, ?, ?, ?)",
                    (layer_name, layer_name, min_x, min_y, max_x, max_y, WGS84_SRS_ID),
                )
                conn.execute(
                    "INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', 'POINT', ?, 0, 0)",
                    (layer_name, WGS84_SRS_ID),
                )

                col_names = [name for name, _ in columns]
                placeholders = ", ".join(["?"] * (1 + len(col_names)))
                insert_sql = (
                    f'INSERT INTO "{layer_name}" (geom, {", ".join(col_names)}) '
                    f"VALUES ({placeholders})"
                )
                for feat in features:
                    geom_blob = make_gpkg_point_blob(feat["lon"], feat["lat"])
                    values = [feat["properties"].get(c, "") for c in col_names]
                    conn.execute(insert_sql, [geom_blob, *values])

            conn.commit()
        finally:
            conn.close()

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CamTrap DP ZIP assembler
# ---------------------------------------------------------------------------


def build_camtrap_dp_zip(
    datapackage_json: bytes,
    deployments_csv: bytes,
    media_csv: bytes,
    observations_csv: bytes,
    thumbnails: dict[str, bytes] | None = None,
) -> bytes:
    """Assemble the four CamTrap DP files into a single ZIP at the root.

    If `thumbnails` is provided, each entry is written under `media/` with
    the dict key as the filename. Callers must rewrite `filePath` in
    `media.csv` to match these relative paths before calling.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("datapackage.json", datapackage_json)
        zf.writestr("deployments.csv", deployments_csv)
        zf.writestr("media.csv", media_csv)
        zf.writestr("observations.csv", observations_csv)
        if thumbnails:
            for name, data in thumbnails.items():
                zf.writestr(f"media/{name}", data)
    return buf.getvalue()


def generate_thumbnail(
    source_path: str,
    max_width: int = 640,
    quality: int = 80,
) -> bytes | None:
    """Produce a JPEG thumbnail as bytes. Returns None if the source
    can't be opened (missing file, unsupported format). Downscales to
    `max_width` preserving aspect ratio; never upscales.
    """
    from PIL import Image, ImageOps

    path = Path(source_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            if im.width > max_width:
                new_h = round(im.height * (max_width / im.width))
                im = im.resize((max_width, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:
        return None
