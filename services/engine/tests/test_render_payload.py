from __future__ import annotations

from geologic_wizard_engine.modules.render_payload import wrap_polygon_geometry


def test_antimeridian_polygon_wrap_splits_to_multipolygon_without_stitching():
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [170.0, 8.0],
                [179.5, 18.0],
                [-179.4, 17.5],
                [-168.0, 7.0],
                [-172.0, -5.0],
                [176.0, -4.0],
                [170.0, 8.0],
            ]
        ],
    }

    wrapped = wrap_polygon_geometry(geometry)

    assert wrapped["type"] == "MultiPolygon"
    assert len(wrapped["coordinates"]) >= 2

    for polygon in wrapped["coordinates"]:
        ring = polygon[0]
        assert len(ring) >= 4
        for index in range(len(ring) - 1):
            lon_a = ring[index][0]
            lon_b = ring[index + 1][0]
            assert -180.0 <= lon_a <= 180.0
            assert -180.0 <= lon_b <= 180.0
            assert abs(lon_b - lon_a) <= 180.0
