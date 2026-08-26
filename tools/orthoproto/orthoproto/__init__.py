"""T14+T15 offline prototype: true-ortho from a FAST-LIVO2 capture.

Pipeline: RTK-anchored frame alignment -> lidar DSM -> backward-projected
true-ortho patches in the geopack CRS. Offline counterpart of the
`geoloc_ortho` runtime node; see README.md.
"""

__version__ = "0.1.0"
