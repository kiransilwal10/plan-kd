"""Geometry helpers for room adjacency via doors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except Exception:  # noqa: BLE001
    Polygon = None  # type: ignore


@dataclass
class PlanElement:
    id: str
    geom: Any
    attrs: Dict[str, Any]


def _to_polygon(poly_like: Any) -> Optional[Any]:
    if Polygon is None:
        return None
    if isinstance(poly_like, Polygon):
        return poly_like
    coords = poly_like.get("coords") if isinstance(poly_like, dict) else None
    if coords:
        try:
            return Polygon(coords)
        except Exception:  # noqa: BLE001
            return None
    return None


def build_adjacency_graph(room_polys: List[Dict[str, Any]], door_polys: List[Dict[str, Any]]) -> nx.Graph:
    """Construct a room adjacency graph using door contacts."""
    graph = nx.Graph()
    for room in room_polys:
        graph.add_node(room.get("id", f"room-{len(graph)}"))

    if not door_polys:
        return graph

    for door in door_polys:
        door_poly = _to_polygon(door)
        touched_rooms: List[str] = door.get("rooms", [])
        if door_poly and Polygon is not None and not touched_rooms:
            overlaps = []
            for room in room_polys:
                room_poly = _to_polygon(room)
                if room_poly and room_poly.buffer(1e-3).intersects(door_poly):
                    overlaps.append(room.get("id"))
            touched_rooms = overlaps

        if len(touched_rooms) == 2:
            a, b = touched_rooms
            graph.add_edge(a, b, door_id=door.get("id"))
        elif len(touched_rooms) == 1:
            graph.add_edge(touched_rooms[0], touched_rooms[0], door_id=door.get("id"))
    return graph


def door_to_rooms(door: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    rooms = door.get("rooms") or door.get("room_ids") or []
    if len(rooms) >= 2:
        return rooms[0], rooms[1]
    if len(rooms) == 1:
        return rooms[0], None
    return None, None
