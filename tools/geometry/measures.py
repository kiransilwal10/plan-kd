"""Geometry measures built on adjacency graphs."""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import networkx as nx

from .graph import build_adjacency_graph, _to_polygon


def count_doors(room_id: str, door_polys: List[Dict[str, Any]]) -> int:
    count = 0
    for door in door_polys:
        rooms = door.get("rooms") or door.get("room_ids") or []
        if room_id in rooms:
            count += 1
    return count


def shortest_path(graph: nx.Graph, room_i: str, room_j: str) -> List[str]:
    if room_i not in graph or room_j not in graph:
        return []
    try:
        return nx.shortest_path(graph, source=room_i, target=room_j)
    except nx.NetworkXNoPath:
        return []


def min_corridor_width(path: List[str], room_polys: List[Dict[str, Any]]) -> Optional[float]:
    """Approximate minimum width along the path using bounding boxes."""
    if not path:
        return None
    widths: List[float] = []
    room_lookup = {room.get("id"): room for room in room_polys}
    for room_id in path:
        room = room_lookup.get(room_id)
        poly = _to_polygon(room) if room else None
        if poly is not None:
            minx, miny, maxx, maxy = poly.bounds
            widths.append(min(maxx - minx, maxy - miny))
    return min(widths) if widths else None


def build_tools(room_polys: List[Dict[str, Any]], door_polys: List[Dict[str, Any]]) -> Dict[str, Any]:
    graph = build_adjacency_graph(room_polys, door_polys)
    return {
        "graph": graph,
        "count_doors": lambda room_id: count_doors(room_id, door_polys),
        "shortest_path": lambda a, b: shortest_path(graph, a, b),
        "min_corridor_width": lambda path: min_corridor_width(path, room_polys),
    }
