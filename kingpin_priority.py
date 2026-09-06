"""
kingpin_priority.py
Fork F: Kingpin Strike Priority & Network Disruption Advisor

Turns the existing /api/graph data (nodes + edges from storage.py's
get_case_graph_data) into a *prescriptive* recommendation: which
entity, if actioned first (arrest / account freeze / vendor takedown),
disrupts the largest share of the network's transaction pathways.

Algorithm: betweenness centrality (Brandes' algorithm, O(VE) unweighted),
implemented from scratch in pure stdlib -- no networkx/numpy dependency,
consistent with the project's zero-external-dependency ground rules.

Each recommendation also carries an "explainability trail" -- the exact
arithmetic behind the score and, crucially, a plain-language caveat
about what would weaken this node's evidentiary/strategic weight, so an
investigating officer (or a cross-examining defence lawyer) can see the
reasoning, not just a black-box ranking.

Usage from server.py:

    from kingpin_priority import compute_strike_priority
    import storage

    graph = storage.get_case_graph_data(case_id)
    result = compute_strike_priority(graph)
"""

from collections import defaultdict, deque
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Brandes' algorithm - unweighted betweenness centrality
# ---------------------------------------------------------------------------

def _betweenness_centrality(adjacency: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Standard Brandes' algorithm for betweenness centrality on an
    undirected, unweighted graph given as an adjacency list.
    Returns a dict of node_id -> raw betweenness score.
    """
    centrality = {node: 0.0 for node in adjacency}

    for s in adjacency:
        # --- BFS from source s ---
        stack = []
        pred = {v: [] for v in adjacency}
        sigma = {v: 0.0 for v in adjacency}
        sigma[s] = 1.0
        dist = {v: -1 for v in adjacency}
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adjacency[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # --- Accumulation (back-propagation) ---
        delta = {v: 0.0 for v in adjacency}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    # Undirected graph: each shortest path counted twice
    for v in centrality:
        centrality[v] /= 2.0

    return centrality


def _build_adjacency(nodes: List[Dict], edges: List[Dict]) -> Dict[str, List[str]]:
    adjacency = {n["id"]: [] for n in nodes}
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if src in adjacency and dst in adjacency:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
    return adjacency


def _degree_map(adjacency: Dict[str, List[str]]) -> Dict[str, int]:
    return {node: len(neighbors) for node, neighbors in adjacency.items()}


# ---------------------------------------------------------------------------
# Disruption simulation - "what % of shortest paths break if this node is removed"
# ---------------------------------------------------------------------------

def _count_reachable_pairs(adjacency: Dict[str, List[str]]) -> int:
    """Count the number of connected (reachable) unordered node pairs."""
    visited_global = set()
    total_pairs = 0
    for start in adjacency:
        if start in visited_global:
            continue
        # BFS to find this component
        component = set()
        queue = deque([start])
        component.add(start)
        while queue:
            v = queue.popleft()
            for w in adjacency[v]:
                if w not in component:
                    component.add(w)
                    queue.append(w)
        visited_global |= component
        n = len(component)
        total_pairs += n * (n - 1) // 2
    return total_pairs


def _simulate_removal_disruption(adjacency: Dict[str, List[str]], node_to_remove: str) -> float:
    """
    Returns the fraction of connected node-pairs that become disconnected
    (or newly require a longer path) if `node_to_remove` is deleted from
    the graph. Used as the headline "% of network disrupted" figure.
    """
    baseline_pairs = _count_reachable_pairs(adjacency)
    if baseline_pairs == 0:
        return 0.0

    pruned = {
        node: [nb for nb in neighbors if nb != node_to_remove]
        for node, neighbors in adjacency.items()
        if node != node_to_remove
    }
    remaining_pairs = _count_reachable_pairs(pruned)

    # Fraction of pairs that were connected before but are no longer
    # (or belong to a node that no longer exists at all).
    disrupted_fraction = 1.0 - (remaining_pairs / baseline_pairs) if baseline_pairs else 0.0
    return max(0.0, min(1.0, disrupted_fraction))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_strike_priority(graph: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """
    graph: the exact dict shape returned by storage.get_case_graph_data(),
           i.e. {"nodes": [...], "edges": [...]}

    Returns a ranked list of strike recommendations with an explainability
    trail per entry, ready to render directly in Panel 3.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return {"status": "empty", "recommendations": [], "message": "No graph data available for this case yet."}

    adjacency = _build_adjacency(nodes, edges)
    betweenness = _betweenness_centrality(adjacency)
    degree = _degree_map(adjacency)
    node_lookup = {n["id"]: n for n in nodes}

    max_betweenness = max(betweenness.values()) if betweenness.values() else 0.0

    scored = []
    for node_id, score in betweenness.items():
        node = node_lookup.get(node_id, {})
        normalized_score = round((score / max_betweenness) * 100, 1) if max_betweenness > 0 else 0.0
        disruption_pct = round(_simulate_removal_disruption(adjacency, node_id) * 100, 1)

        # Caveats: what would weaken confidence in this recommendation
        caveats = []
        if degree.get(node_id, 0) <= 1:
            caveats.append("Low degree (1 connection) - centrality driven by sparse graph data, not confirmed hub role.")
        if node.get("mentions", 0) < 3:
            caveats.append(f"Only {node.get('mentions', 0)} corroborating mention(s) in evidence - verify before acting.")
        if node.get("risk_score", 0) < 60:
            caveats.append("Underlying entity risk score is moderate; corroborate with an independent evidence source before prioritizing.")
        if not caveats:
            caveats.append("No major caveats detected - multiple independent mentions and strong network position.")

        scored.append(
            {
                "entity_id": node_id,
                "label": node.get("label", node_id),
                "entity_type": node.get("type", "UNKNOWN"),
                "betweenness_raw": round(score, 3),
                "betweenness_normalized_pct": normalized_score,
                "network_degree": degree.get(node_id, 0),
                "estimated_disruption_pct": disruption_pct,
                "mention_count": node.get("mentions", 0),
                "risk_score": node.get("risk_score", 0),
                "caveats": caveats,
                "explanation": (
                    f"'{node.get('label', node_id)}' sits on {round(score, 1)} weighted shortest-path "
                    f"intersections between other actors in this case (normalized: {normalized_score}% of the "
                    f"most central node). Simulated removal disconnects an estimated {disruption_pct}% of "
                    f"currently-linked actor pairs, based on {node.get('mentions', 0)} evidence mentions."
                ),
            }
        )

    scored.sort(key=lambda x: (x["betweenness_normalized_pct"], x["estimated_disruption_pct"]), reverse=True)
    top_recommendations = scored[:top_n]

    for i, rec in enumerate(top_recommendations, start=1):
        rec["priority_rank"] = i

    return {
        "status": "success",
        "case_node_count": len(nodes),
        "case_edge_count": len(edges),
        "recommendations": top_recommendations,
        "methodology": (
            "Betweenness centrality (Brandes' algorithm, unweighted) computed over the case's "
            "entity co-occurrence graph. Ranks entities by how often they sit on the shortest "
            "connecting path between other actors - i.e. how much of the network's communication "
            "or financial flow would need to reroute if this entity were removed."
        ),
    }


# ---------------------------------------------------------------------------
# server.py integration - paste inside do_GET() dispatch
# ---------------------------------------------------------------------------
"""
elif path == '/api/strike_priority':
    case_id = params.get('case_id', ['FIR_104_2026'])[0]
    graph = storage.get_case_graph_data(case_id)
    result = compute_strike_priority(graph)
    self._set_json_headers(200)
    self.wfile.write(json.dumps(result).encode('utf-8'))
    return
"""


if __name__ == "__main__":
    # Smoke test with a small synthetic network: a hub-and-spoke vendor
    # plus one bridge node connecting two otherwise-separate clusters.
    sample_graph = {
        "nodes": [
            {"id": "V1", "label": "@Shadow_Sector", "type": "DARKNET_VENDOR", "risk_score": 90, "mentions": 12},
            {"id": "U1", "label": "mule44@ybl", "type": "UPI_ID", "risk_score": 90, "mentions": 8},
            {"id": "U2", "label": "raj@upi", "type": "UPI_ID", "risk_score": 85, "mentions": 3},
            {"id": "P1", "label": "9812345670", "type": "PHONE", "risk_score": 75, "mentions": 5},
            {"id": "L1", "label": "Sector 17", "type": "LOCATION", "risk_score": 50, "mentions": 2},
            {"id": "P2", "label": "9998887776", "type": "PHONE", "risk_score": 60, "mentions": 1},
        ],
        "edges": [
            {"from": "V1", "to": "U1", "label": "8 mentions"},
            {"from": "V1", "to": "U2", "label": "3 mentions"},
            {"from": "U1", "to": "P1", "label": "5 mentions"},
            {"from": "P1", "to": "L1", "label": "2 mentions"},
            {"from": "L1", "to": "P2", "label": "1 mention"},
        ],
    }

    result = compute_strike_priority(sample_graph)
    print(f"Status: {result['status']}")
    for rec in result["recommendations"]:
        print(f"\n#{rec['priority_rank']} {rec['label']} ({rec['entity_type']})")
        print(f"  Disruption if removed: {rec['estimated_disruption_pct']}%")
        print(f"  Explanation: {rec['explanation']}")
        print(f"  Caveats: {rec['caveats']}")
