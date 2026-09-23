from collections import defaultdict
import re


class KnowledgeGraph:
    """Small in-memory KG adapter used by the method loops.

    Real experiments should replace/add the project's fixed biomedical KG.
    Edges are stored as (subject, relation, object).
    """

    def __init__(self, edges=None):
        self.adj = defaultdict(list)
        self.edges = []
        for edge in edges or []:
            self.add_edge(*edge)

    def add_edge(self, subject, relation, obj):
        edge = (subject, relation, obj)
        self.edges.append(edge)
        self.adj[subject].append((relation, obj))

    def neighbors(self, node):
        return self.adj.get(node, [])

    def traverse(self, start_nodes, depth):
        frontier = list(start_nodes)
        visited = set(frontier)
        paths = []
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for relation, obj in self.neighbors(node):
                    paths.append((node, relation, obj))
                    if obj not in visited:
                        visited.add(obj)
                        next_frontier.append(obj)
            frontier = next_frontier
            if not frontier:
                break
        return paths


def normalize_entity(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def seed_entities(question, known_entities):
    q = normalize_entity(question)
    return [e for e in known_entities if normalize_entity(e) in q]
