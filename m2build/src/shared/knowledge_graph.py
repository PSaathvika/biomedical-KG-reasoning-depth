from collections import defaultdict
import csv
import re


class KnowledgeGraph:
    """
    Real PrimeKG-backed biomedical knowledge graph.

    The graph is loaded from PrimeKG and supports one-hop traversal.
    """

    def __init__(self, edges=None, csv_path=None, max_edges=None):
        self.adj = defaultdict(list)
        self.edges = []
        self.nodes = set()

        if csv_path is not None:
            self.load_primekg(
                csv_path,
                max_edges=max_edges,
            )

        for edge in edges or []:
            self.add_edge(*edge)

    def add_edge(self, subject, relation, obj):
        subject = str(subject).strip()
        relation = str(relation).strip()
        obj = str(obj).strip()

        if not subject or not relation or not obj:
            return

        self.edges.append(
            (subject, relation, obj)
        )

        self.adj[subject].append(
            (relation, obj)
        )

        self.nodes.add(subject)
        self.nodes.add(obj)

    def load_primekg(self, csv_path, max_edges=None):
        """
        Load real biomedical relationships from PrimeKG.

        PrimeKG columns include:
            relation
            x_id
            x_name
            x_type
            y_id
            y_name
            y_type
        """

        count = 0

        print()
        print("=" * 70)
        print("Loading REAL PrimeKG")
        print("=" * 70)
        print(f"PrimeKG path: {csv_path}")

        with open(
            csv_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            required = {
                "relation",
                "x_id",
                "x_name",
                "x_type",
                "y_id",
                "y_name",
                "y_type",
            }

            missing = (
                required
                - set(reader.fieldnames or [])
            )

            if missing:
                raise ValueError(
                    "PrimeKG is missing required columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:
                x_name = (
                    row.get("x_name") or ""
                ).strip()

                y_name = (
                    row.get("y_name") or ""
                ).strip()

                relation = (
                    row.get("relation") or ""
                ).strip()

                if (
                    not x_name
                    or not y_name
                    or not relation
                ):
                    continue

                self.add_edge(
                    x_name,
                    relation,
                    y_name,
                )

                count += 1

                if (
                    max_edges is not None
                    and count >= max_edges
                ):
                    break

        print(
            f"PrimeKG edges loaded: "
            f"{len(self.edges):,}"
        )

        print(
            f"PrimeKG nodes loaded: "
            f"{len(self.nodes):,}"
        )

        print("=" * 70)

    def neighbors(self, node):
        return self.adj.get(node, [])

    def traverse(
        self,
        start_nodes,
        depth=1,
    ):
        """
        Traverse the graph up to `depth` hops.
        """

        frontier = list(start_nodes)
        visited = set(frontier)
        paths = []

        for _ in range(depth):

            next_frontier = []

            for node in frontier:

                for relation, obj in self.neighbors(node):

                    paths.append(
                        (
                            node,
                            relation,
                            obj,
                        )
                    )

                    if obj not in visited:

                        visited.add(obj)
                        next_frontier.append(obj)

            frontier = next_frontier

            if not frontier:
                break

        return paths


def normalize_entity(text):
    """
    Normalize biomedical entity text for matching.
    """

    text = str(text or "")

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def seed_entities(
    question,
    known_entities,
):
    """
    Find PrimeKG entities explicitly mentioned
    in the question.
    """

    normalized_question = normalize_entity(
        question
    )

    matches = []

    for entity in known_entities:

        normalized_entity = normalize_entity(
            entity
        )

        if not normalized_entity:
            continue

        if normalized_entity in normalized_question:
            matches.append(entity)

    return matches