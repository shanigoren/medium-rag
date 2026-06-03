"""In-memory Pinecone test doubles.

Shared by `tests/conftest.py` (the `fake_pc` fixture) and `tests/test_vectorstore.py`.
Lives outside both to avoid a cyclic import: conftest imports it, and the test
module imports it for assertions about recorded calls.
"""

from __future__ import annotations

from typing import Any


class _FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.queries: list[dict] = []
        self.deletes: list[dict] = []
        self.stats_response: dict = {"namespaces": {}, "dimension": 1536}
        self.query_response: dict = {"matches": []}

    def upsert(self, vectors: list[dict], namespace: str) -> dict:
        self.upserts.append({"vectors": list(vectors), "namespace": namespace})
        return {"upserted_count": len(vectors)}

    def query(
        self,
        vector: list[float],
        top_k: int,
        namespace: str,
        include_metadata: bool,
        include_values: bool,
    ) -> dict:
        self.queries.append(
            {
                "vector": vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": include_metadata,
                "include_values": include_values,
            }
        )
        return self.query_response

    def delete(self, delete_all: bool, namespace: str) -> dict:
        self.deletes.append({"delete_all": delete_all, "namespace": namespace})
        return {}

    def describe_index_stats(self) -> dict:
        return self.stats_response


class _FakeIndexInfo:
    """Mutable stand-in for pc.describe_index(name) return value."""

    def __init__(
        self, dimension: int = 1536, metric: str = "cosine", ready: bool = True
    ) -> None:
        self.dimension = dimension
        self.metric = metric
        self.status = type("S", (), {"ready": ready})()


class _FakePinecone:
    def __init__(self, api_key: str, **_: Any) -> None:
        self.api_key = api_key
        self.created: list[dict] = []
        self.index_infos: dict[str, _FakeIndexInfo] = {}
        self._index = _FakeIndex()
        self.quota_reached: bool = False
        # If non-None, describe_index() runs this callback before returning the
        # stored info — lets tests flip `status.ready` between polls.
        self.describe_hook = None

    def list_indexes(self):
        class _R:
            def __init__(self, names: list[str]) -> None:
                self._names = names

            def names(self) -> list[str]:
                return self._names

        return _R(list(self.index_infos.keys()))

    def describe_index(self, name: str) -> _FakeIndexInfo:
        if self.describe_hook is not None:
            self.describe_hook(name)
        if name not in self.index_infos:
            raise KeyError(f"index {name!r} not found")
        return self.index_infos[name]

    def create_index(self, name: str, dimension: int, metric: str, spec) -> None:
        if self.quota_reached:
            raise RuntimeError(
                "PROJECT_QUOTA_REACHED: max indexes reached on free tier"
            )
        self.created.append(
            {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "cloud": getattr(spec, "cloud", None),
                "region": getattr(spec, "region", None),
            }
        )
        self.index_infos[name] = _FakeIndexInfo(
            dimension=dimension, metric=metric, ready=True
        )

    def Index(self, name: str) -> _FakeIndex:
        return self._index
