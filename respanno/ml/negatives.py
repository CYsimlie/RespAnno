"""Hard-negative sample manager for ML training feedback loop.

Each time the user deletes or corrects an annotation for a non-phase label,
the deleted region is recorded as a hard negative sample.  During the next
training run these regions are forced to the negative class, which helps the
classifier avoid repeating the same false-positive mistakes.

The manager is a pure-data class — no PyQt / GUI dependency.
"""

from collections import defaultdict


class NegSampleManager:
    """Manages per-label hard-negative segments collected from user deletions.

    Typical lifecycle::

        mgr = NegSampleManager()
        mgr.add("Wheeze", 1.2, 1.8)       # user deleted a Wheeze span
        mgr.count("Wheeze")                 # → 1
        mgr.clear("Wheeze")                 # reset after re-training
        mgr.to_dict()                       # pass to frame label builder
    """

    def __init__(self):
        self._segments = defaultdict(list)
        self._id_counter = 0

    # ---- add / remove --------------------------------------------------------

    def add(self, label: str, start: float, end: float):
        """Record one hard-negative segment and return its (s, e, neg_id).

        Returns ``None`` when *label* is empty.
        """
        try:
            label = str(label)
        except Exception:
            label = ""
        if not label:
            return None
        self._id_counter += 1
        item = (float(start), float(end), int(self._id_counter))
        self._segments[label].append(item)
        return item

    def remove(self, label: str, neg_id: int):
        """Remove a single hard-negative segment by its *neg_id*."""
        lst = self._segments.get(str(label), [])
        for i, it in enumerate(list(lst)):
            try:
                if int(it[2]) == int(neg_id):
                    lst.pop(i)
                    break
            except Exception:
                continue

    # ---- query ---------------------------------------------------------------

    def count(self, label: str) -> int:
        """Return the number of hard-negative segments stored for *label*."""
        return len(self._segments.get(str(label), []))

    def get(self, label: str):
        """Return the list of (s, e, neg_id) for *label* (empty list if none)."""
        return list(self._segments.get(str(label), []))

    # ---- bulk clear ----------------------------------------------------------

    def clear(self, label: str):
        """Remove all hard-negative segments for a single label."""
        self._segments.pop(str(label), None)

    def clear_all(self):
        """Remove hard-negative segments for **every** label."""
        self._segments.clear()
        self._id_counter = 0

    # ---- serialisation -------------------------------------------------------

    def to_dict(self):
        """Return the internal ``{label: [(s, e, neg_id), ...]}`` dict.

        The returned dict is the live data — mutations affect the manager.
        For a snapshot use ``copy.deepcopy()``.
        """
        return self._segments
