from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


# =========================================================
# Cache Item
# =========================================================

@dataclass
class RewriteCacheItem:

    key: str

    query: str

    history_text: str

    rewritten_query: str

    query_type: str

    entities: List[str]

    created_at: float

    hit_count: int = 0


# =========================================================
# Rewrite Cache
# =========================================================

class RewriteCache:
    """
    Lightweight rewrite cache.

    Current Version:
        - Exact cache
        - Semantic-safe validation
        - Entity overlap validation
        - Query type validation

    Future Upgrades:
        - Vector cache DB
        - Learned cache policy
        - Adaptive cache ranking
    """

    def __init__(
        self,
        max_size: int = 1000,
        min_entity_overlap: float = 0.5,
    ) -> None:

        self.max_size = max_size

        self.min_entity_overlap = (
            min_entity_overlap
        )

        self.cache: Dict[
            str,
            RewriteCacheItem
        ] = {}

    # =====================================================
    # Public API
    # =====================================================

    def get(
        self,
        query: str,
        history_text: str,
    ) -> Optional[str]:
        """
        Retrieve cached rewrite.
        """

        key = self.build_cache_key(
            query=query,
            history_text=history_text,
        )

        item = self.cache.get(key)

        if item is None:
            return None

        # ================================================
        # Validation
        # ================================================

        if not self.validate_cache_hit(
            query=query,
            history_text=history_text,
            item=item,
        ):
            return None

        item.hit_count += 1

        return item.rewritten_query

    def set(
        self,
        query: str,
        history_text: str,
        rewritten_query: str,
    ) -> None:
        """
        Store rewrite result.
        """

        key = self.build_cache_key(
            query=query,
            history_text=history_text,
        )

        item = RewriteCacheItem(

            key=key,

            query=query,

            history_text=history_text,

            rewritten_query=rewritten_query,

            query_type=self.detect_query_type(
                query
            ),

            entities=self.extract_entities(
                query + " " + history_text
            ),

            created_at=time.time(),
        )

        self.cache[key] = item

        self.evict_if_needed()

    # =====================================================
    # Cache Validation
    # =====================================================

    def validate_cache_hit(
        self,
        query: str,
        history_text: str,
        item: RewriteCacheItem,
    ) -> bool:
        """
        Validate whether cache item
        is still safe to reuse.
        """

        # ================================================
        # Query type consistency
        # ================================================

        current_query_type = (
            self.detect_query_type(
                query
            )
        )

        if (
            current_query_type
            != item.query_type
        ):
            return False

        # ================================================
        # Entity overlap validation
        # ================================================

        current_entities = (
            self.extract_entities(
                query + " " + history_text
            )
        )

        overlap = self.compute_entity_overlap(
            current_entities,
            item.entities,
        )

        if overlap < self.min_entity_overlap:
            return False

        return True

    # =====================================================
    # Cache Key
    # =====================================================

    def build_cache_key(
        self,
        query: str,
        history_text: str,
    ) -> str:
        """
        Build stable cache key.
        """

        normalized_query = (
            self.normalize_text(query)
        )

        normalized_history = (
            self.normalize_text(
                history_text
            )
        )

        combined = (
            normalized_query
            + " || "
            + normalized_history
        )

        return hashlib.md5(
            combined.encode("utf-8")
        ).hexdigest()

    # =====================================================
    # Query Type Detection
    # =====================================================

    def detect_query_type(
        self,
        query: str,
    ) -> str:
        """
        Lightweight query type detection.
        """

        query = (
            self.normalize_text(query)
        )

        if any(
            keyword in query

            for keyword in [
                "bao nhiêu",
                "mức phạt",
                "giá",
                "chi phí",
            ]
        ):
            return "quantity"

        if any(
            keyword in query

            for keyword in [
                "khi nào",
                "bao giờ",
                "thời gian",
            ]
        ):
            return "temporal"

        if any(
            keyword in query

            for keyword in [
                "ở đâu",
                "nơi nào",
                "địa điểm",
            ]
        ):
            return "location"

        if any(
            keyword in query

            for keyword in [
                "như thế nào",
                "cách",
                "hướng dẫn",
            ]
        ):
            return "procedural"

        if any(
            keyword in query

            for keyword in [
                "khác gì",
                "so sánh",
                "giống nhau",
            ]
        ):
            return "comparison"

        return "factual"

    # =====================================================
    # Entity Extraction
    # =====================================================

    def extract_entities(
        self,
        text: str,
    ) -> List[str]:
        """
        Lightweight entity extraction.

        Current heuristic:
            - Capitalized words
            - Numbers
            - Technical abbreviations
        """

        entities = set()

        # ================================================
        # Capitalized entities
        # ================================================

        for token in text.split():

            token = token.strip()

            if (
                len(token) > 1
                and token[0].isupper()
            ):
                entities.add(
                    token.lower()
                )

        # ================================================
        # Numbers
        # ================================================

        numbers = re.findall(
            r"\d+",
            text,
        )

        for number in numbers:
            entities.add(number)

        # ================================================
        # Technical abbreviations
        # ================================================

        abbreviations = re.findall(
            r"\b[A-Z]{2,10}\b",
            text,
        )

        for abbr in abbreviations:
            entities.add(
                abbr.lower()
            )

        return list(entities)

    # =====================================================
    # Entity Overlap
    # =====================================================

    def compute_entity_overlap(
        self,
        entities_a: List[str],
        entities_b: List[str],
    ) -> float:
        """
        Compute entity overlap ratio.
        """

        set_a = set(entities_a)

        set_b = set(entities_b)

        if not set_a or not set_b:
            return 0.0

        overlap = len(
            set_a & set_b
        )

        return overlap / max(
            len(set_a),
            len(set_b),
        )

    # =====================================================
    # Cache Eviction
    # =====================================================

    def evict_if_needed(
        self,
    ) -> None:
        """
        Evict oldest items if cache is full.
        """

        if (
            len(self.cache)
            <= self.max_size
        ):
            return

        oldest_key = min(

            self.cache,

            key=lambda k: (
                self.cache[k].created_at
            ),
        )

        del self.cache[oldest_key]

    # =====================================================
    # Utilities
    # =====================================================

    def normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize text.
        """

        return " ".join(
            str(text)
            .strip()
            .lower()
            .split()
        )

    # =====================================================
    # Statistics
    # =====================================================

    def stats(
        self,
    ) -> Dict[str, int]:
        """
        Cache statistics.
        """

        return {
            "cache_size": len(
                self.cache
            ),
        }