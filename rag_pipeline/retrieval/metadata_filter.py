"""Metadata filtering for retrieval results"""
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
import operator


class MetadataFilter:
    """Filter search results based on metadata"""

    # Supported operators
    OPERATORS = {
        'eq': operator.eq,
        'ne': operator.ne,
        'lt': operator.lt,
        'le': operator.le,
        'gt': operator.gt,
        'ge': operator.ge,
        'in': lambda x, y: x in y,
        'not_in': lambda x, y: x not in y,
        'contains': lambda x, y: y in x,
        'startswith': lambda x, y: x.startswith(y),
        'endswith': lambda x, y: x.endswith(y),
    }

    @staticmethod
    def filter_results(
        results: List[Tuple[str, float, Dict[str, Any]]],
        filters: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Filter results based on metadata conditions

        Args:
            results: List of (document, score, metadata) tuples
            filters: Dictionary of field -> condition mappings

        Examples:
            # Simple equality
            filters = {"source": "wikipedia"}

            # Operator-based
            filters = {"date": {"gt": "2023-01-01"}}

            # Multiple conditions
            filters = {
                "category": {"in": ["tech", "science"]},
                "score": {"ge": 0.8}
            }

        Returns:
            Filtered list of results
        """
        filtered = []

        for doc, score, metadata in results:
            if MetadataFilter._matches_filters(metadata, filters):
                filtered.append((doc, score, metadata))

        return filtered

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filter conditions"""
        for field, condition in filters.items():
            if not MetadataFilter._matches_condition(metadata, field, condition):
                return False
        return True

    @staticmethod
    def _matches_condition(
        metadata: Dict[str, Any],
        field: str,
        condition: Any
    ) -> bool:
        """Check if metadata field matches a single condition"""
        # Handle nested fields (e.g., "author.name")
        value = MetadataFilter._get_nested_value(metadata, field)

        if value is None:
            return False

        # Simple equality check
        if not isinstance(condition, dict):
            return value == condition

        # Operator-based checks
        for op, target in condition.items():
            if op not in MetadataFilter.OPERATORS:
                raise ValueError(f"Unsupported operator: {op}")

            op_func = MetadataFilter.OPERATORS[op]

            try:
                if not op_func(value, target):
                    return False
            except (TypeError, AttributeError):
                return False

        return True

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], field: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value

    @staticmethod
    def create_date_filter(
        field: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Helper to create date range filters

        Args:
            field: Metadata field containing date
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filter dictionary
        """
        filter_dict = {}

        if start_date:
            filter_dict["ge"] = start_date
        if end_date:
            filter_dict["le"] = end_date

        return {field: filter_dict}

    @staticmethod
    def combine_filters(*filter_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple filter dictionaries (AND logic)"""
        combined = {}
        for f in filter_dicts:
            combined.update(f)
        return combined


class AdvancedMetadataFilter(MetadataFilter):
    """Advanced filtering with custom predicates and OR logic"""

    @staticmethod
    def filter_with_predicate(
        results: List[Tuple[str, float, Dict[str, Any]]],
        predicate: Callable[[Dict[str, Any]], bool]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Filter results using a custom predicate function

        Args:
            results: List of (document, score, metadata) tuples
            predicate: Function that takes metadata and returns bool

        Example:
            predicate = lambda m: m.get('views', 0) > 1000 and 'python' in m.get('tags', [])
        """
        return [
            (doc, score, metadata)
            for doc, score, metadata in results
            if predicate(metadata)
        ]

    @staticmethod
    def filter_or(
        results: List[Tuple[str, float, Dict[str, Any]]],
        *filter_dicts: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Filter with OR logic - match any of the filter dictionaries

        Args:
            results: List of (document, score, metadata) tuples
            filter_dicts: Multiple filter dictionaries

        Returns:
            Results matching any of the filters
        """
        filtered = []

        for doc, score, metadata in results:
            for filters in filter_dicts:
                if MetadataFilter._matches_filters(metadata, filters):
                    filtered.append((doc, score, metadata))
                    break

        return filtered
