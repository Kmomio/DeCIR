"""JSON schema for edit intent validation.

This module defines the expected structure of edit intent JSON objects
returned by the Intent Parser (Stage 1).
"""

# JSON Schema for validating edit intent structure
INTENT_SCHEMA = {
    "type": "object",
    "required": [
        "edit_type",
        "operations",
        "needs_bbox",
        "needs_background_mask",
        "needs_global_redraw"
    ],
    "properties": {
        "edit_type": {
            "type": "string",
            "enum": [
                "subject_only",          # Modify only the main subject
                "background_only",       # Modify only the background
                "subject_and_background",  # Modify both
                "composition_change",    # Change spatial arrangement
                "replacement",           # Replace objects
                "global"                 # Global style/attribute changes
            ],
            "description": "High-level category of the edit"
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "target", "object", "attributes"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "add",          # Add new object
                            "remove",       # Remove existing object
                            "replace",      # Replace object with another
                            "modify",       # Modify attributes
                            "duplicate",    # Duplicate existing object
                            "change_count", # Change number of objects
                            "swap"          # Swap two objects
                        ],
                        "description": "Type of modification action"
                    },
                    "target": {
                        "type": "string",
                        "enum": [
                            "subject",         # Main subject
                            "background",      # Background region
                            "specific_object", # Named object
                            "global"           # Entire image
                        ],
                        "description": "Target region for the action"
                    },
                    "object": {
                        "type": "string",
                        "description": "Object name (e.g., 'person', 'car', 'tree')"
                    },
                    "attributes": {
                        "type": "object",
                        "description": "Key-value pairs of attributes (e.g., {'color': 'blue', 'size': 'large'})",
                        "additionalProperties": True
                    }
                }
            },
            "description": "List of atomic edit operations"
        },
        "needs_bbox": {
            "type": "boolean",
            "description": "Whether bounding box localization is needed"
        },
        "needs_background_mask": {
            "type": "boolean",
            "description": "Whether background mask is needed"
        },
        "needs_global_redraw": {
            "type": "boolean",
            "description": "Whether full image redraw is needed"
        }
    }
}


def validate_intent(intent: dict) -> bool:
    """Validate intent JSON against schema.

    Args:
        intent: Intent dictionary to validate.

    Returns:
        True if valid, False otherwise.

    Example:
        >>> intent = {
        ...     "edit_type": "subject_only",
        ...     "operations": [{
        ...         "action": "modify",
        ...         "target": "subject",
        ...         "object": "person",
        ...         "attributes": {"clothing_color": "blue"}
        ...     }],
        ...     "needs_bbox": True,
        ...     "needs_background_mask": False,
        ...     "needs_global_redraw": False
        ... }
        >>> validate_intent(intent)
        True
    """
    try:
        from jsonschema import validate
        validate(instance=intent, schema=INTENT_SCHEMA)
        return True
    except Exception:
        return False
