from __future__ import annotations


def missing_reference_summary(errors: list[str]) -> str:
    if not errors:
        return ""
    return f"Missing references ({len(errors)}): {errors[0]}"
