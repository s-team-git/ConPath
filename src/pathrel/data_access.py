"""Fail-closed packet eligibility for newly defined development experiments."""


def require_development_packet(row):
    """A reassigned provenance label cannot unlock physical test packet images."""
    if row.get('archive_split') not in {'train', 'validation'}:
        raise ValueError(f"Physical test or unresolved archive split is not eligible: {row.get('global_id')}")
    if row.get('candidate_split') not in {'train', 'calibration', 'validation'}:
        raise ValueError('Explicit candidate development partition required')
    if not row.get('parent_group'):
        raise ValueError('Resolved upstream parent group required before image access')
