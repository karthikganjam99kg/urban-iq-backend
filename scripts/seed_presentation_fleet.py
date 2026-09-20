#!/usr/bin/env python3
"""Compatibility entry point for the unified presentation seeder."""

from seed_presentation_data import main


if __name__ == "__main__":
    print(
        "seed_presentation_fleet.py is deprecated; "
        "running seed_presentation_data.py instead."
    )
    main()
