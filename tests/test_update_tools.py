"""Basic format and transformation checks for the supplementary scripts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from r1_multihot_sensitivity import (  # noqa: E402
    encode_as_multihot,
    encode_by_position,
    fit_encoders,
    normalize_input,
)
from r3_utilization_comparison import lab_test_recorded  # noqa: E402


class EncodingTests(unittest.TestCase):
    def test_multihot_collapses_repeated_family(self):
        sample = pd.DataFrame({
            "age": ["[50-60)", "[60-70)"],
            "diag1_family": ["circulatory", "respiratory"],
            "diag2_family": ["circulatory", "circulatory"],
            "diag3_family": ["missing", "missing"],
        })
        frame = normalize_input(sample)
        age_encoder, positional_encoder, families = fit_encoders(frame)
        positional = encode_by_position(frame, age_encoder, positional_encoder)
        multihot = encode_as_multihot(frame, age_encoder, families)
        family_block = multihot.toarray()[:, len(age_encoder.categories_[0]):]
        self.assertEqual(list(family_block[:, families.index("circulatory")]), [1.0, 1.0])
        self.assertEqual(int(family_block.max()), 1)
        self.assertGreater(positional.shape[1], multihot.shape[1])

    def test_lab_observation_categories(self):
        readings = pd.Series(["None", ">8", "Norm", "?", None, "  "])
        flags = lab_test_recorded(readings)
        self.assertEqual(flags.tolist(), [0, 1, 1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
