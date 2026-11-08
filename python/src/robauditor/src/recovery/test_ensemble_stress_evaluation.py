import unittest

from ensemble_stress_evaluation import summarize


class EnsembleStressEvaluationTests(unittest.TestCase):
    def test_summary_groups_cases_and_counts_invariant_failures(self):
        rows = [
            {"case": "unsafe_state", "status": "VALIDATED", "invariant_pass": True},
            {"case": "unsafe_state", "status": "VALIDATED", "invariant_pass": True},
            {"case": "incomplete_ensemble", "status": "HUMAN_INTERVENTION", "invariant_pass": True},
            {"case": "randomized_consensus", "status": "VALIDATED", "invariant_pass": False},
        ]
        result = summarize(rows)
        self.assertEqual(result["total_cases"], 4)
        self.assertEqual(result["invariant_passes"], 3)
        self.assertEqual(result["invariant_failures"], 1)
        self.assertEqual(result["by_case"]["unsafe_state"]["trials"], 2)
        self.assertEqual(result["by_case"]["unsafe_state"]["invariant_pass_rate"], 1.0)
        self.assertEqual(result["by_case"]["randomized_consensus"]["invariant_failures"], 1)

    def test_empty_summary_is_well_defined(self):
        result = summarize([])
        self.assertEqual(result["total_cases"], 0)
        self.assertEqual(result["invariant_pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
