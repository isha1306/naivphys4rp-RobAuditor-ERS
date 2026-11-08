import unittest

from recovery_evaluation_experiment import (
    parse_run_output,
    summarize,
)


class RepeatedRecoveryEvaluationTests(unittest.TestCase):
    def test_successful_trial_records_stage_and_repeat(self):
        output = """
[recovery] test strategy: GOB
[recovery] manager status: VALIDATED
[recovery-execution] pre-grasp position error: 0.6 mm
[recovery-execution] canister displacement: 1.7 mm
[recovery-backend] status=SUCCEEDED
[recovery-backend] attempts=1
[recovery-backend] postconditions_verified=True
[recovery-backend] executed actions: return_to_pre_grasp
"""
        row = parse_run_output("grasp", 3, 0, 12.5, output)
        self.assertTrue(row["run_success"])
        self.assertEqual(row["repeat_index"], 3)
        self.assertEqual(row["strategy"], "GOB")
        self.assertEqual(row["waypoint_error_mm"], 0.6)
        self.assertEqual(row["object_displacement_mm"], 1.7)

    def test_false_postconditions_are_failure(self):
        output = """
[recovery] manager status: VALIDATED
[recovery-backend] status=SUCCEEDED
[recovery-backend] attempts=1
[recovery-backend] postconditions_verified=False
"""
        row = parse_run_output("transfer", 1, 0, 10.0, output)
        self.assertFalse(row["run_success"])
        self.assertEqual(
            row["failure_reason"],
            "postconditions were not verified as true",
        )

    def test_nonzero_process_exit_is_failure(self):
        row = parse_run_output("insertion", 1, 2, 4.0, "")
        self.assertFalse(row["run_success"])
        self.assertIn("code 2", row["failure_reason"])

    def test_contact_metrics_are_extracted(self):
        output = """
'contact_count': 224
'canister_environment_contact_count': 222
'canister_gripper_contact_count': 2
"""
        row = parse_run_output("placement", 1, 0, 10.0, output)
        self.assertEqual(row["contact_count"], 224)
        self.assertEqual(row["environment_contact_count"], 222)
        self.assertEqual(row["gripper_contact_count"], 2)

    def test_summary_groups_trials_by_stage(self):
        good = parse_run_output("grasp", 1, 0, 10.0, """
[recovery] manager status: VALIDATED
[recovery-backend] status=SUCCEEDED
[recovery-backend] attempts=1
[recovery-backend] postconditions_verified=True
""")
        bad = parse_run_output("grasp", 2, 0, 12.0, "")
        transfer = parse_run_output("transfer", 1, 0, 8.0, """
[recovery] manager status: VALIDATED
[recovery-backend] status=SUCCEEDED
[recovery-backend] attempts=1
[recovery-backend] postconditions_verified=True
""")
        result = summarize([good, bad, transfer])
        self.assertEqual(result["total_trials"], 3)
        self.assertEqual(result["successful_trials"], 2)
        # The production summary rounds rates to six decimal places.
        self.assertAlmostEqual(
            result["overall_success_rate"], 2 / 3, places=6
        )
        self.assertEqual(result["by_stage"]["grasp"]["trials"], 2)
        self.assertEqual(
            result["by_stage"]["grasp"]["successful_trials"], 1
        )
        self.assertEqual(
            result["by_stage"]["transfer"]["success_rate"], 1.0
        )

    def test_metric_summary_reports_spread(self):
        rows = [
            {
                "stage": "grasp",
                "run_success": True,
                "process_elapsed_s": 1.0,
                "waypoint_error_mm": 1.0,
                "object_displacement_mm": 2.0,
                "attachment_error_mm": 0.0,
            },
            {
                "stage": "grasp",
                "run_success": True,
                "process_elapsed_s": 3.0,
                "waypoint_error_mm": 3.0,
                "object_displacement_mm": 4.0,
                "attachment_error_mm": 0.0,
            },
        ]
        result = summarize(rows)
        metrics = result["by_stage"]["grasp"]["waypoint_error_mm"]
        self.assertEqual(metrics["mean"], 2.0)
        self.assertEqual(metrics["median"], 2.0)
        self.assertEqual(metrics["min"], 1.0)
        self.assertEqual(metrics["max"], 3.0)
        self.assertAlmostEqual(metrics["stddev"], 1.414214, places=5)


if __name__ == "__main__":
    unittest.main()
