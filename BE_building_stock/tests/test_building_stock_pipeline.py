from __future__ import annotations

import math
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"


class BuildingStockPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = pd.read_csv(
            DATA / "matrices" / "national" / "base_physical_archetype_matrix.csv"
        )
        cls.states_2025 = pd.read_csv(
            DATA / "scenarios" / "renovation" / "renovation_state_layer.csv"
        )
        cls.priority = pd.read_csv(
            DATA
            / "scenarios"
            / "renovation"
            / "renovation_state_layer_with_allocation.csv"
        )
        cls.scenarios = pd.read_csv(
            DATA
            / "scenarios"
            / "renovation"
            / "archetype_matrix_2050_renovation_scenarios.csv"
        )
        cls.crosscheck = pd.read_csv(
            DATA
            / "scenarios"
            / "renovation"
            / "renovation_scenario_policy_context_2050.csv"
        )
        cls.allocation = pd.read_csv(
            DATA
            / "scenarios"
            / "renovation"
            / "renovation_priority_allocation_2050.csv"
        )

    def test_infiltration_conversion(self) -> None:
        self.assertEqual(len(self.base), 25)
        for row in self.base.itertuples(index=False):
            q50 = row.v50_m3_h_m2 * row.total_building_envelope_area_m2
            n50 = q50 / row.protected_volume_m3
            self.assertTrue(math.isclose(row.q50_m3_h, q50, abs_tol=1e-9))
            self.assertTrue(math.isclose(row.n50_h_1, n50, abs_tol=1e-9))
            self.assertTrue(
                math.isclose(
                    row.infiltration_airflow_normal_m3_h,
                    q50 / 20,
                    abs_tol=1e-9,
                )
            )
            self.assertTrue(
                math.isclose(
                    row.infiltration_ach_normal_h_1,
                    n50 / 20,
                    abs_tol=1e-9,
                )
            )
            self.assertGreater(row.specific_heat_loss_z_W_m2K, 0)

    def test_calibrated_state_null_assumption(self) -> None:
        self.assertEqual(len(self.states_2025), 225)
        expected = {
            "Flemish Region": {
                "TABULA_existing": 0.69,
                "TABULA_standard_B_proxy": 0.22,
                "TABULA_advanced_A_proxy": 0.09,
            },
            "Walloon Region": {
                "TABULA_existing": 0.878,
                "TABULA_standard_B_proxy": 0.11,
                "TABULA_advanced_A_proxy": 0.012,
            },
            "Brussels-Capital Region": {
                "TABULA_existing": 0.9283,
                "TABULA_standard_B_proxy": 0.0562,
                "TABULA_advanced_A_proxy": 0.0155,
            },
        }
        for region, state_values in expected.items():
            region_rows = self.states_2025[self.states_2025["region"] == region]
            for state_id, expected_share in state_values.items():
                values = region_rows.loc[
                    region_rows["state_id"] == state_id, "regional_state_share"
                ].unique()
                self.assertEqual(len(values), 1)
                self.assertTrue(
                    math.isclose(values[0], expected_share, abs_tol=1e-12)
                )
                bound_rows = region_rows[
                    region_rows["state_id"] == state_id
                ]
                self.assertTrue(
                    (
                        bound_rows["regional_state_share_lower"]
                        <= bound_rows["regional_state_share"]
                    ).all()
                )
                self.assertTrue(
                    (
                        bound_rows["regional_state_share"]
                        <= bound_rows["regional_state_share_upper"]
                    ).all()
                )
            for _, group in region_rows.groupby("archetype_id"):
                self.assertTrue(
                    math.isclose(
                        group["regional_state_dwellings_2025"].sum(),
                        group["regional_number_of_dwellings"].iloc[0],
                        abs_tol=1e-5,
                    )
                )

    def test_source_precision_intervals_and_residual(self) -> None:
        expected_bounds = {
            "Flemish Region": {
                "TABULA_existing": (0.68, 0.70),
                "TABULA_standard_B_proxy": (0.215, 0.225),
                "TABULA_advanced_A_proxy": (0.085, 0.095),
            },
            "Walloon Region": {
                "TABULA_existing": (0.877, 0.879),
                "TABULA_standard_B_proxy": (0.1095, 0.1105),
                "TABULA_advanced_A_proxy": (0.0115, 0.0125),
            },
            "Brussels-Capital Region": {
                "TABULA_existing": (0.9282, 0.9284),
                "TABULA_standard_B_proxy": (0.05615, 0.05625),
                "TABULA_advanced_A_proxy": (0.01545, 0.01555),
            },
        }
        for region, state_bounds in expected_bounds.items():
            region_rows = self.states_2025[
                self.states_2025["region"] == region
            ]
            observed = {}
            for state_id, bounds in state_bounds.items():
                row = region_rows[region_rows["state_id"] == state_id].iloc[0]
                observed[state_id] = (
                    row.regional_state_share_lower,
                    row.regional_state_share_upper,
                )
                self.assertTrue(
                    math.isclose(observed[state_id][0], bounds[0], abs_tol=1e-12)
                )
                self.assertTrue(
                    math.isclose(observed[state_id][1], bounds[1], abs_tol=1e-12)
                )
            existing = observed["TABULA_existing"]
            standard = observed["TABULA_standard_B_proxy"]
            advanced = observed["TABULA_advanced_A_proxy"]
            self.assertTrue(
                math.isclose(existing[0], 1 - standard[1] - advanced[1], abs_tol=1e-12)
            )
            self.assertTrue(
                math.isclose(existing[1], 1 - standard[0] - advanced[0], abs_tol=1e-12)
            )

    def test_heat_loss_priority(self) -> None:
        self.assertEqual(len(self.priority), 150)
        for region, group in self.priority.groupby("region", sort=False):
            ordered = group.sort_values(
                "priority_rank_for_advanced_within_region"
            )
            self.assertEqual(
                ordered["priority_rank_for_advanced_within_region"].tolist(),
                list(range(1, 51)),
            )
            z_values = ordered["specific_heat_loss_z_W_m2K"].tolist()
            self.assertTrue(
                all(
                    left >= right - 1e-12
                    for left, right in zip(z_values, z_values[1:])
                ),
                msg=f"{region} priority is not descending in z",
            )
            medium = ordered[ordered["eligible_for_standard_transition"]].sort_values(
                "priority_rank_for_medium_within_region"
            )
            self.assertEqual(
                medium["priority_rank_for_medium_within_region"]
                .astype(int)
                .tolist(),
                list(range(1, 21)),
            )
            self.assertFalse((medium["construction_period"] == "post-2005").any())
            post_2005_existing = ordered[
                (ordered["state_id"] == "TABULA_existing")
                & (ordered["construction_period"] == "post-2005")
            ]
            self.assertEqual(len(post_2005_existing), 5)
            self.assertFalse(post_2005_existing["eligible_for_standard_transition"].any())
            self.assertEqual(
                set(post_2005_existing["eligible_transition_destinations"]),
                {"TABULA_advanced_A_proxy"},
            )

    def test_depth_allocation_audit(self) -> None:
        self.assertEqual(len(self.allocation), 420)
        for (scenario, region, depth), group in self.allocation.groupby(
            ["scenario", "region", "renovation_depth"], sort=False
        ):
            expected_rows = 50 if depth == "advanced" else 20
            self.assertEqual(len(group), expected_rows)
            ordered = group.sort_values("priority_rank_within_region_and_depth")
            self.assertEqual(
                ordered["priority_rank_within_region_and_depth"].tolist(),
                list(range(1, expected_rows + 1)),
            )
            z_values = ordered["specific_heat_loss_z_W_m2K"].tolist()
            self.assertTrue(
                all(
                    left >= right - 1e-12
                    for left, right in zip(z_values, z_values[1:])
                ),
                msg=f"{scenario}/{region}/{depth} priority is not descending in z",
            )
            if depth == "medium":
                self.assertEqual(set(group["source_state_id"]), {"TABULA_existing"})
                self.assertNotIn("post-2005", set(group["construction_period"]))
                self.assertEqual(
                    set(group["destination_state_id"]),
                    {"TABULA_standard_B_proxy"},
                )
            else:
                self.assertEqual(
                    set(group["source_state_id"]),
                    {"TABULA_existing", "TABULA_standard_B_proxy"},
                )
                self.assertEqual(
                    set(group["destination_state_id"]),
                    {"TABULA_advanced_A_proxy"},
                )

    def test_2050_stock_identities_and_arr(self) -> None:
        self.assertEqual(len(self.scenarios), 450)
        self.assertEqual(len(self.crosscheck), 6)
        expected_arr = {"central": 0.028, "high": 0.056}
        expected_depths = {
            "central": (0.4, 0.5, 0.1),
            "high": (0.4, 0.5, 0.1),
        }
        self.assertEqual(set(self.scenarios["scenario"]), {"central", "high"})
        self.assertTrue(
            self.scenarios["stock_geometry_mapping_assumption"]
            .str.contains("conditional on this mapping", regex=False)
            .all()
        )
        for (scenario, region), group in self.scenarios.groupby(
            ["scenario", "region"], sort=False
        ):
            self.assertTrue(
                math.isclose(group["ARR"].iloc[0], expected_arr[scenario])
            )
            observed_depths = (
                group["shallow_share"].iloc[0],
                group["medium_share"].iloc[0],
                group["advanced_share"].iloc[0],
            )
            self.assertEqual(observed_depths, expected_depths[scenario])
            regional_stock = group["regional_modelled_stock_dwellings"].iloc[0]
            self.assertTrue(
                math.isclose(
                    group["state_dwellings_2050"].sum(),
                    regional_stock,
                    abs_tol=1e-5,
                )
            )
            self.assertTrue(
                math.isclose(
                    group["state_share_within_region_2050"].sum(),
                    1.0,
                    abs_tol=1e-9,
                )
            )
            for _, archetype in group.groupby("archetype_id"):
                self.assertTrue(
                    math.isclose(
                        archetype["state_dwellings_2050"].sum(),
                        archetype["regional_number_of_dwellings"].iloc[0],
                        abs_tol=1e-5,
                    )
                )
            for row in group.itertuples(index=False):
                reconstructed = (
                    row.initial_state_dwellings_2025
                    - row.state_outflow_to_standard_2025_2050
                    - row.state_outflow_to_advanced_2025_2050
                    + row.state_inflow_from_medium_2025_2050
                    + row.state_inflow_from_advanced_2025_2050
                )
                self.assertTrue(
                    math.isclose(
                        reconstructed,
                        row.state_dwellings_2050,
                        abs_tol=1e-5,
                    )
                )

        for row in self.crosscheck.itertuples(index=False):
            self.assertTrue(
                math.isclose(
                    row.annual_total_renovation_activity_dwellings,
                    row.ARR * row.regional_modelled_stock_dwellings,
                    abs_tol=1e-5,
                )
            )
            self.assertTrue(
                math.isclose(
                    row.shallow_share + row.medium_share + row.advanced_share,
                    1.0,
                    abs_tol=1e-12,
                )
            )
            self.assertLessEqual(
                row.applied_medium_renovations_to_2050_dwellings,
                row.nominal_medium_renovations_to_2050_dwellings + 1e-5,
            )
            self.assertLessEqual(
                row.applied_advanced_renovations_to_2050_dwellings,
                row.nominal_advanced_renovations_to_2050_dwellings + 1e-5,
            )
            self.assertTrue(
                math.isclose(
                    row.physical_transition_events_to_2050,
                    row.applied_physical_renovations_to_2050_dwellings,
                    abs_tol=1e-5,
                )
            )
            self.assertLessEqual(
                row.minimum_repeat_transition_events_to_2050,
                row.physical_transition_events_to_2050 + 1e-5,
            )
            self.assertTrue(
                math.isclose(
                    row.maximum_unique_dwellings_with_physical_transition,
                    row.physical_transition_events_to_2050
                    - row.minimum_repeat_transition_events_to_2050,
                    abs_tol=1e-5,
                )
            )


if __name__ == "__main__":
    unittest.main()
