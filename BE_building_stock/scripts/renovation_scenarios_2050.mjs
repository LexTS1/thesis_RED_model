import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, "..");
const BASE_YEAR = 2025;
const TARGET_YEAR = 2050;
const YEARS_ELAPSED = TARGET_YEAR - BASE_YEAR;
const STATES = [
  "TABULA_existing",
  "TABULA_standard_B_proxy",
  "TABULA_advanced_A_proxy",
];
const EXISTING = STATES[0];
const STANDARD = STATES[1];
const ADVANCED = STATES[2];
const DEPTHS = ["shallow", "medium", "advanced"];
const SHARE_TOLERANCE = 1e-9;
const DWELLING_TOLERANCE = 1e-5;

const INPUT_PATHS = {
  states: path.join(PROJECT_ROOT, "data/scenarios/renovation/renovation_state_layer.csv"),
  allocation: path.join(PROJECT_ROOT, "data/scenarios/renovation/renovation_state_layer_with_allocation.csv"),
  arr: path.join(PROJECT_ROOT, "data/assumptions/renovation/renovation_scenario_ARR.csv"),
  policy: path.join(PROJECT_ROOT, "data/assumptions/renovation/regional_policy_targets.csv"),
};

const OUTPUT_PATHS = {
  scenarios: path.join(PROJECT_ROOT, "data/scenarios/renovation/archetype_matrix_2050_renovation_scenarios.csv"),
  allocation: path.join(PROJECT_ROOT, "data/scenarios/renovation/renovation_priority_allocation_2050.csv"),
  policyContext: path.join(PROJECT_ROOT, "data/scenarios/renovation/renovation_scenario_policy_context_2050.csv"),
  nationalSummary: path.join(PROJECT_ROOT, "data/scenarios/renovation/renovation_projection_national_summary_2050.csv"),
  trajectory: path.join(PROJECT_ROOT, "data/scenarios/renovation/renovation_state_trajectory_2025_2050.csv"),
};

const STATE_COLUMNS = [
  "scenario", "base_year", "target_year", "ARR", "ARR_percent_per_year",
  "shallow_share", "medium_share", "advanced_share", "region", "archetype_id",
  "dwelling_type", "construction_period", "state_id", "renovation_state",
  "regional_modelled_stock_dwellings", "regional_number_of_dwellings",
  "stock_geometry_mapping_assumption",
  "regional_state_share_2025", "regional_state_share_2025_lower",
  "regional_state_share_2025_upper", "initial_state_dwellings_2025",
  "initial_state_dwellings_2025_lower", "initial_state_dwellings_2025_upper",
  "state_outflow_to_standard_2025_2050", "state_outflow_to_advanced_2025_2050",
  "state_inflow_from_medium_2025_2050", "state_inflow_from_advanced_2025_2050",
  "state_dwellings", "state_dwellings_2050", "state_share_within_region_2050",
  "advanced_fraction_within_archetype_2025", "advanced_fraction_within_archetype_2050",
  "annual_total_renovation_activity_dwellings", "annual_shallow_activity_dwellings",
  "annual_medium_renovations_dwellings", "annual_advanced_renovations_dwellings",
  "nominal_renovations_to_2050_dwellings", "tracked_shallow_activity_to_2050_dwellings",
  "applied_physical_renovations_to_2050_dwellings", "unused_medium_quota_to_2050_dwellings",
  "unused_advanced_quota_to_2050_dwellings", "priority_metric",
  "U_facade_W_m2K", "U_roof_W_m2K", "U_floor_W_m2K", "U_window_W_m2K",
  "U_door_W_m2K", "v50_m3_h_m2", "q50_m3_h", "n50_h_1",
  "infiltration_n_factor", "infiltration_airflow_normal_m3_h",
  "infiltration_ach_normal_h_1", "transmission_heat_loss_H_tr_W_K",
  "infiltration_heat_loss_H_inf_W_K", "specific_heat_loss_z_W_m2K",
  "ventilation_system", "hrv_eta", "summer_bypass", "target_label_2050",
  "target_energy_score_2050_kWh_m2_year", "policy_source",
  "policy_status", "current_policy_context_url", "current_policy_context_locator",
  "state_evidence_source_url", "state_physical_source_url", "state_allocation_assumption",
  "ARR_evidence_classification", "ARR_source_url", "ARR_source_locator", "ARR_caveat",
];

const ALLOCATION_COLUMNS = [
  "scenario", "base_year", "target_year", "ARR", "renovation_depth", "depth_share",
  "region", "priority_rank_within_region_and_depth", "archetype_id", "dwelling_type",
  "construction_period", "source_state_id", "destination_state_id",
  "specific_heat_loss_z_W_m2K", "initial_source_state_dwellings_2025",
  "allocated_renovations_2025_2050", "first_allocation_year", "last_allocation_year",
  "annual_regional_depth_quota_dwellings", "nominal_depth_quota_to_2050_dwellings",
  "applied_depth_renovations_to_2050_dwellings", "unused_depth_quota_to_2050_dwellings",
  "priority_metric", "priority_tie_break", "transition_event_definition",
  "unique_dwelling_count_available",
];

const CROSSCHECK_COLUMNS = [
  "region", "scenario", "base_year", "target_year", "ARR",
  "shallow_share", "medium_share", "advanced_share",
  "annual_total_renovation_activity_dwellings", "annual_shallow_activity_dwellings",
  "annual_medium_renovations_dwellings", "annual_advanced_renovations_dwellings",
  "nominal_renovations_to_2050_dwellings", "nominal_shallow_activity_to_2050_dwellings",
  "nominal_medium_renovations_to_2050_dwellings", "nominal_advanced_renovations_to_2050_dwellings",
  "applied_medium_renovations_to_2050_dwellings", "applied_advanced_renovations_to_2050_dwellings",
  "applied_physical_renovations_to_2050_dwellings", "unused_medium_quota_to_2050_dwellings",
  "unused_advanced_quota_to_2050_dwellings", "regional_modelled_stock_dwellings",
  "existing_to_standard_transition_events_to_2050", "existing_to_advanced_transition_events_to_2050",
  "standard_to_advanced_transition_events_to_2050", "physical_transition_events_to_2050",
  "minimum_repeat_transition_events_to_2050", "maximum_unique_dwellings_with_physical_transition",
  "existing_share_2025", "standard_B_proxy_share_2025", "advanced_A_proxy_share_2025",
  "existing_share_2050", "standard_B_proxy_share_2050", "advanced_A_proxy_share_2050",
  "target_label_2050", "target_energy_score_2050_kWh_m2_year",
  "ARR_evidence_classification", "policy_status", "policy_source", "current_policy_context_url",
  "current_policy_context_locator", "allocation_status", "event_counting_note",
];

const NATIONAL_SUMMARY_COLUMNS = [
  "projection", "base_year", "target_year", "ARR", "ARR_percent_per_year",
  "shallow_share", "medium_share", "advanced_share",
  "national_modelled_stock_dwellings",
  "annual_total_renovation_activity_dwellings", "annual_shallow_activity_dwellings",
  "annual_medium_renovations_dwellings", "annual_advanced_renovations_dwellings",
  "nominal_renovations_to_2050_dwellings", "nominal_shallow_activity_to_2050_dwellings",
  "nominal_medium_renovations_to_2050_dwellings", "nominal_advanced_renovations_to_2050_dwellings",
  "applied_medium_renovations_to_2050_dwellings", "applied_advanced_renovations_to_2050_dwellings",
  "applied_physical_renovations_to_2050_dwellings", "unused_medium_quota_to_2050_dwellings",
  "unused_advanced_quota_to_2050_dwellings",
  "existing_to_standard_transition_events_to_2050", "existing_to_advanced_transition_events_to_2050",
  "standard_to_advanced_transition_events_to_2050", "physical_transition_events_to_2050",
  "minimum_repeat_transition_events_to_2050", "maximum_unique_dwellings_with_physical_transition",
  "existing_share_2025", "standard_B_proxy_share_2025", "advanced_A_proxy_share_2025",
  "existing_share_2050", "standard_B_proxy_share_2050", "advanced_A_proxy_share_2050",
  "shallow_TABULA_representation", "shallow_physical_effect",
  "medium_TABULA_representation", "medium_physical_effect",
  "advanced_TABULA_representation", "advanced_physical_effect",
  "projection_scope", "regional_allocation_method", "ARR_evidence_classification",
  "ARR_source_url", "ARR_source_locator", "ARR_caveat", "event_counting_note",
];

const TRAJECTORY_COLUMNS = [
  "projection", "year", "region", "regional_modelled_stock_dwellings",
  "existing_dwellings", "standard_B_proxy_dwellings", "advanced_A_proxy_dwellings",
  "existing_share", "standard_B_proxy_share", "advanced_A_proxy_share",
  "improved_envelope_dwellings", "improved_envelope_share",
  "improved_envelope_definition",
];

const REQUIRED_COLUMNS = {
  states: [
    "region", "archetype_id", "dwelling_type", "construction_period", "state_id",
    "regional_state_share", "regional_state_share_lower", "regional_state_share_upper",
    "regional_state_dwellings_2025", "regional_state_dwellings_2025_lower",
    "regional_state_dwellings_2025_upper", "regional_number_of_dwellings",
    "regional_modelled_stock_dwellings", "state_allocation_assumption",
    "stock_geometry_mapping_assumption",
    "state_evidence_source_url", "state_physical_source_url", "U_facade_W_m2K",
    "U_roof_W_m2K", "U_floor_W_m2K", "U_window_W_m2K", "U_door_W_m2K",
    "v50_m3_h_m2", "q50_m3_h", "n50_h_1", "infiltration_n_factor",
    "infiltration_airflow_normal_m3_h", "infiltration_ach_normal_h_1",
    "transmission_heat_loss_H_tr_W_K", "infiltration_heat_loss_H_inf_W_K",
    "specific_heat_loss_z_W_m2K", "ventilation_system",
  ],
  allocation: [
    "region", "archetype_id", "dwelling_type", "construction_period", "state_id",
    "regional_state_dwellings_2025", "priority_rank_for_advanced_within_region",
    "priority_rank_for_medium_within_region", "specific_heat_loss_z_W_m2K",
    "priority_metric", "priority_tie_break", "eligible_for_standard_transition",
    "eligible_transition_destinations",
  ],
  arr: [
    "scenario", "ARR", "ARR_percent_per_year", "shallow_share", "medium_share",
    "advanced_share", "evidence_classification", "source_url", "source_locator", "caveat",
  ],
  policy: [
    "region", "target_2050", "target_energy_score_2050_kWh_m2_year",
    "policy_url_primary", "policy_target_note", "policy_status",
    "current_policy_context_url", "current_policy_context_locator",
  ],
};


function assert(condition, message) {
  if (!condition) throw new Error(message);
}


function nearlyEqual(left, right, tolerance = SHARE_TOLERANCE) {
  return Math.abs(Number(left) - Number(right)) <= tolerance;
}


function numberValue(value, label, { min = -Infinity, max = Infinity } = {}) {
  const number = Number(value);
  assert(Number.isFinite(number), `${label} must be finite; found ${value}`);
  assert(number >= min && number <= max, `${label} must be in [${min}, ${max}]; found ${number}`);
  return number;
}


function booleanValue(value, label) {
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "true") return true;
  if (normalized === "false") return false;
  throw new Error(`${label} must be true or false; found ${value}`);
}


function stateKey(region, archetypeId, stateId) {
  return [region, archetypeId, stateId].join("\u0000");
}


function rowStateKey(row) {
  return stateKey(row.region, row.archetype_id, row.state_id);
}


function archetypeKey(region, archetypeId) {
  return [region, archetypeId].join("\u0000");
}


function auditKey(depth, row) {
  return [depth, rowStateKey(row)].join("\u0000");
}


function requireColumns(headers, required, tableName) {
  const observed = new Set(headers);
  const missing = required.filter((column) => !observed.has(column));
  assert(missing.length === 0, `${tableName} is missing columns: ${missing.join(", ")}`);
}


function requireUnique(rows, columns, tableName) {
  const seen = new Set();
  for (const row of rows) {
    const key = columns.map((column) => row[column]).join("\u0000");
    assert(!seen.has(key), `${tableName} has duplicate key ${key}`);
    seen.add(key);
  }
}


function parseCsv(text, filePath) {
  const source = text.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const values = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        cell += character;
      }
    } else if (character === '"') {
      assert(cell === "", `${filePath} has an unexpected quote`);
      quoted = true;
    } else if (character === ",") {
      row.push(cell);
      cell = "";
    } else if (character === "\n") {
      row.push(cell);
      values.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }
  assert(!quoted, `${filePath} has an unterminated quoted field`);
  if (cell !== "" || row.length > 0) {
    row.push(cell);
    values.push(row);
  }
  return values.filter((candidate) => candidate.some((value) => value !== ""));
}


async function readCsv(filePath) {
  const values = parseCsv(await fs.readFile(filePath, "utf8"), filePath);
  assert(values.length >= 1, `${filePath} is empty`);
  const headers = values[0].map((value) => String(value).trim());
  assert(headers.every(Boolean), `${filePath} has a blank header`);
  assert(new Set(headers).size === headers.length, `${filePath} has duplicate headers`);
  return {
    headers,
    rows: values.slice(1).map((valuesRow) =>
      Object.fromEntries(headers.map((header, index) => [header, valuesRow[index] ?? ""])),
    ),
  };
}


function validateInputs(tables) {
  for (const [name, table] of Object.entries(tables)) {
    requireColumns(table.headers, REQUIRED_COLUMNS[name], name);
  }
  const stateRows = tables.states.rows;
  const allocationRows = tables.allocation.rows;
  const arrRows = tables.arr.rows;
  const policyRows = tables.policy.rows;
  assert(stateRows.length === 225, `states must contain 225 rows; found ${stateRows.length}`);
  assert(allocationRows.length === 150, `allocation must contain 150 rows; found ${allocationRows.length}`);
  assert(arrRows.length === 1, `ARR input must contain one canonical projection row; found ${arrRows.length}`);
  assert(policyRows.length === 3, `policy input must contain 3 rows; found ${policyRows.length}`);
  requireUnique(stateRows, ["region", "archetype_id", "state_id"], "states");
  requireUnique(allocationRows, ["region", "archetype_id", "state_id"], "allocation");
  requireUnique(arrRows, ["scenario"], "ARR input");
  requireUnique(policyRows, ["region"], "policy input");

  const regions = [...new Set(stateRows.map((row) => row.region))];
  const scenarios = arrRows.map((row) => row.scenario);
  assert(regions.length === 3, `states must contain three regions; found ${regions.length}`);
  assert(new Set(scenarios).size === 1, "ARR projection identifier must be unique");
  assert(scenarios[0] === "central", `canonical projection identifier must be central; found ${scenarios[0]}`);
  const policyByRegion = new Map(policyRows.map((row) => [row.region, row]));
  assert(regions.every((region) => policyByRegion.has(region)), "policy regions differ from state regions");

  const regionalStock = new Map();
  for (const row of stateRows) {
    assert(STATES.includes(row.state_id), `unknown state ${row.state_id}`);
    const share = numberValue(row.regional_state_share, `${rowStateKey(row)} share`, { min: 0, max: 1 });
    const lower = numberValue(row.regional_state_share_lower, `${rowStateKey(row)} lower`, { min: 0, max: 1 });
    const upper = numberValue(row.regional_state_share_upper, `${rowStateKey(row)} upper`, { min: 0, max: 1 });
    assert(lower <= share && share <= upper, `${rowStateKey(row)} point share is outside bounds`);
    numberValue(row.regional_state_dwellings_2025, `${rowStateKey(row)} dwellings`, { min: 0 });
    numberValue(row.specific_heat_loss_z_W_m2K, `${rowStateKey(row)} z`, { min: 0 });
  }
  for (const row of arrRows) {
    const ARR = numberValue(row.ARR, `${row.scenario} ARR`, { min: 0, max: 1 });
    assert(nearlyEqual(ARR * 100, Number(row.ARR_percent_per_year)), `${row.scenario} ARR percent is inconsistent`);
    const depthTotal = DEPTHS.reduce(
      (sum, depth) => sum + numberValue(row[`${depth}_share`], `${row.scenario}/${depth}`, { min: 0, max: 1 }),
      0,
    );
    assert(nearlyEqual(depthTotal, 1), `${row.scenario} depth shares do not sum to one`);
  }

  for (const region of regions) {
    const regionStates = stateRows.filter((row) => row.region === region);
    const regionAllocation = allocationRows.filter((row) => row.region === region);
    assert(regionStates.length === 75, `${region} must contain 75 state rows`);
    assert(regionAllocation.length === 50, `${region} must contain 50 transition source rows`);
    const stocks = [...new Set(regionStates.map((row) => Number(row.regional_modelled_stock_dwellings)))];
    assert(stocks.length === 1, `${region} must have one regional stock total`);
    regionalStock.set(region, stocks[0]);
    const advancedRanks = regionAllocation
      .map((row) => Number(row.priority_rank_for_advanced_within_region))
      .sort((left, right) => left - right);
    assert(advancedRanks.every((rank, index) => rank === index + 1), `${region} advanced ranks are incomplete`);
    const mediumRows = regionAllocation.filter((row) => booleanValue(
      row.eligible_for_standard_transition,
      `${rowStateKey(row)} eligible_for_standard_transition`,
    ));
    const mediumRanks = mediumRows
      .map((row) => Number(row.priority_rank_for_medium_within_region))
      .sort((left, right) => left - right);
    assert(mediumRanks.length === 20, `${region} must contain 20 medium source rows`);
    assert(mediumRanks.every((rank, index) => rank === index + 1), `${region} medium ranks are incomplete`);
    assert(mediumRows.every((row) => row.state_id === EXISTING && row.construction_period !== "post-2005"), `${region} medium eligibility includes an invalid source cell`);
    for (const row of regionAllocation) {
      const destinations = new Set(String(row.eligible_transition_destinations).split("|"));
      const mediumEligible = booleanValue(
        row.eligible_for_standard_transition,
        `${rowStateKey(row)} eligible_for_standard_transition`,
      );
      if (mediumEligible) {
        assert(destinations.has(STANDARD) && destinations.has(ADVANCED), `${rowStateKey(row)} has incomplete destinations`);
      } else {
        assert(destinations.size === 1 && destinations.has(ADVANCED), `${rowStateKey(row)} has invalid destinations`);
      }
    }
    const archetypes = [...new Set(regionStates.map((row) => row.archetype_id))];
    assert(archetypes.length === 25, `${region} must contain 25 archetypes`);
    for (const archetypeId of archetypes) {
      const rows = regionStates.filter((row) => row.archetype_id === archetypeId);
      assert(rows.length === 3, `${region}/${archetypeId} must contain three states`);
      const total = rows.reduce((sum, row) => sum + Number(row.regional_state_dwellings_2025), 0);
      assert(Math.abs(total - Number(rows[0].regional_number_of_dwellings)) <= DWELLING_TOLERANCE, `${region}/${archetypeId} state stock identity failed`);
    }
  }
  return { regions, scenarios, policyByRegion, regionalStock };
}


function increment(map, key, value) {
  map.set(key, (map.get(key) ?? 0) + value);
}


function quotaResidual(nominal, applied) {
  const residual = Number(nominal) - Number(applied);
  return residual <= DWELLING_TOLERANCE ? 0 : residual;
}


function simulateScenario(arrRow, region, regionStates, regionAllocation, regionalStock) {
  const ARR = Number(arrRow.ARR);
  const depthShare = Object.fromEntries(DEPTHS.map((depth) => [depth, Number(arrRow[`${depth}_share`])]));
  const annualRenovations = Object.fromEntries(
    DEPTHS.map((depth) => [depth, ARR * depthShare[depth] * regionalStock]),
  );
  const nominalRenovationsTo2050 = Object.fromEntries(
    DEPTHS.map((depth) => [depth, annualRenovations[depth] * YEARS_ELAPSED]),
  );
  const appliedRenovationsTo2050 = { shallow: 0, medium: 0, advanced: 0 };
  const counts = new Map(
    regionStates.map((row) => [rowStateKey(row), Number(row.regional_state_dwellings_2025)]),
  );
  const trajectoryRows = [];
  const captureTrajectory = (year) => {
    const stateDwellings = Object.fromEntries(STATES.map((state) => [
      state,
      regionStates
        .filter((row) => row.state_id === state)
        .reduce((sum, row) => sum + (counts.get(rowStateKey(row)) ?? 0), 0),
    ]));
    const improvedEnvelopeDwellings = stateDwellings[STANDARD] + stateDwellings[ADVANCED];
    trajectoryRows.push({
      projection: arrRow.scenario,
      year,
      region,
      regional_modelled_stock_dwellings: regionalStock,
      existing_dwellings: stateDwellings[EXISTING],
      standard_B_proxy_dwellings: stateDwellings[STANDARD],
      advanced_A_proxy_dwellings: stateDwellings[ADVANCED],
      existing_share: stateDwellings[EXISTING] / regionalStock,
      standard_B_proxy_share: stateDwellings[STANDARD] / regionalStock,
      advanced_A_proxy_share: stateDwellings[ADVANCED] / regionalStock,
      improved_envelope_dwellings: improvedEnvelopeDwellings,
      improved_envelope_share: improvedEnvelopeDwellings / regionalStock,
      improved_envelope_definition: "TABULA_standard_B_proxy plus TABULA_advanced_A_proxy",
    });
  };
  captureTrajectory(BASE_YEAR);
  const outflowToStandard = new Map();
  const outflowToAdvanced = new Map();
  const inflowFromMedium = new Map();
  const inflowFromAdvanced = new Map();
  const audit = new Map();

  const advancedRows = [...regionAllocation].sort(
    (left, right) => Number(left.priority_rank_for_advanced_within_region) - Number(right.priority_rank_for_advanced_within_region),
  );
  const mediumRows = regionAllocation
    .filter((row) => booleanValue(
      row.eligible_for_standard_transition,
      `${rowStateKey(row)} eligible_for_standard_transition`,
    ))
    .sort((left, right) => Number(left.priority_rank_for_medium_within_region) - Number(right.priority_rank_for_medium_within_region));
  for (const row of advancedRows) {
    audit.set(auditKey("advanced", row), { allocated: 0, firstYear: "", lastYear: "" });
  }
  for (const row of mediumRows) {
    audit.set(auditKey("medium", row), { allocated: 0, firstYear: "", lastYear: "" });
  }

  for (let year = BASE_YEAR + 1; year <= TARGET_YEAR; year += 1) {
    let advancedRemaining = annualRenovations.advanced;
    for (const row of advancedRows) {
      if (advancedRemaining <= DWELLING_TOLERANCE) break;
      const sourceKey = rowStateKey(row);
      const available = counts.get(sourceKey) ?? 0;
      const allocated = Math.min(advancedRemaining, available);
      if (allocated <= DWELLING_TOLERANCE) continue;
      const destinationKey = stateKey(region, row.archetype_id, ADVANCED);
      counts.set(sourceKey, available - allocated);
      counts.set(destinationKey, (counts.get(destinationKey) ?? 0) + allocated);
      increment(outflowToAdvanced, sourceKey, allocated);
      increment(inflowFromAdvanced, destinationKey, allocated);
      const entry = audit.get(auditKey("advanced", row));
      entry.allocated += allocated;
      if (entry.firstYear === "") entry.firstYear = year;
      entry.lastYear = year;
      advancedRemaining -= allocated;
      appliedRenovationsTo2050.advanced += allocated;
    }

    let mediumRemaining = annualRenovations.medium;
    for (const row of mediumRows) {
      if (mediumRemaining <= DWELLING_TOLERANCE) break;
      const sourceKey = rowStateKey(row);
      const available = counts.get(sourceKey) ?? 0;
      const allocated = Math.min(mediumRemaining, available);
      if (allocated <= DWELLING_TOLERANCE) continue;
      const destinationKey = stateKey(region, row.archetype_id, STANDARD);
      counts.set(sourceKey, available - allocated);
      counts.set(destinationKey, (counts.get(destinationKey) ?? 0) + allocated);
      increment(outflowToStandard, sourceKey, allocated);
      increment(inflowFromMedium, destinationKey, allocated);
      const entry = audit.get(auditKey("medium", row));
      entry.allocated += allocated;
      if (entry.firstYear === "") entry.firstYear = year;
      entry.lastYear = year;
      mediumRemaining -= allocated;
      appliedRenovationsTo2050.medium += allocated;
    }

    const yearTotal = [...counts.values()].reduce((sum, value) => sum + value, 0);
    assert(Math.abs(yearTotal - regionalStock) <= DWELLING_TOLERANCE, `${region}/${arrRow.scenario}/${year} stock identity failed`);
    assert([...counts.values()].every((value) => value >= -DWELLING_TOLERANCE), `${region}/${arrRow.scenario}/${year} has negative stock`);
    captureTrajectory(year);
  }

  const allocationRows = [];
  for (const [depth, rows, destination, rankField] of [
    ["advanced", advancedRows, ADVANCED, "priority_rank_for_advanced_within_region"],
    ["medium", mediumRows, STANDARD, "priority_rank_for_medium_within_region"],
  ]) {
    const appliedDepth = appliedRenovationsTo2050[depth];
    const unusedDepth = quotaResidual(nominalRenovationsTo2050[depth], appliedDepth);
    for (const row of rows) {
      const entry = audit.get(auditKey(depth, row));
      allocationRows.push({
        scenario: arrRow.scenario,
        base_year: BASE_YEAR,
        target_year: TARGET_YEAR,
        ARR,
        renovation_depth: depth,
        depth_share: depthShare[depth],
        region,
        priority_rank_within_region_and_depth: Number(row[rankField]),
        archetype_id: row.archetype_id,
        dwelling_type: row.dwelling_type,
        construction_period: row.construction_period,
        source_state_id: row.state_id,
        destination_state_id: destination,
        specific_heat_loss_z_W_m2K: Number(row.specific_heat_loss_z_W_m2K),
        initial_source_state_dwellings_2025: Number(row.regional_state_dwellings_2025),
        allocated_renovations_2025_2050: entry.allocated,
        first_allocation_year: entry.firstYear,
        last_allocation_year: entry.lastYear,
        annual_regional_depth_quota_dwellings: annualRenovations[depth],
        nominal_depth_quota_to_2050_dwellings: nominalRenovationsTo2050[depth],
        applied_depth_renovations_to_2050_dwellings: appliedDepth,
        unused_depth_quota_to_2050_dwellings: unusedDepth,
        priority_metric: "specific_heat_loss_z_W_m2K_descending_current_state",
        priority_tie_break: "physical-state order then TABULA type number",
        transition_event_definition: `${row.state_id}_to_${destination}`,
        unique_dwelling_count_available: false,
      });
    }
  }

  return {
    counts,
    outflowToStandard,
    outflowToAdvanced,
    inflowFromMedium,
    inflowFromAdvanced,
    annualRenovations,
    nominalRenovationsTo2050,
    appliedRenovationsTo2050,
    trackedShallowActivityTo2050: nominalRenovationsTo2050.shallow,
    allocationRows,
    trajectoryRows,
  };
}


function buildOutputs(tables, context) {
  const scenarioRows = [];
  const allocationOutputRows = [];
  const crosscheckRows = [];
  const trajectoryRows = [];

  for (const arrRow of tables.arr.rows) {
    for (const region of context.regions) {
      const regionStates = tables.states.rows.filter((row) => row.region === region);
      const regionAllocation = tables.allocation.rows.filter((row) => row.region === region);
      const regionalStock = context.regionalStock.get(region);
      const simulation = simulateScenario(arrRow, region, regionStates, regionAllocation, regionalStock);
      allocationOutputRows.push(...simulation.allocationRows);
      trajectoryRows.push(...simulation.trajectoryRows);
      const policy = context.policyByRegion.get(region);
      const ARR = Number(arrRow.ARR);
      const annualTotal = ARR * regionalStock;
      const nominalTotal = annualTotal * YEARS_ELAPSED;
      const appliedPhysical = simulation.appliedRenovationsTo2050.medium + simulation.appliedRenovationsTo2050.advanced;

      for (const row of regionStates) {
        const key = rowStateKey(row);
        const archetypeRows = regionStates.filter((candidate) => candidate.archetype_id === row.archetype_id);
        const initialAdvanced = Number(archetypeRows.find((candidate) => candidate.state_id === ADVANCED).regional_state_dwellings_2025);
        const finalAdvanced = simulation.counts.get(stateKey(region, row.archetype_id, ADVANCED)) ?? 0;
        const archetypeStock = Number(row.regional_number_of_dwellings);
        const finalDwellings = simulation.counts.get(key) ?? 0;
        scenarioRows.push({
          scenario: arrRow.scenario,
          base_year: BASE_YEAR,
          target_year: TARGET_YEAR,
          ARR,
          ARR_percent_per_year: Number(arrRow.ARR_percent_per_year),
          shallow_share: Number(arrRow.shallow_share),
          medium_share: Number(arrRow.medium_share),
          advanced_share: Number(arrRow.advanced_share),
          region,
          archetype_id: row.archetype_id,
          dwelling_type: row.dwelling_type,
          construction_period: row.construction_period,
          state_id: row.state_id,
          renovation_state: row.state_id,
          regional_modelled_stock_dwellings: regionalStock,
          regional_number_of_dwellings: archetypeStock,
          stock_geometry_mapping_assumption: row.stock_geometry_mapping_assumption,
          regional_state_share_2025: Number(row.regional_state_share),
          regional_state_share_2025_lower: Number(row.regional_state_share_lower),
          regional_state_share_2025_upper: Number(row.regional_state_share_upper),
          initial_state_dwellings_2025: Number(row.regional_state_dwellings_2025),
          initial_state_dwellings_2025_lower: Number(row.regional_state_dwellings_2025_lower),
          initial_state_dwellings_2025_upper: Number(row.regional_state_dwellings_2025_upper),
          state_outflow_to_standard_2025_2050: simulation.outflowToStandard.get(key) ?? 0,
          state_outflow_to_advanced_2025_2050: simulation.outflowToAdvanced.get(key) ?? 0,
          state_inflow_from_medium_2025_2050: simulation.inflowFromMedium.get(key) ?? 0,
          state_inflow_from_advanced_2025_2050: simulation.inflowFromAdvanced.get(key) ?? 0,
          state_dwellings: finalDwellings,
          state_dwellings_2050: finalDwellings,
          state_share_within_region_2050: finalDwellings / regionalStock,
          advanced_fraction_within_archetype_2025: initialAdvanced / archetypeStock,
          advanced_fraction_within_archetype_2050: finalAdvanced / archetypeStock,
          annual_total_renovation_activity_dwellings: annualTotal,
          annual_shallow_activity_dwellings: simulation.annualRenovations.shallow,
          annual_medium_renovations_dwellings: simulation.annualRenovations.medium,
          annual_advanced_renovations_dwellings: simulation.annualRenovations.advanced,
          nominal_renovations_to_2050_dwellings: nominalTotal,
          tracked_shallow_activity_to_2050_dwellings: simulation.trackedShallowActivityTo2050,
          applied_physical_renovations_to_2050_dwellings: appliedPhysical,
          unused_medium_quota_to_2050_dwellings: quotaResidual(simulation.nominalRenovationsTo2050.medium, simulation.appliedRenovationsTo2050.medium),
          unused_advanced_quota_to_2050_dwellings: quotaResidual(simulation.nominalRenovationsTo2050.advanced, simulation.appliedRenovationsTo2050.advanced),
          priority_metric: "specific_heat_loss_z_W_m2K_descending_current_state",
          U_facade_W_m2K: Number(row.U_facade_W_m2K),
          U_roof_W_m2K: Number(row.U_roof_W_m2K),
          U_floor_W_m2K: Number(row.U_floor_W_m2K),
          U_window_W_m2K: Number(row.U_window_W_m2K),
          U_door_W_m2K: Number(row.U_door_W_m2K),
          v50_m3_h_m2: Number(row.v50_m3_h_m2),
          q50_m3_h: Number(row.q50_m3_h),
          n50_h_1: Number(row.n50_h_1),
          infiltration_n_factor: Number(row.infiltration_n_factor),
          infiltration_airflow_normal_m3_h: Number(row.infiltration_airflow_normal_m3_h),
          infiltration_ach_normal_h_1: Number(row.infiltration_ach_normal_h_1),
          transmission_heat_loss_H_tr_W_K: Number(row.transmission_heat_loss_H_tr_W_K),
          infiltration_heat_loss_H_inf_W_K: Number(row.infiltration_heat_loss_H_inf_W_K),
          specific_heat_loss_z_W_m2K: Number(row.specific_heat_loss_z_W_m2K),
          ventilation_system: row.ventilation_system,
          hrv_eta: row.hrv_eta,
          summer_bypass: row.summer_bypass,
          target_label_2050: policy.target_2050,
          target_energy_score_2050_kWh_m2_year: Number(policy.target_energy_score_2050_kWh_m2_year),
          policy_source: policy.policy_url_primary,
          policy_status: policy.policy_status,
          current_policy_context_url: policy.current_policy_context_url,
          current_policy_context_locator: policy.current_policy_context_locator,
          state_evidence_source_url: row.state_evidence_source_url,
          state_physical_source_url: row.state_physical_source_url,
          state_allocation_assumption: row.state_allocation_assumption,
          ARR_evidence_classification: arrRow.evidence_classification,
          ARR_source_url: arrRow.source_url,
          ARR_source_locator: arrRow.source_locator,
          ARR_caveat: arrRow.caveat,
        });
      }

      const finalRegionRows = scenarioRows.filter((row) => row.region === region && row.scenario === arrRow.scenario);
      const initialShares = Object.fromEntries(STATES.map((state) => [
        state,
        regionStates.filter((row) => row.state_id === state).reduce((sum, row) => sum + Number(row.regional_state_dwellings_2025), 0) / regionalStock,
      ]));
      const finalShares = Object.fromEntries(STATES.map((state) => [
        state,
        finalRegionRows.filter((row) => row.state_id === state).reduce((sum, row) => sum + Number(row.state_dwellings_2050), 0) / regionalStock,
      ]));
      const unusedMedium = quotaResidual(simulation.nominalRenovationsTo2050.medium, simulation.appliedRenovationsTo2050.medium);
      const unusedAdvanced = quotaResidual(simulation.nominalRenovationsTo2050.advanced, simulation.appliedRenovationsTo2050.advanced);
      const existingToStandardEvents = simulation.appliedRenovationsTo2050.medium;
      const existingToAdvancedEvents = regionStates
        .filter((row) => row.state_id === EXISTING)
        .reduce((sum, row) => sum + (simulation.outflowToAdvanced.get(rowStateKey(row)) ?? 0), 0);
      const standardToAdvancedEvents = regionStates
        .filter((row) => row.state_id === STANDARD)
        .reduce((sum, row) => sum + (simulation.outflowToAdvanced.get(rowStateKey(row)) ?? 0), 0);
      const minimumRepeatEvents = regionStates
        .filter((row) => row.state_id === STANDARD)
        .reduce((sum, row) => {
          const standardOutflow = simulation.outflowToAdvanced.get(rowStateKey(row)) ?? 0;
          return sum + Math.max(0, standardOutflow - Number(row.regional_state_dwellings_2025));
        }, 0);
      const physicalTransitionEvents = existingToStandardEvents + existingToAdvancedEvents + standardToAdvancedEvents;
      const maximumUniqueDwellings = physicalTransitionEvents - minimumRepeatEvents;
      crosscheckRows.push({
        region,
        scenario: arrRow.scenario,
        base_year: BASE_YEAR,
        target_year: TARGET_YEAR,
        ARR,
        shallow_share: Number(arrRow.shallow_share),
        medium_share: Number(arrRow.medium_share),
        advanced_share: Number(arrRow.advanced_share),
        annual_total_renovation_activity_dwellings: annualTotal,
        annual_shallow_activity_dwellings: simulation.annualRenovations.shallow,
        annual_medium_renovations_dwellings: simulation.annualRenovations.medium,
        annual_advanced_renovations_dwellings: simulation.annualRenovations.advanced,
        nominal_renovations_to_2050_dwellings: nominalTotal,
        nominal_shallow_activity_to_2050_dwellings: simulation.nominalRenovationsTo2050.shallow,
        nominal_medium_renovations_to_2050_dwellings: simulation.nominalRenovationsTo2050.medium,
        nominal_advanced_renovations_to_2050_dwellings: simulation.nominalRenovationsTo2050.advanced,
        applied_medium_renovations_to_2050_dwellings: simulation.appliedRenovationsTo2050.medium,
        applied_advanced_renovations_to_2050_dwellings: simulation.appliedRenovationsTo2050.advanced,
        applied_physical_renovations_to_2050_dwellings: appliedPhysical,
        unused_medium_quota_to_2050_dwellings: unusedMedium,
        unused_advanced_quota_to_2050_dwellings: unusedAdvanced,
        regional_modelled_stock_dwellings: regionalStock,
        existing_to_standard_transition_events_to_2050: existingToStandardEvents,
        existing_to_advanced_transition_events_to_2050: existingToAdvancedEvents,
        standard_to_advanced_transition_events_to_2050: standardToAdvancedEvents,
        physical_transition_events_to_2050: physicalTransitionEvents,
        minimum_repeat_transition_events_to_2050: minimumRepeatEvents,
        maximum_unique_dwellings_with_physical_transition: maximumUniqueDwellings,
        existing_share_2025: initialShares[EXISTING],
        standard_B_proxy_share_2025: initialShares[STANDARD],
        advanced_A_proxy_share_2025: initialShares[ADVANCED],
        existing_share_2050: finalShares[EXISTING],
        standard_B_proxy_share_2050: finalShares[STANDARD],
        advanced_A_proxy_share_2050: finalShares[ADVANCED],
        target_label_2050: policy.target_2050,
        target_energy_score_2050_kWh_m2_year: Number(policy.target_energy_score_2050_kWh_m2_year),
        ARR_evidence_classification: arrRow.evidence_classification,
        policy_status: policy.policy_status,
        policy_source: policy.policy_url_primary,
        current_policy_context_url: policy.current_policy_context_url,
        current_policy_context_locator: policy.current_policy_context_locator,
        allocation_status: unusedMedium + unusedAdvanced > DWELLING_TOLERANCE ? "eligible_stock_limited" : "depth_quotas_allocated",
        event_counting_note: "Physical totals count state-transition events. By archetype, Standard-to-Advanced events exceeding the initial Standard stock form a guaranteed lower bound on repeated Medium-then-Advanced renovations; zero means the aggregate flows do not require a repeat.",
      });
    }
  }
  const nationalSummaryRows = tables.arr.rows.map((arrRow) => {
    const regionalRows = crosscheckRows.filter((row) => row.scenario === arrRow.scenario);
    const projectionStates = scenarioRows.filter((row) => row.scenario === arrRow.scenario);
    const nationalStock = regionalRows.reduce(
      (sum, row) => sum + Number(row.regional_modelled_stock_dwellings),
      0,
    );
    const sumField = (field) => regionalRows.reduce((sum, row) => sum + Number(row[field]), 0);
    const stateShare = (stateId, field) => projectionStates
      .filter((row) => row.state_id === stateId)
      .reduce((sum, row) => sum + Number(row[field]), 0) / nationalStock;
    return {
      projection: arrRow.scenario,
      base_year: BASE_YEAR,
      target_year: TARGET_YEAR,
      ARR: Number(arrRow.ARR),
      ARR_percent_per_year: Number(arrRow.ARR_percent_per_year),
      shallow_share: Number(arrRow.shallow_share),
      medium_share: Number(arrRow.medium_share),
      advanced_share: Number(arrRow.advanced_share),
      national_modelled_stock_dwellings: nationalStock,
      annual_total_renovation_activity_dwellings: sumField("annual_total_renovation_activity_dwellings"),
      annual_shallow_activity_dwellings: sumField("annual_shallow_activity_dwellings"),
      annual_medium_renovations_dwellings: sumField("annual_medium_renovations_dwellings"),
      annual_advanced_renovations_dwellings: sumField("annual_advanced_renovations_dwellings"),
      nominal_renovations_to_2050_dwellings: sumField("nominal_renovations_to_2050_dwellings"),
      nominal_shallow_activity_to_2050_dwellings: sumField("nominal_shallow_activity_to_2050_dwellings"),
      nominal_medium_renovations_to_2050_dwellings: sumField("nominal_medium_renovations_to_2050_dwellings"),
      nominal_advanced_renovations_to_2050_dwellings: sumField("nominal_advanced_renovations_to_2050_dwellings"),
      applied_medium_renovations_to_2050_dwellings: sumField("applied_medium_renovations_to_2050_dwellings"),
      applied_advanced_renovations_to_2050_dwellings: sumField("applied_advanced_renovations_to_2050_dwellings"),
      applied_physical_renovations_to_2050_dwellings: sumField("applied_physical_renovations_to_2050_dwellings"),
      unused_medium_quota_to_2050_dwellings: sumField("unused_medium_quota_to_2050_dwellings"),
      unused_advanced_quota_to_2050_dwellings: sumField("unused_advanced_quota_to_2050_dwellings"),
      existing_to_standard_transition_events_to_2050: sumField("existing_to_standard_transition_events_to_2050"),
      existing_to_advanced_transition_events_to_2050: sumField("existing_to_advanced_transition_events_to_2050"),
      standard_to_advanced_transition_events_to_2050: sumField("standard_to_advanced_transition_events_to_2050"),
      physical_transition_events_to_2050: sumField("physical_transition_events_to_2050"),
      minimum_repeat_transition_events_to_2050: sumField("minimum_repeat_transition_events_to_2050"),
      maximum_unique_dwellings_with_physical_transition: sumField("maximum_unique_dwellings_with_physical_transition"),
      existing_share_2025: stateShare(EXISTING, "initial_state_dwellings_2025"),
      standard_B_proxy_share_2025: stateShare(STANDARD, "initial_state_dwellings_2025"),
      advanced_A_proxy_share_2025: stateShare(ADVANCED, "initial_state_dwellings_2025"),
      existing_share_2050: stateShare(EXISTING, "state_dwellings_2050"),
      standard_B_proxy_share_2050: stateShare(STANDARD, "state_dwellings_2050"),
      advanced_A_proxy_share_2050: stateShare(ADVANCED, "state_dwellings_2050"),
      shallow_TABULA_representation: "TABULA_existing_as_is",
      shallow_physical_effect: "No envelope-state change",
      medium_TABULA_representation: "TABULA_standard_B_proxy",
      medium_physical_effect: "Pre-2006 TABULA_existing to TABULA_standard_B_proxy",
      advanced_TABULA_representation: "TABULA_advanced_A_proxy_low_energy",
      advanced_physical_effect: "TABULA_existing or TABULA_standard_B_proxy to TABULA_advanced_A_proxy",
      projection_scope: "Renovation-state projection of the fixed 2025 Belgian R1-R4 dwelling stock",
      regional_allocation_method: "National ARR and depth shares disaggregated in proportion to regional 2025 R1-R4 stock before within-region z ranking",
      ARR_evidence_classification: arrRow.evidence_classification,
      ARR_source_url: arrRow.source_url,
      ARR_source_locator: arrRow.source_locator,
      ARR_caveat: arrRow.caveat,
      event_counting_note: "Physical totals count state-transition events. The repeated-transition field sums, by archetype, Standard-to-Advanced events exceeding the initial Standard stock and is therefore a guaranteed lower bound.",
    };
  });
  return { scenarioRows, allocationOutputRows, crosscheckRows, nationalSummaryRows, trajectoryRows };
}


function validateOutputs(outputs, context) {
  const {
    scenarioRows, allocationOutputRows, crosscheckRows, nationalSummaryRows,
    trajectoryRows,
  } = outputs;
  const projectionCount = context.scenarios.length;
  assert(scenarioRows.length === projectionCount * 225, `projection output must contain ${projectionCount * 225} rows; found ${scenarioRows.length}`);
  assert(allocationOutputRows.length === projectionCount * 210, `allocation output must contain ${projectionCount * 210} rows; found ${allocationOutputRows.length}`);
  assert(crosscheckRows.length === projectionCount * 3, `policy-context output must contain ${projectionCount * 3} rows; found ${crosscheckRows.length}`);
  assert(nationalSummaryRows.length === projectionCount, `national summary must contain ${projectionCount} row; found ${nationalSummaryRows.length}`);
  assert(trajectoryRows.length === projectionCount * context.regions.length * (YEARS_ELAPSED + 1), `trajectory output must contain ${projectionCount * context.regions.length * (YEARS_ELAPSED + 1)} rows; found ${trajectoryRows.length}`);
  requireUnique(scenarioRows, ["scenario", "region", "archetype_id", "state_id"], "scenario output");
  requireUnique(allocationOutputRows, ["scenario", "region", "renovation_depth", "archetype_id", "source_state_id"], "allocation output");
  requireUnique(crosscheckRows, ["scenario", "region"], "cross-check output");
  requireUnique(nationalSummaryRows, ["projection"], "national summary");
  requireUnique(trajectoryRows, ["projection", "region", "year"], "annual state trajectory");

  for (const scenario of context.scenarios) {
    for (const region of context.regions) {
      const rows = scenarioRows.filter((row) => row.scenario === scenario && row.region === region);
      assert(rows.length === 75, `${region}/${scenario} must contain 75 state rows`);
      const stock = context.regionalStock.get(region);
      assert(Math.abs(rows.reduce((sum, row) => sum + Number(row.state_dwellings_2050), 0) - stock) <= DWELLING_TOLERANCE, `${region}/${scenario} stock identity failed`);
      assert(Math.abs(rows.reduce((sum, row) => sum + Number(row.state_share_within_region_2050), 0) - 1) <= SHARE_TOLERANCE, `${region}/${scenario} shares do not sum to one`);
      assert(rows.every((row) => Number(row.state_dwellings_2050) >= -DWELLING_TOLERANCE), `${region}/${scenario} has negative stock`);
      for (const archetypeId of [...new Set(rows.map((row) => row.archetype_id))]) {
        const group = rows.filter((row) => row.archetype_id === archetypeId);
        assert(Math.abs(group.reduce((sum, row) => sum + Number(row.state_dwellings_2050), 0) - Number(group[0].regional_number_of_dwellings)) <= DWELLING_TOLERANCE, `${region}/${scenario}/${archetypeId} stock identity failed`);
      }
      const check = crosscheckRows.find((row) => row.scenario === scenario && row.region === region);
      const depthShares = Number(check.shallow_share) + Number(check.medium_share) + Number(check.advanced_share);
      assert(nearlyEqual(depthShares, 1), `${region}/${scenario} depth shares do not sum to one`);
      assert(nearlyEqual(Number(check.annual_total_renovation_activity_dwellings), Number(check.ARR) * stock, DWELLING_TOLERANCE), `${region}/${scenario} annual ARR identity failed`);
      assert(Number(check.applied_medium_renovations_to_2050_dwellings) <= Number(check.nominal_medium_renovations_to_2050_dwellings) + DWELLING_TOLERANCE, `${region}/${scenario} medium quota exceeded`);
      assert(Number(check.applied_advanced_renovations_to_2050_dwellings) <= Number(check.nominal_advanced_renovations_to_2050_dwellings) + DWELLING_TOLERANCE, `${region}/${scenario} advanced quota exceeded`);
      assert(nearlyEqual(Number(check.physical_transition_events_to_2050), Number(check.applied_physical_renovations_to_2050_dwellings), DWELLING_TOLERANCE), `${region}/${scenario} transition-event identity failed`);
      assert(Number(check.minimum_repeat_transition_events_to_2050) <= Number(check.physical_transition_events_to_2050) + DWELLING_TOLERANCE, `${region}/${scenario} repeat-event lower bound is invalid`);
      const trajectory = trajectoryRows
        .filter((row) => row.projection === scenario && row.region === region)
        .sort((left, right) => Number(left.year) - Number(right.year));
      assert(trajectory.length === YEARS_ELAPSED + 1, `${region}/${scenario} trajectory must contain ${YEARS_ELAPSED + 1} annual rows`);
      assert(trajectory[0].year === BASE_YEAR && trajectory.at(-1).year === TARGET_YEAR, `${region}/${scenario} trajectory endpoints are invalid`);
      for (const row of trajectory) {
        const stateTotal = Number(row.existing_dwellings) + Number(row.standard_B_proxy_dwellings) + Number(row.advanced_A_proxy_dwellings);
        const shareTotal = Number(row.existing_share) + Number(row.standard_B_proxy_share) + Number(row.advanced_A_proxy_share);
        assert(nearlyEqual(stateTotal, stock, DWELLING_TOLERANCE), `${region}/${scenario}/${row.year} trajectory stock identity failed`);
        assert(nearlyEqual(shareTotal, 1), `${region}/${scenario}/${row.year} trajectory shares do not sum to one`);
        assert(nearlyEqual(Number(row.improved_envelope_share), Number(row.standard_B_proxy_share) + Number(row.advanced_A_proxy_share)), `${region}/${scenario}/${row.year} improved-envelope identity failed`);
      }
      assert(trajectory.every((row, index) => index === 0 || Number(row.improved_envelope_share) >= Number(trajectory[index - 1].improved_envelope_share) - SHARE_TOLERANCE), `${region}/${scenario} improved-envelope trajectory is not non-decreasing`);
      assert(nearlyEqual(Number(trajectory[0].existing_share), Number(check.existing_share_2025)), `${region}/${scenario} 2025 trajectory does not match calibration`);
      assert(nearlyEqual(Number(trajectory.at(-1).existing_share), Number(check.existing_share_2050)), `${region}/${scenario} 2050 trajectory does not match final matrix`);
    }
    const summary = nationalSummaryRows.find((row) => row.projection === scenario);
    const nationalStock = [...context.regionalStock.values()].reduce((sum, value) => sum + value, 0);
    assert(nearlyEqual(Number(summary.national_modelled_stock_dwellings), nationalStock, DWELLING_TOLERANCE), `${scenario} national stock identity failed`);
    assert(nearlyEqual(Number(summary.annual_total_renovation_activity_dwellings), Number(summary.ARR) * nationalStock, DWELLING_TOLERANCE), `${scenario} national ARR identity failed`);
    assert(nearlyEqual(Number(summary.existing_share_2025) + Number(summary.standard_B_proxy_share_2025) + Number(summary.advanced_A_proxy_share_2025), 1), `${scenario} national 2025 shares do not sum to one`);
    assert(nearlyEqual(Number(summary.existing_share_2050) + Number(summary.standard_B_proxy_share_2050) + Number(summary.advanced_A_proxy_share_2050), 1), `${scenario} national 2050 shares do not sum to one`);
  }
}


const NUMBER_FORMATTER = new Intl.NumberFormat("en-US", {
  useGrouping: false,
  minimumFractionDigits: 0,
  maximumFractionDigits: 12,
});


function csvCell(value) {
  const text = typeof value === "number"
    ? NUMBER_FORMATTER.format(Math.abs(value) < 5e-13 ? 0 : value)
    : String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}


function toCsv(columns, rows) {
  return `${[
    columns.map(csvCell).join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\n")}\n`;
}


async function writeAtomic(filePath, columns, rows) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}`;
  await fs.writeFile(temporary, toCsv(columns, rows), "utf8");
  const reread = await readCsv(temporary);
  assert(reread.headers.length === columns.length && reread.headers.every((header, index) => header === columns[index]), `${path.basename(filePath)} headers changed during serialization`);
  assert(reread.rows.length === rows.length, `${path.basename(filePath)} row count changed during serialization`);
  await fs.rename(temporary, filePath);
}


async function main() {
  const tables = Object.fromEntries(
    await Promise.all(Object.entries(INPUT_PATHS).map(async ([name, filePath]) => [name, await readCsv(filePath)])),
  );
  const context = validateInputs(tables);
  const outputs = buildOutputs(tables, context);
  validateOutputs(outputs, context);
  await writeAtomic(OUTPUT_PATHS.scenarios, STATE_COLUMNS, outputs.scenarioRows);
  await writeAtomic(OUTPUT_PATHS.allocation, ALLOCATION_COLUMNS, outputs.allocationOutputRows);
  await writeAtomic(OUTPUT_PATHS.policyContext, CROSSCHECK_COLUMNS, outputs.crosscheckRows);
  await writeAtomic(OUTPUT_PATHS.nationalSummary, NATIONAL_SUMMARY_COLUMNS, outputs.nationalSummaryRows);
  await writeAtomic(OUTPUT_PATHS.trajectory, TRAJECTORY_COLUMNS, outputs.trajectoryRows);

  console.log("National renovation-projection validation passed");
  console.log(`- projection-state rows: ${outputs.scenarioRows.length}`);
  console.log(`- depth-allocation rows: ${outputs.allocationOutputRows.length}`);
  console.log(`- regional policy-context rows: ${outputs.crosscheckRows.length}`);
  console.log(`- national summary rows: ${outputs.nationalSummaryRows.length}`);
  console.log(`- annual state-trajectory rows: ${outputs.trajectoryRows.length}`);
  console.log("- ARR uses the fixed 2025 regional R1-R4 stock over 25 annual steps");
  console.log("- advanced transitions precede medium transitions within each year");
  for (const row of outputs.crosscheckRows) {
    console.log(
      `  ${row.region} | ${row.scenario} | ARR=${(row.ARR * 100).toFixed(1)}%/yr | `
      + `mix=${(row.shallow_share * 100).toFixed(0)}/${(row.medium_share * 100).toFixed(0)}/${(row.advanced_share * 100).toFixed(0)} | `
      + `states 2050=${(row.existing_share_2050 * 100).toFixed(2)}/${(row.standard_B_proxy_share_2050 * 100).toFixed(2)}/${(row.advanced_A_proxy_share_2050 * 100).toFixed(2)}`,
    );
  }
}


main().catch((error) => {
  console.error(`Validation failed: ${error.message}`);
  process.exitCode = 1;
});
