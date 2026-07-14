import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, "..");

const INPUT_PATHS = {
  matrix: path.join(
    PROJECT_ROOT,
    "data/matrices/regional/regional_stock_weighted_archetype_matrix.csv",
  ),
  layer: path.join(
    PROJECT_ROOT,
    "data/scenarios/renovation/renovation_state_layer_with_allocation.csv",
  ),
  policy: path.join(
    PROJECT_ROOT,
    "data/assumptions/renovation/regional_policy_targets.csv",
  ),
  priority: path.join(
    PROJECT_ROOT,
    "data/assumptions/renovation/renovation_priority_mapping.csv",
  ),
};

const OUTPUT_PATHS = {
  package: path.join(
    PROJECT_ROOT,
    "data/assumptions/renovation/renovation_physical_package_TABULA_low_energy.csv",
  ),
  scenarios: path.join(
    PROJECT_ROOT,
    "data/scenarios/renovation/archetype_matrix_2050_renovation_scenarios.csv",
  ),
  crosscheck: path.join(
    PROJECT_ROOT,
    "data/scenarios/renovation/renovation_policy_crosscheck_2050.csv",
  ),
};

const BASE_YEAR = 2025;
const TARGET_YEAR = 2050;
const YEARS_ELAPSED = TARGET_YEAR - BASE_YEAR;
const SCENARIOS = ["low", "central", "high"];
const SHARE_TOLERANCE = 1e-9;
const DWELLING_TOLERANCE = 1e-5;
const VALUE_TOLERANCE = 1e-10;

const PACKAGE = {
  renovation_package_id: "TABULA_low_energy",
  U_wall_renovated: 0.25,
  U_roof_renovated: 0.15,
  U_floor_renovated: 0.25,
  U_window_renovated: 1.6,
  v50_renovated: 2.5,
  ventilation_after_renovation:
    "mechanical_ventilation_heat_recovery_eta_0.8_with_bypass",
  hrv_eta: 0.8,
  summer_bypass: true,
  renovation_package_source:
    "Belgian TABULA/VITO Scientific Report, Low Energy upgrade scenario",
};

const PACKAGE_COLUMNS = Object.keys(PACKAGE);

const SCENARIO_COLUMNS = [
  "scenario",
  "target_year",
  "region",
  "archetype_id",
  "dwelling_type",
  "construction_period",
  "renovation_state",
  "regional_number_of_dwellings",
  "regional_modelled_stock_dwellings",
  "allocation_share_within_region",
  "renovated_fraction",
  "state_dwellings",
  "state_share_within_region",
  "annual_renovation_rate_scenario",
  "target_label_2050",
  "target_energy_score_2050_kWh_m2_year",
  ...PACKAGE_COLUMNS,
  "policy_source",
];

const CROSSCHECK_COLUMNS = [
  "region",
  "scenario",
  "target_year",
  "annual_renovation_rate_scenario",
  "renovated_dwellings_region",
  "regional_modelled_stock_dwellings",
  "renovated_fraction_region",
  "target_label_2050",
  "target_energy_score_2050_kWh_m2_year",
  "model_physical_package",
  "cross_check_status",
  "cross_check_note",
];

const REQUIRED_COLUMNS = {
  matrix: [
    "region",
    "archetype_id",
    "dwelling_type",
    "construction_period",
    "regional_archetype_share_within_region",
    "regional_number_of_dwellings",
    "regional_modelled_stock_dwellings",
  ],
  layer: [
    "region",
    "archetype_id",
    "dwelling_type",
    "construction_period",
    "renovation_priority",
    "priority_weight",
    "annual_renovation_rate_low",
    "annual_renovation_rate_central",
    "annual_renovation_rate_high",
    "allocation_share_within_region",
    "target_label_2050",
    "target_energy_score_2050_kWh_m2_year",
    "policy_source",
    "regional_number_of_dwellings",
  ],
  policy: [
    "region",
    "target_2050",
    "target_energy_score_2050_kWh_m2_year",
    "annual_renovation_rate_low",
    "annual_renovation_rate_central",
    "annual_renovation_rate_high",
  ],
  priority: [
    "construction_period",
    "renovation_priority",
    "priority_weight",
  ],
};

const CURRENT_INPUT_HASHES = {
  matrix: "54c9be2ab3808fac5565d0afec25dd0e47d42b9da07ee3f01960f9684ec03e3c",
  layer: "f2c839f6f4cb1a47cfa0ea8edf014f6258e573e0db54f70610490a485a1c0fdf",
  policy: "4abb24e959e2a42df53e05f749440a0020309514bae1ba043cdc7d1b4455ffdd",
  priority: "3fc4771ce594a68705d6f46f32b44cae2c41457242feda8a9031c358a7b3bf0a",
};

const CURRENT_HIGH_FRACTIONS = {
  "Flemish Region": 0.8482461906327716,
  "Walloon Region": 0.8787302750384757,
  "Brussels-Capital Region": 0.9002826809621413,
};

const POLICY_BENCHMARK_NOTE =
  "EPC/PEB targets are policy benchmarks, while TABULA Low Energy defines " +
  "the physical renovated state; the model does not claim exact equivalence " +
  "to EPC A, PEB A, or PEB C+.";

const CSV_NUMBER_FORMATTER = new Intl.NumberFormat("en-US", {
  useGrouping: false,
  minimumFractionDigits: 0,
  maximumFractionDigits: 12,
});


function loadArtifactTool() {
  const runtimeNodeModules = process.env.CODEX_NODE_MODULES;
  const requireFrom = runtimeNodeModules
    ? path.join(runtimeNodeModules, "_codex_runtime_entry.cjs")
    : import.meta.url;
  try {
    const require = createRequire(requireFrom);
    return require("@oai/artifact-tool");
  } catch (error) {
    throw new Error(
      "@oai/artifact-tool is required. Install it locally or set " +
        "CODEX_NODE_MODULES to the bundled runtime node_modules path.",
      { cause: error },
    );
  }
}


const { Workbook } = loadArtifactTool();


function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}


function nearlyEqual(left, right, tolerance = VALUE_TOLERANCE) {
  return Math.abs(left - right) <= tolerance;
}


function keyFor(row) {
  return `${row.region}\u0000${row.archetype_id}`;
}


function groupKey(region, scenario) {
  return `${region}\u0000${scenario}`;
}


function uniqueInOrder(values) {
  return [...new Set(values)];
}


function requireColumns(headers, required, tableName) {
  const headerSet = new Set(headers);
  const missing = required.filter((column) => !headerSet.has(column));
  assert(
    missing.length === 0,
    `${tableName} is missing required columns: ${missing.join(", ")}`,
  );
}


function requireNonblank(rows, columns, tableName) {
  for (const [index, row] of rows.entries()) {
    for (const column of columns) {
      assert(
        String(row[column] ?? "").trim() !== "",
        `${tableName} row ${index + 2} has a blank required value in ${column}`,
      );
    }
  }
}


function requireUnique(rows, keyColumns, tableName) {
  const seen = new Set();
  for (const row of rows) {
    const key = keyColumns.map((column) => row[column]).join("\u0000");
    assert(
      !seen.has(key),
      `${tableName} has a duplicate key for ${keyColumns.join(", ")}: ${key}`,
    );
    seen.add(key);
  }
}


function numberValue(value, label, { min = -Infinity, strictlyPositive = false } = {}) {
  const number = Number(value);
  assert(Number.isFinite(number), `${label} must be a finite number; found ${value}`);
  assert(number >= min, `${label} must be at least ${min}; found ${number}`);
  if (strictlyPositive) {
    assert(number > 0, `${label} must be greater than zero; found ${number}`);
  }
  return number;
}


function booleanValue(value, label) {
  if (value === true || String(value).toLowerCase() === "true") return true;
  if (value === false || String(value).toLowerCase() === "false") return false;
  throw new Error(`${label} must be true or false; found ${value}`);
}


async function sha256(filePath) {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(bytes).digest("hex");
}


async function inputHashes() {
  return Object.fromEntries(
    await Promise.all(
      Object.entries(INPUT_PATHS).map(async ([name, filePath]) => [
        name,
        await sha256(filePath),
      ]),
    ),
  );
}


async function readCsv(filePath, sheetName) {
  const text = await fs.readFile(filePath, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange(true).values;
  assert(values.length >= 1, `${filePath} is empty`);
  const headers = values[0].map((value) => String(value ?? "").trim());
  assert(headers.every((header) => header !== ""), `${filePath} has a blank header`);
  assert(new Set(headers).size === headers.length, `${filePath} has duplicate headers`);
  const rows = values.slice(1).map((valuesRow) =>
    Object.fromEntries(
      headers.map((header, index) => [header, valuesRow[index] ?? ""]),
    ),
  );
  return { headers, rows };
}


function validateInputs(tables) {
  for (const [name, table] of Object.entries(tables)) {
    requireColumns(table.headers, REQUIRED_COLUMNS[name], name);
    requireNonblank(table.rows, REQUIRED_COLUMNS[name], name);
  }

  const matrixRows = tables.matrix.rows;
  const layerRows = tables.layer.rows;
  const policyRows = tables.policy.rows;
  const priorityRows = tables.priority.rows;

  assert(matrixRows.length === 75, `matrix must contain 75 rows; found ${matrixRows.length}`);
  assert(layerRows.length === 75, `layer must contain 75 rows; found ${layerRows.length}`);
  assert(policyRows.length === 3, `policy must contain 3 rows; found ${policyRows.length}`);

  requireUnique(matrixRows, ["region", "archetype_id"], "matrix");
  requireUnique(layerRows, ["region", "archetype_id"], "layer");
  requireUnique(policyRows, ["region"], "policy");
  requireUnique(priorityRows, ["construction_period"], "priority");

  const matrixByKey = new Map(matrixRows.map((row) => [keyFor(row), row]));
  const layerByKey = new Map(layerRows.map((row) => [keyFor(row), row]));
  assert(matrixByKey.size === layerByKey.size, "matrix and layer key counts differ");
  for (const key of matrixByKey.keys()) {
    assert(layerByKey.has(key), `renovation layer is missing matrix key ${key}`);
  }
  for (const key of layerByKey.keys()) {
    assert(matrixByKey.has(key), `matrix is missing renovation-layer key ${key}`);
  }

  const regions = uniqueInOrder(matrixRows.map((row) => row.region));
  assert(regions.length === 3, `matrix must contain 3 regions; found ${regions.length}`);
  const policyByRegion = new Map(policyRows.map((row) => [row.region, row]));
  assert(
    policyByRegion.size === regions.length && regions.every((region) => policyByRegion.has(region)),
    "policy regions must match matrix regions",
  );

  const priorityByPeriod = new Map(
    priorityRows.map((row) => [row.construction_period, row]),
  );
  const periods = new Set(matrixRows.map((row) => row.construction_period));
  assert(
    priorityByPeriod.size === periods.size && [...periods].every((period) => priorityByPeriod.has(period)),
    "priority mapping must cover exactly the construction periods in the matrix",
  );

  for (const row of matrixRows) {
    numberValue(row.regional_archetype_share_within_region, `${keyFor(row)} regional share`, { min: 0 });
    numberValue(row.regional_number_of_dwellings, `${keyFor(row)} archetype stock`, { strictlyPositive: true });
    numberValue(row.regional_modelled_stock_dwellings, `${keyFor(row)} regional stock`, { strictlyPositive: true });
  }

  for (const row of layerRows) {
    numberValue(row.priority_weight, `${keyFor(row)} priority weight`, { strictlyPositive: true });
    numberValue(row.allocation_share_within_region, `${keyFor(row)} allocation share`, { min: 0 });
    numberValue(row.target_energy_score_2050_kWh_m2_year, `${keyFor(row)} target score`, { min: 0 });
    numberValue(row.regional_number_of_dwellings, `${keyFor(row)} layer stock`, { strictlyPositive: true });
    for (const scenario of SCENARIOS) {
      numberValue(row[`annual_renovation_rate_${scenario}`], `${keyFor(row)} ${scenario} rate`, { min: 0 });
    }

    const matrixRow = matrixByKey.get(keyFor(row));
    assert(row.dwelling_type === matrixRow.dwelling_type, `${keyFor(row)} dwelling_type differs between inputs`);
    assert(row.construction_period === matrixRow.construction_period, `${keyFor(row)} construction_period differs between inputs`);
    assert(
      nearlyEqual(Number(row.regional_number_of_dwellings), Number(matrixRow.regional_number_of_dwellings)),
      `${keyFor(row)} regional_number_of_dwellings differs between inputs`,
    );

    const priorityRow = priorityByPeriod.get(row.construction_period);
    assert(row.renovation_priority === priorityRow.renovation_priority, `${keyFor(row)} renovation priority differs from mapping`);
    assert(
      nearlyEqual(Number(row.priority_weight), Number(priorityRow.priority_weight)),
      `${keyFor(row)} priority weight differs from mapping`,
    );
  }

  for (const row of policyRows) {
    numberValue(row.target_energy_score_2050_kWh_m2_year, `${row.region} policy target score`, { min: 0 });
    for (const scenario of SCENARIOS) {
      numberValue(row[`annual_renovation_rate_${scenario}`], `${row.region} policy ${scenario} rate`, { min: 0 });
    }
  }

  const regionalStock = new Map();
  for (const region of regions) {
    const matrixRegion = matrixRows.filter((row) => row.region === region);
    const layerRegion = layerRows.filter((row) => row.region === region);
    assert(matrixRegion.length === 25, `${region} matrix must contain 25 archetypes`);
    assert(layerRegion.length === 25, `${region} layer must contain 25 archetypes`);

    const modelledValues = uniqueInOrder(
      matrixRegion.map((row) => Number(row.regional_modelled_stock_dwellings)),
    );
    assert(modelledValues.length === 1, `${region} must have one regional modeled stock value`);
    const modelled = modelledValues[0];
    const stockSum = matrixRegion.reduce(
      (sum, row) => sum + Number(row.regional_number_of_dwellings),
      0,
    );
    const shareSum = matrixRegion.reduce(
      (sum, row) => sum + Number(row.regional_archetype_share_within_region),
      0,
    );
    const allocationSum = layerRegion.reduce(
      (sum, row) => sum + Number(row.allocation_share_within_region),
      0,
    );
    assert(Math.abs(stockSum - modelled) <= DWELLING_TOLERANCE, `${region} archetype stock does not sum to modeled stock`);
    assert(Math.abs(shareSum - 1) <= SHARE_TOLERANCE, `${region} archetype shares do not sum to 1`);
    assert(Math.abs(allocationSum - 1) <= SHARE_TOLERANCE, `${region} allocation shares do not sum to 1`);
    regionalStock.set(region, modelled);

    const policyRow = policyByRegion.get(region);
    const labels = uniqueInOrder(layerRegion.map((row) => row.target_label_2050));
    const scores = uniqueInOrder(
      layerRegion.map((row) => Number(row.target_energy_score_2050_kWh_m2_year)),
    );
    assert(labels.length === 1 && labels[0] === policyRow.target_2050, `${region} target label differs from policy table`);
    assert(
      scores.length === 1 && nearlyEqual(scores[0], Number(policyRow.target_energy_score_2050_kWh_m2_year)),
      `${region} target score differs from policy table`,
    );
    for (const scenario of SCENARIOS) {
      const rates = uniqueInOrder(
        layerRegion.map((row) => Number(row[`annual_renovation_rate_${scenario}`])),
      );
      assert(rates.length === 1, `${region} has inconsistent ${scenario} rates in the layer`);
      assert(
        nearlyEqual(rates[0], Number(policyRow[`annual_renovation_rate_${scenario}`])),
        `${region} ${scenario} rate differs from policy table`,
      );
    }
  }

  return { matrixByKey, layerByKey, policyByRegion, regions, regionalStock };
}


function buildScenarioRows(matrixRows, layerByKey, regions, regionalStock) {
  const rows = [];
  const capsByGroup = new Map();

  for (const scenario of SCENARIOS) {
    const rateField = `annual_renovation_rate_${scenario}`;
    const regionalFlows = new Map();
    for (const region of regions) {
      const regionLayerRow = [...layerByKey.values()].find((row) => row.region === region);
      const rate = Number(regionLayerRow[rateField]);
      regionalFlows.set(region, rate * YEARS_ELAPSED * regionalStock.get(region));
      capsByGroup.set(groupKey(region, scenario), 0);
    }

    for (const matrixRow of matrixRows) {
      const layerRow = layerByKey.get(keyFor(matrixRow));
      const region = matrixRow.region;
      const modelled = regionalStock.get(region);
      const stock = Number(matrixRow.regional_number_of_dwellings);
      const allocation = Number(layerRow.allocation_share_within_region);
      const allocatedRenovated = regionalFlows.get(region) * allocation;
      const renovated = Math.min(allocatedRenovated, stock);
      const capKey = groupKey(region, scenario);
      if (allocatedRenovated > stock + VALUE_TOLERANCE) {
        capsByGroup.set(capKey, capsByGroup.get(capKey) + 1);
      }
      const asIs = stock - renovated;
      const renovatedFraction = renovated / stock;
      const rate = Number(layerRow[rateField]);

      const common = {
        scenario,
        target_year: TARGET_YEAR,
        region,
        archetype_id: matrixRow.archetype_id,
        dwelling_type: matrixRow.dwelling_type,
        construction_period: matrixRow.construction_period,
        regional_number_of_dwellings: stock,
        regional_modelled_stock_dwellings: modelled,
        allocation_share_within_region: allocation,
        renovated_fraction: renovatedFraction,
        annual_renovation_rate_scenario: rate,
        target_label_2050: layerRow.target_label_2050,
        target_energy_score_2050_kWh_m2_year: Number(
          layerRow.target_energy_score_2050_kWh_m2_year,
        ),
        policy_source: layerRow.policy_source,
      };

      rows.push({
        ...common,
        renovation_state: "as_is_TABULA",
        state_dwellings: asIs,
        state_share_within_region: asIs / modelled,
        ...Object.fromEntries(PACKAGE_COLUMNS.map((column) => [column, ""])),
      });
      rows.push({
        ...common,
        renovation_state: "renovated_TABULA_low_energy",
        state_dwellings: renovated,
        state_share_within_region: renovated / modelled,
        ...PACKAGE,
      });
    }
  }

  return { rows, capsByGroup };
}


function policyNote(region, scenario, renovatedFraction, capsBound) {
  const percent = `${(renovatedFraction * 100).toFixed(1)}%`;
  let interpretation;

  if (region === "Flemish Region") {
    if (scenario === "central") {
      interpretation =
        `The central scenario renovates about ${percent} by 2050. This is a strong ` +
        "trajectory but does not guarantee full label-A compliance unless an existing " +
        "renovated share is included or rates increase.";
    } else if (scenario === "low") {
      interpretation =
        `The low scenario renovates about ${percent} by 2050 and remains below the ` +
        "central policy-compatible trajectory.";
    } else {
      interpretation =
        `The high scenario renovates about ${percent} by 2050. Archetype caps bind; ` +
        "because capped excess is not redistributed, the nominal 100% regional flow " +
        "does not renovate the full stock.";
    }
  } else if (region === "Walloon Region") {
    if (scenario === "central") {
      interpretation =
        "The central scenario is a policy-compatible deep-renovation trajectory toward " +
        "average decarbonised PEB label A. The 3% rate is a scenario assumption unless " +
        "directly sourced as a fixed legal annual rate.";
    } else if (scenario === "low") {
      interpretation =
        `The low scenario renovates about ${percent} by 2050 and represents a ` +
        "conservative current-rate proxy below the central deep-renovation trajectory.";
    } else {
      interpretation =
        `The high scenario renovates about ${percent} by 2050. Archetype caps bind; ` +
        "capped excess is not redistributed, so the nominal 100% regional flow does " +
        "not renovate the full stock. The central 3% rate remains a scenario assumption.";
    }
  } else if (region === "Brussels-Capital Region") {
    if (scenario === "central") {
      interpretation =
        "The central scenario is a cautious policy-compatible trajectory toward average " +
        "PEB C+ / 100 kWh/m²/year. TABULA Low Energy is a physical deep-renovation " +
        "package and may be stronger than the average C+ target for many dwellings.";
    } else if (scenario === "low") {
      interpretation =
        `The low scenario renovates about ${percent} by 2050 and remains below the ` +
        "cautious central policy-compatible trajectory.";
    } else {
      interpretation =
        `The high scenario renovates about ${percent} by 2050. Archetype caps bind; ` +
        "capped excess is not redistributed, so the nominal 100% regional flow does " +
        "not renovate the full stock. TABULA Low Energy may be stronger than the " +
        "average C+ benchmark for many dwellings.";
    }
  } else {
    throw new Error(`No policy interpretation is defined for ${region}`);
  }

  if (scenario === "high" && !capsBound) {
    interpretation = interpretation.replace(/ Archetype caps bind;[^.]*\./, "");
  }
  return `${interpretation} ${POLICY_BENCHMARK_NOTE}`;
}


function buildCrosscheckRows(scenarioRows, regions, capsByGroup) {
  const rows = [];
  for (const region of regions) {
    for (const scenario of SCENARIOS) {
      const groupRows = scenarioRows.filter(
        (row) => row.region === region && row.scenario === scenario,
      );
      const renovatedRows = groupRows.filter(
        (row) => row.renovation_state === "renovated_TABULA_low_energy",
      );
      const renovated = renovatedRows.reduce(
        (sum, row) => sum + Number(row.state_dwellings),
        0,
      );
      const modelled = Number(groupRows[0].regional_modelled_stock_dwellings);
      const fraction = renovated / modelled;
      const capsBound = capsByGroup.get(groupKey(region, scenario)) > 0;
      const status =
        scenario === "low"
          ? "below_policy_compatible_trajectory"
          : scenario === "central"
            ? "policy_compatible_scenario"
            : capsBound
              ? "accelerated_scenario_with_archetype_caps"
              : "accelerated_scenario";

      rows.push({
        region,
        scenario,
        target_year: TARGET_YEAR,
        annual_renovation_rate_scenario: Number(
          groupRows[0].annual_renovation_rate_scenario,
        ),
        renovated_dwellings_region: renovated,
        regional_modelled_stock_dwellings: modelled,
        renovated_fraction_region: fraction,
        target_label_2050: groupRows[0].target_label_2050,
        target_energy_score_2050_kWh_m2_year: Number(
          groupRows[0].target_energy_score_2050_kWh_m2_year,
        ),
        model_physical_package: PACKAGE.renovation_package_id,
        cross_check_status: status,
        cross_check_note: policyNote(region, scenario, fraction, capsBound),
      });
    }
  }
  return rows;
}


function validatePackageRows(rows) {
  assert(rows.length === 1, `package output must contain 1 row; found ${rows.length}`);
  const row = rows[0];
  for (const column of PACKAGE_COLUMNS) {
    assert(String(row[column] ?? "") !== "", `package output has a blank ${column}`);
    if (typeof PACKAGE[column] === "number") {
      assert(
        nearlyEqual(Number(row[column]), PACKAGE[column]),
        `package output has an incorrect ${column}`,
      );
    } else if (typeof PACKAGE[column] === "boolean") {
      assert(
        booleanValue(row[column], `package output ${column}`) === PACKAGE[column],
        `package output has an incorrect ${column}`,
      );
    } else {
      assert(row[column] === PACKAGE[column], `package output has an incorrect ${column}`);
    }
  }
}


function validateScenarioRows(rows, regionalStock) {
  assert(rows.length === 450, `scenario output must contain 450 rows; found ${rows.length}`);
  requireUnique(
    rows,
    ["scenario", "region", "archetype_id", "renovation_state"],
    "scenario output",
  );

  const pairGroups = new Map();
  const regionGroups = new Map();
  for (const row of rows) {
    assert(SCENARIOS.includes(row.scenario), `unknown scenario ${row.scenario}`);
    assert(Number(row.target_year) === TARGET_YEAR, `incorrect target year for ${keyFor(row)}`);
    const pairKey = `${row.scenario}\u0000${row.region}\u0000${row.archetype_id}`;
    if (!pairGroups.has(pairKey)) pairGroups.set(pairKey, []);
    pairGroups.get(pairKey).push(row);
    const rKey = groupKey(row.region, row.scenario);
    if (!regionGroups.has(rKey)) regionGroups.set(rKey, []);
    regionGroups.get(rKey).push(row);
  }
  assert(pairGroups.size === 225, `scenario output must contain 225 archetype/scenario pairs`);

  for (const [pairKey, pair] of pairGroups) {
    assert(pair.length === 2, `${pairKey} must contain exactly two state rows`);
    const asIs = pair.find((row) => row.renovation_state === "as_is_TABULA");
    const renovated = pair.find(
      (row) => row.renovation_state === "renovated_TABULA_low_energy",
    );
    assert(asIs && renovated, `${pairKey} is missing a required state row`);
    const stock = Number(asIs.regional_number_of_dwellings);
    const modelled = Number(asIs.regional_modelled_stock_dwellings);
    const renovatedDwellings = Number(renovated.state_dwellings);
    const asIsDwellings = Number(asIs.state_dwellings);
    const archetypeFraction = renovatedDwellings / stock;
    assert(renovatedDwellings <= stock + VALUE_TOLERANCE, `${pairKey} renovation exceeds archetype stock`);
    assert(asIsDwellings >= -VALUE_TOLERANCE, `${pairKey} has negative as-is dwellings`);
    assert(
      Math.abs(asIsDwellings + renovatedDwellings - stock) <= DWELLING_TOLERANCE,
      `${pairKey} state dwellings do not sum to archetype stock`,
    );
    assert(
      nearlyEqual(Number(asIs.renovated_fraction), Number(renovated.renovated_fraction)),
      `${pairKey} must repeat the same archetype-level renovated_fraction on both rows`,
    );
    assert(
      nearlyEqual(Number(asIs.renovated_fraction), archetypeFraction),
      `${pairKey} renovated_fraction is not the archetype-level renovated fraction`,
    );
    assert(
      nearlyEqual(Number(asIs.state_share_within_region), asIsDwellings / modelled),
      `${pairKey} as-is state share is incorrect`,
    );
    assert(
      nearlyEqual(Number(renovated.state_share_within_region), renovatedDwellings / modelled),
      `${pairKey} renovated state share is incorrect`,
    );
    for (const column of PACKAGE_COLUMNS) {
      assert(String(asIs[column] ?? "") === "", `${pairKey} assigns ${column} to an as-is row`);
      if (typeof PACKAGE[column] === "number") {
        assert(nearlyEqual(Number(renovated[column]), PACKAGE[column]), `${pairKey} has an incorrect renovated ${column}`);
      } else if (typeof PACKAGE[column] === "boolean") {
        assert(
          booleanValue(renovated[column], `${pairKey} renovated ${column}`) === PACKAGE[column],
          `${pairKey} has an incorrect renovated ${column}`,
        );
      } else {
        assert(renovated[column] === PACKAGE[column], `${pairKey} has an incorrect renovated ${column}`);
      }
    }
  }

  assert(regionGroups.size === 9, `scenario output must contain 9 region/scenario groups`);
  for (const [rKey, group] of regionGroups) {
    assert(group.length === 50, `${rKey} must contain 50 state rows`);
    const [region] = rKey.split("\u0000");
    const shareSum = group.reduce(
      (sum, row) => sum + Number(row.state_share_within_region),
      0,
    );
    const dwellingSum = group.reduce(
      (sum, row) => sum + Number(row.state_dwellings),
      0,
    );
    assert(Math.abs(shareSum - 1) <= SHARE_TOLERANCE, `${rKey} state shares do not sum to 1`);
    assert(
      Math.abs(dwellingSum - regionalStock.get(region)) <= DWELLING_TOLERANCE,
      `${rKey} state dwellings do not sum to regional modeled stock`,
    );
  }
}


function validateCrosscheckRows(rows, scenarioRows, regionalStock) {
  assert(rows.length === 9, `cross-check output must contain 9 rows; found ${rows.length}`);
  requireUnique(rows, ["region", "scenario"], "cross-check output");
  for (const row of rows) {
    const fraction = Number(row.renovated_fraction_region);
    assert(fraction >= 0 && fraction <= 1, `${groupKey(row.region, row.scenario)} fraction is outside [0,1]`);
    assert(
      Math.abs(Number(row.regional_modelled_stock_dwellings) - regionalStock.get(row.region)) <= DWELLING_TOLERANCE,
      `${groupKey(row.region, row.scenario)} regional stock differs from scenario matrix`,
    );
    const scenarioRenovated = scenarioRows
      .filter(
        (scenarioRow) =>
          scenarioRow.region === row.region &&
          scenarioRow.scenario === row.scenario &&
          scenarioRow.renovation_state === "renovated_TABULA_low_energy",
      )
      .reduce((sum, scenarioRow) => sum + Number(scenarioRow.state_dwellings), 0);
    assert(
      Math.abs(Number(row.renovated_dwellings_region) - scenarioRenovated) <= DWELLING_TOLERANCE,
      `${groupKey(row.region, row.scenario)} renovated dwellings differ from scenario matrix`,
    );
    assert(
      nearlyEqual(fraction, scenarioRenovated / regionalStock.get(row.region)),
      `${groupKey(row.region, row.scenario)} renovated fraction is incorrect`,
    );
    assert(row.model_physical_package === PACKAGE.renovation_package_id, `${groupKey(row.region, row.scenario)} has an incorrect physical package`);
    assert(
      String(row.cross_check_note).includes(POLICY_BENCHMARK_NOTE),
      `${groupKey(row.region, row.scenario)} does not state the policy/physical distinction`,
    );
  }
}


function applyCurrentInputRegression(crosscheckRows, hashes) {
  const applies = Object.entries(CURRENT_INPUT_HASHES).every(
    ([name, expected]) => hashes[name] === expected,
  );
  if (!applies) return false;

  for (const [region, expected] of Object.entries(CURRENT_HIGH_FRACTIONS)) {
    const row = crosscheckRows.find(
      (candidate) => candidate.region === region && candidate.scenario === "high",
    );
    assert(row, `current-input regression is missing ${region} high scenario`);
    assert(
      Math.abs(Number(row.renovated_fraction_region) - expected) <= 1e-9,
      `${region} high-scenario regression changed: expected ${expected}, found ${row.renovated_fraction_region}`,
    );
  }
  return true;
}


function formatNumber(value) {
  assert(Number.isFinite(value), `cannot serialize non-finite number ${value}`);
  return CSV_NUMBER_FORMATTER.format(Math.abs(value) < 5e-13 ? 0 : value);
}


function csvCell(value) {
  const text =
    typeof value === "number" ? formatNumber(value) : String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}


function toCsv(columns, rows) {
  const lines = [columns.map(csvCell).join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvCell(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}


async function writeAndValidateTemporaryOutput(
  finalPath,
  columns,
  rows,
  sheetName,
  validator,
) {
  await fs.mkdir(path.dirname(finalPath), { recursive: true });
  const temporaryPath = `${finalPath}.tmp-${process.pid}`;
  await fs.writeFile(temporaryPath, toCsv(columns, rows), "utf8");
  const reread = await readCsv(temporaryPath, sheetName);
  assert(
    reread.headers.length === columns.length &&
      reread.headers.every((header, index) => header === columns[index]),
    `${path.basename(finalPath)} headers changed during serialization`,
  );
  validator(reread.rows);
  return temporaryPath;
}


function printSummary(packageRows, scenarioRows, crosscheckRows, regressionApplied) {
  console.log("\nRows created:");
  console.log(`  renovation_physical_package_TABULA_low_energy.csv: ${packageRows.length}`);
  console.log(`  archetype_matrix_2050_renovation_scenarios.csv: ${scenarioRows.length}`);
  console.log(`  renovation_policy_crosscheck_2050.csv: ${crosscheckRows.length}`);

  console.log("\nRegional renovation summary:");
  console.log("region | scenario | renovated_fraction | as_is_dwellings | renovated_dwellings");
  for (const crosscheck of crosscheckRows) {
    const group = scenarioRows.filter(
      (row) => row.region === crosscheck.region && row.scenario === crosscheck.scenario,
    );
    const asIs = group
      .filter((row) => row.renovation_state === "as_is_TABULA")
      .reduce((sum, row) => sum + Number(row.state_dwellings), 0);
    console.log(
      [
        crosscheck.region,
        crosscheck.scenario,
        `${(Number(crosscheck.renovated_fraction_region) * 100).toFixed(4)}%`,
        Number(asIs).toFixed(3),
        Number(crosscheck.renovated_dwellings_region).toFixed(3),
      ].join(" | "),
    );
  }

  console.log("\nValidation passed: YES");
  console.log(
    `Current-input high-scenario regression checks: ${regressionApplied ? "APPLIED AND PASSED" : "SKIPPED (input fingerprints changed)"}`,
  );
  console.log("\nAssumptions used:");
  console.log("  - Existing renovated stock is not added separately.");
  console.log(`  - Annual renovation rates are applied over ${YEARS_ELAPSED} years (${BASE_YEAR}-${TARGET_YEAR}).`);
  console.log("  - renovated_fraction is the archetype-level renovated fraction and is repeated on both state rows.");
  console.log("  - Capped excess is not redistributed; this is a conservative simplification and can leave a nominal 100% flow below full-stock renovation.");
  console.log("  - EPC/PEB targets are policy benchmarks; TABULA Low Energy defines the physical renovated state and is not treated as label-equivalent.");
  console.log("  - Source files and original TABULA as-is parameters are read-only and are not modified.");
}


async function main() {
  const hashesBefore = await inputHashes();
  const tables = Object.fromEntries(
    await Promise.all(
      Object.entries(INPUT_PATHS).map(async ([name, filePath]) => [
        name,
        await readCsv(filePath, name),
      ]),
    ),
  );
  const context = validateInputs(tables);

  const packageRows = [{ ...PACKAGE }];
  const scenarioResult = buildScenarioRows(
    tables.matrix.rows,
    context.layerByKey,
    context.regions,
    context.regionalStock,
  );
  const scenarioRows = scenarioResult.rows;
  const crosscheckRows = buildCrosscheckRows(
    scenarioRows,
    context.regions,
    scenarioResult.capsByGroup,
  );

  validatePackageRows(packageRows);
  validateScenarioRows(scenarioRows, context.regionalStock);
  validateCrosscheckRows(crosscheckRows, scenarioRows, context.regionalStock);
  const regressionApplied = applyCurrentInputRegression(crosscheckRows, hashesBefore);

  const temporaryOutputs = {};
  try {
    temporaryOutputs.package = await writeAndValidateTemporaryOutput(
      OUTPUT_PATHS.package,
      PACKAGE_COLUMNS,
      packageRows,
      "package_output",
      validatePackageRows,
    );
    temporaryOutputs.scenarios = await writeAndValidateTemporaryOutput(
      OUTPUT_PATHS.scenarios,
      SCENARIO_COLUMNS,
      scenarioRows,
      "scenario_output",
      (rows) => validateScenarioRows(rows, context.regionalStock),
    );
    temporaryOutputs.crosscheck = await writeAndValidateTemporaryOutput(
      OUTPUT_PATHS.crosscheck,
      CROSSCHECK_COLUMNS,
      crosscheckRows,
      "crosscheck_output",
      (rows) => validateCrosscheckRows(rows, scenarioRows, context.regionalStock),
    );

    const hashesAfter = await inputHashes();
    assert(
      Object.keys(hashesBefore).every((name) => hashesBefore[name] === hashesAfter[name]),
      "an input file changed during generation; outputs were not replaced",
    );

    for (const name of ["package", "scenarios", "crosscheck"]) {
      await fs.rename(temporaryOutputs[name], OUTPUT_PATHS[name]);
      delete temporaryOutputs[name];
    }
  } finally {
    await Promise.all(
      Object.values(temporaryOutputs).map((temporaryPath) =>
        fs.rm(temporaryPath, { force: true }),
      ),
    );
  }

  printSummary(packageRows, scenarioRows, crosscheckRows, regressionApplied);
}


main().catch((error) => {
  console.error(`Validation failed: ${error.message}`);
  process.exitCode = 1;
});
