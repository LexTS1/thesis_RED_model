import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, "..");

const INPUT_PATHS = {
  states: path.join(
    PROJECT_ROOT,
    "data/scenarios/renovation/archetype_matrix_2050_renovation_scenarios.csv",
  ),
  heatingShares: path.join(
    PROJECT_ROOT,
    "data/assumptions/technology/regional_heating_carrier_shares.csv",
  ),
  packages: path.join(
    PROJECT_ROOT,
    "data/assumptions/technology/technical_system_packages.csv",
  ),
  cooling: path.join(
    PROJECT_ROOT,
    "data/assumptions/technology/regional_cooling_adoption.csv",
  ),
  pv: path.join(
    PROJECT_ROOT,
    "data/assumptions/technology/regional_pv_projection_inputs.csv",
  ),
};

const OUTPUT_PATHS = {
  heating: path.join(
    PROJECT_ROOT,
    "data/scenarios/technology/archetype_heating_system_layer_2050.csv",
  ),
  cooling: path.join(
    PROJECT_ROOT,
    "data/scenarios/technology/archetype_cooling_adoption_layer_2050.csv",
  ),
  pv: path.join(
    PROJECT_ROOT,
    "data/scenarios/technology/archetype_pv_assignment_layer_2050.csv",
  ),
  pvAssignment: path.join(
    PROJECT_ROOT,
    "data/assumptions/technology/regional_pv_assignment.csv",
  ),
};

const OBSERVATION_YEAR = 2024;
const TARGET_YEAR = 2050;
const TRAJECTORY_TARGET_YEAR = 2030;
const REGIONS = [
  "Flemish Region",
  "Walloon Region",
  "Brussels-Capital Region",
];
const SCENARIOS = ["low", "central", "high"];
const RENOVATION_STATES = [
  "as_is_TABULA",
  "renovated_TABULA_low_energy",
];
const COOLING_CASES = [
  "no_active_cooling",
  "current_regional",
  "higher_uptake",
];
const SHARE_TOLERANCE = 1e-10;
const DWELLING_TOLERANCE = 1e-5;
const CAPACITY_TOLERANCE_MW = 1e-6;

const STATE_COLUMNS = [
  "scenario",
  "target_year",
  "region",
  "archetype_id",
  "dwelling_type",
  "construction_period",
  "renovation_state",
  "state_dwellings",
  "regional_modelled_stock_dwellings",
];

const HEATING_COLUMNS = [
  ...STATE_COLUMNS,
  "heating_variant_id",
  "variant_share",
  "variant_dwellings",
  "energy_carrier",
  "space_heating_system",
  "dhw_system",
  "heat_emitter",
  "supply_temperature_c",
  "ventilation_system",
  "hrv_eta",
  "summer_bypass",
  "reversible_cooling_capable",
  "source_id",
  "assumption_id",
];

const COOLING_COLUMNS = [
  ...STATE_COLUMNS,
  "cooling_case",
  "cooling_variant_id",
  "active_cooling",
  "cooling_system",
  "heating_system_reversible_cooling_capable",
  "variant_share",
  "variant_dwellings",
  "source_id",
  "assumption_id",
];

const PV_COLUMNS = [
  ...STATE_COLUMNS,
  "pv_basis_year",
  "dwelling_group",
  "pv_variant_id",
  "pv_installed",
  "house_participation_rate",
  "apartment_participation_rate",
  "variant_share",
  "variant_dwellings",
  "assigned_capacity_kwp_per_participating_dwelling",
  "allocated_capacity_mw",
  "regional_basis_capacity_mw",
  "raw_linear_capacity_mw_2050",
  "rooftop_potential_mw",
  "eligible_stock_capacity_mw",
  "capacity_cap_reason",
  "source_id",
  "assumption_id",
];

const PV_ASSIGNMENT_COLUMNS = [
  "region",
  "pv_basis_year",
  "dwelling_group",
  "participation_rate",
  "capacity_kwp_per_participating_dwelling",
  "regional_basis_capacity_mw",
  "target_capacity_mw_2030",
  "raw_linear_capacity_mw_2050",
  "rooftop_potential_mw",
  "eligible_stock_capacity_mw",
  "capacity_cap_reason",
  "source_id",
  "assumption_id",
];

const REQUIRED_COLUMNS = {
  states: STATE_COLUMNS,
  heatingShares: [
    "region",
    "heating_variant_id",
    "energy_carrier",
    "raw_share",
    "normalization_factor",
    "variant_share",
    "source_id",
    "assumption_id",
  ],
  packages: [
    "package_id",
    "applicable_renovation_state",
    "energy_carrier",
    "space_heating_system",
    "dhw_system",
    "heat_emitter",
    "supply_temperature_c",
    "ventilation_system",
    "hrv_eta",
    "summer_bypass",
    "reversible_cooling_capable",
    "source_id",
    "assumption_id",
  ],
  cooling: [
    "cooling_case",
    "region",
    "active_cooling_share",
    "source_id",
    "assumption_id",
  ],
  pv: [
    "region",
    "observation_year",
    "small_system_threshold_kw",
    "small_installations_2024",
    "small_capacity_mw_2024",
    "total_capacity_mw_2024",
    "target_energy_gwh_2030",
    "baseline_yield_kwh_per_kwp",
    "rooftop_potential_mw",
    "modeled_house_dwellings",
    "modeled_apartment_dwellings",
    "apartment_buildings_2025",
    "apartment_participation_ratio_to_house_2050",
    "threshold_supported",
    "source_id",
    "assumption_id",
    "pv_extract_method",
  ],
};

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
      "@oai/artifact-tool is required. Set CODEX_NODE_MODULES to the bundled runtime node_modules path.",
      { cause: error },
    );
  }
}


const { Workbook } = loadArtifactTool();


function assert(condition, message) {
  if (!condition) throw new Error(message);
}


function numberValue(value, label, { min = -Infinity, max = Infinity } = {}) {
  const number = Number(value);
  assert(Number.isFinite(number), `${label} must be finite; found ${value}`);
  assert(number >= min && number <= max, `${label} must be in [${min}, ${max}]; found ${number}`);
  return number;
}


function booleanValue(value, label) {
  if (value === true || String(value).toLowerCase() === "true") return true;
  if (value === false || String(value).toLowerCase() === "false") return false;
  throw new Error(`${label} must be true or false; found ${value}`);
}


function nearlyEqual(left, right, tolerance = SHARE_TOLERANCE) {
  return Math.abs(Number(left) - Number(right)) <= tolerance;
}


function requireColumns(headers, required, tableName) {
  const headerSet = new Set(headers);
  const missing = required.filter((column) => !headerSet.has(column));
  assert(missing.length === 0, `${tableName} is missing required columns: ${missing.join(", ")}`);
}


function requireUnique(rows, keyColumns, tableName) {
  const seen = new Set();
  for (const row of rows) {
    const key = keyColumns.map((column) => row[column]).join("\u0000");
    assert(!seen.has(key), `${tableName} has duplicate ${keyColumns.join("/")} key: ${key}`);
    seen.add(key);
  }
}


function stateKey(row) {
  return [row.scenario, row.region, row.archetype_id, row.renovation_state].join("\u0000");
}


function stateFields(row) {
  return Object.fromEntries(STATE_COLUMNS.map((column) => [column, row[column]]));
}


function addToGroup(map, key, row) {
  if (!map.has(key)) map.set(key, []);
  map.get(key).push(row);
}


function dwellingGroup(row) {
  return String(row.dwelling_type).startsWith("Apartment") ? "apartment" : "house";
}


async function readCsv(filePath, sheetName) {
  const text = await fs.readFile(filePath, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange(true).values;
  assert(values.length >= 1, `${filePath} is empty`);
  const headers = values[0].map((value) => String(value ?? "").trim());
  assert(headers.every(Boolean), `${filePath} has a blank header`);
  assert(new Set(headers).size === headers.length, `${filePath} has duplicate headers`);
  return {
    headers,
    rows: values.slice(1).map((valuesRow) =>
      Object.fromEntries(headers.map((header, index) => [header, valuesRow[index] ?? ""])),
    ),
  };
}


async function sha256(filePath) {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(bytes).digest("hex");
}


async function hashesFor(paths) {
  return Object.fromEntries(
    await Promise.all(Object.entries(paths).map(async ([name, filePath]) => [name, await sha256(filePath)])),
  );
}


function validateInputs(tables) {
  for (const [name, table] of Object.entries(tables)) {
    requireColumns(table.headers, REQUIRED_COLUMNS[name], name);
  }
  assert(!tables.states.headers.includes("heat_recovery_efficiency"), "obsolete heat_recovery_efficiency remains in renovation scenarios");
  assert(tables.states.headers.includes("hrv_eta"), "renovation scenarios are missing hrv_eta");
  assert(tables.states.headers.includes("summer_bypass"), "renovation scenarios are missing summer_bypass");

  const states = tables.states.rows;
  assert(states.length === 450, `renovation state input must contain 450 rows; found ${states.length}`);
  requireUnique(states, ["scenario", "region", "archetype_id", "renovation_state"], "renovation states");

  const regionalStock = new Map();
  const stateGroups = new Map();
  for (const row of states) {
    assert(SCENARIOS.includes(String(row.scenario)), `unknown scenario ${row.scenario}`);
    assert(REGIONS.includes(String(row.region)), `unknown region ${row.region}`);
    assert(RENOVATION_STATES.includes(String(row.renovation_state)), `unknown renovation state ${row.renovation_state}`);
    assert(Number(row.target_year) === TARGET_YEAR, `${stateKey(row)} has target year ${row.target_year}`);
    numberValue(row.state_dwellings, `${stateKey(row)} state_dwellings`, { min: 0 });
    const stock = numberValue(row.regional_modelled_stock_dwellings, `${stateKey(row)} regional stock`, { min: 0 });
    const rKey = `${row.scenario}\u0000${row.region}`;
    if (!regionalStock.has(rKey)) regionalStock.set(rKey, stock);
    assert(nearlyEqual(regionalStock.get(rKey), stock), `${rKey} has inconsistent regional stock`);
    addToGroup(stateGroups, `${row.scenario}\u0000${row.region}\u0000${row.archetype_id}`, row);
  }
  assert(stateGroups.size === 225, `expected 225 scenario-region-archetype groups; found ${stateGroups.size}`);
  for (const [key, group] of stateGroups) {
    assert(group.length === 2, `${key} must contain two renovation states`);
    assert(RENOVATION_STATES.every((state) => group.some((row) => row.renovation_state === state)), `${key} is missing a renovation state`);
  }
  for (const scenario of SCENARIOS) {
    for (const region of REGIONS) {
      const rows = states.filter((row) => row.scenario === scenario && row.region === region);
      assert(rows.length === 50, `${scenario}/${region} must contain 50 state rows`);
      const sum = rows.reduce((total, row) => total + Number(row.state_dwellings), 0);
      assert(Math.abs(sum - regionalStock.get(`${scenario}\u0000${region}`)) <= DWELLING_TOLERANCE, `${scenario}/${region} does not reconstruct regional modeled stock`);
    }
  }

  assert(tables.heatingShares.rows.length === 12, "heating-share input must contain 12 rows");
  requireUnique(tables.heatingShares.rows, ["region", "heating_variant_id"], "heating shares");
  const heatingSharesByRegion = new Map();
  for (const region of REGIONS) {
    const rows = tables.heatingShares.rows.filter((row) => row.region === region);
    assert(rows.length === 4, `${region} must contain four as-is heating shares`);
    for (const row of rows) {
      numberValue(row.raw_share, `${region}/${row.heating_variant_id} raw_share`, { min: 0, max: 1 });
      numberValue(row.normalization_factor, `${region}/${row.heating_variant_id} normalization_factor`, { min: 0 });
      numberValue(row.variant_share, `${region}/${row.heating_variant_id} variant_share`, { min: 0, max: 1 });
    }
    const rawSum = rows.reduce((sum, row) => sum + Number(row.raw_share), 0);
    const normalizedSum = rows.reduce((sum, row) => sum + Number(row.variant_share), 0);
    if (region === "Walloon Region") {
      assert(nearlyEqual(rawSum, 0.997), `Walloon raw heating shares must retain the published 0.997 total; found ${rawSum}`);
    } else {
      assert(nearlyEqual(rawSum, 1), `${region} raw heating shares must sum to one`);
    }
    assert(nearlyEqual(normalizedSum, 1), `${region} normalized heating shares must sum to one`);
    heatingSharesByRegion.set(region, rows);
  }

  assert(tables.packages.rows.length === 5, "technical-system packages must contain five rows");
  requireUnique(tables.packages.rows, ["package_id"], "technical-system packages");
  const packagesById = new Map(tables.packages.rows.map((row) => [row.package_id, row]));
  for (const row of tables.packages.rows) {
    assert(RENOVATION_STATES.includes(String(row.applicable_renovation_state)), `${row.package_id} has unknown renovation state`);
    assert(String(row.energy_carrier) !== "", `${row.package_id} has blank energy_carrier`);
    assert(String(row.space_heating_system) !== "", `${row.package_id} has blank space_heating_system`);
    booleanValue(row.summer_bypass, `${row.package_id} summer_bypass`);
    booleanValue(row.reversible_cooling_capable, `${row.package_id} reversible_cooling_capable`);
  }
  const renovatedPackage = packagesById.get("renovated_air_water_heat_pump");
  assert(renovatedPackage, "renovated heat-pump package is missing");
  assert(Number(renovatedPackage.supply_temperature_c) === 45, "renovated supply temperature must be 45 C");
  assert(nearlyEqual(renovatedPackage.hrv_eta, 0.8), "renovated hrv_eta must equal 0.80");
  assert(booleanValue(renovatedPackage.summer_bypass, "renovated summer_bypass"), "renovated package must have summer bypass");
  assert(booleanValue(renovatedPackage.reversible_cooling_capable, "renovated reversible capability"), "renovated package must be reversible-cooling capable");
  for (const rows of heatingSharesByRegion.values()) {
    for (const share of rows) {
      const packageRow = packagesById.get(share.heating_variant_id);
      assert(packageRow, `heating package ${share.heating_variant_id} is missing`);
      assert(packageRow.applicable_renovation_state === "as_is_TABULA", `${share.heating_variant_id} must apply to as-is rows`);
      assert(packageRow.energy_carrier === share.energy_carrier, `${share.heating_variant_id} carrier differs between inputs`);
    }
  }

  assert(tables.cooling.rows.length === 9, "cooling input must contain nine rows");
  requireUnique(tables.cooling.rows, ["cooling_case", "region"], "cooling input");
  const coolingByKey = new Map();
  for (const row of tables.cooling.rows) {
    assert(COOLING_CASES.includes(String(row.cooling_case)), `unknown cooling case ${row.cooling_case}`);
    assert(REGIONS.includes(String(row.region)), `unknown cooling region ${row.region}`);
    numberValue(row.active_cooling_share, `${row.cooling_case}/${row.region} active share`, { min: 0, max: 1 });
    coolingByKey.set(`${row.cooling_case}\u0000${row.region}`, row);
  }
  for (const coolingCase of COOLING_CASES) {
    for (const region of REGIONS) {
      assert(coolingByKey.has(`${coolingCase}\u0000${region}`), `cooling input is missing ${coolingCase}/${region}`);
    }
  }

  assert(tables.pv.rows.length === 3, "PV projection input must contain three rows");
  requireUnique(tables.pv.rows, ["region"], "PV projection input");
  const pvByRegion = new Map();
  for (const row of tables.pv.rows) {
    assert(REGIONS.includes(String(row.region)), `unknown PV region ${row.region}`);
    assert(Number(row.observation_year) === OBSERVATION_YEAR, `${row.region} PV observation year must be 2024`);
    assert(booleanValue(row.threshold_supported, `${row.region} threshold_supported`), `${row.region} PV extract does not support the <=10 kW/kVA threshold`);
    assert(Number(row.small_system_threshold_kw) === 10, `${row.region} PV threshold must equal 10`);
    for (const column of [
      "small_installations_2024",
      "small_capacity_mw_2024",
      "total_capacity_mw_2024",
      "target_energy_gwh_2030",
      "baseline_yield_kwh_per_kwp",
      "rooftop_potential_mw",
      "modeled_house_dwellings",
      "modeled_apartment_dwellings",
      "apartment_buildings_2025",
    ]) {
      numberValue(row[column], `${row.region} ${column}`, { min: Number.EPSILON });
    }
    numberValue(row.apartment_participation_ratio_to_house_2050, `${row.region} apartment participation ratio`, { min: 0, max: 1 });
    assert(String(row.pv_extract_method) !== "", `${row.region} has blank pv_extract_method`);
    pvByRegion.set(row.region, row);
  }

  const modeledByRegion = new Map();
  for (const region of REGIONS) {
    const stockValues = [...new Set(states.filter((row) => row.region === region).map((row) => Number(row.regional_modelled_stock_dwellings)))];
    assert(stockValues.length === 1, `${region} has inconsistent modeled stock across scenarios`);
    const pvRow = pvByRegion.get(region);
    const pvModeled = Number(pvRow.modeled_house_dwellings) + Number(pvRow.modeled_apartment_dwellings);
    assert(Math.abs(pvModeled - stockValues[0]) <= DWELLING_TOLERANCE, `${region} PV eligible stock differs from modeled R1-R4 stock`);
    modeledByRegion.set(region, stockValues[0]);
  }

  return {
    states,
    heatingSharesByRegion,
    packagesById,
    coolingByKey,
    pvByRegion,
    modeledByRegion,
  };
}


function buildHeatingRows(context) {
  const rows = [];
  for (const state of context.states) {
    const variants = state.renovation_state === "as_is_TABULA"
      ? context.heatingSharesByRegion.get(state.region).map((share) => ({
          share: Number(share.variant_share),
          shareRow: share,
          packageRow: context.packagesById.get(share.heating_variant_id),
        }))
      : [{
          share: 1,
          shareRow: null,
          packageRow: context.packagesById.get("renovated_air_water_heat_pump"),
        }];
    for (const variant of variants) {
      const packageRow = variant.packageRow;
      rows.push({
        ...stateFields(state),
        heating_variant_id: packageRow.package_id,
        variant_share: variant.share,
        variant_dwellings: Number(state.state_dwellings) * variant.share,
        energy_carrier: packageRow.energy_carrier,
        space_heating_system: packageRow.space_heating_system,
        dhw_system: packageRow.dhw_system,
        heat_emitter: packageRow.heat_emitter,
        supply_temperature_c: packageRow.supply_temperature_c,
        ventilation_system: packageRow.ventilation_system,
        hrv_eta: packageRow.hrv_eta,
        summer_bypass: booleanValue(packageRow.summer_bypass, `${packageRow.package_id} summer_bypass`),
        reversible_cooling_capable: booleanValue(packageRow.reversible_cooling_capable, `${packageRow.package_id} reversible capability`),
        source_id: variant.shareRow?.source_id ?? packageRow.source_id,
        assumption_id: variant.shareRow
          ? `${variant.shareRow.assumption_id}+${packageRow.assumption_id}`
          : packageRow.assumption_id,
      });
    }
  }
  return rows;
}


function buildCoolingRows(context) {
  const rows = [];
  for (const state of context.states) {
    const reversible = state.renovation_state === "renovated_TABULA_low_energy";
    for (const coolingCase of COOLING_CASES) {
      const input = context.coolingByKey.get(`${coolingCase}\u0000${state.region}`);
      const activeShare = Number(input.active_cooling_share);
      for (const active of [false, true]) {
        const share = active ? activeShare : 1 - activeShare;
        rows.push({
          ...stateFields(state),
          cooling_case: coolingCase,
          cooling_variant_id: active ? "active_cooling" : "inactive_cooling",
          active_cooling: active,
          cooling_system: active ? "integrated_reversible_or_split_system_unspecified" : "none_active",
          heating_system_reversible_cooling_capable: reversible,
          variant_share: share,
          variant_dwellings: Number(state.state_dwellings) * share,
          source_id: input.source_id,
          assumption_id: input.assumption_id,
        });
      }
    }
  }
  return rows;
}


function pvProjection(input) {
  const smallInstallations = Number(input.small_installations_2024);
  const smallCapacity = Number(input.small_capacity_mw_2024);
  const totalCapacity = Number(input.total_capacity_mw_2024);
  const houseDwellings = Number(input.modeled_house_dwellings);
  const apartmentDwellings = Number(input.modeled_apartment_dwellings);
  const apartmentBuildings = Number(input.apartment_buildings_2025);
  const apartmentRatio = Number(input.apartment_participation_ratio_to_house_2050);
  const meanSystemKwp = smallCapacity * 1000 / smallInstallations;
  const averageApartmentsPerBuilding = apartmentDwellings / apartmentBuildings;
  const currentHouseRate = smallInstallations / houseDwellings;
  assert(currentHouseRate <= 1 + SHARE_TOLERANCE, `${input.region} current PV installations exceed modeled houses`);
  const targetCapacity2030 = Number(input.target_energy_gwh_2030) * 1000 / Number(input.baseline_yield_kwh_per_kwp);
  const smallCapacityShare = smallCapacity / totalCapacity;
  const targetSmallCapacity2030 = Math.max(
    smallCapacity,
    smallCapacity + smallCapacityShare * (targetCapacity2030 - totalCapacity),
  );
  const rawLinearCapacity2050 = smallCapacity +
    (TARGET_YEAR - OBSERVATION_YEAR) *
      (targetSmallCapacity2030 - smallCapacity) /
      (TRAJECTORY_TARGET_YEAR - OBSERVATION_YEAR);
  const apartmentCapacityPerDwelling = meanSystemKwp / averageApartmentsPerBuilding;
  const eligibleStockCapacity = (
    meanSystemKwp * houseDwellings +
    apartmentRatio * apartmentCapacityPerDwelling * apartmentDwellings
  ) / 1000;
  const rooftopPotential = Number(input.rooftop_potential_mw);
  const projectedCapacity = Math.min(rawLinearCapacity2050, rooftopPotential, eligibleStockCapacity);
  const capReasons = [];
  if (nearlyEqual(projectedCapacity, rawLinearCapacity2050, CAPACITY_TOLERANCE_MW)) capReasons.push("linear_trajectory");
  if (nearlyEqual(projectedCapacity, rooftopPotential, CAPACITY_TOLERANCE_MW)) capReasons.push("official_rooftop_potential");
  if (nearlyEqual(projectedCapacity, eligibleStockCapacity, CAPACITY_TOLERANCE_MW)) capReasons.push("eligible_modeled_stock");
  const projectedHouseRate = projectedCapacity * 1000 /
    (meanSystemKwp * houseDwellings + apartmentRatio * apartmentCapacityPerDwelling * apartmentDwellings);
  const projectedApartmentRate = apartmentRatio * projectedHouseRate;
  assert(projectedHouseRate >= 0 && projectedHouseRate <= 1 + SHARE_TOLERANCE, `${input.region} projected house PV rate is outside [0,1]`);
  assert(projectedApartmentRate >= 0 && projectedApartmentRate <= 1 + SHARE_TOLERANCE, `${input.region} projected apartment PV rate is outside [0,1]`);

  return {
    meanSystemKwp,
    apartmentCapacityPerDwelling,
    current: {
      basisYear: OBSERVATION_YEAR,
      houseRate: currentHouseRate,
      apartmentRate: 0,
      regionalCapacity: smallCapacity,
      targetCapacity2030,
      rawLinearCapacity2050,
      rooftopPotential,
      eligibleStockCapacity,
      capReason: "observed_2024_small_system_stock",
    },
    projected: {
      basisYear: TARGET_YEAR,
      houseRate: projectedHouseRate,
      apartmentRate: projectedApartmentRate,
      regionalCapacity: projectedCapacity,
      targetCapacity2030,
      rawLinearCapacity2050,
      rooftopPotential,
      eligibleStockCapacity,
      capReason: capReasons.join("+"),
    },
  };
}


function buildPvAssignments(context) {
  const assignments = new Map();
  const rows = [];
  for (const region of REGIONS) {
    const input = context.pvByRegion.get(region);
    const projection = pvProjection(input);
    assignments.set(region, projection);
    for (const [basisName, basis] of [["current", projection.current], ["projected", projection.projected]]) {
      for (const group of ["house", "apartment"]) {
        const participationRate = group === "house" ? basis.houseRate : basis.apartmentRate;
        const capacityPerDwelling = participationRate === 0
          ? 0
          : group === "house"
            ? projection.meanSystemKwp
            : projection.apartmentCapacityPerDwelling;
        rows.push({
          region,
          pv_basis_year: basis.basisYear,
          dwelling_group: group,
          participation_rate: participationRate,
          capacity_kwp_per_participating_dwelling: capacityPerDwelling,
          regional_basis_capacity_mw: basis.regionalCapacity,
          target_capacity_mw_2030: basis.targetCapacity2030,
          raw_linear_capacity_mw_2050: basis.rawLinearCapacity2050,
          rooftop_potential_mw: basis.rooftopPotential,
          eligible_stock_capacity_mw: basis.eligibleStockCapacity,
          capacity_cap_reason: basis.capReason,
          source_id: input.source_id,
          assumption_id: input.assumption_id,
        });
      }
    }
  }
  return { assignments, rows };
}


function buildPvRows(context, assignments) {
  const rows = [];
  for (const state of context.states) {
    const group = dwellingGroup(state);
    const projection = assignments.get(state.region);
    const basis = state.renovation_state === "as_is_TABULA" ? projection.current : projection.projected;
    const participationRate = group === "house" ? basis.houseRate : basis.apartmentRate;
    const installedCapacity = participationRate === 0
      ? 0
      : group === "house"
        ? projection.meanSystemKwp
        : projection.apartmentCapacityPerDwelling;
    const input = context.pvByRegion.get(state.region);
    for (const installed of [false, true]) {
      const share = installed ? participationRate : 1 - participationRate;
      const capacityPerDwelling = installed ? installedCapacity : 0;
      const variantDwellings = Number(state.state_dwellings) * share;
      rows.push({
        ...stateFields(state),
        pv_basis_year: basis.basisYear,
        dwelling_group: group,
        pv_variant_id: installed ? "pv_installed" : "no_pv",
        pv_installed: installed,
        house_participation_rate: basis.houseRate,
        apartment_participation_rate: basis.apartmentRate,
        variant_share: share,
        variant_dwellings: variantDwellings,
        assigned_capacity_kwp_per_participating_dwelling: capacityPerDwelling,
        allocated_capacity_mw: variantDwellings * capacityPerDwelling / 1000,
        regional_basis_capacity_mw: basis.regionalCapacity,
        raw_linear_capacity_mw_2050: basis.rawLinearCapacity2050,
        rooftop_potential_mw: basis.rooftopPotential,
        eligible_stock_capacity_mw: basis.eligibleStockCapacity,
        capacity_cap_reason: basis.capReason,
        source_id: input.source_id,
        assumption_id: input.assumption_id,
      });
    }
  }
  return rows;
}


function validateVariantGroups(rows, extraKeyColumns, tableName) {
  const groups = new Map();
  for (const row of rows) {
    const key = [stateKey(row), ...extraKeyColumns.map((column) => row[column])].join("\u0000");
    addToGroup(groups, key, row);
    numberValue(row.variant_share, `${tableName} ${key} variant_share`, { min: 0, max: 1 });
    numberValue(row.variant_dwellings, `${tableName} ${key} variant_dwellings`, { min: 0 });
    assert(Math.abs(Number(row.variant_dwellings) - Number(row.state_dwellings) * Number(row.variant_share)) <= DWELLING_TOLERANCE, `${tableName} ${key} violates variant_dwellings = state_dwellings x variant_share`);
    assert(String(row.source_id) !== "", `${tableName} ${key} has blank source_id`);
    assert(String(row.assumption_id) !== "", `${tableName} ${key} has blank assumption_id`);
  }
  for (const [key, group] of groups) {
    const shareSum = group.reduce((sum, row) => sum + Number(row.variant_share), 0);
    const dwellingSum = group.reduce((sum, row) => sum + Number(row.variant_dwellings), 0);
    assert(nearlyEqual(shareSum, 1), `${tableName} ${key} shares sum to ${shareSum}`);
    assert(Math.abs(dwellingSum - Number(group[0].state_dwellings)) <= DWELLING_TOLERANCE, `${tableName} ${key} does not reconstruct state_dwellings`);
  }
  return groups;
}


function validateHeatingRows(rows, context) {
  assert(rows.length === 1125, `heating output must contain 1125 rows; found ${rows.length}`);
  requireUnique(rows, ["scenario", "region", "archetype_id", "renovation_state", "heating_variant_id"], "heating output");
  const groups = validateVariantGroups(rows, [], "heating output");
  assert(groups.size === 450, `heating output must contain 450 state groups; found ${groups.size}`);
  for (const [key, group] of groups) {
    const asIs = group[0].renovation_state === "as_is_TABULA";
    assert(group.length === (asIs ? 4 : 1), `${key} must contain ${asIs ? 4 : 1} heating variants`);
    for (const row of group) {
      booleanValue(row.summer_bypass, `${key} summer_bypass`);
      booleanValue(row.reversible_cooling_capable, `${key} reversible_cooling_capable`);
      assert(!Object.hasOwn(row, "heat_recovery_efficiency"), `${key} contains obsolete heat_recovery_efficiency`);
    }
  }
  for (const scenario of SCENARIOS) {
    for (const region of REGIONS) {
      const sum = rows.filter((row) => row.scenario === scenario && row.region === region)
        .reduce((total, row) => total + Number(row.variant_dwellings), 0);
      assert(Math.abs(sum - context.modeledByRegion.get(region)) <= DWELLING_TOLERANCE, `heating ${scenario}/${region} does not preserve regional stock`);
    }
  }
}


function validateCoolingRows(rows, context) {
  assert(rows.length === 2700, `cooling output must contain 2700 rows; found ${rows.length}`);
  requireUnique(rows, ["scenario", "region", "archetype_id", "renovation_state", "cooling_case", "cooling_variant_id"], "cooling output");
  const groups = validateVariantGroups(rows, ["cooling_case"], "cooling output");
  assert(groups.size === 1350, `cooling output must contain 1350 state/case groups; found ${groups.size}`);
  for (const [key, group] of groups) {
    assert(group.length === 2, `${key} must contain active and inactive variants`);
    const active = group.find((row) => booleanValue(row.active_cooling, `${key} active_cooling`));
    const inactive = group.find((row) => !booleanValue(row.active_cooling, `${key} active_cooling`));
    assert(active && inactive, `${key} is missing active or inactive cooling`);
    for (const row of group) {
      const capable = booleanValue(row.heating_system_reversible_cooling_capable, `${key} heat-pump capability`);
      assert(capable === (row.renovation_state === "renovated_TABULA_low_energy"), `${key} confuses installed cooling with renovated heat-pump capability`);
    }
  }
  for (const scenario of SCENARIOS) {
    for (const region of REGIONS) {
      for (const coolingCase of COOLING_CASES) {
        const sum = rows.filter((row) => row.scenario === scenario && row.region === region && row.cooling_case === coolingCase)
          .reduce((total, row) => total + Number(row.variant_dwellings), 0);
        assert(Math.abs(sum - context.modeledByRegion.get(region)) <= DWELLING_TOLERANCE, `cooling ${scenario}/${region}/${coolingCase} does not preserve regional stock`);
      }
    }
  }
}


function validatePvAssignmentRows(rows, context) {
  assert(rows.length === 12, `PV assignment output must contain 12 rows; found ${rows.length}`);
  requireUnique(rows, ["region", "pv_basis_year", "dwelling_group"], "PV assignment output");
  for (const region of REGIONS) {
    const currentHouse = rows.find((row) => row.region === region && Number(row.pv_basis_year) === OBSERVATION_YEAR && row.dwelling_group === "house");
    const currentApartment = rows.find((row) => row.region === region && Number(row.pv_basis_year) === OBSERVATION_YEAR && row.dwelling_group === "apartment");
    const projectedHouse = rows.find((row) => row.region === region && Number(row.pv_basis_year) === TARGET_YEAR && row.dwelling_group === "house");
    const projectedApartment = rows.find((row) => row.region === region && Number(row.pv_basis_year) === TARGET_YEAR && row.dwelling_group === "apartment");
    assert(currentHouse && currentApartment && projectedHouse && projectedApartment, `${region} PV assignment rows are incomplete`);
    assert(nearlyEqual(currentApartment.participation_rate, 0), `${region} current apartment PV must be zero`);
    const ratio = Number(context.pvByRegion.get(region).apartment_participation_ratio_to_house_2050);
    assert(nearlyEqual(projectedApartment.participation_rate, ratio * Number(projectedHouse.participation_rate)), `${region} projected apartment participation must equal 25% of the house rate`);
    for (const row of [currentHouse, currentApartment, projectedHouse, projectedApartment]) {
      numberValue(row.participation_rate, `${region}/${row.pv_basis_year}/${row.dwelling_group} participation_rate`, { min: 0, max: 1 });
    }
    const input = context.pvByRegion.get(region);
    const currentCapacity = Number(currentHouse.participation_rate) * Number(input.modeled_house_dwellings) * Number(currentHouse.capacity_kwp_per_participating_dwelling) / 1000;
    assert(Math.abs(currentCapacity - Number(currentHouse.regional_basis_capacity_mw)) <= CAPACITY_TOLERANCE_MW, `${region} current PV assignment does not reconcile with regional small capacity`);
    const projectedCapacity = (
      Number(projectedHouse.participation_rate) * Number(input.modeled_house_dwellings) * Number(projectedHouse.capacity_kwp_per_participating_dwelling) +
      Number(projectedApartment.participation_rate) * Number(input.modeled_apartment_dwellings) * Number(projectedApartment.capacity_kwp_per_participating_dwelling)
    ) / 1000;
    assert(Math.abs(projectedCapacity - Number(projectedHouse.regional_basis_capacity_mw)) <= CAPACITY_TOLERANCE_MW, `${region} projected PV assignment does not reconcile with capped trajectory`);
  }
}


function validatePvRows(rows, context) {
  assert(rows.length === 900, `PV output must contain 900 rows; found ${rows.length}`);
  requireUnique(rows, ["scenario", "region", "archetype_id", "renovation_state", "pv_variant_id"], "PV output");
  const groups = validateVariantGroups(rows, [], "PV output");
  assert(groups.size === 450, `PV output must contain 450 state groups; found ${groups.size}`);
  for (const [key, group] of groups) {
    assert(group.length === 2, `${key} must contain PV and no-PV variants`);
    const installed = group.find((row) => booleanValue(row.pv_installed, `${key} pv_installed`));
    const absent = group.find((row) => !booleanValue(row.pv_installed, `${key} pv_installed`));
    assert(installed && absent, `${key} is missing a PV variant`);
    assert(nearlyEqual(absent.allocated_capacity_mw, 0), `${key} no-PV row carries capacity`);
    assert(nearlyEqual(absent.assigned_capacity_kwp_per_participating_dwelling, 0), `${key} no-PV row carries per-dwelling capacity`);
    const expectedBasis = installed.renovation_state === "as_is_TABULA" ? OBSERVATION_YEAR : TARGET_YEAR;
    assert(Number(installed.pv_basis_year) === expectedBasis, `${key} has wrong PV basis year`);
    if (installed.dwelling_group === "apartment" && expectedBasis === OBSERVATION_YEAR) {
      assert(nearlyEqual(installed.variant_share, 0), `${key} assigns current PV to apartments`);
      assert(nearlyEqual(installed.variant_dwellings, 0), `${key} current apartment PV variant must be retained with zero dwellings`);
      assert(nearlyEqual(installed.assigned_capacity_kwp_per_participating_dwelling, 0), `${key} current apartment PV variant must carry zero assigned capacity`);
    }
    const expectedCapacity = Number(installed.variant_dwellings) * Number(installed.assigned_capacity_kwp_per_participating_dwelling) / 1000;
    assert(Math.abs(expectedCapacity - Number(installed.allocated_capacity_mw)) <= CAPACITY_TOLERANCE_MW, `${key} allocated PV capacity is inconsistent`);
  }
  for (const scenario of SCENARIOS) {
    for (const region of REGIONS) {
      const group = rows.filter((row) => row.scenario === scenario && row.region === region);
      const dwellingSum = group.reduce((sum, row) => sum + Number(row.variant_dwellings), 0);
      assert(Math.abs(dwellingSum - context.modeledByRegion.get(region)) <= DWELLING_TOLERANCE, `PV ${scenario}/${region} does not preserve regional stock`);
      const installedCapacity = group.reduce((sum, row) => sum + Number(row.allocated_capacity_mw), 0);
      const expectedCapacity = context.states
        .filter((row) => row.scenario === scenario && row.region === region)
        .reduce((sum, state) => {
          const installed = group.find((row) => stateKey(row) === stateKey(state) && booleanValue(row.pv_installed, "pv_installed"));
          return sum + Number(installed.variant_dwellings) * Number(installed.assigned_capacity_kwp_per_participating_dwelling) / 1000;
        }, 0);
      assert(Math.abs(installedCapacity - expectedCapacity) <= CAPACITY_TOLERANCE_MW, `PV ${scenario}/${region} capacity does not reconcile with state assignments`);
    }
  }
}


function formatNumber(value) {
  assert(Number.isFinite(value), `cannot serialize non-finite number ${value}`);
  return CSV_NUMBER_FORMATTER.format(Math.abs(value) < 5e-13 ? 0 : value);
}


function csvCell(value) {
  const text = typeof value === "number" ? formatNumber(value) : String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}


function toCsv(columns, rows) {
  return `${[columns.map(csvCell).join(","), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))].join("\n")}\n`;
}


async function writeTemporary(finalPath, columns, rows, sheetName, validator) {
  await fs.mkdir(path.dirname(finalPath), { recursive: true });
  const temporaryPath = `${finalPath}.tmp-${process.pid}`;
  await fs.writeFile(temporaryPath, toCsv(columns, rows), "utf8");
  const reread = await readCsv(temporaryPath, sheetName);
  assert(reread.headers.length === columns.length && reread.headers.every((header, index) => header === columns[index]), `${path.basename(finalPath)} headers changed during serialization`);
  validator(reread.rows);
  return temporaryPath;
}


async function main() {
  const hashesBefore = await hashesFor(INPUT_PATHS);
  const tables = Object.fromEntries(
    await Promise.all(Object.entries(INPUT_PATHS).map(async ([name, filePath]) => [name, await readCsv(filePath, name)])),
  );
  const context = validateInputs(tables);
  const heatingRows = buildHeatingRows(context);
  const coolingRows = buildCoolingRows(context);
  const pvAssignment = buildPvAssignments(context);
  const pvRows = buildPvRows(context, pvAssignment.assignments);

  validateHeatingRows(heatingRows, context);
  validateCoolingRows(coolingRows, context);
  validatePvAssignmentRows(pvAssignment.rows, context);
  validatePvRows(pvRows, context);

  const temporary = {};
  try {
    temporary.heating = await writeTemporary(OUTPUT_PATHS.heating, HEATING_COLUMNS, heatingRows, "heating_output", (rows) => validateHeatingRows(rows, context));
    temporary.cooling = await writeTemporary(OUTPUT_PATHS.cooling, COOLING_COLUMNS, coolingRows, "cooling_output", (rows) => validateCoolingRows(rows, context));
    temporary.pv = await writeTemporary(OUTPUT_PATHS.pv, PV_COLUMNS, pvRows, "pv_output", (rows) => validatePvRows(rows, context));
    temporary.pvAssignment = await writeTemporary(OUTPUT_PATHS.pvAssignment, PV_ASSIGNMENT_COLUMNS, pvAssignment.rows, "pv_assignment_output", (rows) => validatePvAssignmentRows(rows, context));

    const hashesAfter = await hashesFor(INPUT_PATHS);
    assert(Object.keys(hashesBefore).every((name) => hashesBefore[name] === hashesAfter[name]), "an input changed during generation; outputs were not replaced");
    for (const name of ["heating", "cooling", "pv", "pvAssignment"]) {
      await fs.rename(temporary[name], OUTPUT_PATHS[name]);
      delete temporary[name];
    }
  } finally {
    await Promise.all(Object.values(temporary).map((filePath) => fs.rm(filePath, { force: true })));
  }

  const outputHashes = await hashesFor(OUTPUT_PATHS);
  console.log("Technology layer generated and validated.");
  console.log(`Heating rows: ${heatingRows.length}`);
  console.log(`Cooling rows: ${coolingRows.length}`);
  console.log(`PV rows: ${pvRows.length}`);
  console.log(`PV assignment rows: ${pvAssignment.rows.length}`);
  for (const [name, hash] of Object.entries(outputHashes)) console.log(`${name} sha256: ${hash}`);
}


main().catch((error) => {
  console.error(`Validation failed: ${error.message}`);
  process.exitCode = 1;
});
