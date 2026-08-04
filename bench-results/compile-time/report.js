// Compile-time benchmark report: the data pipeline and the Alpine shell.
//
// The pipeline is pure functions -- (runs, filters) -> a table model or
// an ECharts option -- plus small statistics helpers, with no DOM or
// framework dependency, so they are the testable core and the framework
// is swappable. Alpine is a thin shell (state + calling setOption);
// ECharts draws the charts; the headline regression grid is plain HTML.
//
// The report is a DASHBOARD, not a set of filtered single charts: the
// question that dominates ("what regressed, everywhere") wants program
// and metric shown as facets at once, not paged through one at a time.
// So the only global filters are coarse -- which run, which baseline,
// which optimisation level -- and everything else is laid out in full:
//   1. regression grid: programs x metrics, delta vs baseline (HTML)
//   2. metric trends: geomean-over-programs per metric, across runs
//   3. phase breakdown: where our time goes, all programs, one level
//   4. vs clang: ours/clang ratio per program (the competitive gap)
//
// The record shape (one run per runs/<...>.json, see src/bench/record.rs)
// is documented as JSDoc @typedefs below. Metrics are RAW SAMPLE VECTORS;
// every median / delta / significance figure is derived here, never read
// pre-aggregated.

'use strict';

/**
 * @typedef {Object} Run  One benchmark invocation.
 * @property {number} schema_version
 * @property {RunMeta} meta
 * @property {Program[]} programs
 */
/**
 * @typedef {Object} RunMeta
 * @property {number} timestamp_unix
 * @property {string} git_sha
 * @property {boolean} git_dirty
 * @property {string} hostname
 * @property {string} cpu_model
 * @property {string|null} governor
 * @property {string} clang_version
 * @property {number} iters
 * @property {number} warmup
 * @property {boolean} rss_available
 */
/**
 * @typedef {Object} Program
 * @property {string} name
 * @property {number} values  Post-construction graph size.
 * @property {boolean} verified
 * @property {EmittedIr|null} emitted_ir
 * @property {Config[]} configs
 */
/**
 * @typedef {Object} Config  One {compiler, level} cell of the matrix.
 * @property {"Ours"|"Clang"} compiler
 * @property {string} level  "o0" | "o2" | "o3".
 * @property {"Measured"|"Failed"|"TimedOut"} [status]  Absent in v1 records.
 * @property {string|null} [error]  stderr tail when not measured.
 * @property {MetricSamples} end_to_end
 * @property {number|null} peak_rss_bytes
 * @property {number|null} object_size_bytes
 * @property {number|null} cachegrind_ir  Deterministic; absent in schema < 3 for cache/cycles.
 * @property {number|null} [cachegrind_ll_misses]
 * @property {number|null} [cachegrind_total_accesses]  Ir+Dr+Dw; miss-rate denominator (schema >= 4).
 * @property {number|null} [cachegrind_estimated_cycles]
 * @property {PhaseRecord[]} phases  Empty for clang.
 * @property {PassRecord[]} passes  Empty for clang.
 */
/**
 * @typedef {Object} MetricSamples  Raw per-iteration vectors; counters are
 *   null when unmeasured.
 * @property {number[]} wall_ms
 * @property {number[]|null} cycles
 * @property {number[]|null} instructions
 * @property {number[]|null} cache_misses
 * @property {number[]|null} cache_references
 * @property {number[]|null} [allocations]  Rust-heap allocator calls (schema >= 5).
 * @property {number[]|null} [alloc_bytes]  Bytes handed out, cumulative churn (schema >= 5).
 */
/**
 * @typedef {Object} PhaseRecord
 * @property {string} phase
 * @property {MetricSamples} samples
 */
/**
 * @typedef {Object} PassRecord
 * @property {string} name
 * @property {number[]} wall_ms
 */
/**
 * @typedef {Object} EmittedIr
 * @property {number} functions
 * @property {number} basic_blocks
 * @property {number} instructions
 * @property {number} phis
 */
/**
 * @typedef {Object} Metric  A viewable metric (an entry in METRICS).
 * @property {string} label
 * @property {string} unit
 * @property {(c: Config) => (number[]|null)} samples  Raw vector, or null when point-valued.
 * @property {(c: Config) => (number|null)} point  Single value from a config.
 */
/**
 * @typedef {Object} Cell  One (program, metric) cell of the regression grid.
 * @property {number|null} deltaPct
 * @property {number|null} curV
 * @property {number|null} baseV
 * @property {boolean|null} sig  Significant (wall only); null when not tested.
 * @property {boolean} has  Both sides present and comparable.
 * @property {string|false} failed  "fail"/"timeout" label, or false.
 */
/**
 * @typedef {Object} Row  One regression-grid row.
 * @property {string} name
 * @property {Object<string, Cell>} cells  Keyed by metric key.
 * @property {string|null} failStatus  "Failed"/"TimedOut" if the ours config failed.
 */
/**
 * @typedef {Object} Failure  A non-measured config, for the failures panel.
 * @property {string} program
 * @property {string} config
 * @property {string} status
 * @property {string} error
 */

// -- statistics ------------------------------------------------------------

/**
 * @param {number[]} xs
 * @returns {number[]} ascending copy (input untouched)
 */
function sorted(xs) {
  return xs.slice().sort((a, b) => a - b);
}

/**
 * @param {number[]|null|undefined} xs
 * @returns {number|null} the median, or null for an empty/absent input
 */
function median(xs) {
  if (!xs || xs.length === 0) return null;
  const s = sorted(xs);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * Linear-interpolated quantile, used for the bootstrap CI bounds.
 * @param {number[]|null|undefined} xs
 * @param {number} q  quantile in [0, 1]
 * @returns {number|null}
 */
function quantile(xs, q) {
  if (!xs || xs.length === 0) return null;
  const s = sorted(xs);
  if (s.length === 1) return s[0];
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  // q == 1 lands on the last index; there is no s[lo+1] to interpolate to.
  if (lo + 1 >= s.length) return s[lo];
  return s[lo] + (s[lo + 1] - s[lo]) * (pos - lo);
}

/**
 * @param {number[]} ratios  must be strictly positive (log-space mean)
 * @returns {number|null}
 */
function geomean(ratios) {
  if (!ratios || ratios.length === 0) return null;
  let sumLog = 0;
  for (const r of ratios) sumLog += Math.log(r);
  return Math.exp(sumLog / ratios.length);
}

// Deterministic PRNG (mulberry32) so a bootstrap CI is reproducible: the
// same runs give the same interval, or a "significant" flag would flicker
// between page loads.
/**
 * @param {number} seed
 * @returns {() => number} a generator of floats in [0, 1)
 */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** @param {number} x @returns {number} */
function erf(x) {
  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 -- enough for a p-value
  // we only threshold at 0.05.
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return sign * y;
}

/** @param {number} z @returns {number} standard-normal CDF at z */
function normalCdf(z) {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

// Mann-Whitney U with the normal approximation and tie correction. Chosen
// over a t-test because compile-time samples are not normal (a warmup
// tail skews them) and this is rank-based, distribution-free.
/**
 * @param {number[]} a
 * @param {number[]} b
 * @returns {{u: number, p: number}|null} two-sided p-value; null when
 *   either group has fewer than 2 samples
 */
function mannWhitneyU(a, b) {
  if (!a || !b || a.length < 2 || b.length < 2) return null;
  const na = a.length;
  const nb = b.length;
  const all = a.map((v) => [v, 0]).concat(b.map((v) => [v, 1]));
  all.sort((x, y) => x[0] - y[0]);

  const ranks = new Array(all.length);
  let tieTerm = 0;
  let i = 0;
  while (i < all.length) {
    let j = i;
    while (j + 1 < all.length && all[j + 1][0] === all[i][0]) j++;
    const avg = (i + j) / 2 + 1; // ranks are 1-based
    for (let k = i; k <= j; k++) ranks[k] = avg;
    const t = j - i + 1;
    if (t > 1) tieTerm += t * t * t - t;
    i = j + 1;
  }

  let rankSumA = 0;
  for (let k = 0; k < all.length; k++) if (all[k][1] === 0) rankSumA += ranks[k];
  const uA = rankSumA - (na * (na + 1)) / 2;
  const u = Math.min(uA, na * nb - uA);

  const n = na + nb;
  const meanU = (na * nb) / 2;
  const varU = ((na * nb) / 12) * (n + 1 - tieTerm / (n * (n - 1)));
  if (varU <= 0) return { u, p: 1 };
  const z = (u - meanU) / Math.sqrt(varU);
  return { u, p: Math.min(1, 2 * normalCdf(z)) };
}

// Bootstrap CI for the difference of medians (b - a). The CI excluding 0
// is the effect-size half of significance: Mann-Whitney says "different",
// this says "by how much, and are we sure of the sign".
/**
 * @param {number[]} a
 * @param {number[]} b
 * @param {number} resamples
 * @param {number} alpha  e.g. 0.05 for a 95% CI
 * @returns {[number, number]|null} [low, high] of (median b - median a)
 */
function bootstrapMedianDiffCI(a, b, resamples, alpha) {
  if (!a || !b || a.length < 2 || b.length < 2) return null;
  const rng = mulberry32(0x9e3779b9 ^ (a.length << 8) ^ b.length);
  const resample = (xs) => {
    const out = new Array(xs.length);
    for (let i = 0; i < xs.length; i++) out[i] = xs[(rng() * xs.length) | 0];
    // `out` has xs.length (>= 2) elements, so the median is never null.
    return /** @type {number} */ (median(out));
  };
  /** @type {number[]} */
  const diffs = new Array(resamples);
  for (let r = 0; r < resamples; r++) diffs[r] = resample(b) - resample(a);
  const lo = quantile(diffs, alpha / 2);
  const hi = quantile(diffs, 1 - alpha / 2);
  return lo == null || hi == null ? null : [lo, hi];
}

// -- record access ---------------------------------------------------------

const RESAMPLES = 2000;
const ALPHA = 0.05;
const LEVELS = ['o0', 'o2', 'o3'];
const PHASES = ['parse', 'construct', 'optimise', 'lower', 'codegen'];

// Metrics the grid/charts show. `point` extracts one value from a config;
// `samples` the raw vector when one exists (only whole-compile wall does,
// so only wall gets significance -- the others are point-valued today).
//
// The cg_* metrics are DETERMINISTIC Cachegrind counts (same code -> same
// number, run to run, machine to machine), which is why they are the
// regression signals; wall/RSS are contention-sensitive and informational.
// The cg_* counts isolate our compiler's work (frontend children
// excluded), so they are not clang-comparable and stay out of the vs-clang
// panel.
/** @type {Object<string, Metric>} */
const METRICS = {
  // Instructions executed: the primary deterministic signal -- a direct,
  // unmodelled count of the compiler's work, and what iai-callgrind /
  // rustc-perf headline for stable regression detection.
  cg_ir: { label: 'instr', unit: 'Mi', samples: () => null, point: (c) => (c.cachegrind_ir == null ? null : c.cachegrind_ir / 1e6) },
  // Last-level cache misses: the deterministic locality signal, a raw count
  // like instr (each miss is roughly a round trip to RAM). As a raw count it
  // scales with program size like instr, but its cross-run delta moves with
  // access patterns instruction count does not capture.
  cg_cache: { label: 'cache misses', unit: 'k', samples: () => null, point: (c) => (c.cachegrind_ll_misses == null ? null : c.cachegrind_ll_misses / 1e3) },
  // Estimated cycles: a cost-model roll-up of the two above,
  // L1_hits + 5*LL_hits + 35*RAM_hits, using cachegrind's generic latency
  // weights (NOT calibrated to this CPU, so not a literal cycle count).
  // Secondary: on a cache-friendly compile it is ~instr times a constant,
  // so its only content beyond instr is the locality penalty.
  cg_cycles: { label: 'cycles', unit: 'Mcyc', samples: () => null, point: (c) => (c.cachegrind_estimated_cycles == null ? null : c.cachegrind_estimated_cycles / 1e6) },
  // Rust-heap allocator calls across the in-process phases (a --wall
  // metric). Iteration-deterministic like the Cachegrind counts -- same
  // compile, same count -- so it reads as a regression signal, but it
  // sees only OUR Rust heap: LLVM's C++ allocations are invisible.
  allocs: { label: 'allocs', unit: 'k', samples: () => null, point: (c) => phaseCounterTotal(c, 'allocations', 1e3) },
  // Bytes handed out over the same window (cumulative churn: transient
  // scratch and Vec growth-doubling count in full; a realloc counts its
  // whole new size). The signal for allocator-pressure work that the
  // live/peak RSS view cannot show.
  alloc_bytes: { label: 'alloc churn', unit: 'MiB', samples: () => null, point: (c) => phaseCounterTotal(c, 'alloc_bytes', 1024 * 1024) },
  wall: { label: 'wall', unit: 'ms', samples: (c) => c.end_to_end && c.end_to_end.wall_ms, point: (c) => median(c.end_to_end && c.end_to_end.wall_ms) },
  peak_rss: { label: 'RSS', unit: 'MiB', samples: () => null, point: (c) => (c.peak_rss_bytes == null ? null : c.peak_rss_bytes / (1024 * 1024)) },
  obj_size: { label: 'obj', unit: 'KiB', samples: () => null, point: (c) => (c.object_size_bytes == null ? null : c.object_size_bytes / 1024) },
};
const GRID_METRICS = ['cg_ir', 'cg_cache', 'cg_cycles', 'allocs', 'alloc_bytes', 'wall', 'peak_rss', 'obj_size'];
const TREND_METRICS = ['cg_ir', 'cg_cache', 'cg_cycles', 'allocs'];
// Native, contention-sensitive metrics: their grid cells are dimmed (never
// coloured at full magnitude like a deterministic Cachegrind regression),
// unless wall passed a significance test.
const NOISY_METRICS = ['wall', 'peak_rss'];
const CLANG_METRICS = ['wall', 'peak_rss', 'obj_size'];

// Sum of per-phase medians of one allocation counter, scaled. Null when
// any phase lacks the counter (pre-v5 records, deterministic-only runs,
// clang configs -- which have no phases at all).
/**
 * @param {Config} config
 * @param {"allocations"|"alloc_bytes"} field
 * @param {number} scale
 * @returns {number|null}
 */
function phaseCounterTotal(config, field, scale) {
  if (!config.phases || config.phases.length === 0) return null;
  let total = 0;
  for (const rec of config.phases) {
    const m = median(rec.samples && rec.samples[field]);
    if (m == null) return null;
    total += m;
  }
  return total / scale;
}

/** @param {Config} config @returns {string} e.g. "ours-o0" */
function configKey(config) {
  return config.compiler.toLowerCase() + '-' + config.level;
}

// Schema v2 added `status`; a v1 record has none, so treat its absence as
// "Measured" (v1 could not record a failure). "Failed" / "TimedOut" mean
// the end-to-end compile did not produce data.
/** @param {Config|null} [config] @returns {string} */
function configStatus(config) {
  return (config && config.status) || 'Measured';
}
/** @param {Config|null} [config] @returns {boolean} */
function configFailed(config) {
  return configStatus(config) !== 'Measured';
}

// Every non-measured config in a run, across both compilers and all
// levels -- the "did anything break" list. Ordered by program then
// config for a stable table.
/** @param {Run} run @returns {Failure[]} */
function runFailures(run) {
  const out = [];
  for (const program of run.programs) {
    for (const config of program.configs) {
      if (configFailed(config)) {
        out.push({
          program: program.name,
          config: configLabel(config),
          status: configStatus(config),
          error: config.error || '',
        });
      }
    }
  }
  return out;
}

/** @param {Config} config @returns {string} e.g. "ours O0" */
function configLabel(config) {
  return config.compiler.toLowerCase() + ' ' + config.level.toUpperCase();
}

/** @param {Program|null} program @param {string} level @returns {Config|null} */
function oursConfig(program, level) {
  return (program && program.configs.find((c) => configKey(c) === `ours-${level}`)) || null;
}
/** @param {Program|null} program @param {string} level @returns {Config|null} */
function clangConfig(program, level) {
  return (program && program.configs.find((c) => configKey(c) === `clang-${level}`)) || null;
}
/**
 * @param {Program|null} program
 * @param {string} level
 * @param {string} key  a METRICS key
 * @param {"ours"|"clang"} compiler
 * @returns {number|null}
 */
function metricAt(program, level, key, compiler) {
  const cfg = (compiler === 'clang' ? clangConfig : oursConfig)(program, level);
  return cfg ? METRICS[key].point(cfg) : null;
}

/** @param {Run} run @returns {string} short git label, "*" if dirty */
function runLabel(run) {
  const dirty = run.meta && run.meta.git_dirty ? '*' : '';
  return (run.meta && run.meta.git_sha ? run.meta.git_sha : 'nogit') + dirty;
}

/** @param {Run|null} run @param {string} name @returns {Program|null} */
function findProgram(run, name) {
  return (run && run.programs.find((p) => p.name === name)) || null;
}

// -- regression grid model (pure) ------------------------------------------

// One row per program present in the current run: for each metric, the
// delta of our compiler's value at `level` versus the baseline run, plus
// significance (wall only, from its samples). Baseline-absent programs get
// has=false cells (shown as "-").
/**
 * @param {Run} current
 * @param {Run} baseline
 * @param {string} level
 * @returns {Row[]}
 */
function regressionRows(current, baseline, level) {
  return current.programs.map((cur) => {
    const base = findProgram(baseline, cur.name);
    const curCfg = oursConfig(cur, level);
    const baseCfg = oursConfig(base, level);
    // The failure is a property of the config, so a missing current value
    // reads as "fail"/"timeout" rather than "no data".
    const failStatus = configFailed(curCfg) ? configStatus(curCfg) : null;
    /** @type {Object<string, Cell>} */
    const cells = {};
    for (const key of GRID_METRICS) {
      const curV = curCfg ? METRICS[key].point(curCfg) : null;
      const baseV = baseCfg ? METRICS[key].point(baseCfg) : null;
      /** @type {number|null} */
      let deltaPct = null;
      /** @type {boolean|null} */
      let sig = null;
      // Inline the null checks (rather than a derived `has` boolean) so the
      // type checker can narrow curV/baseV to non-null in the body.
      if (curV != null && baseV != null && baseV !== 0) {
        deltaPct = (curV / baseV - 1) * 100;
        const cs = curCfg && METRICS[key].samples(curCfg);
        const bs = baseCfg && METRICS[key].samples(baseCfg);
        if (cs && bs) {
          const mw = mannWhitneyU(bs, cs);
          const ci = bootstrapMedianDiffCI(bs, cs, RESAMPLES, ALPHA);
          const ciExcludesZero = ci && (ci[0] > 0 || ci[1] < 0);
          sig = !!(mw && mw.p < ALPHA && ciExcludesZero);
        }
      }
      const has = deltaPct != null;
      // A value missing because this config's compile failed shows the
      // failure label, not a bare dash (a failed config nulls every metric).
      const failed =
        curV == null && failStatus != null
          ? failStatus === 'TimedOut'
            ? 'timeout'
            : 'fail'
          : false;
      cells[key] = { deltaPct, curV, baseV, sig, has, failed };
    }
    return { name: cur.name, cells, failStatus };
  });
}

// Net geomean delta per metric across the programs that compare.
/**
 * @param {Row[]} rows
 * @returns {Object<string, number|null>} percent delta per metric key
 */
function regressionGeomean(rows) {
  /** @type {Object<string, number|null>} */
  const geo = {};
  for (const key of GRID_METRICS) {
    /** @type {number[]} */
    const ratios = [];
    for (const r of rows) {
      const c = r.cells[key];
      // Both sides must be positive: a zero current value has no
      // representable ratio (log 0 = -Inf), and one such cell would
      // otherwise collapse the whole metric's geomean to -100%.
      if (c.has && c.baseV != null && c.baseV > 0 && c.curV != null && c.curV > 0) {
        ratios.push(c.curV / c.baseV);
      }
    }
    const g = ratios.length ? geomean(ratios) : null;
    geo[key] = g == null ? null : (g - 1) * 100;
  }
  return geo;
}

// -- ECharts option builders (pure) ----------------------------------------

const AXIS = '#adb5bd';
const GRID = 'rgba(173,181,189,0.15)';
const GEOMEAN_COLOR = '#adb5bd';
const FOCUS_COLOR = '#e8590c';
const PHASE_COLORS = ['#4dabf7', '#3bc9db', '#38d9a9', '#ffd43b', '#ff922b'];
const CLANG_SERIES_COLORS = { wall: '#f4845f', RSS: '#4dabf7', obj: '#38d9a9' };

function rotateFor(names) {
  return names.length > 6 ? 40 : 0;
}

// Shared layout for the category-bar panels (phase breakdown, vs-clang):
// identical grid, rotated category axis, and a scroll zoom past 8 bars.
// Each caller adds its own title/tooltip/legend/yAxis/series.
/**
 * @param {string[]} names
 * @returns {object} partial ECharts option (grid, xAxis, dataZoom)
 */
function categoryBarBase(names) {
  return {
    grid: { left: 8, right: 16, top: 58, bottom: 64, containLabel: true },
    xAxis: {
      type: 'category',
      data: names,
      axisLabel: { rotate: rotateFor(names), fontSize: 10 },
      axisLine: { lineStyle: { color: AXIS } },
    },
    dataZoom: names.length > 8 ? [{ type: 'slider', bottom: 6, height: 14 }] : undefined,
  };
}

// One small line chart for a metric: the geomean over programs across
// runs (the aggregate regression-over-commits signal), plus the focused
// program's own line when one is picked in the grid.
/**
 * @param {Run[]} runs
 * @param {string} metricKey
 * @param {string} level
 * @param {string|null} focus  program to overlay, or null
 * @returns {object} ECharts option
 */
function metricTrendOption(runs, metricKey, level, focus) {
  const metric = METRICS[metricKey];
  const xs = runs.map(runLabel);
  const geo = runs.map((run, i) => {
    const vals = /** @type {number[]} */ (
      run.programs.map((p) => metricAt(p, level, metricKey, 'ours')).filter((v) => v != null && v > 0)
    );
    return [i, vals.length ? geomean(vals) : null];
  });
  const series = [
    { name: 'geomean', type: 'line', showSymbol: true, symbolSize: 5, data: geo, lineStyle: { width: 2, color: GEOMEAN_COLOR }, itemStyle: { color: GEOMEAN_COLOR } },
  ];
  if (focus) {
    const f = runs.map((run, i) => [i, metricAt(findProgram(run, focus), level, metricKey, 'ours')]);
    series.push({ name: focus, type: 'line', showSymbol: true, symbolSize: 5, data: f, lineStyle: { width: 2, color: FOCUS_COLOR }, itemStyle: { color: FOCUS_COLOR } });
  }
  return {
    title: { text: metric.unit ? `${metric.label} (${metric.unit})` : metric.label, left: 'center', textStyle: { fontSize: 12 } },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v) => (v == null ? '-' : v.toFixed(2) + (metric.unit ? ' ' + metric.unit : '')),
    },
    legend: focus ? { bottom: 0, textStyle: { fontSize: 10 }, data: ['geomean', focus] } : undefined,
    grid: { left: 8, right: 12, top: 30, bottom: focus ? 30 : 20, containLabel: true },
    xAxis: { type: 'category', data: xs, axisLabel: { show: false }, axisLine: { lineStyle: { color: AXIS } } },
    yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: GRID } }, axisLabel: { fontSize: 10 } },
    series,
  };
}

// What the phase-breakdown panel can stack: wall time or the allocation
// counters (the latter only present in schema >= 5 --wall runs; their
// bars simply blank out on older records).
/** @type {Object<string, {label: string, unit: string, field: string, scale: number}>} */
const PHASE_METRICS = {
  wall: { label: 'wall', unit: 'ms', field: 'wall_ms', scale: 1 },
  allocs: { label: 'allocations', unit: 'k', field: 'allocations', scale: 1e3 },
  alloc_bytes: { label: 'alloc churn', unit: 'MiB', field: 'alloc_bytes', scale: 1024 * 1024 },
};

// How the phase bars are scaled. `raw` is the metric as measured;
// `per_value` divides by graph size so differently-sized programs
// compare on unit cost; `share` stacks each program to 100% so the
// question "which phase dominates" is independent of both program size
// and machine noise (a uniformly slow first program shares out the
// same as a fast one).
/** @type {Object<string, string>} */
const NORM_MODES = { raw: 'raw', per_value: '/ value', share: '% of total' };

// Where our time (or allocator traffic) goes: stacked per-phase medians,
// one bar per program, at the chosen level.
/**
 * @param {Run} run
 * @param {string} level
 * @param {string} normMode  a NORM_MODES key
 * @param {string} metricKey  a PHASE_METRICS key
 * @returns {object} ECharts option
 */
function phaseBreakdownOption(run, level, normMode, metricKey) {
  const pm = PHASE_METRICS[metricKey] || PHASE_METRICS.wall;
  const programs = run.programs;
  const names = programs.map((p) => p.name);
  const phaseMedian = (p, ph) => {
    const c = oursConfig(p, level);
    const rec = c && c.phases && c.phases.find((x) => x.phase === ph);
    return rec ? median(rec.samples && rec.samples[pm.field]) : null;
  };
  // Per-program totals for the share mode; null (blanking the bar) when
  // any phase is unmeasured, so a partial bar never masquerades as 100%.
  const totals = programs.map((p) => {
    let total = 0;
    for (const ph of PHASES) {
      const m = phaseMedian(p, ph);
      if (m == null) return null;
      total += m;
    }
    return total > 0 ? total : null;
  });
  const series = PHASES.map((ph, idx) => ({
    name: ph,
    type: 'bar',
    stack: 'phase',
    itemStyle: { color: PHASE_COLORS[idx] },
    data: programs.map((p, pi) => {
      const raw = phaseMedian(p, ph);
      if (raw == null) return null;
      if (normMode === 'share') {
        return totals[pi] == null ? null : (raw / totals[pi]) * 100;
      }
      const m = raw / pm.scale;
      // When normalizing by size, a zero-size program has no meaningful
      // per-value figure -- blank it rather than show a misleading bar.
      if (normMode === 'per_value') return p.values ? m / p.values : null;
      return m;
    }),
  }));
  const unit =
    normMode === 'share' ? '% of compile' : normMode === 'per_value' ? `${pm.unit} / value` : pm.unit;
  return {
    title: { text: `phase breakdown -- ours ${level.toUpperCase()}, ${pm.label} (${unit})`, left: 'center', textStyle: { fontSize: 13 } },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: normMode === 'share' ? (v) => (v == null ? '-' : v.toFixed(1) + '%') : undefined,
    },
    legend: { top: 26 },
    ...categoryBarBase(names),
    yAxis: {
      type: 'value',
      max: normMode === 'share' ? 100 : undefined,
      splitLine: { lineStyle: { color: GRID } },
    },
    series,
  };
}

// The competitive gap: ours/clang per program at the chosen level, for the
// metrics clang also reports. A ratio > 1 means we are slower/bigger; the
// parity line at 1 is the target.
/**
 * @param {Run} run
 * @param {string} level
 * @returns {object} ECharts option
 */
function clangRatioOption(run, level) {
  const programs = run.programs.filter((p) => oursConfig(p, level) && clangConfig(p, level));
  const names = programs.map((p) => p.name);
  const series = CLANG_METRICS.map((key, idx) => {
    const label = METRICS[key].label;
    return {
      name: label,
      type: 'bar',
      itemStyle: { color: CLANG_SERIES_COLORS[label] },
      data: programs.map((p) => {
        const o = metricAt(p, level, key, 'ours');
        const c = metricAt(p, level, key, 'clang');
        return o != null && c != null && c > 0 ? o / c : null;
      }),
      ...(idx === 0
        ? { markLine: { silent: true, symbol: 'none', data: [{ yAxis: 1 }], lineStyle: { color: '#868e96', type: 'dashed' }, label: { show: true, position: 'start', formatter: 'parity', fontSize: 10, color: '#868e96' } } }
        : {}),
    };
  });
  return {
    title: { text: `ours / clang at ${level.toUpperCase()} (>1 = we are slower/bigger)`, left: 'center', textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => (v == null ? '-' : v.toFixed(2) + 'x') },
    legend: { top: 26 },
    ...categoryBarBase(names),
    yAxis: { type: 'value', name: 'ratio', splitLine: { lineStyle: { color: GRID } } },
    series,
  };
}

// -- Alpine shell ----------------------------------------------------------
// State + wiring only; all logic is in the pure functions above.

function reportComponent() {
  return {
    /** @type {Run[]} */
    runs: [],
    empty: false,
    runIdx: 0,
    baselineIdx: 0,
    level: 'o0',
    /** @type {string|null} */
    focus: null, // program drilled into from the grid
    normMode: 'raw',
    normModeKeys: Object.keys(NORM_MODES),
    phaseMetric: 'wall',
    phaseMetricKeys: Object.keys(PHASE_METRICS),
    sort: { key: 'cg_ir', dir: -1 }, // dir -1 = biggest |delta| first; cg_ir is always populated (wall needs --wall)
    /** @type {Object<string, any>} */
    charts: {},
    /** @type {{ key: string, rows: Row[], geo: Object<string, number|null> }} */
    matrixCache: { key: '', rows: [], geo: {} },

    gridMetrics: GRID_METRICS,
    trendMetrics: TREND_METRICS,
    levels: LEVELS,

    init() {
      this.runs = Array.isArray(window.BENCH_RUNS) ? window.BENCH_RUNS : [];
      this.empty = this.runs.length === 0;
      if (this.empty) return;
      this.runIdx = this.runs.length - 1;
      this.baselineIdx = Math.max(0, this.runs.length - 2);

      // `$nextTick` is injected by Alpine onto the reactive `this`, so it
      // is not a declared member of this plain object literal.
      const alpine = /** @type {any} */ (this);
      alpine.$nextTick(() => {
        this.initCharts();
        this.renderCharts();
      });
    },

    // Create each chart once and attach a ResizeObserver: whenever a
    // container's box changes -- window resize, layout settling, a hidden
    // ancestor being revealed -- the observer resizes the chart. This is
    // what keeps a chart from latching a zero/tiny width measured before
    // layout was ready (the width bug); no manual resize timing needed.
    initCharts() {
      // Derived from TREND_METRICS so the containers, init, and render
      // never drift out of sync when the metric set changes.
      const ids = [...TREND_METRICS.map((m) => `trend-${m}`), 'phases', 'clang'];
      for (const id of ids) {
        const el = document.getElementById(`chart-${id}`);
        if (!el || this.charts[id]) continue;
        const chart = window.echarts.init(el, null, { renderer: 'canvas' });
        this.charts[id] = chart;
        new ResizeObserver(() => chart.resize()).observe(el);
      }
    },

    currentRun() {
      return this.runs[this.runIdx];
    },
    baselineRun() {
      return this.runs[this.baselineIdx];
    },

    // -- regression grid (reactive HTML, memoized by run/baseline/level) --
    matrix() {
      const key = `${this.runIdx}-${this.baselineIdx}-${this.level}`;
      if (this.matrixCache.key !== key) {
        const rows = regressionRows(this.currentRun(), this.baselineRun(), this.level);
        this.matrixCache = { key, rows, geo: regressionGeomean(rows) };
      }
      return this.matrixCache;
    },
    matrixRows() {
      const rows = this.matrix().rows.slice();
      const { key, dir } = this.sort;
      if (key === '__name') {
        rows.sort((a, b) => a.name.localeCompare(b.name) * dir);
        return rows;
      }
      rows.sort((a, b) => {
        const av = a.cells[key].has ? Math.abs(a.cells[key].deltaPct ?? 0) : -Infinity;
        const bv = b.cells[key].has ? Math.abs(b.cells[key].deltaPct ?? 0) : -Infinity;
        return (av - bv) * dir;
      });
      return rows;
    },
    geoRow() {
      return this.matrix().geo;
    },
    // Non-measured configs in the current run, for the failures panel.
    failures() {
      return runFailures(this.currentRun());
    },
    setSort(key) {
      if (this.sort.key === key) this.sort.dir *= -1;
      else this.sort = { key, dir: -1 };
    },
    /** @param {Cell} cell @param {string} metricKey @returns {string} inline CSS */
    cellStyle(cell, metricKey) {
      // A compile failure is a distinct, neutral-but-noticeable amber, not
      // on the red/green regression scale (it is not a magnitude).
      if (cell.failed) return 'background: rgba(240, 173, 78, 0.25);';
      if (!cell.has || cell.deltaPct == null || Math.abs(cell.deltaPct) < 0.05) return '';
      // Alpha scales with magnitude so a big move reads darker (deterministic
      // metrics reach 0.8).
      let alpha = Math.min(0.8, 0.1 + Math.abs(cell.deltaPct) / 25);
      // Contention-sensitive metrics (wall, RSS) never read as loud as a
      // deterministic Cachegrind regression: capped at 0.4 when wall passed a
      // significance test, dimmed to 0.1 otherwise (RSS has no significance).
      if (NOISY_METRICS.includes(metricKey)) {
        alpha = cell.sig === true ? Math.min(alpha, 0.4) : 0.1;
      }
      const rgb = cell.deltaPct > 0 ? '224, 49, 49' : '47, 158, 68';
      return `background: rgba(${rgb}, ${alpha.toFixed(3)});`;
    },
    /** @param {Cell} cell @returns {string} */
    /** @param {Cell} cell @param {string} metricKey @returns {string} */
    cellText(cell, metricKey) {
      if (cell.failed) return cell.failed;
      // No distinct baseline (e.g. a single run loaded): a delta against the
      // same run is always 0%, so show the absolute value instead.
      if (this.runIdx === this.baselineIdx) {
        if (cell.curV == null) return '-';
        const u = METRICS[metricKey].unit;
        return cell.curV.toFixed(2) + (u ? ' ' + u : '');
      }
      if (!cell.has || cell.deltaPct == null) return '-';
      const s = (cell.deltaPct >= 0 ? '+' : '') + cell.deltaPct.toFixed(1) + '%';
      return cell.sig === true ? (cell.deltaPct > 0 ? s + ' ▲' : s + ' ▼') : s;
    },
    /** @param {Cell} cell @param {string} metricKey @returns {string} tooltip */
    cellTitle(cell, metricKey) {
      if (cell.failed) return 'compile failed -- see the failures panel above';
      const u = METRICS[metricKey].unit;
      // No baseline (single run): show the absolute value, not "X -> X".
      if (this.runIdx === this.baselineIdx) {
        return cell.curV == null ? 'no data' : `${cell.curV.toFixed(2)} ${u}`;
      }
      if (!cell.has) return 'not comparable';
      const sig = cell.sig === null ? '' : cell.sig ? ' (significant)' : ' (not significant)';
      // has is true here, so both are non-null; `?? 0` satisfies the checker.
      return `${(cell.baseV ?? 0).toFixed(2)} -> ${(cell.curV ?? 0).toFixed(2)} ${u}${sig}`;
    },
    geoText(key) {
      // No baseline to divide against -> no meaningful geomean delta.
      if (this.runIdx === this.baselineIdx) return '-';
      const g = this.geoRow()[key];
      return g == null ? '-' : (g >= 0 ? '+' : '') + g.toFixed(1) + '%';
    },
    metricLabel(key) {
      return METRICS[key].label;
    },

    focusProgram(name) {
      this.focus = this.focus === name ? null : name;
      this.renderTrends();
    },

    setLevel(level) {
      this.level = level;
      this.renderCharts();
    },

    onFilterChange() {
      this.renderCharts();
    },

    // The phase-breakdown and vs-clang panels only have data from a --wall
    // run; a default (deterministic-only) run has neither, so their sections
    // are hidden rather than shown empty under a confident heading.
    hasPhaseData() {
      return this.currentRun().programs.some((p) => p.configs.some((c) => c.phases && c.phases.length > 0));
    },
    hasClangData() {
      return this.currentRun().programs.some((p) =>
        p.configs.some((c) => c.compiler === 'Clang' && (c.end_to_end?.wall_ms?.length ?? 0) > 0)
      );
    },

    setPhaseMetric(key) {
      this.phaseMetric = key;
      this.renderCharts();
    },
    phaseMetricLabel(key) {
      return PHASE_METRICS[key].label;
    },
    setNormMode(key) {
      this.normMode = key;
      this.renderCharts();
    },
    normModeLabel(key) {
      return NORM_MODES[key];
    },

    renderCharts() {
      this.renderTrends();
      this.setChart('phases', () => phaseBreakdownOption(this.currentRun(), this.level, this.normMode, this.phaseMetric));
      this.setChart('clang', () => clangRatioOption(this.currentRun(), this.level));
    },
    renderTrends() {
      for (const m of TREND_METRICS) {
        this.setChart(`trend-${m}`, () => metricTrendOption(this.runs, m, this.level, this.focus));
      }
    },
    setChart(id, build) {
      const chart = this.charts[id];
      if (!chart) return;
      chart.clear();
      chart.setOption(build());
      chart.resize();
    },

    // -- meta lines --
    runLabelText(run) {
      const m = run.meta || {};
      const when = m.timestamp_unix ? new Date(m.timestamp_unix * 1000).toISOString().slice(0, 16).replace('T', ' ') : '?';
      return `${runLabel(run)}  ${when}`;
    },
    runMetaLine(run) {
      const m = run.meta || {};
      return `${m.cpu_model || 'cpu?'} [${m.governor || 'gov?'}]  ${m.clang_version || ''}  iters=${m.iters}/${m.warmup}`;
    },
  };
}

document.addEventListener('alpine:init', () => {
  window.Alpine.data('report', reportComponent);
});

// Expose the pure pipeline for console poking / future unit tests.
window.BENCH_REPORT = {
  median,
  quantile,
  geomean,
  mannWhitneyU,
  bootstrapMedianDiffCI,
  regressionRows,
  regressionGeomean,
  metricTrendOption,
  phaseBreakdownOption,
  clangRatioOption,
  phaseCounterTotal,
  METRICS,
  PHASE_METRICS,
  NORM_MODES,
};
