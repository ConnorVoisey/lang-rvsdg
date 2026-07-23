// Ambient declarations for the report's runtime globals. ECharts and
// Alpine are loaded via CDN <script> tags (untyped here), `BENCH_RUNS`
// comes from the generated data.js, and `BENCH_REPORT` is the pipeline we
// expose for the console. Typed loosely on purpose -- the value of the
// checker is the pure functions in report.js, not this glue.
interface Window {
  echarts: any;
  Alpine: any;
  BENCH_RUNS?: any[];
  BENCH_REPORT?: any;
}
