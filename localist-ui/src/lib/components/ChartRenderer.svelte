<script lang="ts">
  /**
   * ChartRenderer.svelte
   *
   * Renders a chart from the same chart_config shape both the backend's
   * PNG rendering (mcp_server/chart.py) and the frontend consume — one
   * chart-config schema in the whole system, nothing to drift apart.
   *
   * Colors are read from this app's existing CSS custom properties
   * (--accent, --success, --warning, --error, --text-*, --border-*) rather
   * than hardcoded hex values, so the chart tracks the active light/dark
   * theme the same way every other surface in this UI does.
   */

  import { onMount, onDestroy } from 'svelte';
  import { Chart, registerables } from 'chart.js';

  Chart.register(...registerables);

  export let config: {
    chart_type: 'bar' | 'line' | 'pie';
    title: string;
    labels: string[];
    datasets: { label: string; data: number[] }[];
  };

  let canvasEl: HTMLCanvasElement;
  let chart: Chart | null = null;

  // Fixed order, drawn from this app's own semantic tokens rather than a
  // separate dataviz palette — every hue here is already used elsewhere
  // in the UI for something else, so a chart never introduces a color
  // vocabulary the rest of the app doesn't have.
  const SERIES_VARS = ['--accent', '--success', '--warning', '--error'] as const;

  function cssVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function buildChart(): void {
    const seriesColors = SERIES_VARS.map(cssVar);
    const textSecondary = cssVar('--text-secondary');
    const textPrimary = cssVar('--text-primary');
    const gridColor = cssVar('--border-soft');
    const bgRaised = cssVar('--bg-raised');
    const border = cssVar('--border');

    const isPie = config.chart_type === 'pie';

    const datasets = config.datasets.map((ds, i) => {
      const color = seriesColors[i % seriesColors.length];
      if (isPie) {
        return {
          label: ds.label,
          data: ds.data,
          backgroundColor: config.labels.map(
            (_, li) => seriesColors[li % seriesColors.length]
          ),
          borderColor: bgRaised,
          borderWidth: 2
        };
      }
      return {
        label: ds.label,
        data: ds.data,
        backgroundColor: config.chart_type === 'bar' ? color : `${color}33`,
        borderColor: color,
        borderWidth: config.chart_type === 'line' ? 2 : 1,
        pointBackgroundColor: color,
        tension: 0.25
      };
    });

    chart = new Chart(canvasEl, {
      type: config.chart_type,
      data: {
        labels: config.labels,
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: !!config.title,
            text: config.title,
            color: textPrimary,
            font: { size: 13, weight: 600 }
          },
          legend: {
            display: isPie || config.datasets.length > 1,
            labels: { color: textSecondary, boxWidth: 12, font: { size: 11 } }
          },
          tooltip: {
            backgroundColor: bgRaised,
            titleColor: textPrimary,
            bodyColor: textSecondary,
            borderColor: border,
            borderWidth: 1,
            padding: 8
          }
        },
        scales: isPie
          ? {}
          : {
              x: {
                ticks: { color: textSecondary, font: { size: 11 } },
                grid: { color: gridColor, display: false }
              },
              y: {
                ticks: { color: textSecondary, font: { size: 11 } },
                grid: { color: gridColor },
                beginAtZero: true
              }
            }
      }
    });
  }

  onMount(buildChart);

  onDestroy(() => {
    chart?.destroy();
    chart = null;
  });
</script>

<div class="chart-card">
  <div class="chart-canvas-wrap">
    <canvas bind:this={canvasEl} />
  </div>
</div>

<style>
  .chart-card {
    margin-top: var(--sp-3);
    padding: var(--sp-4);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }

  .chart-canvas-wrap {
    position: relative;
    width: 100%;
    height: 260px;
  }
</style>
