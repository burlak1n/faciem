<script lang="ts">
  import { onMount } from 'svelte';
  import Chart from 'chart.js/auto';
  import 'chartjs-adapter-date-fns';
  import { ru } from 'date-fns/locale';
  import type { ChartData, ChartOptions, Point } from 'chart.js';

  let embeddingChart: Chart | null = null;
  let progressChart: Chart | null = null;
  let embeddingTimesData: number[] = [];
  let progressOverTimeData: Point[] = [];

  export function addEmbeddingTime(time: number) {
    embeddingTimesData.push(time);
    updateEmbeddingChart();
  }

  export function addProgressPoint(count: number) {
    progressOverTimeData.push({ x: new Date().getTime(), y: count });
    updateProgressChart();
  }

  function updateEmbeddingChart() {
    if (!embeddingTimesData.length) return;

    const bins = [0, 50, 100, 150, 200, 300, 500, 1000, Infinity] as const;
    const labels = bins.slice(0, -1).map((b, i) => {
      const nextBin = bins[i + 1];
      return `${b}-${nextBin === Infinity ? 'inf' : nextBin} мс`;
    });
    const dataCounts = Array(labels.length).fill(0);

    embeddingTimesData.forEach(time => {
      for (let i = 0; i < bins.length - 1; i++) {
        const currentBin = bins[i];
        const nextBin = bins[i + 1];
        if (currentBin !== undefined && nextBin !== undefined && time >= currentBin && time < nextBin) {
          dataCounts[i]++;
          break;
        }
      }
    });

    const chartData: ChartData = {
      labels,
      datasets: [{
        label: 'Распределение времени обработки лиц (мс)',
        data: dataCounts,
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }]
    };

    if (embeddingChart) {
      embeddingChart.data = chartData;
      embeddingChart.update();
    } else {
      const ctx = document.getElementById('embeddingTimeChart') as HTMLCanvasElement;
      if (!ctx) return;

      embeddingChart = new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
          scales: {
            y: {
              beginAtZero: true,
              title: { display: true, text: 'Количество лиц' }
            },
            x: {
              title: { display: true, text: 'Время обработки (мс)' }
            }
          },
          animation: { duration: 0 }
        }
      });
    }
  }

  function updateProgressChart() {
    if (!progressOverTimeData.length) return;

    const chartData: ChartData = {
      datasets: [{
        label: 'Обработано изображений (накопительно)',
        data: progressOverTimeData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.1
      }]
    };

    const options: ChartOptions = {
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'minute',
            tooltipFormat: 'dd.MM.yyyy HH:mm:ss',
            displayFormats: {
              minute: 'HH:mm',
              hour: 'HH:mm'
            }
          },
          adapters: {
            date: {
              locale: ru
            }
          },
          title: { display: true, text: 'Время (МСК)' }
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: 'Кол-во обработанных' }
        }
      },
      animation: { duration: 200 }
    };

    if (progressChart) {
      progressChart.data = chartData;
      progressChart.options = options;
      progressChart.update();
    } else {
      const ctx = document.getElementById('progressOverTimeChart') as HTMLCanvasElement;
      if (!ctx) return;

      progressChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options
      });
    }
  }

  onMount(() => {
    updateEmbeddingChart();
    updateProgressChart();

    return () => {
      if (embeddingChart) {
        embeddingChart.destroy();
      }
      if (progressChart) {
        progressChart.destroy();
      }
    };
  });
</script>

<div class="charts-container">
  <h4>Графики:</h4>
  <div class="chart-wrapper">
    <canvas id="embeddingTimeChart"></canvas>
  </div>
  <div class="chart-wrapper">
    <canvas id="progressOverTimeChart"></canvas>
  </div>
</div>

<style>
  .charts-container {
    margin-top: 20px;
    padding: 15px;
    background-color: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 0 5px rgba(0,0,0,0.05);
  }

  .chart-wrapper {
    margin-top: 20px;
    position: relative;
    height: 300px;
  }

  .chart-wrapper:first-child {
    margin-top: 0;
  }
</style> 