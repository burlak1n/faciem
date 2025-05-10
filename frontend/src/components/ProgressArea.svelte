<script lang="ts">
  import { onMount } from 'svelte';
  
  interface LogItem {
    photo_id: string;
    status: string;
    image_base64?: string;
    url?: string;
    photo_date?: string;
    faces_count?: number;
    face_confidences?: number[];
    error_message?: string;
    face_index_processed?: number;
    embedding_duration_ms?: number;
    total_count?: number;
  }

  let eventSource: EventSource | null = null;
  let logItems: LogItem[] = [];
  let totalImagesToProcess = 0;
  let processedImagesCount = 0;
  let completedSuccessfullyCount = 0;
  let errorCount = 0;
  let overallProgress = '';
  let totalEmbeddingTimeMs = 0;
  let processedFacesForTiming = 0;
  let startIndexingTime: number | null = null;

  function getStatusClass(status: string): string {
    if (status.startsWith('error')) return 'status-error';
    if (status === 'extraction_completed') return 'status-completed';
    if (status === 'downloaded_processing') return 'status-processing';
    return 'status-pending';
  }

  function getStatusText(status: string): string {
    switch (status) {
      case 'downloaded_processing': return 'Загружено, обработка лиц...';
      case 'extraction_completed': return 'Детекция лиц завершена';
      case 'error_download': return 'Ошибка загрузки';
      case 'error_decode': return 'Ошибка декодирования';
      case 'error_extraction': return 'Ошибка детекции лиц';
      case 'error_embedding_face': return 'Ошибка получения эмбеддинга';
      case 'error_represent_chunk':
      case 'error_embedding_mismatch':
      case 'error_milvus_insert_chunk':
      case 'error_internal_logic': 
        return 'Ошибка обработки чанка';
      default: return status;
    }
  }

  function formatUnixTimestamp(timestampStr: string | number): string {
    if (!timestampStr) return 'N/A';
    if (typeof timestampStr === 'string' && !/^\d+$/.test(timestampStr)) {
      const directDate = new Date(timestampStr);
      if (!isNaN(directDate.getTime())) {
        return directDate.toLocaleDateString() + ' ' + directDate.toLocaleTimeString();
      }
      return timestampStr;
    }
    const timestamp = parseInt(String(timestampStr), 10);
    if (isNaN(timestamp)) {
      return String(timestampStr);
    }
    const date = new Date(timestamp * 1000);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
  }

  function listenForProgress() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource('/index/stream_status');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data) as LogItem;
      
      if (data.status === 'started') {
        totalImagesToProcess = data.total_count || 0;
        overallProgress = `Индексация запущена. Всего изображений для обработки: ${totalImagesToProcess}`;
        logItems = [];
        return;
      }

      if (data.status === 'finished') {
        overallProgress = 'Индексация полностью завершена!';
        if (eventSource) {
          eventSource.close();
        }
        return;
      }

      if (data.status === 'error_critical') {
        overallProgress = `Критическая ошибка сервера: ${data.error_message || 'Неизвестно'}. Индексация остановлена.`;
        if (eventSource) {
          eventSource.close();
        }
        return;
      }

      if (data.photo_id === "CHUNK_ERROR") {
        console.warn("Chunk error received:", data.error_message);
        return;
      }

      const existingItemIndex = logItems.findIndex(item => item.photo_id === data.photo_id);
      if (existingItemIndex === -1) {
        logItems = [data, ...logItems];
        processedImagesCount++;
      } else {
        logItems[existingItemIndex] = { ...logItems[existingItemIndex], ...data };
        logItems = [...logItems];
      }

      if (data.status === 'extraction_completed') {
        completedSuccessfullyCount++;
      } else if (data.status.startsWith('error')) {
        errorCount++;
      }

      if (data.embedding_duration_ms) {
        totalEmbeddingTimeMs += data.embedding_duration_ms;
        processedFacesForTiming++;
      }
    };

    eventSource.onerror = () => {
      console.error("EventSource failed");
      if (eventSource) {
        eventSource.close();
      }
    };
  }

  onMount(() => {
    listenForProgress();
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  });
</script>

<div class="progress-container">
  <h2>Прогресс индексации:</h2>
  <div class="overall-progress">{overallProgress}</div>
  
  <div class="summary-counts">
    <div>Всего изображений: {totalImagesToProcess}</div>
    <div>Обработано: {processedImagesCount}</div>
    <div>Успешно: {completedSuccessfullyCount}</div>
    <div>Ошибок: {errorCount}</div>
  </div>

  <div class="metrics-area">
    <h4>Метрики:</h4>
    <div>Среднее время обработки лица: {processedFacesForTiming ? (totalEmbeddingTimeMs / processedFacesForTiming).toFixed(1) : '-'} мс</div>
    <div>Общее время: {startIndexingTime ? ((Date.now() - startIndexingTime) / 1000).toFixed(1) : '-'} сек</div>
  </div>

  <div class="log-items">
    {#each logItems as item (item.photo_id)}
      <div class="log-item {getStatusClass(item.status)}">
        {#if item.image_base64}
          <button 
            class="image-button"
            on:click={() => window.open(item.url, '_blank')}
            on:keydown={(e) => e.key === 'Enter' && window.open(item.url, '_blank')}
          >
            <img 
              src="data:image/jpeg;base64,{item.image_base64}" 
              alt="Фото ID: {item.photo_id}" 
              class="log-image"
              title="Нажмите, чтобы открыть оригинал"
            />
          </button>
        {:else}
          <button 
            class="image-button placeholder"
            on:click={() => window.open(item.url, '_blank')}
            on:keydown={(e) => e.key === 'Enter' && window.open(item.url, '_blank')}
            title="Нажмите, чтобы открыть оригинал"
          >
            Фото<br>(ID: {item.photo_id})
          </button>
        {/if}

        <div class="info">
          <div class="photo-id">
            <span class="status-dot"></span>
            <strong>ID: {item.photo_id}</strong>
          </div>
          <div class="details status-text">
            Статус: {getStatusText(item.status)}
          </div>
          
          {#if item.photo_date}
            <div class="details photo-date">
              Дата фото: {formatUnixTimestamp(item.photo_date)}
            </div>
          {/if}

          {#if item.faces_count !== undefined}
            <div class="details faces-info">
              Найдено лиц: {item.faces_count}
              {#if item.face_confidences?.length}
                (Conf: [{item.face_confidences.map(c => c.toFixed(2)).join(', ')}])
              {/if}
            </div>
          {/if}

          {#if item.face_index_processed !== undefined && item.embedding_duration_ms !== undefined}
            <div class="details embedding-details">
              Лицо #{item.face_index_processed}: эмбеддинг за {item.embedding_duration_ms.toFixed(1)} мс.
            </div>
          {/if}

          {#if item.error_message}
            <div class="error-message">
              Ошибка ({item.status}): {item.error_message}
            </div>
          {/if}
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  .progress-container {
    margin-top: 20px;
  }

  .overall-progress {
    font-size: 1.1em;
    margin-bottom: 15px;
  }

  .summary-counts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 15px 0;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 4px;
  }

  .metrics-area {
    margin: 15px 0;
    padding: 10px;
    background-color: #e9ecef;
    border-radius: 4px;
  }

  .log-items {
    margin-top: 20px;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #eee;
    border-radius: 4px;
  }

  .log-item {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    display: flex;
    align-items: flex-start;
    gap: 15px;
  }

  .log-item:last-child {
    border-bottom: none;
  }

  .log-image {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border: 1px solid #ccc;
    cursor: pointer;
  }

  .placeholder {
    background: #eee;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 0.9em;
    color: #666;
  }

  .info {
    flex-grow: 1;
  }

  .status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
  }

  .status-processing .status-dot { background-color: #ffc107; }
  .status-completed .status-dot { background-color: #28a745; }
  .status-error .status-dot { background-color: #dc3545; }
  .status-pending .status-dot { background-color: #6c757d; }

  .error-message {
    color: #721c24;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: .75rem 1.25rem;
    margin-top: 5px;
    border-radius: .25rem;
  }

  .details {
    margin: 5px 0;
    font-size: 0.9em;
  }

  .image-button {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    width: 100px;
    height: 100px;
  }

  .image-button.placeholder {
    background: #eee;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 0.9em;
    color: #666;
  }

  .image-button img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border: 1px solid #ccc;
  }
</style> 