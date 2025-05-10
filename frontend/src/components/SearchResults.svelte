<script lang="ts">
  interface SearchResult {
    match_photo_id: string;
    match_face_index: number;
    query_face_index: number;
    similarity: number;
    match_url?: string;
    match_date?: string;
  }

  export let results: SearchResult[] = [];
  export let loading = false;
  export let error = '';

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
</script>

<div class="search-results">
  {#if loading}
    <div class="loading">Поиск похожих лиц...</div>
  {:else if error}
    <div class="error-message">{error}</div>
  {:else if results.length === 0}
    <div class="no-results">Похожие лица не найдены.</div>
  {:else}
    <div class="results-grid">
      {#each results as result}
        <div class="result-item">
          {#if result.match_url}
            <button 
              class="image-button"
              on:click={() => window.open(result.match_url, '_blank')}
              on:keydown={(e) => e.key === 'Enter' && window.open(result.match_url, '_blank')}
            >
              <img 
                src={result.match_url} 
                alt="Match ID: {result.match_photo_id}"
                title="Нажмите, чтобы открыть оригинал"
              />
            </button>
          {:else}
            <div class="placeholder">
              Нет URL<br>(ID: {result.match_photo_id})
            </div>
          {/if}
          
          <div class="result-info">
            <p>Photo ID: {result.match_photo_id}</p>
            <p>Face Idx: {result.match_face_index}</p>
            <p>Query Face: {result.query_face_index}</p>
            <p>Similarity: {result.similarity}</p>
            {#if result.match_date}
              <p>Дата: {formatUnixTimestamp(result.match_date)}</p>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .search-results {
    margin-top: 20px;
  }

  .loading {
    text-align: center;
    color: #666;
    padding: 20px;
  }

  .error-message {
    color: #721c24;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: .75rem 1.25rem;
    border-radius: .25rem;
  }

  .no-results {
    text-align: center;
    color: #666;
    padding: 20px;
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 20px;
    margin-top: 15px;
  }

  .result-item {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    background-color: #fff;
    text-align: center;
  }

  .image-button {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    width: 100%;
  }

  .image-button img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 4px;
    margin-bottom: 10px;
  }

  .placeholder {
    width: 100%;
    height: 200px;
    background: #eee;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9em;
    color: #666;
    border-radius: 4px;
    margin-bottom: 10px;
  }

  .result-info p {
    margin: 5px 0;
    font-size: 0.9em;
    word-wrap: break-word;
  }
</style> 