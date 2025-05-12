<script lang="ts">
  import { onMount } from 'svelte';
  import AlbumPhotos from './AlbumPhotos.svelte';
  
  interface Photo {
    id: number;
    url: string;
    date?: string;
    status: string;
  }
  
  interface Group {
    id: string;
    name: string;
  }

  interface Album {
    id: number;
    title: string;
    size?: number;
    processed_count?: number;
    sample_photos?: Photo[];
  }
  
  let groups: Group[] = [];
  let albums: Album[] = [];
  let selectedGroupId = '';
  let selectedAlbumId: number | null = null;
  let skipExisting = false;
  let loading = false;
  let showPhotos = false;

  onMount(async () => {
    await fetchGroups();
  });

  async function fetchGroups() {
    try {
      const response = await fetch('/api/groups');
      if (!response.ok) throw new Error(`Ошибка загрузки групп: ${response.status}`);
      const data = await response.json();
      groups = data.groups || [];
    } catch (error) {
      console.error('Не удалось загрузить группы:', error);
    }
  }

  async function fetchAlbums(groupId: string) {
    if (!groupId) {
      albums = [];
      return;
    }

    try {
      const response = await fetch(`/api/albums/${groupId}`);
      if (!response.ok) throw new Error(`Ошибка загрузки альбомов: ${response.status}`);
      const data = await response.json();
      albums = data.albums || [];
    } catch (error) {
      console.error('Не удалось загрузить альбомы:', error);
    }
  }

  async function startIndexing() {
    if (!selectedGroupId || selectedAlbumId === null) {
      alert('Пожалуйста, выберите группу и альбом');
      return;
    }

    loading = true;
    try {
      const response = await fetch('/index/start_by_album', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          album_id: selectedAlbumId,
          skip_existing: skipExisting
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Неизвестная ошибка');
      }
    } catch (error) {
      console.error('Ошибка при запуске индексации:', error);
      alert(`Ошибка: ${error instanceof Error ? error.message : 'Неизвестная ошибка'}`);
    } finally {
      loading = false;
    }
  }
  
  function getAlbumStatus(album: Album): string {
    if (!album.processed_count || !album.size) return 'не обработан';
    
    const percent = Math.round((album.processed_count / album.size) * 100);
    if (percent >= 100) return 'полностью обработан';
    if (percent > 0) return `обработано ${percent}%`;
    return 'не обработан';
  }

  $: if (selectedGroupId) {
    fetchAlbums(selectedGroupId);
  }
  
  $: if (selectedAlbumId !== null) {
    console.log(`IndexingForm: выбран альбом ${selectedAlbumId}`);
    showPhotos = true;
  } else {
    showPhotos = false;
  }

  // Обработчик изменения альбома отдельно от реактивного выражения
  function handleAlbumChange() {
    console.log(`IndexingForm: изменен альбом на ${selectedAlbumId}`);
    // Если нужно еще какие-то действия при изменении альбома
  }
</script>

<form on:submit|preventDefault={startIndexing}>
  <div class="form-group">
    <label for="group_select">Выберите группу:</label>
    <select 
      id="group_select" 
      bind:value={selectedGroupId} 
      required
      disabled={loading}
    >
      <option value="">-- Выберите группу --</option>
      {#each groups as group}
        <option value={group.id}>{group.name}</option>
      {/each}
    </select>
  </div>

  <div class="form-group">
    <label for="album_select">Выберите альбом:</label>
    <select 
      id="album_select" 
      bind:value={selectedAlbumId} 
      on:change={handleAlbumChange}
      required
      disabled={!selectedGroupId || loading}
    >
      <option value={null}>-- Выберите альбом --</option>
      {#each albums as album}
        <option value={album.id}>
          {album.title} ({album.size === undefined ? 'N/A' : album.size} фото, {getAlbumStatus(album)})
        </option>
      {/each}
    </select>
  </div>

  <div class="form-group checkbox">
    <input 
      type="checkbox" 
      id="skip_existing_album" 
      bind:checked={skipExisting}
      disabled={loading}
    >
    <label for="skip_existing_album">
      Пропускать существующие фото (по photo_id в Milvus)
    </label>
  </div>

  <button type="submit" disabled={loading || !selectedAlbumId}>
    {loading ? 'Запуск...' : 'Начать индексацию альбома'}
  </button>
</form>

{#if showPhotos && selectedAlbumId !== null}
  <div class="album-photos-container">
    <h3>Фотографии альбома</h3>
    <AlbumPhotos albumId={selectedAlbumId} key={selectedAlbumId} />
  </div>
{/if}

<style>
  .form-group {
    margin-bottom: 15px;
  }

  label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
  }

  select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
  }

  .checkbox {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .checkbox label {
    margin: 0;
  }

  button {
    background-color: #007bff;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.2s;
  }

  button:hover:not(:disabled) {
    background-color: #0056b3;
  }

  button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }

  .album-photos-container {
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #ddd;
  }
  
  h3 {
    margin-bottom: 15px;
    color: #333;
  }
</style> 