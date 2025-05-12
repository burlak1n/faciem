<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  
  export let albumId: number | null = null;
  export let showUnprocessedOnly: boolean = false;
  export let key: number | string | null = null;
  
  let photos: any[] = [];
  let loading = false;
  let error: string | null = null;
  let pageData = {
    current: 1,
    total: 0,
    size: 20,
    totalItems: 0
  };
  let pageSizeOptions = [10, 20, 50, 100];
  
  // Добавляем состояние для выбранных фотографий
  let selectedPhotos: Record<number | string, boolean> = {};
  let selectAllChecked = false;
  let selectedCount = 0; // Отдельная переменная для отслеживания количества выбранных фото
  
  // Новая переменная для фильтра статуса
  let statusFilter: string = "all"; // "all", "processed", "not_processed"
  
  // Используем обычную переменную вместо реактивных выражений
  let loadTrigger = {
    albumId: null,
    page: 1,
    pageSize: 20,
    filter: "all", // Меняем на строковое значение
    timestamp: Date.now()  // Для отслеживания изменений
  };
  
  // Таймер для предотвращения мгновенной повторной загрузки
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  
  onMount(() => {
    console.log(`AlbumPhotos: onMount с albumId=${albumId}, key=${key}`);
    if (albumId) {
      initAlbum(albumId);
    }
  });
  
  onDestroy(() => {
    console.log(`AlbumPhotos: onDestroy с albumId=${albumId}, key=${key}`);
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
  });
  
  // Инициализация альбома с правильными начальными параметрами
  function initAlbum(newAlbumId: number | null) {
    console.log(`initAlbum: Инициализация альбома ${newAlbumId}`);
    // Очищаем фотографии перед загрузкой новых
    photos = [];
    
    loadTrigger = {
      albumId: newAlbumId,
      page: 1,
      pageSize: pageData.size,
      filter: statusFilter,
      timestamp: Date.now()
    };
    
    console.log(`initAlbum: Загрузка с параметрами:`, loadTrigger);
    loadPhotos();
  }
  
  // Удаляем реактивное выражение $: if (albumId !== loadTrigger.albumId), заменяем явной функцией
  // Функция вызывается из IndexingForm при изменении albumId
  export function updateAlbumId(newAlbumId: number | null) {
    console.log(`updateAlbumId: ${loadTrigger.albumId} -> ${newAlbumId}`);
    if (newAlbumId !== loadTrigger.albumId) {
      initAlbum(newAlbumId);
    }
  }
  
  function updatePage(newPage: number) {
    if (newPage !== loadTrigger.page && newPage >= 1 && newPage <= pageData.total && !loading) {
      console.log(`updatePage: ${loadTrigger.page} -> ${newPage}`);
      loadTrigger = {
        ...loadTrigger,
        page: newPage,
        timestamp: Date.now()
      };
      loadPhotos();
    }
  }
  
  function updatePageSize(newSize: number) {
    if (newSize !== loadTrigger.pageSize) {
      console.log(`updatePageSize: ${loadTrigger.pageSize} -> ${newSize}`);
      loadTrigger = {
        ...loadTrigger,
        page: 1,  // При изменении размера сбрасываем на первую страницу
        pageSize: newSize,
        timestamp: Date.now()
      };
      loadPhotos();
    }
  }
  
  // Полностью переписанная функция обновления фильтра без реактивности
  function handleFilterChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const newFilter = target.value;
    console.log(`handleFilterChange: Выбран новый фильтр: ${newFilter}`);
    
    if (newFilter !== loadTrigger.filter) {
      // Обновляем локальную переменную для UI
      statusFilter = newFilter;
      
      // Создаем новый объект загрузки с новым фильтром
      loadTrigger = {
        ...loadTrigger,
        filter: newFilter,
        page: 1, // Сбрасываем на первую страницу при изменении фильтра
        timestamp: Date.now()
      };
      
      console.log(`handleFilterChange: Установлен новый loadTrigger:`, loadTrigger);
      
      // Запускаем загрузку с новым фильтром
      loadPhotos();
    }
  }
  
  // Обработка пропа showUnprocessedOnly не используется
  // Удаляем реактивное выражение, которое его отслеживало
  
  async function loadPhotos() {
    if (!loadTrigger.albumId) {
      console.log("loadPhotos: albumId отсутствует, загрузка отменена");
      return;
    }
    
    // Всегда отменяем предыдущий таймер
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    
    // Отменяем загрузку, если уже идет загрузка
    if (loading) {
      console.log("loadPhotos: уже идет загрузка, операция отложена");
    }
    
    console.log(`loadPhotos: планирование загрузки для альбома ${loadTrigger.albumId}, фильтр: ${loadTrigger.filter}`);
    
    // Используем debounce для предотвращения быстрых последовательных запросов
    debounceTimer = setTimeout(async () => {
      if (loading) {
        console.log("loadPhotos: загрузка уже идет, пропускаем");
        return;
      }
      
      console.log(`loadPhotos: начало загрузки для альбома ${loadTrigger.albumId}, применяемый фильтр: ${loadTrigger.filter}`);
      loading = true;
      error = null;
      
      const currentTrigger = {...loadTrigger}; // Копируем текущие параметры загрузки
      
      try {
        const url = `/api/albums/${currentTrigger.albumId}/photos?page=${currentTrigger.page}&page_size=${currentTrigger.pageSize}&status_filter=${currentTrigger.filter}`;
        
        console.log(`loadPhotos: выполняем запрос: ${url}`);
        const response = await fetch(url);
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Ошибка при загрузке фотографий');
        }
        
        const data = await response.json();
        console.log(`loadPhotos: получено ${data.photos?.length || 0} фотографий`);
        
        // Проверяем, что параметры загрузки не изменились за время запроса
        if (
          currentTrigger.albumId === loadTrigger.albumId && 
          currentTrigger.page === loadTrigger.page && 
          currentTrigger.pageSize === loadTrigger.pageSize && 
          currentTrigger.filter === loadTrigger.filter
        ) {
          photos = data.photos;
          pageData = {
            current: data.pagination.page,
            total: data.pagination.total_pages,
            size: currentTrigger.pageSize,
            totalItems: data.pagination.total
          };
        } else {
          console.log("loadPhotos: параметры загрузки изменились, результат проигнорирован");
        }
        
      } catch (err) {
        error = err instanceof Error ? err.message : 'Неизвестная ошибка';
        console.error('Ошибка при загрузке фотографий:', err);
      } finally {
        loading = false;
        debounceTimer = null;
        console.log(`loadPhotos: загрузка завершена, получено ${photos.length} фотографий`);
      }
    }, 50); // Небольшая задержка для дебаунса
  }
  
  function goToPage(page: number) {
    updatePage(page);
  }
  
  function getStatusClass(status: string) {
    return status === 'processed' ? 'processed' : 'not-processed';
  }
  
  function getPageNumbers() {
    const pages = [];
    const maxVisiblePages = 5;
    let startPage = Math.max(1, pageData.current - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(pageData.total, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    
    return pages;
  }
  
  // Функция для обработки URL изображений VK
  function processVkImageUrl(url: string, useMaxSize: boolean = false): string {
    try {
      // Проверяем, является ли URL адресом из VK
      if (!url || !url.includes('userapi.com')) {
        return url;
      }
      
      // Разбиваем URL на части
      const urlObj = new URL(url);
      const params = new URLSearchParams(urlObj.search);
      
      // Получаем параметр 'as', который содержит размеры
      const asSizes = params.get('as');
      if (!asSizes) {
        return url;
      }
      
      // Разбиваем строку размеров на массив
      const sizes = asSizes.split(',');
      
      if (sizes.length === 0) {
        return url;
      }
      
      // Выбираем размер в зависимости от параметра
      let targetSize: string;
      
      if (useMaxSize) {
        // Выбираем максимальный размер (последний в списке)
        targetSize = sizes[sizes.length - 1];
      } else {
        // Выбираем средний размер для предпросмотра
        const middleIndex = Math.floor(sizes.length / 2);
        targetSize = sizes[middleIndex];
      }
      
      // Обновляем параметр 'cs' (current size)
      params.set('cs', targetSize);
      
      // Собираем новый URL
      urlObj.search = params.toString();
      return urlObj.toString();
    } catch (e) {
      console.error('Ошибка при обработке URL:', e);
      return url; // В случае ошибки возвращаем исходный URL
    }
  }
  
  // Получаем полноразмерный URL для перехода
  function getFullSizeUrl(photo: any): string {
    return processVkImageUrl(photo.url, true);
  }
  
  // Получаем URL среднего размера для предпросмотра
  function getPreviewUrl(photo: any): string {
    return processVkImageUrl(photo.url, false);
  }
  
  // Функция для выбора/снятия выбора фотографии
  function togglePhotoSelection(photoId: number | string) {
    console.log(`Переключение выбора фото ID: ${photoId}, текущее состояние:`, selectedPhotos[photoId]);
    
    // Создаем новый объект для обеспечения реактивности
    const newSelected = { ...selectedPhotos };
    newSelected[photoId] = !newSelected[photoId];
    selectedPhotos = newSelected;
    
    // Пересчитываем выбранные
    selectedCount = Object.values(selectedPhotos).filter(Boolean).length;
    console.log(`После переключения, количество выбранных: ${selectedCount}`, selectedPhotos);
    
    updateSelectAllState();
  }
  
  // Функция для выбора/снятия выбора всех фотографий на текущей странице
  function toggleSelectAll() {
    console.log("Переключение выбора всех фото, текущее состояние:", selectAllChecked);
    
    const newSelected = { ...selectedPhotos };
    
    if (selectAllChecked) {
      // Снимаем выбор со всех фотографий текущей страницы
      photos.forEach(photo => {
        newSelected[photo.id] = false;
      });
    } else {
      // Выбираем все фотографии текущей страницы
      photos.forEach(photo => {
        newSelected[photo.id] = true;
      });
    }
    
    selectedPhotos = newSelected;
    selectAllChecked = !selectAllChecked;
    
    // Пересчитываем выбранные
    selectedCount = Object.values(selectedPhotos).filter(Boolean).length;
    console.log(`После переключения всех, количество выбранных: ${selectedCount}`);
  }
  
  // Обновляем состояние "выбрать все" на основе выбранных фотографий
  function updateSelectAllState() {
    if (photos.length === 0) {
      selectAllChecked = false;
      return;
    }
    
    selectAllChecked = photos.every(photo => selectedPhotos[photo.id]);
  }
  
  // Возвращает количество выбранных фотографий
  function getSelectedCount() {
    return selectedCount;
  }
  
  // При изменении списка фотографий (например, при смене страницы)
  // нужно также обновить счетчик выбранных
  $: if (photos) {
    // Пересчитываем выбранные при изменении списка фотографий
    setTimeout(() => {
      selectedCount = Object.values(selectedPhotos).filter(Boolean).length;
      updateSelectAllState();
    }, 0);
  }
  
  // Обработчик нажатия кнопки для разметки выбранных фотографий
  async function handleMarkSelected() {
    // Получаем ID всех выбранных фотографий
    const selectedIds = Object.entries(selectedPhotos)
      .filter(([_, selected]) => selected)
      .map(([id, _]) => id);
    
    if (selectedIds.length === 0) {
      alert('Не выбрано ни одной фотографии');
      return;
    }
    
    console.log(`Отмечено ${selectedIds.length} фото для разметки:`, selectedIds);
    
    // Показываем диалог для выбора типа разметки
    if (!confirm(`Вы уверены, что хотите разметить ${selectedIds.length} фотографий?`)) {
      return;
    }
    
    const markAsProcessed = confirm('Выберите действие:\n\nОК = полная обработка (извлечение лиц, эмбеддингов и запись в Milvus)\nОтмена = удаление записей из Milvus');
    
    try {
      loading = true;
      
      if (markAsProcessed) {
        // Если делаем полную обработку, показываем предупреждение
        alert('Запускается полная обработка выбранных фотографий. Этот процесс может занять некоторое время.');
      }
      
      const response = await fetch('/api/photos/mark', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          photo_ids: selectedIds,
          mark_as_processed: markAsProcessed,
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ошибка при разметке фотографий');
      }
      
      const result = await response.json();
      console.log("Результат разметки:", result);
      
      // Показываем результат
      let resultMessage = `Разметка завершена:\nУспешно: ${result.successful}\nС ошибками: ${result.failed}`;
      
      // Если была обработка лиц, показываем сколько лиц найдено
      if (markAsProcessed && 'faces_found' in result) {
        resultMessage += `\nНайдено лиц: ${result.faces_found}`;
      }
      
      alert(resultMessage);
      
      // Обновляем список фотографий для отображения новых статусов
      loadPhotos();
      
      // Сбрасываем выбор фотографий
      selectedPhotos = {};
      selectedCount = 0;
      selectAllChecked = false;
      
    } catch (error) {
      console.error('Ошибка при разметке фотографий:', error);
      alert(`Ошибка: ${error.message || 'Не удалось разметить фотографии'}`);
    } finally {
      loading = false;
    }
  }
</script>

<div class="album-photos">
  <div class="controls">
    <div class="filter">
      <label for="status-filter">Статус обработки:</label>
      <select 
        id="status-filter" 
        value={statusFilter}
        on:change={handleFilterChange}
      >
        <option value="all">Все фотографии</option>
        <option value="processed">Только обработанные</option>
        <option value="not_processed">Только необработанные</option>
      </select>
    </div>
    
    <div class="page-size">
      <label>
        Фотографий на странице:
        <select on:change={(e) => updatePageSize(Number(e.target.value))} value={loadTrigger.pageSize}>
          {#each pageSizeOptions as size}
            <option value={size}>{size}</option>
          {/each}
        </select>
      </label>
    </div>
  </div>
  
  {#if error}
    <div class="error-message">
      {error}
    </div>
  {/if}
  
  {#if photos.length === 0 && !loading}
    <div class="no-photos">
      {#if albumId}
        В этом альбоме нет фотографий
      {:else}
        Выберите альбом для просмотра фотографий
      {/if}
    </div>
  {/if}
  
  {#if photos.length > 0}
    <div class="selection-controls">
      <div class="select-all">
        <label>
          <input 
            type="checkbox" 
            checked={selectAllChecked}
            on:change={toggleSelectAll}
          >
          Выбрать все на странице
        </label>
      </div>
      
      <div class="selected-count">
        Выбрано: {selectedCount}
      </div>
      
      <div class="action-buttons">
        <button 
          type="button" 
          class="mark-button"
          disabled={selectedCount === 0}
          on:click={handleMarkSelected}
        >
          Разметить выбранные
        </button>
      </div>
    </div>
  {/if}
  
  <div class="photos-grid">
    {#each photos as photo (photo.id)}
      <div class="photo-card {photo.status !== 'processed' ? 'not-processed-card' : ''}">
        <div class="photo-checkbox">
          <input 
            type="checkbox"
            checked={selectedPhotos[photo.id] || false}
            on:change={() => togglePhotoSelection(photo.id)}
          />
        </div>
        <div class="photo-status {getStatusClass(photo.status)}">
          {photo.status === 'processed' ? 'Обработано' : 'Не обработано'}
        </div>
        <a href={getFullSizeUrl(photo)} target="_blank" rel="noopener noreferrer" class="photo-link">
          <img 
            src={getPreviewUrl(photo)} 
            alt="Фото {photo.id}" 
            loading="lazy"
          />
        </a>
        <div class="photo-info">
          ID: {photo.id}
        </div>
      </div>
    {/each}
  </div>
  
  {#if loading}
    <div class="loading">
      Загрузка фотографий...
    </div>
  {/if}
  
  {#if pageData.total > 1}
    <div class="pagination">
      <div class="pagination-info">
        Страница {pageData.current} из {pageData.total} (всего {pageData.totalItems} фото)
      </div>
      
      <div class="pagination-controls">
        <button 
          type="button"
          class="page-button"
          disabled={pageData.current === 1 || loading} 
          on:click={() => goToPage(1)}
        >
          &laquo;
        </button>
        
        <button 
          type="button"
          class="page-button"
          disabled={pageData.current === 1 || loading} 
          on:click={() => goToPage(pageData.current - 1)}
        >
          &lsaquo;
        </button>
        
        {#each getPageNumbers() as page}
          <button 
            type="button"
            class="page-button {page === pageData.current ? 'active' : ''}"
            disabled={loading || page === pageData.current}
            on:click={() => goToPage(page)}
          >
            {page}
          </button>
        {/each}
        
        <button 
          type="button"
          class="page-button"
          disabled={pageData.current === pageData.total || loading} 
          on:click={() => goToPage(pageData.current + 1)}
        >
          &rsaquo;
        </button>
        
        <button 
          type="button"
          class="page-button"
          disabled={pageData.current === pageData.total || loading} 
          on:click={() => goToPage(pageData.total)}
        >
          &raquo;
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .album-photos {
    margin-top: 20px;
  }
  
  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }
  
  .filter {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  .filter select {
    padding: 4px;
    border-radius: 4px;
    border: 1px solid #ddd;
    min-width: 200px;
  }
  
  .page-size {
    display: flex;
    align-items: center;
  }
  
  .page-size select {
    margin-left: 8px;
    padding: 4px;
    border-radius: 4px;
    border: 1px solid #ddd;
  }
  
  .photos-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
  }
  
  .photo-card {
    border: 1px solid #ddd;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  
  /* Красная рамка для необработанных фотографий */
  .not-processed-card {
    border: 2px solid #dc3545;
  }
  
  .photo-card img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    display: block;
  }
  
  .photo-status {
    position: absolute;
    top: 0;
    right: 0;
    padding: 4px 8px;
    color: white;
    font-size: 12px;
  }
  
  .processed {
    background-color: #28a745;
  }
  
  .not-processed {
    background-color: #dc3545;
  }
  
  .photo-info {
    padding: 8px;
    background-color: #f8f9fa;
    font-size: 14px;
  }
  
  .loading,
  .no-photos,
  .error-message {
    text-align: center;
    padding: 15px;
    margin: 10px 0;
  }
  
  .error-message {
    color: #dc3545;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 4px;
  }
  
  .no-photos {
    color: #6c757d;
    font-style: italic;
  }
  
  .pagination {
    margin-top: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }
  
  .pagination-info {
    font-size: 14px;
    color: #6c757d;
  }
  
  .pagination-controls {
    display: flex;
    gap: 5px;
  }
  
  .page-button {
    background-color: #f8f9fa;
    border: 1px solid #ddd;
    color: #495057;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    min-width: 40px;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .page-button:hover:not(:disabled) {
    background-color: #e9ecef;
  }
  
  .page-button.active {
    background-color: #007bff;
    color: white;
    border-color: #007bff;
  }
  
  .page-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .photo-link {
    display: block;
    cursor: pointer;
    overflow: hidden;
  }
  
  /* Стили для элементов выбора фотографий */
  .selection-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 4px;
    border: 1px solid #ddd;
  }
  
  .select-all label {
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
  }
  
  .selected-count {
    font-weight: bold;
    color: #495057;
  }
  
  .action-buttons {
    display: flex;
    gap: 10px;
  }
  
  .mark-button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 14px;
  }
  
  .mark-button:hover:not(:disabled) {
    background-color: #0056b3;
  }
  
  .mark-button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }
  
  .photo-checkbox {
    position: absolute;
    top: 0;
    left: 0;
    padding: 5px;
    z-index: 10;
  }
  
  .photo-checkbox input[type="checkbox"] {
    width: 18px;
    height: 18px;
  }
</style> 