<script lang="ts">
  import IndexingForm from './components/IndexingForm.svelte';
  import SearchForm from './components/SearchForm.svelte';
  import ProgressArea from './components/ProgressArea.svelte';
  import Charts from './components/Charts.svelte';
  import SearchResults from './components/SearchResults.svelte';
  import './app.css';

  let searchLoading = false;
  let searchError = '';
  let searchResults: any[] = [];

  async function handleSearch(event: CustomEvent<{ file: File }>) {
    const file = event.detail.file;
    searchLoading = true;
    searchError = '';
    searchResults = [];

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/search/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Неизвестная ошибка');
      }

      const data = await response.json();
      searchResults = data.results || [];
    } catch (error) {
      console.error('Ошибка при поиске:', error);
      searchError = error instanceof Error ? error.message : 'Произошла ошибка при поиске';
    } finally {
      searchLoading = false;
    }
  }
</script>

<main>
  <div class="container">
    <h1>Milvus VK Pipeline</h1>
    
    <IndexingForm />
    <ProgressArea />
    <Charts />
  </div>
  <div class="container search-container">
    <h1>Поиск по фото</h1>
    <SearchForm on:search={handleSearch} />
    <SearchResults 
      results={searchResults}
      loading={searchLoading}
      error={searchError}
    />
  </div>
</main>

<style>
  main {
    font-family: Arial, sans-serif;
    margin: 20px;
    background-color: #f4f4f4;
    color: #333;
  }

  .container {
    background-color: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
  }

  h1 {
    color: #333;
    margin-bottom: 20px;
  }

  .search-container {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #ccc;
  }
</style>
