<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{
    search: { file: File };
  }>();

  let fileInput: HTMLInputElement;
  let loading = false;
  let error = '';

  async function handleSubmit() {
    const file = fileInput.files?.[0];
    if (!file) {
      error = 'Пожалуйста, выберите файл изображения.';
      return;
    }

    error = '';
    loading = true;
    dispatch('search', { file });
  }
</script>

<form on:submit|preventDefault={handleSubmit}>
  <div class="form-group">
    <label for="searchImage">Выберите изображение для поиска:</label>
    <input 
      type="file" 
      id="searchImage" 
      accept="image/*" 
      required
      bind:this={fileInput}
      disabled={loading}
    />
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  <button type="submit" disabled={loading}>
    {loading ? 'Поиск...' : 'Найти похожие'}
  </button>
</form>

<style>
  .form-group {
    margin-bottom: 15px;
  }

  label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
  }

  input[type="file"] {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
  }

  .error-message {
    color: #721c24;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: .75rem 1.25rem;
    margin-bottom: 15px;
    border-radius: .25rem;
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
</style> 