from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Модели для эндпоинтов FaceManager ---

class SearchResultItem(BaseModel):
    match_photo_id: str # Теперь это основной ID фото
    match_face_index: int # Индекс лица на фото
    query_face_index: int
    similarity: float
    # match_url и match_date можно будет добавить позже, получая их из SQLite по match_photo_id
    match_url: Optional[str] = None 
    match_date: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# Модели для удаленных эндпоинтов (оставлены на всякий случай)
class AddResponse(BaseModel):
    message: str
    added_pks: Optional[List[int]] = None
    face_ids_added: Optional[List[str]] = None

class ClusterRequest(BaseModel):
    mode: str = Field(default="incremental", description="'full' или 'incremental'")
    eps: float = Field(default=0.4, description="Параметр eps для DBSCAN")
    min_samples: int = Field(default=2, description="Параметр min_samples для DBSCAN")

class ClusterResponse(BaseModel):
    message: str
    mode: str

# --- Модели для эндпоинтов работы с vk_faces.db ---

class GroupItem(BaseModel):
    id: str
    name: str

class GroupListResponse(BaseModel):
    groups: List[GroupItem]

class AlbumItem(BaseModel):
    id: int
    title: str
    size: int
    # processed_count: Optional[int] = None # Пока не возвращаем

class AlbumListResponse(BaseModel):
    albums: List[AlbumItem]

# Можно добавлять сюда другие модели по мере необходимости 