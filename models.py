# models.py
from pydantic import BaseModel
from typing import List, Optional

# --- Estrutura da Requisição para o Ranking ---
class RankingRequest(BaseModel):
    """Modelo para a requisição de busca do ranking."""
    municipio_id: str
    curso_nome: str

# --- Estrutura do Resultado do Ranking ---
class IESRanking(BaseModel):
    """Modelo para um item do ranking final."""
    nome_curso: str
    nome_ies: str
    id_ies: str
    tipo_organizacao_administrativa: str
    score_qualidade: float
    target_desempenho: str
    idd_continuo: float
    igc_continuo: float
    taxa_conclusao: float
    taxa_evasao: float
    taxa_concorrencia: float