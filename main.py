from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from decisao_service import DecisaoService 
from fastapi.middleware.cors import CORSMiddleware 
import os

# A remoção da definição explícita de os.environ["GOOGLE_APPLICATION_CREDENTIALS"] 
# aqui garante que o BigQuery.Client use a lógica do decisao_service.py

# --- Modelos de Pydantic ---
class RankingRequest(BaseModel):
    municipio_id: str
    curso_nome: str
    top_n: Optional[int] = 3 

# A classe de serviço será inicializada UMA VEZ ao iniciar a API.
try:
    decisao_service = DecisaoService()
except Exception as e:
    print(f"⚠️ Erro Crítico ao inicializar DecisaoService: {e}")
    decisao_service = None
    
# --- Instância do FastAPI ---
app = FastAPI(
    title="Sistema de Apoio à Decisão de Cursos",
    description="API que utiliza dados do Censo, IGC e IDD para ranquear cursos."
)

# --- 🎯 Configuração do CORS ---
origins = [
    "http://localhost:3000",
    "https://rankeduc-frontend-drmm.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Função Auxiliar para Obter o Serviço
# ----------------------------------------------------
def get_decisao_service():
    """Verifica a disponibilidade do serviço ao receber uma requisição."""
    global decisao_service 

    if decisao_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de Análise de Dados indisponível. Falha na inicialização ou no acesso aos dados."
        )
    if decisao_service.df_base_treinamento.empty:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço indisponível: Dados insuficientes carregados para o treinamento do modelo."
        )
    return decisao_service

# ----------------------------------------------------
# Rota 1: Obter Ranking (Decisão) 🏆
# ----------------------------------------------------
@app.post(
    "/ranking/", 
    response_model=Dict[str, Any],
    summary="Obtém o ranking das IES para um curso/município (Retorno Completo)."
)
async def get_ranking_completo(request: RankingRequest):
    service = get_decisao_service() 
    
    resultado = service.obter_ranking_api(
        municipio_id=request.municipio_id,
        curso_nome=request.curso_nome
    )
    
    if not resultado.get("ranking_top_ies"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=resultado.get("mensagem") or "Nenhuma IES encontrada para o curso/município especificado."
        )
        
    return resultado

# ----------------------------------------------------
# Rota 2: Listar Cursos por Município 📚
# ----------------------------------------------------
@app.get(
    "/cursos/{municipio_id}", 
    response_model=List[str],
    summary="Lista todos os cursos disponíveis para um município específico."
)
async def list_cursos_por_municipio(municipio_id: str):
    service = get_decisao_service() 
    
    try:
        cursos = service.obter_cursos_por_municipio(municipio_id) 
    except Exception as e:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao buscar cursos: {e}"
        )
    
    if not cursos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum curso encontrado para o ID de município: {municipio_id}"
        )
    return cursos

# ----------------------------------------------------
# Execução (Para rodar localmente)
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)