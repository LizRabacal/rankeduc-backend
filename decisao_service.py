import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import statsmodels.api as sm 
from typing import List, Dict, Any
import json 
import re 
import logging
import os
from google.oauth2 import service_account 
from google.cloud import bigquery 
import pandas_gbq as gbq 
import basedosdados as bd # Mantido apenas para evitar erros de importação em outros locais do seu projeto

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURAÇÕES GLOBAIS ---
PROJECT_ID = "teak-flash-421822" 
ARQUIVO_IGC = "IGC_2023.xlsx"
ARQUIVO_IDD = "IDD_2023.xlsx" 
ANO_REF = 2023 

FEATURES_COLS = [
    'id_municipio', 'nome_curso', 'taxa_concorrencia',
    'taxa_evasao', 'idd_normalizado', 'igc_normalizado'
]
TARGET_COL = 'target_desempenho'

# Instância global do cliente BigQuery
BQ_CLIENT = None 

class DecisaoService:
    def __init__(self):
        DecisaoService._autenticar_bigquery_gcp()
        
        self.regressao_pesos = {'CONCLUSAO': 0.30, 'EVASAO_INVERSA': 0.15, 'IDD': 0.10, 'IGC': 0.45} 
        
        self.df_igc, self.df_idd = self._carregar_dados_auxiliares()
        self.df_base_treinamento = self._carregar_amostra_para_treinamento()
        self.rf_model = None
        self.le = None
        self.X_train_cols = []
        
        if not self.df_base_treinamento.empty:
            self.regressao_pesos = self._calcular_pesos_regressao() 
            self.df_base_treinamento = self._executar_feature_engineering(self.df_base_treinamento)
            self._treinar_modelo()
            logging.info("Serviço de Decisão inicializado e modelo treinado com sucesso.")
        else:
            logging.warning("Não foi possível carregar dados para treinamento. Serviço limitado a ranking por score (usando pesos default).")
            self.regressao_pesos = self.regressao_pesos 
            self._treinar_modelo = lambda: logging.warning("Skipping model training due to empty data.") 
            self.X_train_cols = []
    
    @staticmethod
    def _autenticar_bigquery_gcp():
        """
        Função estática para forçar a autenticação usando a Chave JSON Secreta (Render) 
        ou a variável GOOGLE_APPLICATION_CREDENTIALS (Local/Docker).
        """
        global BQ_CLIENT
        
        # 1. Tenta carregar do JSON STRING (Método Render/Segurança)
        json_key_string = os.environ.get('GCP_SERVICE_KEY_JSON') 
        
        if json_key_string:
            try:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(json_key_string),
                    scopes=['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/bigquery']
                )
                BQ_CLIENT = bigquery.Client(project=PROJECT_ID, credentials=credentials)
                logging.info("Autenticação BigQuery via JSON String (Render) estabelecida com sucesso.")
                return 
            except Exception as e:
                logging.error(f"Falha na autenticação via JSON String: {e}")
        
        # 2. Tenta Autenticação Padrão (Local ou GOOGLE_APPLICATION_CREDENTIALS de arquivo)
        try:
            BQ_CLIENT = bigquery.Client(project=PROJECT_ID)
            logging.info("Autenticação BigQuery via GOOGLE_APPLICATION_CREDENTIALS (Padrão) estabelecida com sucesso.")
            return
        except Exception as e:
            logging.error(f"Falha final na autenticação do BigQuery: {e}")
            
        logging.error("Falha final na autenticação do BigQuery. Serviço de dados desabilitado.")


    def _carregar_dados_auxiliares(self):
        """Carrega dados IGC e IDD com limpeza rigorosa nas chaves (Do XLSX)."""
        try:
            df_igc = pd.read_excel(ARQUIVO_IGC)
            df_igc.rename(columns={' Código da IES': 'id_ies', ' Nome da IES': 'nome_ies', ' IGC (Contínuo)': 'igc_continuo'}, inplace=True)
            df_igc['id_ies'] = df_igc['id_ies'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_igc = df_igc[['id_ies', 'nome_ies', 'igc_continuo']].drop_duplicates(subset=['id_ies'])

            df_idd = pd.read_excel(ARQUIVO_IDD)
            df_idd.rename(columns={' Código da IES': 'id_ies', ' Código do Curso': 'id_curso', ' IDD (Contínuo)': 'idd_continuo' }, inplace=True)
            
            df_idd['id_ies'] = df_idd['id_ies'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_idd['id_curso'] = df_idd['id_curso'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_idd['id_curso'] = df_idd['id_curso'].apply(lambda x: x.lstrip('0') if x.isdigit() else x)
            
            df_idd = df_idd[['id_ies', 'id_curso', 'idd_continuo']].drop_duplicates(subset=['id_ies', 'id_curso'])
            logging.info("Dados auxiliares (IGC/IDD) carregados e chaves limpas.")
            return df_igc, df_idd
        except Exception as e:
            logging.error(f"Erro no carregamento de IGC/IDD (XLSX): {e}")
            return pd.DataFrame({'id_ies': [], 'nome_ies': [], 'igc_continuo': []}), \
                   pd.DataFrame({'id_ies': [], 'id_curso': [], 'idd_continuo': []})
    
    def _carregar_amostra_para_treinamento(self) -> pd.DataFrame:
        """Carrega amostra real do BigQuery e mescla IGC/IDD dos XLSX."""
        if BQ_CLIENT is None:
            logging.error("Cliente BigQuery não inicializado. Não é possível carregar amostra.")
            return pd.DataFrame()
            
        try:
            query_amostra = f"""
                SELECT
                    CAST(id_municipio AS STRING) as id_municipio,
                    nome_curso,
                    CAST(id_ies AS STRING) as id_ies,
                    CAST(id_curso AS STRING) as id_curso,
                    quantidade_vagas,
                    quantidade_inscritos,
                    quantidade_matriculas,
                    quantidade_concluintes,
                    quantidade_alunos_situacao_trancada,
                    quantidade_alunos_situacao_desvinculada,
                    tipo_organizacao_administrativa
                FROM `basedosdados.br_inep_censo_educacao_superior.curso`
                WHERE ano = {ANO_REF} AND quantidade_matriculas > 0
                LIMIT 5000 
            """
            logging.info("Consultando AMOSRA para treinamento na Base dos Dados...")
            
            df = pd.io.gbq.read_gbq(query_amostra, project_id=PROJECT_ID, credentials=BQ_CLIENT._credentials)
            
            logging.info(f"Amostra carregada para treino. Total de linhas: {len(df)}")
            
            # Limpeza e Mesclagem de dados
            df['id_ies'] = df['id_ies'].astype(str).str.strip()
            df['id_curso'] = df['id_curso'].astype(str).str.strip()
            df['id_curso'] = df['id_curso'].apply(lambda x: x.lstrip('0') if x.isdigit() else x)
            
            df = pd.merge(df, self.df_igc[['id_ies', 'nome_ies', 'igc_continuo']], on='id_ies', how='left')
            df = pd.merge(df, self.df_idd[['id_ies', 'id_curso', 'idd_continuo']], on=['id_ies', 'id_curso'], how='left')
            
            df['igc_continuo'] = df['igc_continuo'].fillna(0)
            df['idd_continuo'] = df['idd_continuo'].fillna(0) 
            df.drop(columns=['nome_ies'], inplace=True, errors='ignore')

            return self._executar_feature_engineering(df)
            
        except Exception as e:
            logging.error(f"Erro ao carregar AMOSRA para treinamento: {e}. Retornando DataFrame vazio.")
            return pd.DataFrame()

    def _calcular_pesos_regressao(self):
        if self.df_base_treinamento.empty:
            return {'CONCLUSAO': 0.30, 'EVASAO_INVERSA': 0.15, 'IDD': 0.10, 'IGC': 0.45}

        logging.info("Determinando pesos das variáveis de qualidade via Regressão (OLSM)...")
        
        df_reg = self.df_base_treinamento.copy()
        
        scaler = MinMaxScaler()
        df_reg['y_taxa_conclusao_norm'] = scaler.fit_transform(df_reg[['taxa_conclusao']])
        df_reg['x_idd_norm'] = df_reg['idd_normalizado']
        df_reg['x_igc_norm'] = df_reg['igc_normalizado']

        X = df_reg[['x_idd_norm', 'x_igc_norm']] 
        y = df_reg['y_taxa_conclusao_norm']
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X, missing='drop') 
            results = model.fit()
            
            pesos_base = {
                'IDD': abs(results.params.get('x_idd_norm', 0.1)),
                'IGC': abs(results.params.get('x_igc_norm', 0.1)),
                'EVASAO_INVERSA': abs(results.params.get('x_idd_norm', 0.1)), 
                'CONCLUSAO': abs(results.params.get('const', 0.50)), 
            }
            
            soma_pesos = sum(pesos_base.values())
            pesos_normalizados = {k: v / soma_pesos for k, v in pesos_base.items()}

            logging.info(f"Pesos calculados por Regressão: {json.dumps(pesos_normalizados, indent=2)}")
            return pesos_normalizados
            
        except Exception as e:
            logging.error(f"Erro na Regressão para calcular pesos: {e}. Usando pesos default.")
            return {'CONCLUSAO': 0.30, 'EVASAO_INVERSA': 0.15, 'IDD': 0.10, 'IGC': 0.45} 

    def _executar_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df['idd_continuo'] = df['idd_continuo'].fillna(0)
        df['igc_continuo'] = df['igc_continuo'].fillna(0)
        df.fillna(0, inplace=True)
        
        df['taxa_concorrencia'] = df['quantidade_inscritos'] / df['quantidade_vagas']
        df['taxa_concorrencia'].replace([np.inf, -np.inf], 0, inplace=True)
        
        df['total_evasao'] = df['quantidade_alunos_situacao_trancada'] + df['quantidade_alunos_situacao_desvinculada']
        df['taxa_evasao'] = np.where(df['quantidade_matriculas'] > 0, df['total_evasao'] / df['quantidade_matriculas'], 0)
        df['taxa_conclusao'] = np.where(df['quantidade_matriculas'] > 0, df['quantidade_concluintes'] / df['quantidade_matriculas'], 0)

        pesos = self.regressao_pesos 
        
        df['idd_normalizado'] = df['idd_continuo'] / 5.0
        df['igc_normalizado'] = df['igc_continuo'] / 5.0

        df['score_qualidade'] = (
            (df['taxa_conclusao'] * pesos['CONCLUSAO']) +
            ((1 - df['taxa_evasao']) * pesos['EVASAO_INVERSA']) +
            (df['idd_normalizado'] * pesos['IDD']) +
            (df['igc_normalizado'] * pesos['IGC'])
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf, np.nan], 0).round(4)
        
        if len(df) > 10 and not df['score_qualidade'].empty:
            q1 = df['score_qualidade'].quantile(0.66)
            q2 = df['score_qualidade'].quantile(0.33)

            def categorizar_score(score):
                if score >= q1: return 'Classe 3: Melhor Desempenho'
                elif score >= q2: return 'Classe 2: Desempenho Intermediário'
                else: return 'Classe 1: Desempenho Básico'

            df['target_desempenho'] = df['score_qualidade'].apply(categorizar_score)
        else:
             df['target_desempenho'] = 'Classe 2: Desempenho Intermediário' 

        return df

    def _treinar_modelo(self):
        # 1. Obtenção das Features
        X = self.df_base_treinamento[FEATURES_COLS].copy()
        y = self.df_base_treinamento[TARGET_COL]

        # 2. CORREÇÃO CRÍTICA: Forçar conversão de tipos para string e preencher NaNs
        # Isso garante que LabelEncoder só veja strings, resolvendo ['int', 'str'].
        # Use .apply(str) e .astype(str) juntas para máxima segurança.
        X['id_municipio'] = X['id_municipio'].apply(str).astype(str).fillna('0000000') 
        X['nome_curso'] = X['nome_curso'].apply(str).astype(str).fillna('DESCONHECIDO')

        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        
        X_copy = X.copy()
        
        self.le_municipio = LabelEncoder()
        self.le_curso = LabelEncoder()
        X_copy['id_municipio'] = self.le_municipio.fit_transform(X_copy['id_municipio'])
        X_copy['nome_curso'] = self.le_curso.fit_transform(X_copy['nome_curso'])
        
        X_encoded = pd.get_dummies(X_copy, columns=['id_municipio', 'nome_curso'], drop_first=True)
        self.X_train_cols = X_encoded.columns.tolist()

        if X_encoded.shape[0] < 2:
            return

        X_train, _, y_train, _ = train_test_split(
            X_encoded, y_encoded, test_size=0.3, random_state=42
        )

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train, y_train)
        
    def obter_ranking(self, municipio_id: str, curso_nome: str) -> List[Dict[str, Any]]:
        if BQ_CLIENT is None:
             logging.error("BigQuery Client não está pronto para consulta de ranking.")
             return []
             
        curso_sql = re.sub(r'[^\w\s]', '', curso_nome).strip().upper()
        
        try:
            query_filtrada = f"""
                SELECT
                    CAST(id_municipio AS STRING) as id_municipio,
                    nome_curso,
                    CAST(id_ies AS STRING) as id_ies,
                    CAST(id_curso AS STRING) as id_curso,
                    tipo_organizacao_administrativa,
                    quantidade_vagas,
                    quantidade_inscritos,
                    quantidade_matriculas,
                    quantidade_concluintes,
                    quantidade_alunos_situacao_trancada,
                    quantidade_alunos_situacao_desvinculada
                FROM `basedosdados.br_inep_censo_educacao_superior.curso`
                WHERE 
                    ano = {ANO_REF} AND 
                    quantidade_matriculas > 0 AND
                    id_municipio = '{municipio_id}' AND 
                    UPPER(nome_curso) = '{curso_sql}'
            """
            logging.info(f"Consulta para Ranking: Município={municipio_id}, Curso='{curso_nome}'")
            
            df_filtrado = pd.io.gbq.read_gbq(query_filtrada, project_id=PROJECT_ID, credentials=BQ_CLIENT._credentials)
            
            if df_filtrado.empty:
                return []
                
            df_filtrado['id_ies'] = df_filtrado['id_ies'].astype(str).str.strip()
            df_filtrado['id_curso'] = df_filtrado['id_curso'].astype(str).str.strip()
            df_filtrado['id_curso'] = df_filtrado['id_curso'].apply(lambda x: x.lstrip('0') if x.isdigit() else x)
            
            df_filtrado = pd.merge(df_filtrado, self.df_igc[['id_ies', 'nome_ies', 'igc_continuo']], on='id_ies', how='left')
            df_filtrado = pd.merge(df_filtrado, self.df_idd[['id_ies', 'id_curso', 'idd_continuo']], on=['id_ies', 'id_curso'], how='left')
            
            df_filtrado['igc_continuo'] = df_filtrado['igc_continuo'].fillna(0)
            df_filtrado['idd_continuo'] = df_filtrado['idd_continuo'].fillna(0) 

            df_filtrado = self._executar_feature_engineering(df_filtrado)
            
            df_filtrado.sort_values(by='score_qualidade', ascending=False, inplace=True)
            decisoes = df_filtrado.head(15)

            resultados = []
            for row in decisoes.itertuples():
                nome_ies_value = getattr(row, 'nome_ies', 'N/D') 
                nome_ies_final = nome_ies_value if not pd.isna(nome_ies_value) else 'N/D'

                resultados.append({
                    'nome_curso': row.nome_curso,
                    'nome_ies': nome_ies_final,
                    'id_ies': row.id_ies,
                    'tipo_organizacao_administrativa': row.tipo_organizacao_administrativa,
                    'score_qualidade': float(row.score_qualidade),
                    'target_desempenho': row.target_desempenho,
                    'idd_continuo': float(row.idd_continuo),
                    'igc_continuo': float(row.igc_continuo),
                    'taxa_conclusao': float(row.taxa_conclusao),
                    'taxa_evasao': float(row.taxa_evasao),
                    'taxa_concorrencia': float(row.taxa_concorrencia)
                })
            
            return resultados
            
        except Exception as e:
            logging.error(f"Erro na consulta dinâmica ou processamento: {e}")
            return []
    
    def obter_cursos_por_municipio(self, municipio_id: str) -> List[str]:
        if BQ_CLIENT is None:
             logging.error("BigQuery Client não está pronto para consulta de cursos.")
             return []
             
        query_cursos = f"""
            SELECT DISTINCT
                nome_curso
            FROM `basedosdados.br_inep_censo_educacao_superior.curso`
            WHERE 
                ano = {ANO_REF} AND 
                id_municipio = '{municipio_id}' AND
                quantidade_matriculas > 0
            ORDER BY nome_curso
        """
        
        logging.info(f"Consultando cursos para o município: {municipio_id}...")
        
        try:
            df_cursos = pd.io.gbq.read_gbq(query_cursos, project_id=PROJECT_ID, credentials=BQ_CLIENT._credentials)
            
            if df_cursos.empty:
                return []
            
            cursos_list = df_cursos['nome_curso'].astype(str).tolist()
            return cursos_list
            
        except Exception as e:
            logging.error(f"Erro ao listar cursos por município no BigQuery: {e}")
            return []

    def obter_ranking_api(self, municipio_id: str, curso_nome: str) -> Dict[str, Any]:
        ranking_list = self.obter_ranking(municipio_id, curso_nome)

        if not ranking_list:
            return {
                "status": "sucesso",
                "parametros_busca": {"municipio_id": municipio_id, "curso_nome": curso_nome, "ano": ANO_REF},
                "total_cursos_encontrados": 0,
                "mensagem": f"Nenhuma IES/Curso '{curso_nome}' encontrada no município '{municipio_id}' para o ano {ANO_REF}.",
                "ranking_top_ies": []
            }
        
        return {
            "status": "sucesso",
            "parametros_busca": {"municipio_id": municipio_id, "curso_nome": curso_nome, "ano": ANO_REF},
            "total_cursos_encontrados": len(ranking_list),
            "ranking_top_ies": ranking_list
        }