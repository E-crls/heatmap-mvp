# INPI Image Retriever

Este projeto é uma API para busca visual de marcas do INPI (Instituto Nacional da Propriedade Industrial) utilizando modelos de Deep Learning (SigLIP e DINOv3) e o banco de dados vetorial Qdrant.

## Funcionalidades

*   **Busca Visual**: Encontre marcas visualmente similares a uma imagem de consulta.
*   **Modelos de IA**: Suporte para Google SigLIP e Meta DINOv3.
*   **Busca Híbrida**: Combinação dos resultados dos dois modelos para maior precisão.
*   **Filtros de Metadados**: Filtre buscas por campos como Classe Nice, Número do Processo, etc.
*   **Interface Web**: Interface gráfica amigável para testar as buscas.
*   **API REST**: Endpoints documentados para integração.

## Pré-requisitos

*   Python 3.10+
*   Docker (opcional, para rodar em container)
*   Acesso a uma instância do Qdrant
*   Token do Hugging Face (para baixar os modelos)

## Configuração

1. Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
    ```env
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    HF_TOKEN=seu_token_hugging_face
    
    # Nomes das coleções (opcional, padrões abaixo)
    DINOV3_COLLECTION=inpi_marcas_dinov3
    SIGLIP_COLLECTION=inpi_marcas_siglip2
    
    # Configurações da API
    API_PORT=8000
    TOP_K=10
    ```

## Execução com Docker

1.  Construa a imagem:
    ```bash
    docker build -t vision-retriever .
    ```

2.  Execute o container (passando o arquivo .env):
    ```bash
    docker run -p 8000:8000 --env-file .env vision-retriever
    ```

## Uso da API

### Interface Web
Acesse `http://localhost:8000/` no seu navegador para usar a interface gráfica de busca.

### Endpoints Principais

#### 1. Busca (`POST /search`)
Realiza a busca visual.

*   **Parâmetros (Form Data)**:
    *   `file`: Arquivo de imagem (opcional se `image_base64` for enviado).
    *   `image_base64`: String base64 da imagem (opcional se `file` for enviado).
    *   `model`: Modelo a ser usado (`siglip`, `dino` ou `hybrid`). Padrão: `dino`.
    *   `top_k`: Número de resultados. Padrão: 10.
    *   `filters`: String JSON com filtros de metadados (ex: `{"classeNice": [9, 35]}`).


#### 2. Detalhes da Marca (`GET /brand`)
Recupera os dados completos de uma marca pelo ID (ponto no Qdrant).

*   **Parâmetros (Query)**:
    *   `id`: ID da marca.
    *   `collection_name`: Nome da coleção (opcional).

*   **Exemplo**:
    ```bash
    curl "http://localhost:8000/brand?id=12345"
    ```

#### 3. Health Check (`GET /health`)
Verifica o status da API e a conexão com o Qdrant.

*   **Exemplo**:
    ```bash
    curl "http://localhost:8000/health"
    ```

## Estrutura do Projeto

*   `api/`: Definição das rotas FastAPI.
*   `classes/`: Lógica de negócio (Extração de features, Serviço Qdrant, Interface INPI).
*   `settings/`: Configurações e variáveis de ambiente.
*   `static/`: Arquivos estáticos (HTML/JS/CSS) para o frontend.
*   `main.py`: Ponto de entrada da aplicação.
