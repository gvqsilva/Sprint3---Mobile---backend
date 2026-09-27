from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from groq import Groq
from dotenv import load_dotenv


# ==========================================
# CONFIGURAÇÃO
# ==========================================

load_dotenv()

app = FastAPI(title="AutoIntel AI API")


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# GROQ
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY não encontrada. "
        "Verifique o arquivo .env."
    )

client = Groq(api_key=GROQ_API_KEY)


# ==========================================
# MODELOS
# ==========================================

class PesquisaRequest(BaseModel):
    marca: str
    categoria: str
    modelo: str
    versao: str


class DueloRequest(BaseModel):
    nome_completo: str
    preco_concorrente: str
    perfil_uso: str

class CopilotoRequest(BaseModel):
    mensagem: str

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def health_check():
    return {
        "status": "AutoIntel API Operacional",
        "database": "Local no dispositivo",
        "storage": "AsyncStorage"
    }


# ==========================================
# PESQUISA DE VEÍCULO
# ==========================================

@app.post("/api/v1/pesquisa")
async def mapear_especificacoes(req: PesquisaRequest):

    prompt = f"""
Você é um analista sênior do mercado automotivo brasileiro.

Analise o seguinte veículo:

Marca: {req.marca}
Categoria: {req.categoria}
Modelo: {req.modelo}
Versão: {req.versao}

Gere dados técnicos realistas para o mercado brasileiro.

Retorne ESTRITAMENTE JSON válido no seguinte formato:

{{
    "nomeCompleto": "Nome completo do veículo",
    "especificacoes": [
        {{
            "atributo": "Preço Estimado Atual (R$)",
            "valor": "R$ 000.000"
        }},
        {{
            "atributo": "Potência (cv)",
            "valor": "000 cv"
        }},
        {{
            "atributo": "Torque (kgfm)",
            "valor": "00 kgfm"
        }},
        {{
            "atributo": "Motorização",
            "valor": "..."
        }},
        {{
            "atributo": "Transmissão/Câmbio",
            "valor": "..."
        }},
        {{
            "atributo": "Tração",
            "valor": "..."
        }},
        {{
            "atributo": "Capacidade de Carga/Porta-Malas",
            "valor": "..."
        }},
        {{
            "atributo": "Suspensão/Amortecedores",
            "valor": "..."
        }},
        {{
            "atributo": "Ângulo de Ataque",
            "valor": "..."
        }},
        {{
            "atributo": "Modos de Condução",
            "valor": "..."
        }}
    ],
    "scores": {{
        "forca": 0,
        "tecnologia": 0,
        "offRoad": 0,
        "custoBeneficio": 0
    }}
}}

As notas devem ser números de 0 a 10.
"""


    try:

        resposta = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um especialista automotivo "
                        "do mercado brasileiro. "
                        "Retorne exclusivamente JSON válido."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        conteudo = resposta.choices[0].message.content

        if not conteudo:
            raise ValueError(
                "A Groq retornou uma resposta vazia."
            )

        resultado = json.loads(conteudo)

        return resultado

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="A IA retornou um JSON inválido."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao consultar a IA: {str(e)}"
        )


# ==========================================
# DUELO ESTRATÉGICO
# ==========================================

@app.post("/api/v1/duelo")
async def gerar_duelo(req: DueloRequest):

    prompt = f"""
Analise tecnicamente a comparação entre:

VEÍCULO CONCORRENTE:
{req.nome_completo}

PREÇO DO CONCORRENTE:
{req.preco_concorrente}

VEÍCULO DE REFERÊNCIA:
Ford Ranger Raptor V6

PERFIL DE USO:
{req.perfil_uso}

Produza uma análise executiva curta.

Considere:
- desempenho;
- motorização;
- capacidade;
- tecnologia;
- uso pretendido;
- diferença de preço;
- custo-benefício.

Retorne ESTRITAMENTE JSON válido:

{{
    "veredito": "Texto da análise"
}}

O texto deve ter no máximo 4 linhas.
"""


    try:

        resposta = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um especialista automotivo. "
                        "Responda exclusivamente em JSON válido, "
                        "sem Markdown."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        conteudo = resposta.choices[0].message.content

        if not conteudo:
            raise ValueError(
                "A Groq retornou uma resposta vazia."
            )

        resultado = json.loads(conteudo)

        return resultado

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="A IA retornou um JSON inválido."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar duelo: {str(e)}"
        )


# ==========================================
# COPILOTO IA
# ==========================================

@app.post("/api/v1/copiloto")
async def gerar_copiloto(req: CopilotoRequest):

    if not req.mensagem.strip():
        raise HTTPException(
            status_code=400,
            detail="A mensagem não pode estar vazia."
        )

    prompt = f"""
Você é o COPILOTO IA da plataforma AutoIntel.

Seu objetivo é auxiliar um analista/vendedor automotivo
com argumentação comercial e análise competitiva.

REGRAS:
- Seja objetivo e executivo.
- Responda em português do Brasil.
- Evite respostas excessivamente longas.
- Use argumentos técnicos e comerciais.
- Não invente especificações técnicas.
- Quando não tiver informação suficiente, deixe isso claro.
- Não use Markdown excessivo.
- Para objeções comerciais, entregue argumentos práticos.
- Para comparações, destaque as diferenças relevantes.
- Mantenha uma linguagem profissional.

SOLICITAÇÃO DO USUÁRIO:

{req.mensagem}

Responda diretamente à solicitação.
"""

    try:

        resposta = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é o Copiloto IA do AutoIntel, "
                        "especialista em mercado automotivo, "
                        "produto e estratégia comercial."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=400
        )

        conteudo = resposta.choices[0].message.content

        if not conteudo:
            raise ValueError(
                "A Groq retornou uma resposta vazia."
            )

        return {
            "resposta": conteudo.strip()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Erro ao consultar a IA: {str(e)}"
        )


@app.get("/api/v1/teste")
def teste():
    return {
        "status": "ok",
        "mensagem": "AutoIntel backend funcionando"
    }
