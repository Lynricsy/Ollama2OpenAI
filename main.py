import os
import json
import asyncio
import traceback
from fastapi import FastAPI, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic 
import httpx
import uvicorn
from typing import Dict, List, Optional, Union
from config import config
from pydantic import BaseModel
from datetime import datetime, timezone

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0))
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# 数据模型
class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    format: Optional[str] = None
    options: Optional[Dict] = None

class EmbeddingRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    options: Optional[Dict] = None

class ShowRequest(BaseModel):
    model: str
    verbose: Optional[bool] = False

# 会话管理（简单实现，生产环境建议使用更安全的方式）
sessions = set()

def is_authenticated(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise RedirectResponse(url="/login", status_code=302)
    return True

def del_key(d: Dict, key: str):
    if key in d:
        del d[key]

@app.get("/", response_class=JSONResponse)
async def root_status():
    return {"status": "Ollama is running"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == config.admin_password:
        session_id = os.urandom(16).hex()
        sessions.add(session_id)
        response = RedirectResponse(url="/config", status_code=302)
        response.set_cookie(key="session_id", value=session_id)
        return response
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "密码错误"},
        status_code=401
    )

@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request, _=Depends(is_authenticated)):
    return templates.TemplateResponse(
        "config.html",
        {
            "request": request,
            "config": config,
            "model_mapping_json": json.dumps(config.model_mapping, indent=2)
        }
    )

@app.post("/config")
async def save_config(
    request: Request,
    admin_password: str = Form(...),
    openai_api_key: str = Form(...),
    ollama_api_key: str = Form(None),
    openai_api_base: str = Form(...),
    model_mapping: str = Form(...),
    _=Depends(is_authenticated)
):
    try:
        model_mapping_dict = json.loads(model_mapping)
        config.admin_password = admin_password
        config.openai_api_key = openai_api_key
        config.ollama_api_key = ollama_api_key if ollama_api_key else None
        config.openai_api_base = openai_api_base
        config.model_mapping = model_mapping_dict
        config.save()
        
        return templates.TemplateResponse(
            "config.html",
            {
                "request": request,
                "config": config,
                "model_mapping_json": model_mapping,
                "success": "配置已保存"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "config.html",
            {
                "request": request,
                "config": config,
                "model_mapping_json": model_mapping,
                "error": f"保存失败: {str(e)}"
            }
        )

@app.get("/api/tags")
async def list_models():
    try:
        # 获取OpenAI可用模型列表
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        response = await client.get(
            f"{config.openai_api_base}/v1/models",
            headers=headers
        )
        openai_models = response.json()
        
        models = []
        # 修改后的 model_details 基础模板
        base_model_details = {
            "parent_model": "",
            "format": "gguf",
            "family": "openai", 
            "families": ["openai"], 
            "parameter_size": "N/A", 
            "quantization_level": "N/A"
        }
        
        # 添加所有原始模型
        for model in openai_models.get("data", []):
            model_id = model.get("id", "")
            if not model_id: # 如果 model_id 为空则跳过
                continue

            if ":" not in model_id:
                name_with_tag = f"{model_id}:latest"
            else:
                name_with_tag = model_id
            created_timestamp = model.get("created")
            
            if created_timestamp:
                # 将 Unix 时间戳转换为带时区的 ISO 8601 格式字符串
                modified_at_iso = datetime.fromtimestamp(created_timestamp, timezone.utc).isoformat()
            else:
                # 如果 OpenAI API 没有提供 created 时间戳，使用当前时间并格式化
                modified_at_iso = datetime.now(timezone.utc).isoformat()

            current_details = base_model_details.copy()
            # 可选：更细致的 family 推断可以后续添加
            # if "gpt" in model_id.lower():
            #     current_details["family"] = "gpt"
            #     current_details["families"] = ["gpt"]
            
            models.append({
                "name": name_with_tag,
                "model": name_with_tag, 
                "modified_at": modified_at_iso,
                "size": 0,
                "digest": "", 
                "details": current_details
            })
            
            # 添加映射的别名
            aliases = [k for k, v in config.model_mapping.items() if v == model_id]
            for alias in aliases:
                alias_details = base_model_details.copy()
                # 别名也使用相同的 details 结构和时间戳
                models.append({
                    "name": f"{alias}:latest",
                    "model": f"{alias}:latest",
                    "modified_at": modified_at_iso, # 使用原始模型的时间戳
                    "size": 0,
                    "digest": "",
                    "details": alias_details 
                })
        
        return {"models": models}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # 将Ollama格式转换为OpenAI格式
        openai_body = {
            "model": config.model_mapping.get(request.model, request.model),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": request.stream
        }
        
        if request.options:
            # 转换Ollama选项到OpenAI参数
            if "temperature" in request.options:
                openai_body["temperature"] = request.options["temperature"]
            if "top_p" in request.options:
                openai_body["top_p"] = request.options["top_p"]
            if "num_ctx" in request.options:
                openai_body["max_tokens"] = request.options["num_ctx"]
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"发送请求到 OpenAI: {openai_body}")  # 添加日志
        
        # 发送请求到OpenAI兼容接口
        response = await client.post(
            f"{config.openai_api_base}/v1/chat/completions",
            json=openai_body,
            headers=headers
        )
        
        if response.status_code != 200:
            error_detail = await response.text()
            print(f"OpenAI API 错误: {error_detail}")  # 添加日志
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API 错误: {error_detail}"
            )
        
        if request.stream:
            # 处理流式响应
            async def stream_response():
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            chunk = chunk.removeprefix("data: ")
                            if chunk.strip() != "[DONE]": # 结束标志不处理
                                data = json.loads(chunk)
                                iso_time = datetime.fromtimestamp(data.get("created", ""), timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        if choice.get("finish_reason") is None:
                                            yield json.dumps({
                                                "model": request.model,
                                                "created_at": iso_time,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": choice["delta"]["content"]
                                                },
                                                "done": False
                                            }) + "\n"
                                        else:
                                            # 发送最后一个完成消息
                                            yield json.dumps({
                                                "model": request.model,
                                                "created_at": iso_time,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": choice["delta"]["content"]
                                                },
                                                "done_reason": choice.get("finish_reason"),
                                                "done": True,
                                                "total_duration": 0,
                                                "load_duration": 0,
                                                "prompt_eval_count": 0,
                                                "prompt_eval_duration": 0,
                                                "eval_count": 0,
                                                "eval_duration": 0 
                                            }) + "\n"
                        except json.JSONDecodeError as e:
                            print(f"JSON 解析错误: {e}, chunk: {chunk}")  # 添加日志
                            continue
            
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        else:
            # 处理非流式响应
            data = response.json()
            iso_time = datetime.fromtimestamp(data.get("created", ""), timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            if "choices" in data and len(data["choices"]) > 0:
                return {
                    "model": request.model,
                    "created_at": iso_time,
                    "message": {
                        "role": "assistant",
                        "content": data["choices"][0]["message"]["content"]
                    },
                    "done_reason": data["choices"][0].get("finish_reason"),
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0, 
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0
                }
            else:
                print(f"OpenAI 响应缺少 choices: {data}")  # 添加日志
                raise HTTPException(status_code=500, detail="OpenAI API 返回了无效的响应格式")
        
    except httpx.RequestError as e:
        print(f"请求错误: {str(e)}")  # 添加日志
        raise HTTPException(status_code=500, detail=f"请求 OpenAI API 失败: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {str(e)}")  # 添加日志
        raise HTTPException(status_code=500, detail=f"解析 OpenAI 响应失败: {str(e)}")
    except Exception as e:
        print(f"未知错误: {str(e)}")  # 添加日志
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        # 构建消息列表
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        
        # 准备OpenAI请求体
        openai_body = {
            "model": config.model_mapping.get(request.model, request.model),
            "messages": messages,
            "stream": request.stream
        }
        
        if request.options:
            # 转换Ollama选项到OpenAI参数
            if "temperature" in request.options:
                openai_body["temperature"] = request.options["temperature"]
            if "top_p" in request.options:
                openai_body["top_p"] = request.options["top_p"]
            if "num_ctx" in request.options:
                openai_body["max_tokens"] = request.options["num_ctx"]
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # 发送请求到OpenAI兼容接口
        response = await client.post(
            f"{config.openai_api_base}/v1/chat/completions",
            json=openai_body,
            headers=headers
        )
        
        if request.stream:
            # 处理流式响应
            async def stream_response():
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            chunk = chunk.removeprefix("data: ")
                            if chunk.strip() != "[DONE]": # 结束标志不处理
                                data = json.loads(chunk)
                                iso_time = datetime.fromtimestamp(data.get("created", ""), timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        if choice.get("finish_reason") is None:
                                            yield json.dumps({
                                                "model": request.model,
                                                "created_at": iso_time,
                                                "response": choice["delta"]["content"],
                                                "done": False
                                            }) + "\n"
                                        else:
                                            # 发送最后一个完成消息
                                            yield json.dumps({
                                                "model": request.model,
                                                "created_at": iso_time,
                                                "response": choice.get("delta", {}).get("content", ""),
                                                "done": True,
                                                "done_reason": choice.get("finish_reason"),
                                                "context": [],
                                                "total_duration": 0,
                                                "load_duration": 0,
                                                "prompt_eval_count": 0,
                                                "prompt_eval_duration": 0,
                                                "eval_count": 0,
                                                "eval_duration": 0
                                            }) + "\n"
                        except json.JSONDecodeError:
                            print(f"JSON 解析错误: {e}, chunk: {chunk}")
                            continue
            
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        else:
            # 处理非流式响应
            data = response.json()
            iso_time = datetime.fromtimestamp(data.get("created", ""), timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            if "choices" in data and len(data["choices"]) > 0:
                return {
                    "model": request.model,
                    "created_at": iso_time,
                    "response": data["choices"][0]["message"]["content"],
                    "done": True,
                    "done_reason": data["choices"][0].get("finish_reason"),
                    "context": [],
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0, 
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0
                }
            else:
                raise HTTPException(status_code=500, detail="No response from model")
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        # 将 Ollama 格式转换为 OpenAI 格式
        openai_body = {
            "model": config.model_mapping.get(request.model, request.model),
            "input": request.prompt
        }
        
        if request.options:
            # 转换 Ollama 选项到 OpenAI 参数
            if "dimensions" in request.options:
                openai_body["dimensions"] = request.options["dimensions"]
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"发送 embedding 请求到 OpenAI: {openai_body}")  # 添加日志
        
        # 发送请求到 OpenAI 兼容接口
        response = await client.post(
            f"{config.openai_api_base}/v1/embeddings",
            json=openai_body,
            headers=headers
        )
        
        if response.status_code != 200:
            error_detail = await response.text()
            print(f"OpenAI API 错误: {error_detail}")  # 添加日志
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API 错误: {error_detail}"
            )
        
        # 处理响应
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            embeddings = [item["embedding"] for item in data["data"]]
            return {
                "embedding": embeddings[0] if isinstance(request.prompt, str) else embeddings
            }
        else:
            print(f"OpenAI 响应缺少 embeddings: {data}")  # 添加日志
            raise HTTPException(status_code=500, detail="OpenAI API 返回了无效的响应格式")
        
    except httpx.RequestError as e:
        print(f"请求错误: {str(e)}")  # 添加日志
        raise HTTPException(status_code=500, detail=f"请求 OpenAI API 失败: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {str(e)}")  # 添加日志
        raise HTTPException(status_code=500, detail=f"解析 OpenAI 响应失败: {str(e)}")
    except Exception as e:
        print(f"未知错误: {str(e)}")  # 添加日志
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.post("/api/show")
async def show_model(req: ShowRequest):
    # 从模板文件读取响应
    template_path = os.path.join("templates", "response-api-show.json")
    with open(template_path, "r", encoding="utf-8") as f:
        resp = json.load(f)
    # 修改一些必要信息
    resp["model_info"]["general.basename"] = req.model
    if req.verbose:
        resp["model_info"]["tokenizer.ggml.merges"] = []
        resp["model_info"]["tokenizer.ggml.token_type"] = []
        resp["model_info"]["tokenizer.ggml.tokens"] = []
    else:
        resp["model_info"]["tokenizer.ggml.merges"] = None
        resp["model_info"]["tokenizer.ggml.token_type"] = None
        resp["model_info"]["tokenizer.ggml.tokens"] = None
    return resp

@app.post("/v1/chat/completions")
async def forward_chat(request: Request):
    try:
        # 调整请求头
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "authorization"]}
        headers["authorization"] = f"Bearer {config.openai_api_key}"

        # 调整请求体
        openai_body = await request.json()
        openai_body["model"] = config.model_mapping.get(openai_body["model"], openai_body["model"])

        # 转发请求到上游
        response = await client.post(
            f"{config.openai_api_base}/v1/chat/completions",
            json=openai_body,
            headers=headers
        )

        # 处理响应数据
        if response.headers.get("content-type") == "text/event-stream":
            # 处理流式响应
            async def stream_response():
                await asyncio.sleep(0.3)  # magic wait
                async for chunk in response.aiter_lines():
                    if chunk: # 忽略按行读取时的空行
                        try:
                            chunk = chunk.removeprefix("data: ")
                            if chunk.strip() != "[DONE]": # 结束标志不处理
                                # 修改响应格式
                                data = json.loads(chunk)
                                data["id"] = "chatcmpl-133"
                                data["system_fingerprint"] = "fp_ollama"
                                if "choices" in data:
                                    for choice in data["choices"]:
                                        del_key(choice, "text")
                                chunk = json.dumps(data)

                            yield f"data: {chunk}\n\n"
                        except json.JSONDecodeError as e:
                            print(f"JSON 解析错误: {e}, chunk: {chunk}")  # 添加日志
                            continue

            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            # 修改非流式响应
            data = response.json()
            data["id"] = "chatcmpl-106"
            data["system_fingerprint"] = "fp_ollama"

            return data
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"转发请求失败: {str(e)}")

@app.get("/v1/models")
async def get_models(request: Request):
    try:
        # 调整请求头
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "authorization"]}
        headers["authorization"] = f"Bearer {config.openai_api_key}"

        # 转发请求到上游
        response = await client.get(
            f"{config.openai_api_base}/v1/models",
            headers=headers
        )

        # 处理响应数据
        resp = response.json()
        resp["object"] = "list"
        if "data" in resp:
            for _d in resp["data"]:
                del_key(_d, "permission")
                del_key(_d, "root")
                del_key(_d, "parent")
        del_key(resp, "success")
        
        return resp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"转发请求失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 