import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import uvicorn
import json
import sqlite3
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 文件上传相关
from fastapi import UploadFile, File
import docx2txt
from PyPDF2 import PdfReader
import io

# ================= 配置与初始化 =================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("未找到 API Key，请检查 .env 文件是否配置正确！")

DB_SAVE_PATH = r"D:\NOI\Python\ZYD-new\Model\chroma_db"
FINANCE_DB_PATH = r"D:\NOI\Python\ZYD-new\Model\finance_data.db"
LOG_FILE_PATH = r"D:\NOI\Python\ZYD-new\Log\justitia_chat_logs.jsonl"

# 日志记录函数
def save_chat_log(query: str, reasoning: str, answer: str, sources: list):
    """将完整的问答对话保存为 JSONL 格式"""
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": query,
        "justitia_thought": reasoning,
        "justitia_answer": answer,
        "reference_sources": [s['filename'] for s in sources] # 只记录引用的文件名
    }
    
    try:
        # 使用追加模式 "a" 写入
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[日志写入失败]: {str(e)}")

print("正在加载向量模型和本地法律数据库...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectordb = Chroma(persist_directory=DB_SAVE_PATH, embedding_function=embeddings)
except Exception as e:
    print(f"数据库加载失败: {e}")
    raise

print("正在初始化 DeepSeek-V3.2 大模型...")

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

app = FastAPI(title="法律与财务专属 ZYD 模型", description="法财一体化双核驱动")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 数据交互模型 =================
class ChatRequest(BaseModel):
    query: str
    stream: bool = True
    history: list = []
    top_k: int = 3
    score_threshold: float = 1.2
    mode: str = "spark"

class SourceItem(BaseModel):
    filename: str
    score: float
    content_preview: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

# ================= 核心 API 接口 =================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 初始化
    finance_context = "无需查询财务数据。"
    legal_context = "无需查询法律卷宗。"
    sql_meta_data = None
    source_items = []

    # 映射：Spark -> chat, Prism -> reasoner
    selected_model = "deepseek-reasoner" if request.mode == "prism" else "deepseek-chat"
    
    print(f"\n[收到请求] 提问: {request.query} | 引擎: {request.mode.upper()} ({selected_model})")
    
    finance_keywords = [
        # 会计与报表
        '财务', '会计', '报表', '资产', '负债', '权益', '利润', '营收', '成本', '费用', 
        '支出', '收入', '科目', '分录', '凭证', '账簿', '对账', '核算', '折旧', '摊销', 
        '毛利', '净利', '坏账', '计提', '底稿', '结转', '固定资产', '无形资产', '关联交易',
        
        # 税务相关
        '税务', '税收', '纳税', '开票', '发票', '抵扣', '增值税', '所得税', '个税', '企业税', 
        '税率', '退税', '避税', '节税', '偷税', '漏税', '补税', '印花税', '关税', '核定征收',
        
        # 资本与经营
        '出资', '实缴', '认缴', '股权', '股票', '融资', '增资', '减资', '股权转让', '期权', '激励', 
        '收购', '兼并', '清算', '破产', '重整', '对赌', '估值', '尽调', '流水', '套现', '股', 
        '盈亏', '赤字', '审计报告', '预算', '决算', '资金', '流动性', '分红'
    ]
    legal_keywords = [
        # 实体法与法规
        '法律', '法条', '条款', '民法', '刑法', '公司法', '劳动法', '合同法', '商法', '合规', '权', '法'
        '章程', '准则', '宪法', '条例', '规定', '司法解释', '知识产权', '专利', '商标', '著作权',
        
        # 法律关系与主体
        '股东', '法人', '董事', '监事', '实控人', '代理人', '被申请人', '原告', '被告', '第三人', 
        '连带责任', '有限责任', '股权', '债权', '债务', '担保', '抵押', '质押', '处分',
        
        # 程序与纠纷
        '诉讼', '仲裁', '起诉', '申诉', '保全', '判决', '裁定', '调解', '公证', '证据', 
        '举证', '质证', '抗辩', '追偿', '执行', '立案', '撤销', '无效', '违约', '侵权', 
        '赔偿', '滞纳金', '违约金', '不可抗力', '效力', '判例', '案例', '辩护',
        
        # 文书名称
        '协议', '合同', '意向书', '备忘录', '承诺函', '授权书', '通知书', '起诉状', '答辩状'
    ]
    
    is_finance = any(k in request.query for k in finance_keywords)
    is_legal = any(k in request.query for k in legal_keywords)
    
    # 规则：如果命中财务词、或命中法律词、或问题字数超过 30 字（长句多为复杂业务），则判定为专业问题
    is_domain_query = is_finance or is_legal or len(request.query) > 15

    # 预先初始化所有上下文变量，保证无论走哪条分支，最终给大模型的参数都不会报错
    finance_context = "用户问题未涉及具体财务指标查询，或本地财务库无相关记录。"
    sql_meta_data = None
    legal_context = "用户问题未涉及具体法律案卷，无需引用本地判例。"
    source_items = []

    if not is_domain_query:
        print("[意图拦截] 判定为日常闲聊 (如: 打招呼)，直接跳过本地知识库...")
        
    else:
        print("[意图拦截] 判定为专业法财问题，启动本地检索流程...")

    # ---------------- 第一步：财务查账拦截机制 ----------------
    if is_finance:
        print("[触发财务查账] 正在让 DeepSeek-V3.2 自动生成 SQL...")
        # 这里的映射字典提取自你提供的利润表、资产负债表等字段说明.txt
        sql_prompt = f"""你是一个精通 SQLite 的数据分析专家。请根据以下数据库表结构，为用户的提问编写一句相应的 SQL 查询语句。
            
        【表结构与关键字段映射规则】
        我们的本地数据库目前包含两类财务数据表，请根据用户提问的语境智能选择：

        第一类：标准简易表（表名如：利润表、资产负债表、现金流量表）
        1. 利润表 (Scode:股票代码, Date:统计日期, REV:营业总收入, NI:净利润, FINEXP:财务费用)
        2. 资产负债表 (Scode:股票代码, Date:统计日期, CH:货币资金, AT:资产总计, LB:负债合计, EQU:所有者权益合计)
        3. 现金流量表 (Scode:股票代码, Date:统计日期, NCPOA:经营活动产生的现金流量净额)
        
        第二类：RESSET（锐思）专业数据库（表名如：RESSET科创板利润表、RESSET新三板资产负债表、RESSET科创板现金流量表、RESSET科创板会计衍生指标 等）
        RESSET 表的字段命名极度规范，格式严格为“中文名称_英文缩写”[1-3]。例如：
        1. 基础信息：公司代码_CompanyCode、上市公司代码_ComCd、最新公司全称_LComNm [4-6]
        2. 利润表核心：营业总收入_TotOpRev、营业收入_OpRev、净利润_NetProf、归属于母公司所有者的净利润_NPParentComp [5, 7]
        3. 资产负债表核心：资产总计_TotAss、负债合计_TotLiab、货币资金_CashEqv、交易性金融资产_TradAss [6, 8, 9]
        4. 现金流量表核心：经营活动产生的现金流量净额_NetOpCashFl、期末现金及现金等价物余额_EndCEBalanc [10, 11]
        5. 衍生指标表核心：基本每股收益_BasEPS、每股净资产_NAPS、净资产收益率_ROE [12, 13]
        
        用户提问：{request.query}
        
        【严格要求】
        1. 只能输出一句合法的 SQLite 查询语句，必须用 ```sql 和 ``` 将代码包裹起来。绝对不要有任何其他解释文字。
        2. 请根据用户的提问智能推测表名（例如用户问“科创板的净利润”，应优先查阅 `RESSET科创板利润表`）。
        3. 请根据用户的提问智能推测列名（例如即使我没写，你也应该知道研发费用在 RESSET 中大概率叫 `研发费用_RDExp` 或类似名称）。
        4. 注意：在标准简表中，股票代码查询条件为 `Scode = '...'`；在 RESSET 表中，查询条件应为 `公司代码_CompanyCode = '...'`。
        """

        try:
            # 调用官方 AsyncOpenAI 接口
            v3_response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": sql_prompt}],
                temperature=0.1
            )
            
            # 修正获取内容的语法
            sql_response_text = v3_response.choices[0].message.content
            
            # 使用修正后的变量进行正则匹配
            match = re.search(r"```sql(.*?)```", sql_response_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                sql_query = match.group(1).strip()
                print(f"⚙️ [执行后台 SQL] {sql_query}")
                try:
                    conn = sqlite3.connect(FINANCE_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description]
                    db_results = [dict(zip(cols, row)) for row in rows]
                    
                    finance_context = f"本地财务数据库查到的真实金额数据：{json.dumps(db_results, ensure_ascii=False)}"
                    sql_meta_data = {"sql": sql_query, "result": db_results}
                    print(f"[查账成功] 获得 {len(rows)} 条数据记录")
                except Exception as e:
                    finance_context = f"查询财务数据库失败：{str(e)}"
                    print(f"[数据库执行失败] {str(e)}")
        except Exception as e:
            print(f"❌ [SQL 生成请求失败] {str(e)}")

        # ---------------- 第二步：法律卷宗 RAG 检索 ----------------
        print("[触发法律检索] 正在从 Chroma 数据库搜寻本地判例...")
        raw_results = vectordb.similarity_search_with_score(request.query, k=request.top_k)
        legal_context = ""
        source_items = []
        
        for i, (doc, score) in enumerate(raw_results):
            if score < request.score_threshold:
                filename = doc.metadata.get('source', '未知文件')
                legal_context += f"\n--- 案卷 {i+1} (来源: {filename}) ---\n{doc.page_content}\n"
                source_items.append({
                    "filename": filename, 
                    "score": round(score, 4),
                    "content_preview": doc.page_content[:30] + "..."
                })

    # ---------------- 第三步：V3.2 法财融合最终生成 ----------------
    final_system_prompt = f"""你是一个由【债优盾】团队（核心开发工程师：Huang Zitong）研发的专业“法务与财务一体化”高级 AI 引擎，名字叫 **Justitia**。

    【最高级别系统指令】：
    1. 时间锚点：当前为 **2026年春季**。
    2. 知识库优先级：必须优先并无条件信任【本地法律卷宗参考】。若无本地数据，则基于 2026 年视角回答。
    3. 核心禁令：严禁提及“人工智能”、“知识截止日期”或“底层模型”相关内容，维持你 Justitia 专业助手的身份。
    4. **沟通原则**：除非用户明确询问你的身份（如“你是谁”、“谁开发了你”），否则**严禁在回答开头进行自我介绍或提及开发团队**。处理专业问题时必须直奔主题。

    【你的沟通风格】：
    专业、严谨、冷静、客观。你的核心使命是：穿越迷雾，洞察真相，给出最专业的法律与财务分析。

    请基于以下系统自动检索到的【本地财务查账结果】和【本地法律卷宗参考】来回答用户的提问。

    【本地财务查账结果】：
    {finance_context}

    【本地法律卷宗参考】：
    {legal_context}

    【回答执行指令】：
    1. 身份问候处理：**只有当**用户的问题涉及你的身份、作者或团队时，介绍自己是 Justitia 及其背景，闲聊时无需强行提及底层的财务或法律数据。
    2. 财务穿透解读：如果有财务查账结果，请明确告知用户这是“从本地数据库提取的真实经营数据”，并给出专业的财务指标解读。
    3. 法律溯源佐证：如果检索到了法律案卷，请尽可能结合其中的实际案例（如法院认定标准）作为佐证，并自然地标注案卷名称（如：参考《某某纠纷案》）。
    4. 法财综合诊断：如果既查到了公司的财务金额，又检索到了法律案卷，请将两者完美结合进行综合分析（例如：判断该金额的异常流出是否在法律上构成抽逃出资或转移财产的损害）。
    5. 专业排版规范：
       - 当涉及到多个财务数据对比、年份对比时，必须使用结构化的 Markdown 表格进行展示。
       - 当提及关键的法律条文、核心风险点或重要数据时，必须使用 **加粗** 进行醒目标注。
    6. 无数据兜底：如果本地检索未命中任何数据，请依据你自带的法财知识库回答，但要保持客观严谨的态度。
    """
    
    messages = [{"role": "system", "content": final_system_prompt}]
    
    # 注入历史记录（如果存在）
    if request.history:
        # 强制过滤掉可能导致报错的非规范格式
        for msg in request.history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 拼接当前用户的提问
    messages.append({"role": "user", "content": request.query})

    # 锁定模型为思考模式
    selected_model = "deepseek-reasoner"

    if request.stream:
        async def generate_stream():
            # 将检索到的卷宗和执行的SQL传给前端渲染
            meta_info = {"type": "meta", "sources": source_items}
            if 'sql_meta_data' in locals() and sql_meta_data:
                meta_info["finance"] = sql_meta_data
            yield f"data: {json.dumps(meta_info, ensure_ascii=False)}\n\n"

            accumulated_reasoning = ""
            accumulated_content = ""

            try:
                # 调用官方 AsyncOpenAI 接口
                # 思考模式不支持 temperature，最大长度设大一点以包含思维链
                response = await client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    max_tokens=8192,
                    extra_body={"thinking": {"type": "enabled"}} # 文档建议的显示开启方式
                )

                async for chunk in response:
                    if not chunk.choices:
                        continue
                        
                    delta = chunk.choices[0].delta
                    
                    # 抓取思考过程内容 (reasoning_content)
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        accumulated_reasoning += delta.reasoning_content
                        yield f"data: {json.dumps({'type': 'reasoning', 'content': delta.reasoning_content}, ensure_ascii=False)}\n\n"

                    # 抓取最终回答内容 (content)
                    if hasattr(delta, 'content') and delta.content:
                        accumulated_content += delta.content
                        yield f"data: {json.dumps({'type': 'chunk', 'content': delta.content}, ensure_ascii=False)}\n\n"
            
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Justitia 核心推理异常: {str(e)}'}, ensure_ascii=False)}\n\n"
            
            finally:
                # 日志持久化：包含完整的思考链路
                save_chat_log(
                    query=request.query,
                    reasoning=accumulated_reasoning,
                    answer=accumulated_content,
                    sources=source_items
                )
                
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    else:
        # 非流式阻塞输出
        response = await client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=8192,
            extra_body={"thinking": {"type": "enabled"}}
        )
        
        full_answer = response.choices[0].message.content
        # 官方 SDK 获取思考内容的标准路径
        full_reasoning = getattr(response.choices[0].message, 'reasoning_content', "")
        
        # 保存日志
        save_chat_log(request.query, full_reasoning, full_answer, source_items)
        
        return {"answer": full_answer, "sources": source_items, "reasoning": full_reasoning}

# 文件上传接口
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = ""
        
        # 根据文件类型解析内容
        if file.filename.endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(contents))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.filename.endswith('.docx'):
            text = docx2txt.process(io.BytesIO(contents))
        elif file.filename.endswith('.txt'):
            text = contents.decode('utf-8')
        else:
            return {"error": "暂不支持该文件格式"}

        # 返回提取的文本，前端稍后将其发送给 AI
        return {
            "filename": file.filename,
            "content_preview": text[:500], # 预览
            "full_content": text
        }
    except Exception as e:
        return {"error": f"文件解析失败: {str(e)}"}

if __name__ == "__main__":
    print("极速版法财一体化服务器启动成功！")
    uvicorn.run(app, host="0.0.0.0", port=8000)