import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

try:
    from pyngrok import ngrok          # needed for the public ngrok URL
except ImportError:
    ngrok = None


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")   # leave empty on Render
PORT = int(os.getenv("PORT", "8000"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")


# ============================================================
# 1B. LANGSMITH TRACING (turns on only if a key is present)
# ============================================================

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "it-helpdesk-agent"

if LANGSMITH_API_KEY:
    # Older LangChain reads LANGCHAIN_*, newer reads LANGSMITH_* -> set both
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    print(f"LangSmith tracing ON  -> project: {LANGSMITH_PROJECT}")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing OFF (set LANGSMITH_API_KEY in .env to enable)")


# ============================================================
# 2. MOCK IT KNOWLEDGE BASE  (only source of fixes and policy)
# ============================================================

KNOWLEDGE_BASE = {
    "KB-101": {
        "title": "Reset password / unlock account",
        "keywords": ["password", "reset", "locked", "unlock", "login", "forgot"],
        "steps": (
            "Go to password.example-corp.com and verify with MFA, then choose a new "
            "password (12+ characters). Policy: accounts lock for 15 minutes after "
            "5 failed attempts."
        ),
    },
    "KB-102": {
        "title": "Connect to the corporate VPN",
        "keywords": ["vpn", "remote", "tunnel"],
        "steps": (
            "Open GlobalConnect, select 'Corp-VPN', sign in with SSO + MFA. "
            "If it fails, fully quit and reopen the client, then retry."
        ),
    },
    "KB-103": {
        "title": "Outlook / email not syncing",
        "keywords": ["email", "outlook", "mail", "sync", "inbox"],
        "steps": (
            "Restart Outlook. If still failing, remove and re-add the account under "
            "File > Account Settings."
        ),
    },
    "KB-104": {
        "title": "Wi-Fi problems in the office",
        "keywords": ["wifi", "wi-fi", "wireless", "internet", "network"],
        "steps": (
            "Forget 'Corp-WiFi' and reconnect with SSO credentials. "
            "Guest devices must use 'Corp-Guest' only."
        ),
    },
    "KB-105": {
        "title": "Request new software",
        "keywords": ["software", "install", "license", "app", "application"],
        "steps": (
            "Approved apps: install from Software Center, no ticket needed. "
            "Policy: any app NOT in Software Center needs manager approval and a "
            "ticket (3 business days). Users do not have local admin rights."
        ),
    },
    "KB-106": {
        "title": "Phishing, lost device or suspected compromise",
        "keywords": ["phishing", "suspicious", "hacked", "compromised", "lost",
                     "stolen", "laptop", "malware", "virus"],
        "steps": (
            "Do NOT click links or open attachments. Use the 'Report Phishing' button "
            "in Outlook. Policy: this is a security incident and MUST be escalated "
            "immediately as P1."
        ),
    },
}


# ============================================================
# 3. MOCK SYSTEM STATUS + TICKETS
# ============================================================

SYSTEM_STATUS = {
    "vpn":   {"status": "operational", "note": "No known issues."},
    "sso":   {"status": "operational", "note": "No known issues."},
    "email": {"status": "degraded", "note": "Delayed delivery 10-15 min. ETA 14:00 UTC."},
    "wifi":  {"status": "outage", "note": "Building B floor 3 access points down. ETA 16:00 UTC."},
}


TICKETS = {}


# ============================================================
# 4. KNOWLEDGE BASE TOOL
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the IT knowledge base for troubleshooting steps and IT policy.
    Use this for every issue before answering."""
    query = query.lower()

    for kb_id, article in KNOWLEDGE_BASE.items():
        if any(keyword in query for keyword in article["keywords"]):
            return f"[{kb_id}] {article['title']}: {article['steps']}"

    return "NO_MATCH: no knowledge base article covers this. Do not guess; create a ticket."


# ============================================================
# 5. SYSTEM STATUS TOOL
# ============================================================
@tool
def check_system_status(system: str) -> str:
    """Check live status of an IT system. Known systems: vpn, sso, email, wifi."""
    info = SYSTEM_STATUS.get(system.strip().lower())

    if info is None:
        return f"UNKNOWN: no status information for '{system}'. Known systems: vpn, sso, email, wifi."

    return f"{system.upper()}: {info['status']}. {info['note']}"

# @tool
# def check_system_status(system: str) -> str:
#     """Check live status of an IT system. Known systems: vpn, sso, email, wifi."""
#     system = system.lower()

#     for name, info in SYSTEM_STATUS.items():
#         if name in system:
#             return f"{name.upper()}: {info['status']}. {info['note']}"

#     return f"UNKNOWN: no status information for '{system}'. Known systems: vpn, sso, email, wifi."



# ============================================================
# 6. TICKET TOOL
# ============================================================

@tool
def create_ticket(
    summary: str,
    priority: Literal["P1", "P2", "P3", "P4"],
    escalate: bool = False,
) -> str:
    """Create an IT ticket. Set escalate=True for security incidents and P1 issues.
    P1 = security or critical outage, P2 = user blocked, P3 = degraded, P4 = request."""
    ticket_id = f"INC-{1001 + len(TICKETS)}"
    team = "On-call Engineer" if escalate else "Service Desk"

    TICKETS[ticket_id] = {
        "id": ticket_id,
        "summary": summary,
        "priority": priority,
        "status": "Escalated" if escalate else "Open",
        "assigned_team": team,
    }
    return f"Ticket {ticket_id} created. Priority {priority}. Assigned to {team}."


# ============================================================
# 7. OPENAI MODEL
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================
# 8. AGENT: create_tool_calling_agent + AgentExecutor
# ============================================================

tools = [search_knowledge_base, check_system_status, create_ticket]

SYSTEM_PROMPT = """
You are an internal IT helpdesk triage agent for a large organization.

For every request:
1. Call search_knowledge_base.
2. For VPN, login (sso), email or Wi-Fi issues, also call check_system_status.
3. Decide:
   - RESOLVE: KB has an answer and the system is operational -> give the steps and cite the KB ID.
   - OUTAGE/DEGRADED: tell the user the status and ETA. Do not make them troubleshoot.
   - TICKET: KB returns NO_MATCH, or the request needs approval -> call create_ticket.
   - ESCALATE: phishing, lost/stolen device, suspected compromise -> call create_ticket
     with priority=P1 and escalate=True immediately.

HARD RULES:
- Never invent policy, steps, system status, ETAs or ticket numbers. Use only tool results.
- If a tool says NO_MATCH or UNKNOWN, say so honestly and create a ticket.
- Be concise. End with the ticket ID if you created one, or the KB ID if you resolved it.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,     # lets us report which tools were used
)


# ============================================================
# 9. NGROK LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    tunnel = None

    if NGROK_AUTH_TOKEN and ngrok is not None:
        try:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            tunnel = ngrok.connect(PORT, "http")
            print("\n" + "=" * 60)
            print(f"Public ngrok URL : {tunnel.public_url}")
            print(f"Swagger docs     : {tunnel.public_url}/docs")
            print(f"Chat endpoint    : {tunnel.public_url}/chat")
            print("=" * 60 + "\n")
        except Exception as exc:
            print(f"Ngrok connection failed: {exc}")
    elif ngrok is None:
        print("pyngrok is not installed. Run: pip install pyngrok")
    else:
        print("NGROK_AUTH_TOKEN not found in .env -> running on localhost only.")

    yield

    if tunnel is not None:
        try:
            ngrok.disconnect(tunnel.public_url)
        except Exception as exc:
            print(f"Ngrok disconnect warning: {exc}")


# ============================================================
# 10. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="IT Helpdesk Triage API",
    description="LangChain AgentExecutor + OpenAI + FastAPI + LangSmith + ngrok",
    version="1.1",
    lifespan=lifespan,
)


# ============================================================
# 11. REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Employee's IT issue, for example: I can't connect to the VPN",
    )


# ============================================================
# 12. HOME + HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "IT Helpdesk Triage API is running",
        "chat_endpoint": "/chat",
        "langsmith_tracing": bool(LANGSMITH_API_KEY),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
# 13. CHAT ENDPOINT (returns latency + LangSmith run_id)
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):
    run_id = uuid.uuid4()            # search this ID in LangSmith to find the trace
    start = time.perf_counter()

    try:
        result = executor.invoke(
            {"input": request.question},
            config={
                "run_id": run_id,
                "run_name": "it_helpdesk_chat",
                "tags": ["it-helpdesk"],
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error while processing the request: {str(exc)}",
        )

    latency = time.perf_counter() - start
    tools_used = [action.tool for action, _ in result["intermediate_steps"]]

    return {
        "question": request.question,
        "answer": result["output"],
        "tools_used": tools_used,
        "latency_seconds": round(latency, 2),
        "run_id": str(run_id),
    }


# ============================================================
# 14. TICKETS ENDPOINT (see what the agent created)
# ============================================================

@app.get("/tickets")
def list_tickets():
    return {"count": len(TICKETS), "tickets": list(TICKETS.values())}


# ============================================================
# 15. RUN UVICORN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
