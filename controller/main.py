from typing import List, Tuple, Optional, Annotated, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Json
import uvicorn
import argparse
import requests
import json
import mysql.connector
import datetime
import secrets
import tomllib
import uuid
import base64

app = FastAPI()
cors_origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://serving-dev.colearn.cloud",
    # "*",
]
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"],)
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
cnx = mysql.connector.connect(user=config["db"]["user"], password=config["db"]["password"], host=config["db"]["host"], port=config["db"]["port"],
                              database=config["db"]["database"], time_zone='-8:00', pool_name='fleece', pool_size=32)
cnx.close()


class LayersRequest(BaseModel):
    layer_names: List[str]


def get_email(api_token: Annotated[str, Header()]):
    try:
        if api_token is None:
            raise HTTPException(status_code=403, detail="No valid Authorization header.")
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT email FROM user_api_token WHERE token=%s")
            cursor.execute(query, [api_token])
            res = cursor.fetchall()
        finally:
            cursor.close()
            cnx.close()
        if len(res) != 1:
            raise HTTPException(status_code=403, detail="User not found.")
        return res[0][0]
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


def get_worker_id(worker_id: Annotated[str, Header()], api_token: Annotated[str, Header()]):
    try:
        if worker_id is None or api_token is None:
            raise HTTPException(status_code=403, detail="No valid Authorization header.")
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT email FROM user_api_token WHERE token=%s")
            cursor.execute(query, [api_token])
            res = cursor.fetchall()
            if len(res) != 1:
                raise HTTPException(status_code=403, detail="User not found.")
            email = res[0][0]
            query = ("SELECT owner_email FROM worker WHERE id=%s")
            cursor.execute(query, [worker_id])
            res = cursor.fetchall()
            if len(res) != 1:
                raise HTTPException(status_code=403, detail="Worker not found.")
            owner_email = res[0][0]
        finally:
            cursor.close()
            cnx.close()
        if email != owner_email:
            raise HTTPException(status_code=403, detail="Forbidden.")
        return worker_id
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# def get_task_manager_name(api_token: Annotated[str, Header()]):
#     try:
#         if api_token is None:
#             raise HTTPException(status_code=403, detail="No valid Authorization header.")
#         cnx = mysql.connector.connect(pool_name='fleece')
#         cursor = cnx.cursor()
#         query = ("SELECT name FROM task_manager WHERE api_token=%s")
#         cursor.execute(query, [api_token])
#         res = cursor.fetchall()
#         cursor.close()
#         cnx.close()
#         if len(res) != 1:
#             raise HTTPException(status_code=403, detail="User not found.")
#         return res[0][0]
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/oauth")
def oauth(code: str):
    try:
        data = {
            "client_id": config["slack"]["client_id"],
            "client_secret": config["slack"]["client_secret"],
            "code": code
        }
        r = requests.post("https://slack.com/api/openid.connect.token", data=data)
        res = json.loads(base64.urlsafe_b64decode(json.loads(r.content)["id_token"].split(".")[1]+"=="))
        if res["https://slack.com/team_id"] != config["slack"]["team_id"] or res["email_verified"] == False:
            raise HTTPException(status_code=403, detail="Multiple token.")
        email = res["email"]
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT token FROM user_api_token WHERE email=%s")
            cursor.execute(query, [email])
            res = cursor.fetchall()
            if len(res) < 1:
                token = secrets.token_hex(16)
                query = ("INSERT INTO user_api_token (email, token) VALUES (%s, %s)")
                cursor.execute(query, [email, token])
                query = "INSERT INTO credit_transaction (email, event_type, amount, event_detail) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, [email, "bonus", 1000000000, "sign up bonus"])
                cnx.commit()
                response = RedirectResponse(url=f"https://serving-dev.colearn.cloud/login?token={token}", status_code=302)
                return response
        finally:
            cursor.close()
            cnx.close()
        if len(res) > 1:
            raise HTTPException(status_code=403, detail="Multiple token.")
        api_token = res[0][0]
        response = RedirectResponse(url=f"https://serving-dev.colearn.cloud/login?token={api_token}", status_code=302)
        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class User(BaseModel):
    email: str
    api_token: str


@app.get("/get_user", response_model=User)
def get_user(email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT token FROM user_api_token WHERE email=%s")
            cursor.execute(query, [email])
            res = cursor.fetchall()
            if len(res) < 1:
                token = secrets.token_hex(16)
                query = ("INSERT INTO user_api_token (email, token) VALUES (%s, %s)")
                cursor.execute(query, [email, token])
                cnx.commit()
                return User(email=email, api_token=token)
        finally:
            cursor.close()
            cnx.close()
        if len(res) > 1:
            raise HTTPException(status_code=403, detail="Multiple token.")
        return User(email=email, api_token=res[0][0])
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/refresh_api_token", response_model=User)
def refresh_api_token(email: Annotated[str, Depends(get_email)]):
    try:
        try:
            token = secrets.token_hex(16)
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("UPDATE user_api_token SET token=%s WHERE email=%s")
            cursor.execute(query, [token, email])
            cnx.commit()
        finally:
            cursor.close()
            cnx.close()
        return User(email=email, api_token=token)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class CreditTransaction(BaseModel):
    timestamp: int
    event_type: str
    amount: int
    event_detail: str


class CreditTransactionList(BaseModel):
    credit_transactions: List[CreditTransaction] = []


class RemainingCredit(BaseModel):
    remaining_credit: int


@app.get("/get_remaining_credit", response_model=RemainingCredit)
def get_remaining_credit(email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT SUM(amount) FROM credit_transaction WHERE email=%s")
            cursor.execute(query, [email])
            res = cursor.fetchone()[0] or 0
        finally:
            cursor.close()
            cnx.close()
        return RemainingCredit(remaining_credit=res)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/get_credit_transactions", response_model=CreditTransactionList)
def get_credit_transactions(email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT timestamp, event_type, amount, event_detail FROM credit_transaction WHERE email=%s")
            cursor.execute(query, [email])
            res = cursor.fetchall()
        finally:
            cursor.close()
            cnx.close()
        ans = []
        for trans in res:
            timestamp = trans[0].replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-8))).timestamp()
            ans.append(CreditTransaction(timestamp=timestamp, event_type=trans[1], amount=trans[2], event_detail=trans[3]))
        return CreditTransactionList(credit_transactions=ans)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class NetworkServer(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class NetworkServers(BaseModel):
    signaling: NetworkServer
    turn: List[NetworkServer] = []
    stun: List[NetworkServer] = []


@app.get("/get_network_servers", response_model=NetworkServers)
def get_worker_list(email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT type, url, username, password FROM network_server")
            cursor.execute(query, [])
            res = cursor.fetchall()
        finally:
            cursor.close()
            cnx.close()
        signaling = None
        turn = []
        stun = []
        for x in res:
            if x[0] == "signaling":
                signaling = NetworkServer(url=x[1], username=x[2], password=x[3])
            elif x[0] == "turn":
                turn.append(NetworkServer(url=x[1], username=x[2], password=x[3]))
            elif x[0] == "stun":
                stun.append(NetworkServer(url=x[1]))
        if signaling == None:
            raise HTTPException(status_code=404, detail="Network servers not found.")
        return NetworkServers(signaling=signaling, turn=turn, stun=stun)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class Worker(BaseModel):
    worker_id: str
    owner_email: str
    url: str
    version: str
    nickname: str = ""
    created_at: int
    last_seen: int
    gpu_model: str
    gpu_total_memory: int
    gpu_remaining_memory: int
    loaded_layers: List


class WorkerList(BaseModel):
    workers: List[Worker] = []


@app.get("/get_worker_list", response_model=WorkerList)
def get_worker_list(email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT id, owner_email, url, version, nickname, created_at, last_seen, gpu_model, gpu_total_memory, gpu_remaining_memory, loaded_layers FROM worker")
            cursor.execute(query, [])
            res = cursor.fetchall()
        finally:
            cursor.close()
            cnx.close()
        ans = []
        for trans in res:
            created_at = trans[5].replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-8))).timestamp()
            last_seen = trans[6].replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-8))).timestamp()
            ans.append(Worker(worker_id=trans[0], owner_email=trans[1], url=trans[2], version=trans[3], nickname=trans[4], created_at=int(created_at), last_seen=int(last_seen),
                              gpu_model=trans[7], gpu_total_memory=trans[8], gpu_remaining_memory=trans[9], loaded_layers=json.loads(trans[10])))
        return WorkerList(workers=ans)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class ChatId(BaseModel):
    chat_id: str


class ChatPlan(BaseModel):
    chat_plans: List


@app.post("/get_chat_info", response_model=ChatPlan)
def get_chat_info(req: ChatId, email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT email, plans FROM chat_history WHERE id=%s")
            cursor.execute(query, [req.chat_id])
            res = cursor.fetchone()
            if res is not None:
                chat_owner_email, plans = res
            else:
                raise HTTPException(status_code=404, detail="Not found.")
        finally:
            cursor.close()
            cnx.close()
        if chat_owner_email != email:
            raise HTTPException(status_code=403, detail="Forbidden.")
        plans = json.loads(plans)
        return ChatPlan(chat_plans=plans)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class GetNetworkLatency(BaseModel):
    worker_pairs: List[Tuple[str, str]] = None


class NetworkLatency(BaseModel):
    from_worker_id: str
    to_worker_id: str
    latency: Optional[float]


class NetworkLatencyList(BaseModel):
    network_latencies: List[NetworkLatency]


@app.post("/get_network_latency", response_model=NetworkLatencyList)
def get_network_latency(req: GetNetworkLatency, email: Annotated[str, Depends(get_email)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            ans = []
            for x in req.worker_pairs:
                query = ("SELECT from_worker_id, to_worker_id, latency FROM perf_network WHERE from_worker_id=%s AND to_worker_id=%s")
                cursor.execute(query, [x[0], x[1]])
                res = cursor.fetchone()
                if res is not None:
                    ans.append(NetworkLatency(from_worker_id=res[0], to_worker_id=res[1], latency=res[2]))
                else:
                    ans.append(NetworkLatency(from_worker_id=x[0], to_worker_id=x[1], latency=None))
        finally:
            cursor.close()
            cnx.close()
        return NetworkLatencyList(network_latencies=ans)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class RegisterWorker(BaseModel):
    url: str
    version: str
    nickname: str = ""
    gpu_model: str = ""
    gpu_total_memory: int = 0
    gpu_remaining_memory: int = 0


class WorkerId(BaseModel):
    id: str


class WorkerHeartbeat(BaseModel):
    info_update: Optional[Json] = None


class TaskManagerPubkeyList(BaseModel):
    pubkeys: Dict[str, str] = []


@app.post("/register_worker", response_model=WorkerId)
def register_worker(req: RegisterWorker, email: Annotated[str, Depends(get_email)]):
    try:
        worker_id = str(uuid.uuid4())
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("INSERT INTO worker (id, owner_email, url, version, nickname, gpu_model, gpu_total_memory, gpu_remaining_memory, loaded_layers) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")
            cursor.execute(query, [worker_id, email, req.url, req.version, req.nickname, req.gpu_model, req.gpu_total_memory, req.gpu_remaining_memory, "[]"])
            cnx.commit()
        finally:
            cursor.close()
            cnx.close()
        return WorkerId(id=worker_id)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# @app.post("/deregister_worker")
# def deregister_worker(req: WorkerURL, email: Annotated[str, Depends(get_email)]):
#     try:
#         cnx = mysql.connector.connect(pool_name='fleece')
#         cursor = cnx.cursor()
#         query = ("DELETE FROM worker where owner_email=%s AND url=%s")
#         cursor.execute(query, [email, req.url])
#         cnx.commit()
#         cursor.close()
#         cnx.close()
#         return None
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/worker_heartbeat", response_model=TaskManagerPubkeyList)
def worker_heartbeat(req: WorkerHeartbeat, worker_id: Annotated[str, Depends(get_worker_id)]):
    try:
        try:
            cnx = mysql.connector.connect(pool_name='fleece')
            cursor = cnx.cursor()
            query = ("SELECT url, pubkey FROM task_manager")
            cursor.execute(query)
            res = cursor.fetchall()

            if "loaded_layers" in req.info_update:
                loaded_layers = req.info_update["loaded_layers"]
            else:
                loaded_layers = "[]"
            if "gpu_remaining_memory" in req.info_update:
                query = ("UPDATE worker SET last_seen = CURRENT_TIMESTAMP, loaded_layers = %s, gpu_remaining_memory = %s  WHERE id=%s")
                cursor.execute(query, [loaded_layers, req.info_update["gpu_remaining_memory"], worker_id])
            else:
                query = ("UPDATE worker SET last_seen = CURRENT_TIMESTAMP, loaded_layers = %s WHERE id=%s")
                cursor.execute(query, [loaded_layers, worker_id])
            # query = ("DELETE FROM worker WHERE created_at < (NOW() - INTERVAL 30 SECOND) AND last_seen < (NOW() - INTERVAL 30 SECOND)")
            # cursor.execute(query)

            if "perf_computation" in req.info_update:
                data = []
                for x in req.info_update["perf_computation"]:
                    data.append([worker_id, x["layers"], x["input_shape"], x["latency"]])
                query = ("REPLACE INTO perf_computation SET worker_id=%s, layers=%s, input_shape=%s, latency=%s")
                cursor.executemany(query, data)
            if "perf_network" in req.info_update:
                data = []
                for x in req.info_update["perf_network"]:
                    data.append([worker_id, x["to_worker_id"], x["latency"]])
                query = ("REPLACE INTO perf_network SET from_worker_id=%s, to_worker_id=%s, latency=%s")
                cursor.executemany(query, data)

            cnx.commit()
        finally:
            cursor.close()
            cnx.close()
        ans = {}
        for x in res:
            ans[x[0]] = x[1]
        return TaskManagerPubkeyList(pubkeys=ans)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# class AddCreditTransaction(BaseModel):
#     email: str
#     description: str
#     amount: int


# @app.post("/add_credit_transaction")
# def add_credit_transaction(req: AddCreditTransaction, task_manager_name: Annotated[str, Depends(get_task_manager_name)]):
#     try:
#         note = "Created by {}".format(task_manager_name)
#         cnx = mysql.connector.connect(pool_name='fleece')
#         cursor = cnx.cursor()
#         query = ("INSERT INTO credit_transaction (email, description, amount, note) VALUES (%s, %s, %s, %s)")
#         cursor.execute(query, [req.email, req.description, req.amount, note])
#         cnx.commit()
#         cursor.close()
#         cnx.close()
#         return None
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # uvicorn.run(app, host="0.0.0.0", port=8080, access_log=True)
    uvicorn.run(app, host="0.0.0.0", port=8443, access_log=True, ssl_keyfile="key.pem", ssl_certfile="fullchain.pem")
