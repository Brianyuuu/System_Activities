import os,json,time,hashlib,sqlite3,urllib.request,urllib.error,threading,zlib
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work/pipeline';OUT=ROOT/'outputs/pipeline'
CONFIG=json.loads((ROOT/'config.json').read_text());SOURCE_ROWS=CONFIG['source_rows'];RUN_ID=CONFIG['run_id']
RUN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
RATES={'glm':(.10,.50),'embedding':(.02,0),'gemini_batch':(.125,.75),'gemini_sync':(.25,1.5)}
CAP=float(CONFIG['budget_cap_usd'])
TRACK_GLM_COSTS=os.environ.get('GLM_TRACK_COSTS','0')=='1'
ACTIVE_SIMILARITY_THRESHOLD=0.33
def is_glm_stage(stage):
    return stage.startswith(('glm_', 'filter_pilot')) or stage in ('merge_audit','merge_audit_strict')
def dump(p,x):
    tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
# API workers remain concurrent; only durable ledger transactions are serialized.
# A persistent connection avoids 128 VAST WAL open/close/checkpoint races.
_DB_LOCK=threading.RLock()
_DB_CONNECTIONS={}
_DB_METRICS={'transactions':0,'lock_wait_seconds':0.0,'transaction_seconds':0.0}
def close_databases():
    with _DB_LOCK:
        for connection in _DB_CONNECTIONS.values():connection.close()
        _DB_CONNECTIONS.clear()
import atexit
atexit.register(close_databases)
def db_metrics():
    with _DB_LOCK:return dict(_DB_METRICS)
@contextmanager
def db():
    started=time.monotonic()
    with _DB_LOCK:
        acquired=time.monotonic();path=str((RUN/'budget.sqlite').resolve())
        c=_DB_CONNECTIONS.get(path)
        if c is None:
            c=sqlite3.connect(path,timeout=60,check_same_thread=False)
            try:
                c.execute('PRAGMA journal_mode=WAL')
                c.execute('PRAGMA synchronous=FULL')
                c.execute('CREATE TABLE IF NOT EXISTS charges (id TEXT PRIMARY KEY,stage TEXT,status TEXT,reserved REAL,cost REAL,input_tokens INTEGER,output_tokens INTEGER,note TEXT)')
                c.execute('CREATE TABLE IF NOT EXISTS response_cache (key TEXT PRIMARY KEY, value BLOB)')
                c.commit()
            except BaseException:
                c.close();raise
            _DB_CONNECTIONS[path]=c
        try:
            with c:yield c
        finally:
            _DB_METRICS['transactions']+=1
            _DB_METRICS['lock_wait_seconds']+=acquired-started
            _DB_METRICS['transaction_seconds']+=time.monotonic()-acquired
# Durable grouped writes for unpriced internal GLM only. Paid API budgeting
# keeps the original synchronous path. Every caller waits for the FULL WAL commit.
import queue
_GLM_WRITE_QUEUE=queue.Queue()
_GLM_WRITER_START=threading.Lock()
_GLM_WRITER=None
_GLM_WRITE_STATS={'commits':0,'operations':0,'failed_batches':0}
def _glm_writer_loop():
    while True:
        batch=[_GLM_WRITE_QUEUE.get()]
        deadline=time.monotonic()+0.05
        while len(batch)<64:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            try:batch.append(_GLM_WRITE_QUEUE.get(timeout=remaining))
            except queue.Empty:break
        error=None
        try:
            with db() as c:
                c.execute('BEGIN IMMEDIATE')
                for operation,event,outcome in batch:operation(c)
            # db() has committed successfully before any event is released.
            _GLM_WRITE_STATS['commits']+=1
            _GLM_WRITE_STATS['operations']+=len(batch)
        except Exception as exc:
            error=exc;_GLM_WRITE_STATS['failed_batches']+=1
        for operation,event,outcome in batch:
            outcome['error']=error;event.set();_GLM_WRITE_QUEUE.task_done()
def _glm_durable_write(operation):
    global _GLM_WRITER
    with _GLM_WRITER_START:
        if _GLM_WRITER is None:
            _GLM_WRITER=threading.Thread(target=_glm_writer_loop,name='glm-durable-writer',daemon=True)
            _GLM_WRITER.start()
    event=threading.Event();outcome={}
    _GLM_WRITE_QUEUE.put((operation,event,outcome));event.wait()
    if outcome['error'] is not None:raise outcome['error']
def glm_group_reserve(stage,rid,upper):
    assert is_glm_stage(stage) and not TRACK_GLM_COSTS and upper==0
    def operation(c):
        if c.execute('SELECT 1 FROM charges WHERE id=?',(rid,)).fetchone():
            raise RuntimeError('Request reservation already exists; reconcile before resend: '+rid)
        total_locked(c);scoped_total_locked(c)
        c.execute('INSERT INTO charges VALUES (?,?,?,?,NULL,0,0,?)',(rid,stage,'reserved',0,''))
        # Both cached totals stay unchanged: this unpriced internal charge is zero.
    _glm_durable_write(operation)
def glm_group_settle(rid,cost,inp=0,out=0,note=''):
    assert cost==0 and not TRACK_GLM_COSTS
    def operation(c):
        row=c.execute('SELECT reserved,cost,stage FROM charges WHERE id=?',(rid,)).fetchone()
        assert row and is_glm_stage(row[2]) and row[0]==0 and row[1] in (None,0)
        c.execute('UPDATE charges SET status=?,cost=?,input_tokens=?,output_tokens=?,note=? WHERE id=?',('settled',0,inp,out,note,rid))
    _glm_durable_write(operation)
def glm_group_cache(key,saved):
    assert not TRACK_GLM_COSTS
    blob=zlib.compress(json.dumps(saved).encode())
    _glm_durable_write(lambda c:c.execute('INSERT OR REPLACE INTO response_cache VALUES (?,?)',(key,blob)))

def total_locked(c):
    c.execute('CREATE TABLE IF NOT EXISTS budget_total (id INTEGER PRIMARY KEY, amount REAL)')
    row=c.execute('SELECT amount FROM budget_total WHERE id=1').fetchone()
    if row:return row[0]
    amount=c.execute('SELECT COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE cost END),0) FROM charges').fetchone()[0]
    c.execute('INSERT INTO budget_total VALUES (1,?)',(amount,));return amount
def scoped_total_locked(c):
    # Separate cache permits migration without changing historical charge rows.
    c.execute('CREATE TABLE IF NOT EXISTS budget_scoped_total (id INTEGER PRIMARY KEY, amount REAL)')
    row=c.execute('SELECT amount FROM budget_scoped_total WHERE id=1').fetchone()
    if row:return row[0]
    amount=sum(value for stage,value in c.execute('SELECT stage,CASE WHEN cost IS NULL THEN reserved ELSE cost END FROM charges') if not is_glm_stage(stage))
    c.execute('INSERT INTO budget_scoped_total VALUES (1,?)',(amount,));return amount
def reserve(stage,rid,upper):
    with db() as c:
        c.execute('BEGIN IMMEDIATE');old=c.execute('SELECT status FROM charges WHERE id=?',(rid,)).fetchone()
        if old:raise RuntimeError('Request reservation already exists; reconcile before resend: '+rid)
        total_locked(c);total=scoped_total_locked(c)
        if not is_glm_stage(stage) and total+upper>CAP:raise BudgetStop(f'Gemini/embedding budget would exceed ${CAP}: committed={total:.6f}, next_bound={upper:.6f}')
        c.execute('INSERT INTO charges VALUES (?,?,?,?,NULL,0,0,?)',(rid,stage,'reserved',upper,''))
        c.execute('UPDATE budget_total SET amount=amount+? WHERE id=1',(upper,))
        if not is_glm_stage(stage):c.execute('UPDATE budget_scoped_total SET amount=amount+? WHERE id=1',(upper,))
def settle(rid,cost,inp=0,out=0,note=''):
    with db() as c:
        c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT reserved,cost,stage FROM charges WHERE id=?',(rid,)).fetchone();assert row
        total_locked(c);scoped_total_locked(c)
        if cost>row[0]+1e-8:raise BudgetStop('Cost exceeded reserved bound; stop all paid work')
        previous=row[0] if row[1] is None else row[1]
        c.execute('UPDATE charges SET status=?,cost=?,input_tokens=?,output_tokens=?,note=? WHERE id=?',('settled',cost,inp,out,note,rid))
        c.execute('UPDATE budget_total SET amount=amount+? WHERE id=1',(cost-previous,))
        if not is_glm_stage(row[2]):c.execute('UPDATE budget_scoped_total SET amount=amount+? WHERE id=1',(cost-previous,))
def budget():
    with db() as c:
        rs=c.execute('''SELECT stage,COUNT(*),SUM(COALESCE(cost,0)),
            SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),
            SUM(input_tokens),SUM(output_tokens) FROM charges GROUP BY stage''').fetchall()
    scoped=[r for r in rs if not is_glm_stage(r[0])];glm_rows=[r for r in rs if is_glm_stage(r[0])]
    def totals(rows):
        settled=sum(r[2] for r in rows);reserved=sum(r[3] for r in rows)
        return dict(settled_cost_usd=settled,outstanding_or_uncertain_reservation_usd=reserved,committed_upper_usd=settled+reserved,requests=sum(r[1] for r in rows))
    out={'cap_usd':CAP,'budget_scope':'Gemini and embedding only; GLM tracked separately',**totals(scoped),'glm_separate':totals(glm_rows),'all_services':totals(rs),'by_stage':{}}
    out['committed_upper_usd']=out['settled_cost_usd']+out['outstanding_or_uncertain_reservation_usd'];out['remaining_uncommitted_usd']=CAP-out['committed_upper_usd']
    for stage,count,cost,reserved,inp,out_tokens in rs:
        out['by_stage'][stage]={'requests':count,'settled_cost':cost,'reserved_unsettled':reserved,'input_tokens':inp,'output_tokens':out_tokens}
    dump(OUT/'budget.json',out);return out
class BudgetStop(RuntimeError):pass
class GLMTransportError(RuntimeError):pass
class GLMRecoveryLimitError(RuntimeError):pass
GLM_NETWORK_DOWN=threading.Event()
GLM_DEFER_TRANSPORT=os.environ.get('GLM_DEFER_TRANSPORT','1')=='1'
def post(url,body,key,header='Authorization',timeout=180):
    req=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json',header:('Bearer '+key if header=='Authorization' else key)})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r),r.headers.get('x-litellm-response-cost')
def glm(stage,key,prompt,payload,validator,max_tokens=4096,schema=None,disable_thinking=False,compact_cache=False):
    cache=RUN/stage;cache.mkdir(exist_ok=True)
    body={'model':'parcc/GLM','messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],'temperature':0,'reasoning_effort':'low','max_tokens':max_tokens,'response_format':{'type':'json_object'}}
    if schema:body['response_format']={'type':'json_schema','json_schema':{'name':'validated_batch','strict':True,'schema':schema}}
    if disable_thinking:
        body.pop('reasoning_effort',None);body['chat_template_kwargs']={'enable_thinking':False}
    fingerprint=digest(body);path=cache/(key+'.json')
    if compact_cache:
        with db() as c:
            c.execute('CREATE TABLE IF NOT EXISTS response_cache (key TEXT PRIMARY KEY, value BLOB)')
            row=c.execute('SELECT value FROM response_cache WHERE key=?',(stage+':'+key,)).fetchone()
        if row:
            saved=json.loads(zlib.decompress(row[0]));assert saved['fingerprint']==fingerprint;validator(saved['result']);return saved['result']
    def save_result(saved):
        if is_glm_stage(stage) and not TRACK_GLM_COSTS:saved.pop('cost_usd',None)
        if compact_cache:
            try:
                if is_glm_stage(stage) and not TRACK_GLM_COSTS:glm_group_cache(stage+':'+key,saved)
                else:
                    with db() as c:c.execute('INSERT OR REPLACE INTO response_cache VALUES (?,?)',(stage+':'+key,zlib.compress(json.dumps(saved).encode())))
            except sqlite3.Error:
                # Keep a validated, paid answer even if compact cache storage fails.
                # The normal file-cache path will reuse it on restart without a call.
                dump(path,saved)
        else:dump(path,saved)
    if path.exists():
        saved=json.loads(path.read_text());assert saved['fingerprint']==fingerprint
        validator(saved['result']);return saved['result']
    request_path=cache/(key+'.request.json')
    if not compact_cache:dump(request_path,body)
    # This PARCC vLLM adapter can place a non-thinking JSON answer entirely in
    # reasoning_content. Accept only a complete STOP response parsing as the exact schema.
    def parse_answer(env):
        choice=env['choices'][0]
        if choice['finish_reason']!='stop':raise ValueError('Truncated or non-stop response: '+str(choice['finish_reason']))
        message=choice['message'];text=message.get('content')
        if not text and disable_thinking:text=message.get('reasoning_content') or (message.get('provider_specific_fields') or {}).get('reasoning')
        result=json.loads(text);validator(result);return result
    for previous in ([] if compact_cache else sorted(cache.glob(key+'.attempt*.json'))):
        try:
            env=json.loads(previous.read_text());result=parse_answer(env)
            dump(path,dict(fingerprint=fingerprint,result=result,usage=env['usage'],recovered_from=previous.name));return result
        except (ValueError,TypeError,KeyError,AssertionError):pass
    # UTF-8 byte count plus generous framing overhead is a conservative token upper bound.
    upper=0 if is_glm_stage(stage) and not TRACK_GLM_COSTS else (len(json.dumps(body,ensure_ascii=False).encode())+2048)*.10/1e6+max_tokens*.50/1e6
    # Transport failures must not consume the four malformed-answer attempts or
    # trigger recursive candidate splitting. Attempt IDs remain unique on resume.
    recovery_mode=compact_cache and is_glm_stage(stage)
    attempt_limit=8 if recovery_mode else 4
    request_prefix=f'{stage}:{key}:'
    with db() as c:
        prior=c.execute('SELECT status FROM charges WHERE id>=? AND id<?',(request_prefix,request_prefix+'~')).fetchall() if recovery_mode else []
    validation_failures=sum(status=='settled' for (status,) in prior)
    transient_failures=0
    for attempt in range(attempt_limit):
        if validation_failures>=4:break
        rid=f'{stage}:{key}:{attempt}'
        with db() as c:old=c.execute('SELECT status FROM charges WHERE id=?',(rid,)).fetchone()
        if old:continue  # do not resend an uncertain or failed previously charged attempt
        if not GLM_DEFER_TRANSPORT and GLM_NETWORK_DOWN.is_set():raise GLMTransportError('GLM transport unavailable; preserve pending pairs until recovery')
        reserve_started=time.monotonic()
        reserve_write=glm_group_reserve if compact_cache and is_glm_stage(stage) and not TRACK_GLM_COSTS else reserve
        settle_write=glm_group_settle if compact_cache and is_glm_stage(stage) and not TRACK_GLM_COSTS else settle
        reserve_write(stage,rid,upper);reserve_seconds=time.monotonic()-reserve_started;started=time.time()
        try:
            env,cost_header=post('https://litellm.parcc.upenn.edu/v1/chat/completions',body,os.environ['OSS_API_KEY'])
            api_seconds=time.time()-started
            use=env.get('usage',{});inp=use.get('prompt_tokens');out=use.get('completion_tokens')
            if inp is None or out is None:raise RuntimeError('Missing billable usage; reservation retained')
            cost=inp*.10/1e6+out*.50/1e6 if TRACK_GLM_COSTS else 0
            if cost_header and TRACK_GLM_COSTS:cost=max(cost,float(cost_header))
            settle_started=time.monotonic();settle_write(rid,cost,inp,out);settle_seconds=time.monotonic()-settle_started
            if not compact_cache:dump(cache/(key+f'.attempt{attempt}.json'),env)
            result=parse_answer(env)
            save_result(dict(fingerprint=fingerprint,result=result,usage=use,elapsed_seconds=time.time()-started,api_seconds=api_seconds,reserve_seconds=reserve_seconds,settle_seconds=settle_seconds,cost_usd=cost,response_id=env.get('id')));return result
        except urllib.error.HTTPError as e:
            detail=e.read().decode()[:500];dump(cache/(key+f'.error{attempt}.json'),dict(status=e.code,detail=detail))
            if e.code in (400,401,403,404,422):
                settle_write(rid,0,note=f'HTTP {e.code}, rejected request')
                raise RuntimeError(f'GLM HTTP {e.code}: {detail}') from None
            if recovery_mode and GLM_DEFER_TRANSPORT and e.code in (408,429,500,502,503,504):
                error=GLMTransportError(f'GLM HTTP {e.code}; deferred to end of sweep')
                retry_after=e.headers.get('Retry-After','0') if e.headers else '0'
                try:error.retry_after=max(0,float(retry_after))
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    try:error.retry_after=max(0,parsedate_to_datetime(retry_after).timestamp()-time.time())
                    except (ValueError,TypeError,OverflowError):error.retry_after=0
                raise error from None
            print('GLM retry',key,e.code,flush=True)
            if recovery_mode and e.code in (429,500,502,503,504):
                transient_failures+=1
                if e.code in (429,503) or transient_failures>=2:
                    GLM_NETWORK_DOWN.set()
                    raise GLMTransportError(f'GLM HTTP {e.code}; checkpoint and cool down') from None
        except (ValueError,KeyError,AssertionError) as e:
            validation_failures+=1
            print('GLM validation retry',key,str(e)[:180],flush=True)
        except BudgetStop:raise
        except (urllib.error.URLError,TimeoutError,ConnectionError) as e:
            if not (recovery_mode and GLM_DEFER_TRANSPORT):GLM_NETWORK_DOWN.set()
            dump(cache/(key+f'.uncertain{attempt}.json'),dict(error=type(e).__name__,reason_type=type(getattr(e,'reason',None)).__name__))
            raise GLMTransportError('GLM transport failure: '+type(e).__name__+('; deferred to end of sweep' if recovery_mode and GLM_DEFER_TRANSPORT else '; checkpoint and retry after connectivity recovers')) from None
        except Exception as e:
            print('GLM uncertain request',key,type(e).__name__,flush=True)
            dump(cache/(key+f'.uncertain{attempt}.json'),dict(error=type(e).__name__))
        time.sleep(min(2**attempt*2,20))
    if recovery_mode and validation_failures<4:
        raise GLMRecoveryLimitError('GLM transport recovery attempt limit reached: '+key+'; preserve pending pairs for inspection')
    raise RuntimeError('GLM failed after bounded attempts: '+key)
