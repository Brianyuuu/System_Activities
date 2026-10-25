"""Observe HTTP attempts, including SDK retries and urllib, without logging content."""
import os,json,time,uuid,sys,threading
_local=threading.local()
def emit(event):
    p=os.environ.get('PHRASE_HTTP_EVENTS')
    if p:
        event.update(pid=os.getpid(),stage=os.path.basename(sys.argv[0]),workers=os.environ.get('PHRASE_API_WORKERS'))
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:os.write(fd,(json.dumps(event,separators=(',',':'))+'\n').encode())
        finally:os.close(fd)
def wrap(fn,kind):
    def call(*a,**kw):
        if getattr(_local,'inside',False):return fn(*a,**kw)
        _local.inside=True;rid=uuid.uuid4().hex;t=time.time()
        emit(dict(event='start',id=rid,time=t,transport=kind))
        try:
            value=fn(*a,**kw)
            emit(dict(event='finish',id=rid,time=time.time(),seconds=time.time()-t,status_code=getattr(value,'status_code',getattr(value,'status',None))))
            return value
        except BaseException as e:
            emit(dict(event='error',id=rid,time=time.time(),seconds=time.time()-t,type=type(e).__name__,status_code=getattr(e,'code',getattr(e,'status_code',None))))
            raise
        finally:_local.inside=False
    return call
try:
    import httpx
    httpx.Client.send=wrap(httpx.Client.send,'httpx')
except ImportError:pass
import urllib.request
urllib.request.OpenerDirector.open=wrap(urllib.request.OpenerDirector.open,'urllib')
# Preserve the existing returned-token observer independently of HTTP-attempt metrics.
from pathlib import Path
original=Path(__file__).resolve().parents[1]/'usage_observer/sitecustomize.py'
if original.is_file():exec(compile(original.read_text(),str(original),'exec'),{})
