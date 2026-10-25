import os,json,time,uuid
from pathlib import Path
try:
 from openai._base_client import SyncAPIClient
 _original=SyncAPIClient.request
 def _record(event):
  p=os.environ.get('GLM_USAGE_EVENTS')
  if p:
   data=(json.dumps(event,separators=(',',':'))+'\n').encode()
   fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
   try:os.write(fd,data)
   finally:os.close(fd)
 def _request(self,*a,**k):
  rid=uuid.uuid4().hex;t=time.time();_record(dict(event='start',id=rid,time=t,pid=os.getpid()))
  try:
   result=_original(self,*a,**k);u=getattr(result,'usage',None)
   _record(dict(event='finish',id=rid,time=time.time(),seconds=time.time()-t,pid=os.getpid(),usage=u.model_dump() if hasattr(u,'model_dump') else None))
   return result
  except BaseException as e:
   _record(dict(event='error',id=rid,time=time.time(),seconds=time.time()-t,pid=os.getpid(),type=type(e).__name__,status_code=getattr(e,'status_code',None)))
   raise
 SyncAPIClient.request=_request
except ImportError:pass
