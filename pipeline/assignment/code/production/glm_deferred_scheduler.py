"""Bounded concurrent sweeps; transient failures wait until the sweep ends."""
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

class DeferredRequestsRemaining(RuntimeError):
    pass

def run_sweeps(items, run, accept, checkpoint, report, target_workers,
               max_workers, transient_errors, terminal_errors,
               max_sweeps=9, retry_delay=60, sleep=time.sleep):
    started=time.time()
    failures={}
    stats=dict(completed_requests_this_run=0, completed_pairs_this_run=0,
               deferred_pairs_this_run=0, failed_attempts_this_run=0,
               total_pending_requests_at_start=len(items), sweep=0)
    pending={}
    last_report=started
    since_report=0
    retry_not_before=0
    def save(status):
        nonlocal last_report,since_report
        checkpoint()
        report(dict(stats,status=status,elapsed_seconds=time.time()-started,
                    pending_requests=len(pending),retry_queue_requests=len(failures),
                    target_workers=target_workers(),updated_at=time.time()),dict(failures))
        last_report=time.time();since_report=0
    def consume(future,i):
        nonlocal since_report,retry_not_before
        try:
            result=future.result()
        except transient_errors+terminal_errors as error:
            prior=failures.get(i,{})
            failures[i]=dict(request_key=f'F{i:06}',reason=str(error),
                             terminal=isinstance(error,terminal_errors),
                             failed_attempts_this_run=prior.get('failed_attempts_this_run',0)+1,
                             updated_at=time.time())
            stats['failed_attempts_this_run']+=1
            retry_not_before=max(retry_not_before,time.time()+getattr(error,'retry_after',0))
        else:
            valid,deferred=accept(result)
            stats['completed_requests_this_run']+=1
            stats['completed_pairs_this_run']+=valid
            stats['deferred_pairs_this_run']+=deferred
            failures.pop(i,None)
        since_report+=1
    current=list(items)
    try:
        for sweep in range(max_sweeps):
            stats['sweep']=sweep
            stats['requests_in_sweep']=len(current)
            save('first_pass' if sweep==0 else 'retry_pass')
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                iterator=iter(current);exhausted=False
                while True:
                    target=target_workers()
                    while len(pending)<target and not exhausted:
                        i=next(iterator,None)
                        if i is None:exhausted=True;break
                        pending[executor.submit(run,i)]=i
                    if not pending:break
                    done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                    for future in done:
                        i=pending.pop(future)
                        consume(future,i)
                    if since_report>=100 or time.time()-last_report>=30:
                        save('first_pass' if sweep==0 else 'retry_pass')
            save('sweep_finished')
            if not failures:
                save('complete');return stats
            current=[i for i,x in failures.items() if not x['terminal']]
            if not current or sweep+1==max_sweeps:break
            # The first retry begins immediately after all untouched work is done.
            # Later retries have a short bounded pause, also respecting Retry-After.
            delay=max(0,retry_not_before-time.time(),retry_delay if sweep else 0)
            if delay:
                stats['retry_at']=time.time()+delay
                save('retry_wait');sleep(delay);stats.pop('retry_at',None)
        save('retry_limit_reached')
        raise DeferredRequestsRemaining(f'{len(failures)} GLM requests remain unresolved after bounded sweeps; pending decisions preserved')
    except BaseException:
        # Executor shutdown drains active calls. Save every validated success.
        for future,i in list(pending.items()):
            if future.done() and not future.cancelled():
                try:consume(future,i)
                except Exception:pass
        pending.clear()
        save('interrupted_or_failed')
        raise
