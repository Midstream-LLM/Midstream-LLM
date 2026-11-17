#!/usr/bin/env python3
"""Read CUDA logs independently and publish only allowlisted public metrics.
No torch/W&B imports, checkpoint loading, or training-process changes.
"""
import argparse
import datetime as dt
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import urllib.request

FIELDS = re.compile(r'([\w/]+)=([^\s]+)')
KEYS = {'loss': 'trainLoss', 'lr': 'learningRate', 'tok/s': 'throughput'}
TZ = dt.timezone(dt.timedelta(hours=8))


def sample(points, limit=800):
    """Retain endpoints and bucket extrema, including short spikes."""
    if len(points) <= limit:
        return points
    result = [points[0]]
    inner = points[1:-1]
    buckets = (limit - 2) // 2
    for i in range(buckets):
        group = inner[len(inner)*i//buckets:len(inner)*(i+1)//buckets]
        pair = {p['step']: p for p in (min(group, key=lambda p:p['value']), max(group, key=lambda p:p['value']))}
        result.extend(sorted(pair.values(), key=lambda p:p['step']))
    return result + [points[-1]]


class LogFeed:
    def __init__(self, run):
        self.run = Path(run)
        self.config = json.loads((self.run/'resolved_config.json').read_text())
        self.offset = 0
        self.identity = None
        self.latest = None
        self.validation = None
        self.started_at = None
        self.history = {key: {} for key in ('trainLoss','valLoss','learningRate','throughput')}
        self.state = 'live'

    def read(self):
        path = self.run/'trainlog.txt'
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity != identity or stat.st_size < self.offset:
            self.offset = 0
            self.latest = self.validation = None
            self.history = {key:{} for key in self.history}
            self.state = 'live'
        self.identity = identity
        with path.open('rb') as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line or not line.endswith(b'\n'):
                    break
                self.offset = f.tell()
                self.parse(line.decode('utf-8', errors='replace'))

    def parse(self, line):
        fields = dict(FIELDS.findall(line))
        try:
            stamp = dt.datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=TZ)
        except ValueError:
            stamp = None
        if stamp is not None and getattr(self, 'started_at', None) is None:
            self.started_at = stamp
        if 'target reached' in line:
            self.state = 'finished'
        elif 'training failed' in line or 'Traceback (most recent call last)' in line:
            self.state = 'error'
        elif 'stopped step=' in line:
            self.state = 'paused'
        try:
            step = int(fields['step'])
            if stamp is None:
                return
            if 'validation step=' in line:
                value = float(fields['loss'])
                if math.isfinite(value):
                    self.validation = value
                    self.history['valLoss'][step] = value
            elif 'tok/s=' in line and 'tokens=' in line:
                values = {key: float(fields[src]) for src,key in KEYS.items()}
                if not all(math.isfinite(v) for v in values.values()):
                    return
                # A resumed checkpoint may roll back; discard superseded future points.
                if self.latest and step < self.latest['step']:
                    for history in self.history.values():
                        for old in list(history):
                            if old > step:
                                del history[old]
                self.latest = dict(step=step, tokens=int(fields['tokens']), stamp=stamp, **values)
                for key,value in values.items():
                    self.history[key][step] = value
                self.state = 'live'
        except (KeyError, ValueError):
            return

    def payload(self):
        self.read()
        if not self.latest:
            return None
        row = self.latest
        pretraining = self.config.get('pretraining')
        recipe = pretraining or self.config.get('continuation')
        if not recipe:
            raise ValueError('Run has no pretraining or continuation token budget')
        target = recipe['target_tokens']
        train = self.config['training']
        wandb_file = self.run/'wandb_run.json'
        wandb = json.loads(wandb_file.read_text()) if wandb_file.exists() else {}
        state = self.state
        if state == 'live' and time.time()-row['stamp'].timestamp() > 180:
            state = 'paused'
        return dict(
            title='击水 · 200M', subtitle='古汉语原生语言模型 · 从零预训练',
            runId=self.run.name, runName=('200M · 1945年前语料' if pretraining else '200M → 4B tokens'), runUrl=wandb.get('url') or '',
            startedAt=self.started_at.isoformat() if self.started_at else row['stamp'].isoformat(),
            status=state, timestamp=row['stamp'].isoformat(),
            config=dict(model='击水 200M', dataset=('中流语料库 · 成书年代上界 < 1945' if pretraining else '古汉语混合语料'), batchSize=train['batch_size']*train['grad_accum_steps'],sequenceLength=train['seq_len']),
            progress=dict(step=row['step'],tokens=row['tokens'],targetTokens=target),
            metrics={**{key:row[key] for key in KEYS.values()},'valLoss':self.validation},
            charts={key:sample([dict(step=step,value=value) for step,value in sorted(history.items())]) for key,history in self.history.items()},
            replaceHistory=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/jishui-200m-cuda-4b'))
    parser.add_argument('--endpoint', default=os.environ.get('TRAINING_DASHBOARD_INGEST_URL') or os.environ.get('TRAINING_PUSH_URL'))
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    token = os.environ.get('TRAINING_INGEST_TOKEN','')
    if not args.dry_run and (not args.endpoint or not token):
        parser.error('Set TRAINING_DASHBOARD_INGEST_URL (or TRAINING_PUSH_URL) and TRAINING_INGEST_TOKEN')
    if not args.dry_run and not args.endpoint.startswith('https://') and not args.endpoint.startswith('http://127.0.0.1:'):
        parser.error('Use HTTPS for remote pushes')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    feed = LogFeed(args.run_dir)
    while True:
        try:
            payload = feed.payload()
            if payload is None:
                raise RuntimeError('No training metrics yet')
            if args.dry_run:
                print(json.dumps(payload, ensure_ascii=False))
                return
            request = urllib.request.Request(args.endpoint, json.dumps(payload, allow_nan=False).encode(), headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.load(response)
                if not result.get('ok'):
                    raise RuntimeError('Ingest did not acknowledge update')
            logging.info('Published step=%s status=%s', payload['progress']['step'],payload['status'])
        except Exception as error:
            logging.error('Push failed (%s); will retry', type(error).__name__)
            if args.once:
                raise SystemExit(1)
        if args.once:
            return
        time.sleep(max(10,args.interval))

if __name__ == '__main__':
    main()
