import importlib.util
from pathlib import Path
import unittest
import datetime as dt
import json
import tempfile
spec=importlib.util.spec_from_file_location('dashboard',Path(__file__).parents[1]/'scripts/push_training_dashboard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class DashboardTests(unittest.TestCase):
 def test_pretraining_payload_uses_budget_and_preserves_start_across_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   (root/'resolved_config.json').write_text(json.dumps({'pretraining':{'target_tokens':4000000000},'continuation':None,'training':{'batch_size':1,'grad_accum_steps':8,'seq_len':2048}}))
   (root/'wandb_run.json').write_text(json.dumps({'url':None}))
   now=dt.datetime.now(m.TZ)
   start=(now-dt.timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S,%f')[:23]
   latest=now.strftime('%Y-%m-%d %H:%M:%S,%f')[:23]
   (root/'trainlog.txt').write_text(f'{start} INFO device=CUDA\n{start} INFO validation step=0 loss=10.5\n{start} INFO stopped step=20 tokens=327680\n{latest} INFO step=30 loss=9.2 lr=9e-6 tokens=491520 tok/s=7700\n')
   payload=m.LogFeed(root).payload()
   self.assertEqual(payload['progress']['targetTokens'],4000000000)
   self.assertEqual(payload['progress']['step'],30)
   self.assertEqual(payload['status'],'live')
   self.assertEqual(payload['runUrl'],'')
   self.assertEqual(payload['config']['batchSize'],8)
   self.assertIn('1945',payload['config']['dataset'])
   self.assertEqual(payload['startedAt'],dt.datetime.strptime(start,'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=m.TZ).isoformat())
   self.assertEqual(payload['startedAt'],m.LogFeed(root).payload()['startedAt'])
   self.assertEqual(payload['charts']['valLoss'],[{'step':0,'value':10.5}])
 def test_run_name_and_subtitle_follow_resolved_config(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   now=dt.datetime.now(m.TZ).strftime('%Y-%m-%d %H:%M:%S,%f')[:23]
   (root/'trainlog.txt').write_text(f'{now} INFO step=10 loss=9.2 lr=1e-6 tokens=163840 tok/s=6300\n{now} INFO step=10 mtp_loss=9.5 weight=0.30\n')
   base={'pretraining':{'target_tokens':4000006144,'sampling':'shuffled_blocks'},'sampling':{'mode':'shuffled_blocks'},
         'data_policy':{'date_policy':'exclude_known_late'},'training':{'batch_size':1,'grad_accum_steps':8,'seq_len':2048}}
   (root/'resolved_config.json').write_text(json.dumps(dict(base,model={'mtp_depth':1},origin={'format':'random','seed':42})))
   payload=m.LogFeed(root).payload()
   self.assertEqual(payload['runName'],'200M · 年代宽筛 · 分块打乱 · MTP')
   self.assertTrue(payload['subtitle'].endswith('从零预训练'))
   self.assertEqual(payload['metrics']['trainLoss'],9.2)
   self.assertEqual(len(payload['charts']['trainLoss']),1)
   pre=dict(base['pretraining'],attention='document',optimizer={'name':'muon'})
   (root/'resolved_config.json').write_text(json.dumps(dict(base,pretraining=pre,model={},distributed={'world_size':8},origin={'format':'weights_only'})))
   payload=m.LogFeed(root).payload()
   self.assertEqual(payload['runName'],'200M · 年代宽筛 · 分块打乱 · 文档隔断 · Muon · 8卡')
   self.assertTrue(payload['subtitle'].endswith('继续预训练'))
   self.assertEqual(payload['config']['batchSize'],64)
 def test_sampling_retains_spike_and_endpoints(self):
  points=[dict(step=i,value=999 if i==573 else 1) for i in range(2000)]
  result=m.sample(points,100)
  self.assertLessEqual(len(result),100)
  self.assertEqual(result[0],points[0]);self.assertEqual(result[-1],points[-1]);self.assertIn(points[573],result)
 def test_validation_not_repeated_and_rollback(self):
  feed=m.LogFeed.__new__(m.LogFeed)
  feed.latest=None;feed.validation=None;feed.state='live';feed.history={k:{} for k in ('trainLoss','valLoss','learningRate','throughput')}
  feed.parse('2026-10-02 20:00:00,000 INFO validation step=100 loss=4.2 ppl=66')
  for step in (101,102,101):
   feed.parse(f'2026-10-02 20:00:01,000 INFO step={step} loss=5.1 lr=1e-4 tokens=10000 tok/s=7000')
  self.assertEqual(list(feed.history['valLoss']),[100])
  self.assertEqual(list(feed.history['trainLoss']),[101])
if __name__=='__main__':unittest.main()
