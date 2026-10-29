import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('dashboard',Path(__file__).parents[1]/'scripts/push_training_dashboard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class DashboardTests(unittest.TestCase):
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
