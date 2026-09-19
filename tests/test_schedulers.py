import unittest

import torch

from engine.schedulers import WarmupStepDecayScheduler


class SchedulerTest(unittest.TestCase):
    def test_state_restores_iteration_and_epoch(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = WarmupStepDecayScheduler(
            optimizer, warmup_iters=2, milestones=(1,), gamma=0.1
        )
        scheduler.step_epoch(1)
        scheduler.step_iter()

        restored = WarmupStepDecayScheduler(
            optimizer, warmup_iters=2, milestones=(1,), gamma=0.1
        )
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.iteration, 1)
        self.assertEqual(restored.epoch, 1)


if __name__ == "__main__":
    unittest.main()
