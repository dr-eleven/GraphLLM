# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def make_trainer(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "keep_best_and_last": True,
                "best_checkpoint_metric": "val-core/global/acc/mean@1",
                "resume_mode": "auto",
            }
        }
    )
    trainer.best_checkpoint_metric = "val-core/global/acc/mean@1"
    trainer.best_checkpoint_value = None
    trainer.best_checkpoint_step = None
    trainer.global_steps = 50
    return trainer


def create_checkpoint(root, step):
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "model.pt").write_text("weights")
    (checkpoint / "data.pt").write_text("dataloader")


def test_retains_only_best_and_latest_complete_directories(tmp_path):
    trainer = make_trainer(tmp_path)
    for step in (50, 100, 150, 200):
        create_checkpoint(tmp_path, step)

    trainer.best_checkpoint_step = 100
    trainer.best_checkpoint_value = 0.4
    trainer.global_steps = 200
    trainer._apply_best_and_last_retention()

    retained = sorted(path.name for path in tmp_path.glob("global_step_*"))
    assert retained == ["global_step_100", "global_step_200"]
    assert (tmp_path / "global_step_100" / "data.pt").exists()
    assert (tmp_path / "global_step_200" / "actor" / "model.pt").exists()


def test_new_best_replaces_old_best_and_persists_across_resume(tmp_path):
    trainer = make_trainer(tmp_path)
    create_checkpoint(tmp_path, 50)
    create_checkpoint(tmp_path, 100)

    assert trainer._update_best_checkpoint({"val-core/global/acc/mean@1": 0.25})
    trainer._save_best_checkpoint_state()

    trainer.global_steps = 100
    assert trainer._update_best_checkpoint({"val-core/global/acc/mean@1": 0.3})
    trainer._save_best_checkpoint_state()
    trainer._apply_best_and_last_retention()

    state = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert state == {
        "metric": "val-core/global/acc/mean@1",
        "value": 0.3,
        "step": 100,
    }
    assert sorted(path.name for path in tmp_path.glob("global_step_*")) == ["global_step_100"]

    resumed_trainer = make_trainer(tmp_path)
    resumed_trainer._load_best_checkpoint_state()
    assert resumed_trainer.best_checkpoint_step == 100
    assert resumed_trainer.best_checkpoint_value == 0.3


def test_accuracy_ties_keep_the_earlier_best_step(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.best_checkpoint_step = 50
    trainer.best_checkpoint_value = 0.3
    trainer.global_steps = 100

    assert not trainer._update_best_checkpoint({"val-core/global/acc/mean@1": 0.3})
    assert trainer.best_checkpoint_step == 50
