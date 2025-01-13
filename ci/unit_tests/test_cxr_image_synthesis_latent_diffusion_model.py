# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import unittest
from parameterized import parameterized
import numpy as np
from monai.bundle import ConfigWorkflow
from utils import check_workflow

TEST_CASE_1 = [  # inference
    {
        "bundle_root": "models/cxr_image_synthesis_latent_diffusion_model",
        "prompt": "Big right-sided pleural effusion. Normal left lung.",
        "guidance_scale": 7.0,
    }
]

class TestCXRLatentDiffusionInference(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_inference(self, params):
        bundle_root = params["bundle_root"]
        inference_file = os.path.join(bundle_root, "configs/inference.json")
        trainer = ConfigWorkflow(
            workflow_type="inference",
            config_file=inference_file,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **params,
        )
        check_workflow(trainer, check_properties=True)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    unittest.main(testLoader=loader)
