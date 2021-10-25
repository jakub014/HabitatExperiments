# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from trainer import resnet_entry_point as rep
from trainer import inception_entry_point as iep
from trainer import dcgan_entry_point as dep
import torch
#import entry_point as rep

def main():
    """Setup / Start the experiment
    """

    batch_sizes = [16, 32, 48, 64, 96, 128]
    batch_sizes_inception = [4, 8, 16, 24]
    sample_size = 1024
    replication = 3
    print("Hihi")
    runtimes = []
    for batch_size in batch_sizes_inception:
        # INCEPTION ---------------------------------------------
        for k in range(replication):
            model = iep.skyline_model_provider()
            iteration = iep.skyline_iteration_provider(model)

            iterations = int(sample_size / batch_size)
            inputs = []
            for i in range(iterations):
                inputs.append(iep.skyline_input_provider(batch_size=batch_size))

            start = time.time()
            for itInput in list(inputs):
                iterationInput = itInput
                def runnable():
                    iteration(*iterationInput)
                runnable()
                inputs.remove(itInput)
                torch.cuda.empty_cache()

            del model
            torch.cuda.empty_cache()
            endTrainModel = time.time()
            runtime = endTrainModel - start
            runtimes.append(("Inception", batch_size, runtime))
    for batch_size in batch_sizes:
        # Resnet ---------------------------------------------

        for k in range(replication):
            model = rep.skyline_model_provider()
            iteration = rep.skyline_iteration_provider(model)

            iterations = int(sample_size / batch_size)
            inputs = []
            for i in range(iterations):
                inputs.append(rep.skyline_input_provider(batch_size=batch_size))

            start = time.time()
            for itInput in list(inputs):
                iterationInput = itInput
                def runnable():
                    iteration(*iterationInput)
                runnable()
                inputs.remove(itInput)
                torch.cuda.empty_cache()

            del model
            torch.cuda.empty_cache()
            endTrainModel = time.time()
            runtime = endTrainModel - start
            runtimes.append(("Resnet", batch_size, runtime))

        # dcgan ---------------------------------------------

        for k in range(replication):
            models = dep.skyline_model_provider()
            iteration = dep.skyline_iteration_provider(*models)

            iterations = int(sample_size / batch_size)
            inputs = []
            for i in range(iterations):
                inputs.append(dep.skyline_input_provider(batch_size=batch_size))

            start = time.time()
            for itInput in list(inputs):
                iterationInput = itInput

                def runnable():
                    iteration(*iterationInput)

                runnable()
                inputs.remove(itInput)
                torch.cuda.empty_cache()

            del models
            torch.cuda.empty_cache()
            endTrainModel = time.time()
            runtime = endTrainModel - start
            runtimes.append(("dcgan", batch_size, runtime))
    print('\n'.join(map(str, runtimes)))

    def runnable():
        iteration(*inputs)


if __name__ == '__main__':
    main()
