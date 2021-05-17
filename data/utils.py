# Copyright 2021 Zhejiang University of Techonology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import torch
from torch.utils.data.dataset import Dataset


def one_batch_dataset(dataset, batch_size):
    print("==> Grabbing a single batch")

    perm = torch.randperm(len(dataset))

    one_batch = [dataset[idx.item()] for idx in perm[:batch_size]]

    class _OneBatchWrapper(Dataset):
        def __init__(self):
            self.batch = one_batch

        def __getitem__(self, index):
            return self.batch[index]

        def __len__(self):
            return len(self.batch)

    return _OneBatchWrapper()