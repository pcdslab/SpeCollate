from torch.utils.data import Sampler
import random

from src.atlesconfig import config


class PSMSampler(Sampler):
    r"""Samples labeled spectra randomly within a mega batch.

    Args:
        masses (Dataset): used to get appropriate indices.
    """
#     mega_batches: Sized

    def __init__(self, masses):
        self.batch_size = config.get_config(section="ml", key="batch_size")
        self.batch_queue = []
        
        self.masses = masses

        min_mass_mb1 = min(self.masses) + 0.2

        mb_size = 2 * self.batch_size
        self.mega_batches_1 = []
        self.mega_batches_2 = []
        prev = cutoff = 0
        # TODO: check for the end of the list and empty range
        for idx, mass in enumerate(self.masses):
            if idx < prev:
                continue
            if mass > min_mass_mb1:
                while cutoff <= idx:
                    cutoff = prev + mb_size
                while cutoff > len(self.masses):
                    cutoff -= self.batch_size
                if cutoff - prev == 0:
                    break
                self.mega_batches_1.append((prev, cutoff))
                prev = cutoff
                min_mass_mb1 = self.masses[cutoff - 1] + 0.2

        start = 0
        for mb1_range in self.mega_batches_1:
            end = int( ((mb1_range[1] - mb1_range[0]) / 2) + mb1_range[0])
            if mb1_range[1] - mb1_range[0] == self.batch_size:
                end = mb1_range[1]
            self.mega_batches_2.append((start, end))
            start = end
        if start < self.mega_batches_1[-1][1]:
            self.mega_batches_2.append((start, self.mega_batches_1[-1][1]))
        
        self.current_mb = []
        self.initial_len = int(self.mega_batches_1[-1][1])
        

    def __iter__(self):
        if not self.batch_queue:
            if self.current_mb is self.mega_batches_1:
                self.current_mb = self.mega_batches_2
            else:
                self.current_mb = self.mega_batches_1
            random.shuffle(self.current_mb)
            
            for mb in self.current_mb:
                mb_range = list(range(mb[0], mb[1]))
                random.shuffle(mb_range)
                assert len(mb_range) % self.batch_size == 0
                for i in range(0, len(mb_range), self.batch_size):
                    self.batch_queue.append(mb_range[i:i+self.batch_size])
            
        for batch in random.sample(self.batch_queue, len(self.batch_queue)):
            for idx in batch:
                yield idx
        self.batch_queue = []


    def __len__(self) -> int:
        return int(self.current_mb[-1][1]) if self.current_mb else self.initial_len