import torch
import random
from TTS.tts.models.xtts import load_audio

torch.set_num_threads(1)

def key_samples_by_col(samples, col):
    """Returns a dictionary of samples keyed by language."""
    samples_by_col = {}
    for sample in samples:
        col_val = sample[col]
        assert isinstance(col_val, str)
        if col_val not in samples_by_col:
            samples_by_col[col_val] = []
        samples_by_col[col_val].append(sample)
    return samples_by_col

class DVAEDataset(torch.utils.data.Dataset):
    def __init__(self, samples, sample_rate, is_eval, max_wav_len=255995):
        self.sample_rate = sample_rate
        self.is_eval = is_eval
        self.max_wav_len = max_wav_len
        self.samples = samples
        self.training_seed = 1
        self.failed_samples = set()
        if not is_eval:
            random.seed(self.training_seed)
            # random.shuffle(self.samples)
            random.shuffle(self.samples)
            # order by language
            self.samples = key_samples_by_col(self.samples, "language")
            print(" > Sampling by language:", self.samples.keys())
        else:
            # for evaluation load and check samples that are corrupted to ensures the reproducibility
            self.check_eval_samples()

    def check_eval_samples(self):
        print(" > Filtering invalid eval samples!!")
        new_samples = []
        for sample in self.samples:
            try:
                _, wav = self.load_item(sample)
            except:
                continue
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            if (
                wav is None
                or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
            ):
                continue
            new_samples.append(sample)
        self.samples = new_samples
        print(" > Total eval samples after filtering:", len(self.samples))

    def load_item(self, sample):
        audiopath = sample["audio_file"]
        wav = load_audio(audiopath, self.sample_rate)
        if wav is None or wav.shape[-1] < (0.5 * self.sample_rate):
            # Ultra short clips are also useless (and can cause problems within some models).
            raise ValueError

        return audiopath, wav
    
    def __getitem__(self, index):
        if self.is_eval:
            sample = self.samples[index]
            sample_id = str(index)
        else:
            # select a random language
            lang = random.choice(list(self.samples.keys()))
            # select random sample
            index = random.randint(0, len(self.samples[lang]) - 1)
            sample = self.samples[lang][index]
            # a unique id for each sampel to deal with fails
            sample_id = lang + "_" + str(index)

        # ignore samples that we already know that is not valid ones
        if sample_id in self.failed_samples:
            # call get item again to get other sample
            return self[1]

        # try to load the sample, if fails added it to the failed samples list
        try:
            audiopath, wav = self.load_item(sample)
        except:
            self.failed_samples.add(sample_id)
            return self[1]

        # check if the audio and text size limits and if it out of the limits, added it failed_samples
        if (
            wav is None
            or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
        ):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            self.failed_samples.add(sample_id)
            return self[1]

        res = {
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
        }
        return res
    
    def __len__(self):
        if self.is_eval:
            return len(self.samples)
        return sum([len(v) for v in self.samples.values()])

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])

        max_wav_len = batch["wav_lengths"].max()

        # create padding tensors
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        wav_padded = wav_padded.zero_()
        for i in range(B):
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded
        return batch