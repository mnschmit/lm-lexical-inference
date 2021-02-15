from typing import Tuple, Union, Dict, Optional, List, Iterable
from torch.utils.data import Dataset
from overrides import overrides
from .common import PREM_KEY, HYPO_KEY, LABEL_KEY, PATTERNS, ANTIPATTERNS,\
    SENT_KEY, ANTI_KEY, load_patterns, chunks


class LevyHoltBase(Dataset):
    def __init__(self, txt_file):
        self.data = self.load_dataset(txt_file)

    def load_dataset(self, txt_file):
        data = []
        with open(txt_file) as f:
            for line in f:
                hypo, prem, label = line.strip().split('\t')
                hypo = tuple(h.strip() for h in hypo.split(','))
                prem = tuple(p.strip() for p in prem.split(','))
                label = label == 'True'
                data.extend(self.create_instances(prem, hypo, label))
        return data

    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool
    ) -> List[Dict[str, Union[bool, str]]]:
        raise NotImplementedError(
            "You have to implement `create_instances` in a subclass inheriting from `LevyHoltBase`"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError(
            "You have to implement `__getitem__` in a subclass inheriting from `LevyHoltBase`"
        )


class LevyHoltSentences(LevyHoltBase):

    @overrides
    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool
    ) -> List[Dict[str, Union[bool, str]]]:
        inst = {
            PREM_KEY: ' '.join(prem),
            HYPO_KEY: ' '.join(hypo),
            LABEL_KEY: label
        }
        return [inst]

    @overrides
    def __getitem__(self, index):
        inst = self.data[index]
        return inst[PREM_KEY], inst[HYPO_KEY], inst[LABEL_KEY]


class LevyHoltPattern(LevyHoltBase):

    def __init__(self, txt_file: str, pattern_file: Optional[str] = None,
                 antipattern_file: Optional[str] = None, best_k_patterns: Optional[int] = None,
                 pattern_chunk_size: int = 5, training: bool = False,
                 curated_auto: bool = False):
        self.training = training
        self.pattern_chunk_size = pattern_chunk_size

        if pattern_file is None:
            self.patterns = PATTERNS
            self.antipatterns = ANTIPATTERNS
            self.handcrafted = True
        else:
            if best_k_patterns is not None and best_k_patterns % pattern_chunk_size != 0:
                print(
                    "WARNING: best_k_patterns should be a"
                    + " multiple of pattern_chunk_size ({})".format(pattern_chunk_size))
            self.patterns = load_patterns(pattern_file, best_k_patterns)
            assert antipattern_file is not None,\
                "pattern_file and antipattern_file must either"\
                + " both be None or both set to file paths."
            self.antipatterns = load_patterns(
                antipattern_file, best_k_patterns)
            self.handcrafted = curated_auto

        super().__init__(txt_file)

    def create_sent_from_pattern(self, pattern: str, prem: Tuple[str, str, str],
                                 hypo: Tuple[str, str, str]) -> str:
        if self.handcrafted:
            sent = pattern.format(pal=prem[0], prem=prem[1], par=prem[2],
                                  hal=hypo[0], hypo=hypo[1], har=hypo[2])
        else:
            sent = pattern.format(prem=prem[1], hypo=hypo[1])
        return sent

    def create_single_instance(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool,
            patterns: Iterable[str],
            antipatterns: Iterable[str]) -> Dict[str, Union[bool, str]]:
        inst = {}

        inst[SENT_KEY] = [
            self.create_sent_from_pattern(pat, prem, hypo)
            for pat in patterns
        ]
        inst[ANTI_KEY] = [
            self.create_sent_from_pattern(pat, prem, hypo)
            for pat in antipatterns
        ]
        inst[LABEL_KEY] = label

        return inst

    @overrides
    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool
    ) -> List[Dict[str, Union[bool, str]]]:
        instances = []

        if self.training:
            chunked = [
                chunks(self.patterns, self.pattern_chunk_size),
                chunks(self.antipatterns, self.pattern_chunk_size)
            ]
            for pattern_chunk, antipattern_chunk in zip(*chunked):
                inst = self.create_single_instance(
                    prem, hypo, label, pattern_chunk, antipattern_chunk)
                instances.append(inst)
        else:
            inst = self.create_single_instance(
                prem, hypo, label, self.patterns, self.antipatterns)
            instances.append(inst)

        return instances

    @overrides
    def __getitem__(self, index):
        inst = self.data[index]
        return inst[SENT_KEY], inst[ANTI_KEY], inst[LABEL_KEY]
