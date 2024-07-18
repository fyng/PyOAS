from abc import ABC, abstractmethod
from dataclasses import dataclass

from pathlib import Path
import pickle
import json
import ast
import re
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

from pyoas.utils import vocab


@dataclass
class Antibody():
    sequence: list[str] = None
    vdj_label: list[str] = None
    anarci_label: list[str] = None
    position: list[int] = None
    metadata: dict = None

    def __str__(self) -> str:
        return(f"{''.join(self.sequence)}\n{''.join(self.vdj_label)}\n{''.join(self.anarci_label)}\n{self.position}\n{self.metadata}")

@dataclass
class AntibodyTensor():
    sequence: torch.Tensor = None
    vdj_label: torch.Tensor = None
    anarci_label: torch.Tensor = None
    position: torch.Tensor = None
    metadata: dict = None    
    

class DataModule(ABC):
    @abstractmethod
    def load_data_folder(self):
        raise NotImplementedError

    @abstractmethod
    def to_array(self):
        raise NotImplementedError

    @abstractmethod
    def to_dataframe(self):
        raise NotImplementedError

    @abstractmethod
    def load_saved(self):
        raise NotImplementedError

class OASDataModule(DataModule):
    pass


class OTSDataModule(DataModule):
    '''
    Data module for (Observed TCR Space)[https://opig.stats.ox.ac.uk/webapps/ots]
    '''
    def __init__(self) -> None:
        self.df = None
        self.metadata_dict = {}
        self.data = None

    def _process_anarci_pos(self, pos):
        # FIXME: is this really correct? right now we shift the alignment upon insertion, but we want to preserve the alignment
        match = re.match(r'(\d+)([A-Za-z].*)?', pos)
        if match:
            return int(match.group(1)), 1
        else:
            return int(pos), 0

    def _parse_anarci_dict(self, x):
        dict = ast.literal_eval(x) if isinstance(x, str) else x

        label = []
        sequence = []
        position = []
        pos_offset = 0

        for region, dict in dict.items():
            for pos, aa in dict.items():
                sequence.append(aa)
                label.append(region)
                pos, offset = self._process_anarci_pos(pos)
                pos_offset += offset
                position.append(pos + pos_offset - 1)

        return sequence, label, position
    
    def _parse_vdj_label(self, seq, v, d, j):
        label = [vocab.UNK_STR] * len(seq)
        idx = 0
        if v:
            match = re.search(v, seq)
            if match:
                start, end = match.span()
                idx = end
                label[start:end] = 'V' * (end - start)
        if d:
            match = re.search(d, seq)
            if match:
                start, end = match.span()
                start += idx
                end += idx
                label[start:end] = 'D' * (end - start)
                idx = end
        if j:
            match = re.search(j, seq)
            if match:
                start, end = match.span()
                start += idx
                end += idx
                label[start:end] = 'J' * (end - start)
        
        return label

    def parse_data(self):
        if self.df is None:
            raise ValueError('No data loaded')

        columns_to_convert = [
            'ANARCI_numbering_alpha',
            'ANARCI_numbering_beta',
            'v_sequence_alignment_aa_alpha',
            'j_sequence_alignment_aa_alpha',
            'd_sequence_alignment_aa_alpha',
            'v_sequence_alignment_aa_beta',
            'j_sequence_alignment_aa_beta',
            'd_sequence_alignment_aa_beta'
        ]
        # TODO: discard NAs in key columns
        self.df[columns_to_convert] = self.df[columns_to_convert].astype(str)

        anarci_dict_a = self.df['ANARCI_numbering_alpha'].to_list()
        anarci_dict_b = self.df['ANARCI_numbering_beta'].to_list()
        v_a = self.df['v_sequence_alignment_aa_alpha'].to_list()
        j_a = self.df['j_sequence_alignment_aa_alpha'].to_list()
        d_a = self.df['d_sequence_alignment_aa_alpha'].to_list()
        v_b = self.df['v_sequence_alignment_aa_beta'].to_list()
        j_b = self.df['j_sequence_alignment_aa_beta'].to_list()
        d_b = self.df['d_sequence_alignment_aa_beta'].to_list()
        meta_idx = self.df['metadata_idx'].to_list()

        data = []
        for i, (aa, ab, va, ja, da, vb, jb, db) in enumerate(tqdm(
            zip(anarci_dict_a, anarci_dict_b, v_a, j_a, d_a, v_b, j_b, d_b), total=len(anarci_dict_a)
        )):
            seq_a, anarci_label_a, pos_a = self._parse_anarci_dict(aa)
            vdj_label_a = self._parse_vdj_label(''.join(seq_a), va, da, ja)
            seq_b, anarci_label_b, pos_b = self._parse_anarci_dict(ab)
            vdj_label_b = self._parse_vdj_label(''.join(seq_b), vb, db, jb)

            seq_a.append(vocab.CHAIN_BREAK_STR)
            seq_a.extend(seq_b)
            anarci_label_a.append(vocab.CHAIN_BREAK_STR)
            anarci_label_a.extend(anarci_label_b)
            pos_a.append(0)
            pos_a.extend(pos_b)
            vdj_label_a.append(vocab.CHAIN_BREAK_STR)
            vdj_label_a.extend(vdj_label_b)

            ab = Antibody(
                sequence=seq_a, 
                vdj_label=seq_b, 
                anarci_label=anarci_label_a, 
                position=pos_a,
                metadata=self.metadata_dict[meta_idx[i]]
            )    
            data.append(ab)
        self.data = data

    def load_data_folder(self, data_path = None):
        dataframes = []
        metadata_dict = {}
        metadata_idx = 0

        data_path = Path(data_path)
        files = list(data_path.glob('*.csv.gz'))
        for filename in tqdm(files):
            df = pd.read_csv(filename, header=1)
            metadata = json.loads(','.join(pd.read_csv(filename, nrows=0).columns))
            df['metadata_idx'] = metadata_idx
            for k, v in metadata.items():
                df[k] = v 
            dataframes.append(df)
            metadata_dict[metadata_idx] = metadata
            metadata_idx += 1

        self.df = pd.concat(dataframes, ignore_index=True)
        self.metadata_dict = metadata_dict

    def to_dataframe(self):
        return self.df

    def to_array(self, aligned=True, save_dir=None):
        pass

    def to_antibody(self, save=True):
        # 10mins on single thread
        if self.data is None:
            self.parse_data()

        if save:
            with open('output/ab.pkl', 'wb') as f:
                pickle.dump(self.data, f)

        return self.data

    def load_saved(self, path = None):
        fp = Path(path)
        with open(fp, 'rb') as f:
            self.data = pickle.load(f)
        