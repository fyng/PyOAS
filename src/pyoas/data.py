from pathlib import Path
import json
import ast
import re
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

class OASData():
    def __init__(self):
        self.df = None
        self.annot_map = {
            'fwk1': '1',
            'cdrk1': '2',
            'fwk2': '3',
            'cdrk2': '4',
            'fwk3': '5',
            'cdrk3': '6',
            'fwk4': '7',
            'fwl1': '1',
            'cdrl1': '2',
            'fwl2': '3',
            'cdrl2': '4',
            'fwl3': '5',
            'cdrl3': '6',
            'fwl4': '7',
            'fwh1': '1',
            'cdrh1': '2',
            'fwh2': '3',
            'cdrh2': '4',
            'fwh3': '5',
            'cdrh3': '6',
            'fwh4': '7',
        }

    def load_data_from_files(self, data_path = None):
        sequences = []
        metadata = {}

        meta_idx = 0
        data_path = Path(data_path)
        files = list(data_path.glob('*.csv.gz'))
        for filename in tqdm(files):
            metadata[meta_idx] = json.loads(','.join(pd.read_csv(filename, nrows=0).columns))
            seq = pd.read_csv(filename, header=1)
            seq['meta_idx'] = meta_idx
            sequences.append(seq)
            meta_idx += 1
        self.df = pd.concat(sequences, ignore_index=True)


    def _row_anarci_aa_seq_label(self, row):
        anarci_dict_light = ast.literal_eval(row['ANARCI_numbering_light'])
        label_heavy = []
        seq_heavy = []
        pos_heavy = []
        anarci_dict_heavy = ast.literal_eval(row['ANARCI_numbering_heavy'])
        label_light = []
        seq_light = []
        pos_light = []
        offset_light = 0
        offset_heavy = 0

        for region, dict in anarci_dict_light.items():
            for pos, aa in dict.items():
                seq_light.append(aa)
                label_light.append(self.annot_map[region])
                pos, offset = self._process_anarci_pos(pos)
                offset_light += offset
                pos_light.append(pos + offset_light - 1)
        for region, dict in anarci_dict_heavy.items():
            for pos, aa in dict.items():
                seq_heavy.append(aa)
                label_heavy.append(self.annot_map[region])
                pos, offset = self._process_anarci_pos(pos)
                offset_heavy += offset
                pos_heavy.append(pos + offset_heavy - 1)
                
        return label_heavy, seq_heavy, pos_heavy, label_light, seq_light, pos_light
    

    def _process_anarci_pos(self, pos):
        if re.search(r'[a-zA-Z]$', pos):
            return int(pos[:-1]), 1
        else:
            return int(pos), 0


    def get_vdj_label_aa_sequence_match(self, row):
        # TODO: this whole dictionary thing is disgusting. Make a tokenizer class to handle the functional annotations, and probably store an array instead of a string
        alignment_aa_heavy = row['sequence_alignment_aa_heavy']
        label_heavy = ['-'] * len(alignment_aa_heavy)
        idx = 0

        if not pd.isna(row['v_sequence_alignment_aa_heavy']):
            match = re.search(row['v_sequence_alignment_aa_heavy'], alignment_aa_heavy)
            if match:
                start, end = match.span()
                idx = end
                label_heavy[start:end] = 'V' * (end - start)
        if not pd.isna(row['d_sequence_alignment_aa_heavy']):
            match = re.search(row['d_sequence_alignment_aa_heavy'], alignment_aa_heavy[idx:])
            if match:
                start, end = match.span()
                start += idx
                end += idx
                label_heavy[start:end] = 'D' * (end - start)
                idx = end
        if not pd.isna(row['j_sequence_alignment_aa_heavy']):
            match = re.search(row['j_sequence_alignment_aa_heavy'], alignment_aa_heavy[idx:])
            if match:
                start, end = match.span()
                start += idx
                end += idx
                label_heavy[start:end] = 'J' * (end - start)
            label_heavy = ''.join(label_heavy)

        alignment_aa_light = row['sequence_alignment_aa_light']
        label_light = ['-'] * len(alignment_aa_light)
        idx = 0

        if not pd.isna(row['v_sequence_alignment_aa_light']):
            match = re.search(row['v_sequence_alignment_aa_light'], alignment_aa_light)
            if match:
                start, end = match.span()
                idx = end
                label_light[start:end] = 'V' * (end - start)
        if not pd.isna(row['d_sequence_alignment_aa_light']):
            match = re.search(row['d_sequence_alignment_aa_light'], alignment_aa_light[idx:])
            if match:
                start, end = match.span()
                start += idx
                end += idx
                label_light[start:end] = 'D' * (end - start)
                idx = end
        if not pd.isna(row['j_sequence_alignment_aa_light']):  
            match = re.search(row['j_sequence_alignment_aa_light'], alignment_aa_light[idx:])
            if match:
                start, end = match.span()
                start += idx
                end += idx
                label_light[start:end] = 'J' * (end - start)
        label_light = ''.join(label_light)

        return label_heavy, label_light


    def df_row_to_array_aligned(self, row, to_tensor=False, aligned=True):
        label_vdj_heavy, label_vdj_light = self.get_vdj_label_aa_sequence_match(row) # arrays
        label_anarci_heavy, seq_heavy, pos_heavy, label_anarci_light, seq_light, pos_light = self._row_anarci_aa_seq_label(row)

        # # protein sequence must match between ANARCI dict entry and recorded entry
        # assert(''.join(seq_heavy) == row['sequence_alignment_aa_heavy'])
        # assert(''.join(seq_light) == row['sequence_alignment_aa_light'])

        # map to standardized position
        # assume that IMGT Ab has max 128 positions 
        # https://github.com/oxpig/ANARCI
        seq_data_heavy = np.full((3,168), '.', ) # '.' token for gaps
        seq_data_light = np.full((3,168), '.') # '.' token for gaps

        if aligned:
            seq_data_heavy[0, pos_heavy] = seq_heavy
            seq_data_heavy[1, pos_heavy] = label_anarci_heavy
            seq_data_heavy[2, pos_heavy] = label_vdj_heavy
            
            seq_data_light[0, pos_light] = seq_light
            seq_data_light[1, pos_light] = label_anarci_light
            seq_data_light[2, pos_light] = label_vdj_light
        else:
            # TODO: postpend a SEP token

            seq_data_heavy[0, len(seq_heavy)] = seq_heavy 
            seq_data_heavy[1, len(label_anarci_heavy)] = label_anarci_heavy
            seq_data_heavy[2, len(label_vdj_heavy)] = label_vdj_heavy
            
            seq_data_light = np.full((3,168), '.') # '.' token for gaps
            seq_data_light[0, len(seq_light)] = seq_light
            seq_data_light[1, len(label_anarci_light)] = label_anarci_light
            seq_data_light[2, len(label_vdj_light)] = label_vdj_light

        if to_tensor:
            seq_data_heavy = torch.tensor(seq_data_heavy)
            seq_data_light = torch.tensor(seq_data_light)
        
        return seq_data_heavy, seq_data_light

    def to_array(self, aligned=True, save_dir=None):
        res = self.df.apply(lambda x: self.df_row_to_array_aligned(x, aligned=aligned), axis=1)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            res.to_pickle(save_dir / 'oas_arr.pkl')

        return res
    
    
        