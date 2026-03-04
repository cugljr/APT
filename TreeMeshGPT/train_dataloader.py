import torch
import os
from os import path
from typing import Dict
from fns import quantize_verts
import pickle
import random


class EdgesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        quant_bit: int = 7,

    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.quant_bit = quant_bit     
        self.data_path = os.listdir(data_dir)

    def __len__(self) -> int:
        return len(self.data_path)
    
    def load_data_with_retry(self, model_path, max_attempts=10000, max_n_seq=18000, log_file='corrupted_files.txt'):
        attempt = 0
        while attempt < max_attempts:
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                n_seq = data['n_seq']
                
                if n_seq < max_n_seq:
                    return data
                else:
                    # If the number of sequences exceeds the max, try a new file
                    model_path = os.path.join(self.data_dir, random.choice(self.data_path))
                
            except (EOFError, pickle.UnpicklingError) as e:
                # Log the corrupted file into a text file
                with open(log_file, 'a') as log:
                    log.write(f"Corrupted file: {model_path}\n")
                
                # Try another file
                model_path = os.path.join(self.data_dir, random.choice(self.data_path))
            
            attempt += 1
        raise RuntimeError(f"Exceeded maximum attempts ({max_attempts}) to load a valid pickle file.")

    def __getitem__(self, idx: int) -> Dict:
        model_path = path.join(self.data_dir, self.data_path[idx])     
        batch = {}
        
        data = self.load_data_with_retry(model_path)
        pc = data['pc']

        vertices = data['vertices']
        vertices = quantize_verts(vertices, self.quant_bit)
        
        # load edges sub-partitions data
        edges = data['edges']
        gt = data['gt']
        
        # construct data
        for i in range(len(edges)):
            gt[i].insert(0, edges[i][0][1])
            gt[i].insert(0, edges[i][0][0])
            edges[i].insert(0,[edges[i][0][0], -1])
            edges[i].insert(0, [-1, -1])
            
        gt_combine = [elem for sublist in gt for elem in sublist]
        edges_combine = [elem for sublist in edges for elem in sublist]

        gt_combine.append(-2) # eos
        edges_combine.append([-1, -1])
                
        batch['vertices'] = torch.tensor(vertices)
        batch['edges'] = torch.tensor(edges_combine)
        batch['gt_ind'] = torch.tensor(gt_combine)
        batch['pc'] = torch.tensor(pc)
        
        return batch

