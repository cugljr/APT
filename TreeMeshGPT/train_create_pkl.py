from tokenizer import tokenize
import os
import pickle

MESH_IN = "dummy_data/mesh"
PKL_OUT = "dummy_data/pkl"

QUANT_BIT = 7
N_TRIAL = 10
MAX_N_FACES = 5500
AUGMENTATION = True

mesh_fns = os.listdir(MESH_IN)

if not os.path.exists(PKL_OUT):
    os.makedirs(PKL_OUT)

while True:
    for fn in mesh_fns:
        mesh_path = os.path.join(MESH_IN, fn)
        io_dict = tokenize(mesh_path, QUANT_BIT, N_TRIAL, MAX_N_FACES, AUGMENTATION)
        
        if io_dict is not None:
            out_path = os.path.join(PKL_OUT, fn[:-4] + ".pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(io_dict, f)
                