import ast
from pathlib import Path
root=Path(__file__).resolve().parent
base=ast.parse((root.parent/'initial_submission/baseline.py').read_text())
pilot=ast.parse((root.parent/'embedding_experiment/helpers.py').read_text())
def grab(tree,names):
 return '\n\n'.join(ast.unparse(n) for n in tree.body if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)))
header='''import os, json, time, gc, sqlite3, hashlib, shutil, pickle, re, unicodedata
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
'''
text=header+grab(base,{'COLS','LEGAL','ADDR_ALIAS','FEATURES','norm','canonical_address','grams','unit_id','prepared','jaccard','containment','features','log'})+'\n\n'+grab(pilot,{'read_chunks','reproduce_reference_sample','clean_embedding_text','text_view'})+'\n\n'
(root/'pipeline.py').write_text(text)
