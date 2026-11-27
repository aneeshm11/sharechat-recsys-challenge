"""Assemble a single-file Kaggle script: python pack.py <pipeline.py> <MODE> <out_dir> <kernel_slug> [dataset ...]
Kernel sources (other notebooks' outputs) are passed as  k:owner/slug ."""
import base64, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
pipe, mode, out, slug = sys.argv[1:5]
srcs = sys.argv[5:]
job = open(os.path.join(here, 'xgbjob.py'), 'rb').read()
body = open(os.path.join(here, pipe), encoding='utf-8').read()
extra = ''
for name, var in (('feats2.py', 'FEATS2_B64'), ('catjob.py', 'CATJOB_B64')):
    p = os.path.join(here, name)
    if os.path.exists(p) and (var in body):
        extra += f"{var} = '{base64.b64encode(open(p, 'rb').read()).decode()}'\n"
head = f"MODE = '{mode}'\nXGBJOB_B64 = '{base64.b64encode(job).decode()}'\n" + extra
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'main.py'), 'w', encoding='utf-8', newline='\n').write(head + body)
meta = {
    "id": f"cassinicraft/{slug}", "title": slug, "code_file": "main.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": [s for s in srcs if not s.startswith('k:')], "kernel_sources": [s[2:] for s in srcs if s.startswith('k:')],
    "competition_sources": ["sharechat"], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
}
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print('packed', out, len(head) + len(body), 'chars')
