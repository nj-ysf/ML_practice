"""Exécuter les cellules Python en ordre et enregistrer sorties et figures.

Petit exécuteur pour ce notebook sans magies, sans dépendance nbclient/nbformat.
Les fonctions display/show capturent les sorties dans le format Jupyter standard.
"""
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path
import traceback

os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
import IPython.display

sys.stdout.reconfigure(encoding='utf-8')

root = Path(__file__).resolve().parents[1]
os.chdir(root)
path = next(root.glob('*polynomiale*.ipynb'))
notebook = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
shell = InteractiveShell.instance()
outputs = []
buffer = io.StringIO()


def flush():
    content = buffer.getvalue()
    if content:
        outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': content})
        buffer.seek(0)
        buffer.truncate(0)


def display(*objects, **kwargs):
    flush()
    for obj in objects:
        data, metadata = shell.display_formatter.format(obj)
        outputs.append({'output_type': 'display_data', 'data': data, 'metadata': metadata})


def show(*args, **kwargs):
    flush()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format='png', dpi=115, bbox_inches='tight')
        outputs.append({'output_type': 'display_data', 'data': {
            'image/png': base64.b64encode(image_buffer.getvalue()).decode('ascii'),
            'text/plain': repr(fig)}, 'metadata': {}})
    plt.close('all')


IPython.display.display = display
plt.show = show
counter = 0
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    counter += 1
    outputs = []
    buffer = io.StringIO()
    cell['execution_count'] = counter
    print(f'Cellule {index}, exécution {counter}', flush=True)
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(''.join(cell['source']), f'{path.name}:cell-{index}', 'exec'), namespace)
        flush()
    except Exception as exc:
        flush()
        outputs.append({'output_type': 'error', 'ename': type(exc).__name__, 'evalue': str(exc),
                        'traceback': traceback.format_exc().splitlines()})
        cell['outputs'] = outputs
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
        raise
    cell['outputs'] = outputs
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')

print(namespace['comparison'].to_string())
print('Degré :', namespace['selected_degree'], 'Recommandation :', namespace['recommended'])
print('Toutes les cellules exécutées avec succès.')
