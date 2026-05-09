"""Push notebooks to Kaggle sequentially, waiting for each to complete before pushing the next."""
import json, os, tempfile, shutil, time, sys

with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    creds = json.load(f)
os.environ['KAGGLE_API_TOKEN'] = creds['key']

import kaggle
api = kaggle.api

notebooks = [
    ('01_Data_Ingestion', False, []),
    ('02_Feature_Engineering', False, ['bishalsarkarr/forex-ml-01-data-ingestion']),
    ('03_Model_Training', True, ['bishalsarkarr/forex-ml-02-feature-engineering']),
    ('04_Backtesting_Evaluation', False, ['bishalsarkarr/forex-ml-02-feature-engineering', 'bishalsarkarr/forex-ml-03-model-training']),
    ('05_Daily_Signal', False, ['bishalsarkarr/forex-ml-03-model-training']),
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def push_and_wait(file_name, gpu, sources):
    slug = file_name.replace('_', '-').lower()
    kernel_id = f'bishalsarkarr/forex-ml-{slug}'
    title = f'Forex ML - {file_name.replace("_", " ")}'
    nb_path = os.path.join(REPO_ROOT, 'ml', 'notebooks', f'{file_name}.ipynb')

    tmpdir = tempfile.mkdtemp()
    meta = {
        'id': kernel_id,
        'title': title,
        'code_file': f'{file_name}.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': gpu,
        'enable_internet': True,
        'dataset_sources': [],
        'kernel_sources': sources,
    }
    with open(os.path.join(tmpdir, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    shutil.copy2(nb_path, os.path.join(tmpdir, f'{file_name}.ipynb'))

    print(f'\n{"="*60}')
    print(f'Pushing {file_name}...')
    api.kernels_push(tmpdir)
    print(f'Push done. Waiting for completion...')
    shutil.rmtree(tmpdir)

    # Poll for completion
    max_wait = 1800  # 30 minutes max (N3 GPU can take a while)
    poll_interval = 20
    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        try:
            status = api.kernels_status(kernel_id)
            s = str(status.status).replace('KernelWorkerStatus.', '')
            print(f'  [{waited}s] Status: {s}')
            if s == 'COMPLETE':
                print(f'  ✅ {file_name} completed!')
                return True
            if s in ('ERROR', 'FAILED'):
                print(f'  ❌ {file_name} failed!')
                # Try to get logs
                try:
                    logs = api.kernels_logs(kernel_id)
                    text = logs.text if hasattr(logs, 'text') else str(logs)
                    # Extract first error
                    for line in text.split('\n'):
                        if 'Traceback' in line or 'Error' in line:
                            print(f'     {line[:200]}')
                except:
                    pass
                return False
        except Exception as e:
            print(f'  [{waited}s] Status check failed: {e}')

    print(f'  ⏰ {file_name} timed out after {max_wait}s')
    return False

start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Push notebooks starting from start_from index
for file_name, gpu, sources in notebooks[start_from:]:
    if not push_and_wait(file_name, gpu, sources):
        print(f'\n❌ {file_name} failed. Stopping.')
        sys.exit(1)

print('\n✅ All notebooks completed successfully!')
