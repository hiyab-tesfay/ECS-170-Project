"""Package code, measured results, report, and trained weights for manual grading."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / 'report/stage_4/Stage_4_Submission.zip'


def main():
    checks = json.loads((ROOT / 'result/stage_4_result/submission_checks.json').read_text())
    if not checks.get('passed'):
        raise RuntimeError('Run check_submission.py successfully before packaging')
    files = [ROOT / n for n in ['.gitignore', '.gitattributes', 'README.md', 'requirements.txt', 'requirements-stage4.txt', 'local_code/__init__.py']]
    for folder in ['local_code/stage_4_code', 'script/stage_4_script', 'report/stage_4', 'result/stage_4_result']:
        for path in (ROOT / folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or 'cache' in path.relative_to(ROOT).parts:
                continue
            if path == DESTINATION or path.name.endswith(('.zip', '.log', '.tmp')):
                continue
            # Keep the selected classifier and both generator checkpoints.
            if path.name == 'last.pt' and 'classification' in path.parts:
                continue
            files.append(path)
    files.append(ROOT / 'tests/test_stage4.py')
    manifest = []
    with zipfile.ZipFile(DESTINATION, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(set(files)):
            data = path.read_bytes()
            relative = str(path.relative_to(ROOT))
            manifest.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            z.writestr(relative, data)
        z.writestr('SUBMISSION_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(DESTINATION) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f'Corrupt archive entry {bad}')
    print(f'{DESTINATION}: {DESTINATION.stat().st_size / 1e6:.1f} MB, {len(manifest)} files')


if __name__ == '__main__':
    main()
