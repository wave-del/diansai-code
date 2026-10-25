"""All deployed geometry consumers must share solver2's types and implementation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'vision/src'),str(ROOT/'handover-puzzle')]


def test_task_and_adapter_contracts_use_the_same_solver2_types():
    import solver2
    from ec_26_e.config import task1 as config1, task2 as config2
    from ec_26_e.tasks.task1 import contracts as task1
    from ec_26_e.tasks.task2 import contracts as task2
    import random_white_solver_adapter as white
    import random_patterned_solver_adapter as patterned
    assert white.PuzzlePiece is solver2.PuzzlePiece
    assert white.PlanningOptions is solver2.PlanningOptions
    assert patterned.PuzzlePiece is solver2.PuzzlePiece
    assert task1.PlanningOptions is solver2.PlanningOptions
    assert task2.PlanningOptions is solver2.PlanningOptions
    assert config1.PlanningOptions is solver2.PlanningOptions
    assert config2.PlanningOptions is solver2.PlanningOptions


def test_old_control_package_is_removed():
    assert not (ROOT/'vision/src/ec_26_e/control').exists()
    assert importlib.util.find_spec('ec_26_e.control') is None


def test_source_task_entry_resolves_solver2_from_vision_working_directory():
    env=dict(os.environ,PYTHONPATH=str(ROOT/'vision/src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    code="""from pathlib import Path
import ec_26_e
import solver2
from ec_26_e.tasks.task1.runner import PuzzlePiece
assert PuzzlePiece is solver2.PuzzlePiece
assert Path(solver2.__file__).resolve().parent == Path.cwd().parent/'solver2'
"""
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT/'vision',env=env,check=True)


def test_web_worker_and_adapter_use_solver2_in_a_fresh_process():
    worker=ROOT.parent/'npu_q6a_test/web_preview/app/q6a_web_solver.py'
    code="""import importlib.util,sys
spec=importlib.util.spec_from_file_location('web_worker',sys.argv[1])
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
import solver2
import random_white_solver_adapter as white
assert white.PuzzlePiece is solver2.PuzzlePiece
assert white.solve_observations.__module__.startswith('solver2.')
assert solver2.__file__.startswith(str(worker.ROOT/'solver2'))
assert not any(k=='ec_26_e.control' or k.startswith('ec_26_e.control.') for k in sys.modules)
"""
    subprocess.run([sys.executable,'-B','-c',code,str(worker)],cwd='/tmp',check=True)


def test_handover_archive_keeps_solver2_beside_vision(tmp_path):
    tools=tmp_path/'vision/tools';tools.mkdir(parents=True)
    shutil.copy2(ROOT/'vision/tools/create_handover_archive.sh',tools/'create_handover_archive.sh')
    (tmp_path/'vision/README.md').write_text('vision')
    solver=tmp_path/'solver2';solver.mkdir();(solver/'__init__.py').write_text('')
    (solver/'outputs').mkdir();(solver/'outputs/stale.json').write_text('{}')
    archive=tmp_path/'handover.zip'
    subprocess.run(['bash',str(tools/'create_handover_archive.sh'),str(archive)],check=True)
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        assert 'vision/README.md' in names and 'solver2/__init__.py' in names
        assert not any('outputs/' in n for n in names)
        assert bundle.testzip() is None
