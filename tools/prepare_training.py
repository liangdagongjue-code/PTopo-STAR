"""Create the directory layout expected by the unchanged training scripts."""
from pathlib import Path
import argparse, shutil
from restore_experimental_data import get_source

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,required=True)
parser.add_argument('--source-zip',type=Path)
args=parser.parse_args()
package=Path(__file__).resolve().parents[1];run=args.run_root.resolve()
if run==package or run.is_relative_to(package):parser.error('Use a run root outside the public package.')
code=run/'experiments/ptopo_star_full'
if code.exists():parser.error('Training code directory already exists; choose a new run root.')
get_source(run,args.source_zip)
shutil.copytree(package/'supplementary_software',code)
print('Training layout created at '+str(run)+'. No training was started.')
