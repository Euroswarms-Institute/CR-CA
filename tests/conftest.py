import sys
from pathlib import Path


# Ensure repository root is importable when running pytest from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
root_s = str(REPO_ROOT)
# Prepend repo root even if already present (e.g. via .pth); otherwise another
# entry can win first and bind a non-package `aixi` before subpackages resolve.
while root_s in sys.path:
    sys.path.remove(root_s)
sys.path.insert(0, root_s)

