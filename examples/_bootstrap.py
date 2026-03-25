from pathlib import Path
import sys


def setup_example_path() -> None:
    """把项目根目录和 src 目录加入导入路径，保证 examples 下脚本可直接运行。"""
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    src_dir = root_dir / "src"

    for path in (str(current_dir), str(src_dir), str(root_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
