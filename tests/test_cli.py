import subprocess
import sys


def test_cli_benchmark_mode_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenegraph_navbench.cli",
            "--benchmark",
            "samples/benchmark_manifest.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "SceneGraphNavBench Benchmark" in result.stdout
    assert "Scenes: 5" in result.stdout
    assert "must_include required relation recall: 1.000" in result.stdout
