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


def test_cli_agent_demo_prints_context_answers_and_evidence() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenegraph_navbench.cli",
            "samples/room_scene.json",
            "--navigation",
            "samples/room_scene_navigation.json",
            "--show-agent-demo",
            "--relation-limit",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Robot context" in result.stdout
    assert "frame: map" in result.stdout
    assert "Questions" in result.stdout
    assert "Deterministic answers" in result.stdout
    assert "closest_object: ok" in result.stdout
    assert "exit_target: ok" in result.stdout
    assert "objects_near_exit: ok" in result.stdout
    assert "blockers_for_exit: ok" in result.stdout
    assert "object_to_avoid: ok" in result.stdout
    assert "Evidence traces" in result.stdout
    assert "distance=" in result.stdout
    assert "frame_id=map" in result.stdout


def test_cli_agent_demo_requires_navigation_context() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenegraph_navbench.cli",
            "samples/room_scene.json",
            "--show-agent-demo",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--show-agent-demo requires --navigation" in result.stderr
