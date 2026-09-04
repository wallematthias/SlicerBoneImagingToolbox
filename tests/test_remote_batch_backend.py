from __future__ import annotations

import json
from pathlib import Path

import pytest

from SlicerBoneImagingToolboxLib.remote_batch import (
    RemoteBatchConfig,
    SshSlurmBatchBackend,
    load_remote_batch_config,
)


def test_remote_batch_config_loads_private_json_and_maps_dataset_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "arc.json"
    local_root = tmp_path / "local"
    config_path.write_text(
        json.dumps(
            {
                "name": "arc",
                "host": "arc.ucalgary.ca",
                "remote_root": "/arc/project/sample",
                "local_root": str(local_root),
                "python": "/home/mwalle/miniforge3/envs/bone/bin/python",
                "work_dir": "/home/mwalle/bone-batch",
            }
        ),
        encoding="utf-8",
    )

    config = load_remote_batch_config(config_path)

    assert config.name == "arc"
    assert config.host == "arc.ucalgary.ca"
    assert config.remote_dataset_root(local_root) == "/arc/project/sample"
    assert config.remote_args(["-m", "tool.cli", "run-batch", str(local_root), "--subject", "001"]) == [
        "-m",
        "tool.cli",
        "run-batch",
        "/arc/project/sample",
        "--subject",
        "001",
    ]


def test_remote_batch_config_requires_explicit_private_config(monkeypatch) -> None:
    monkeypatch.delenv("SLICER_BONE_BATCH_REMOTE_CONFIG", raising=False)

    with pytest.raises(ValueError, match="SLICER_BONE_BATCH_REMOTE_CONFIG"):
        load_remote_batch_config()


def test_remote_batch_config_parses_sbatch_options_with_colons(tmp_path: Path) -> None:
    config_path = tmp_path / "arc.yml"
    config_path.write_text(
        "\n".join(
            [
                "name: arc",
                "host: arc.ucalgary.ca",
                "remote_root: /arc/project/sample",
                "python: /home/mwalle/miniforge3/envs/bone/bin/python",
                "work_dir: /home/mwalle/bone-batch",
                "sbatch_options:",
                "  - --time=01:00:00",
            ]
        ),
        encoding="utf-8",
    )

    config = load_remote_batch_config(config_path)

    assert config.sbatch_options == ("--time=01:00:00",)


def test_slurm_backend_builds_ssh_sbatch_wait_submission_without_partition() -> None:
    backend = SshSlurmBatchBackend(
        RemoteBatchConfig(
            name="arc",
            host="arc.ucalgary.ca",
            remote_root="/remote/data",
            python="/env/bin/python",
            work_dir="/remote/work",
        )
    )

    argv = backend.submit_argv(["-m", "bone_microarchitecture.cli", "run-batch", "/remote/data"], job_name="micro_001")

    assert argv[:2] == ["ssh", "arc.ucalgary.ca"]
    script = argv[2]
    assert "sbatch --wait" in script
    assert "--partition" not in script
    assert "/env/bin/python -m bone_microarchitecture.cli run-batch /remote/data" in script
    assert "cat " in script


def test_slurm_backend_syncs_output_family_to_local_dataset_root() -> None:
    backend = SshSlurmBatchBackend(
        RemoteBatchConfig(
            name="arc",
            host="arc.ucalgary.ca",
            remote_root="/remote/data",
            local_root="/local/data",
            python="/env/bin/python",
            work_dir="/remote/work",
        )
    )

    argv = backend.sync_output_argv("Microarchitecture")

    assert argv == [
        "rsync",
        "-az",
        "arc.ucalgary.ca:/remote/data/derivatives/Microarchitecture/",
        "/local/data/derivatives/Microarchitecture/",
    ]
