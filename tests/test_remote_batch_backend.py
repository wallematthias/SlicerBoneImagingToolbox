from __future__ import annotations

import json
from pathlib import Path

import pytest

from SlicerBoneImagingToolboxLib.remote_batch import (
    RemoteBatchConfig,
    SshSlurmBatchBackend,
    load_remote_batch_config,
)


def test_batch_backend_registry_exposes_local_backend() -> None:
    from SlicerBoneImagingToolboxLib.batch_backends import available_batch_backends, get_batch_backend

    assert "local" in available_batch_backends()
    assert get_batch_backend("local").label == "Local"


def test_batch_backend_registry_accepts_private_backend() -> None:
    from SlicerBoneImagingToolboxLib.batch_backends import available_batch_backends, get_batch_backend, register_batch_backend

    class PrivateBackend:
        key = "arc-slurm"
        label = "ARC / SLURM"

    register_batch_backend(PrivateBackend())

    assert available_batch_backends()["arc-slurm"].label == "ARC / SLURM"
    assert get_batch_backend("arc-slurm").key == "arc-slurm"


def test_batch_backend_registry_knows_private_toolbox_module_name() -> None:
    source = (Path(__file__).resolve().parents[1] / "SlicerBoneImagingToolboxLib" / "batch_backends.py").read_text(
        encoding="utf-8"
    )

    assert '"SlicerBoneImagingToolboxPrivateLib.batch_backends"' in source


def test_batch_backend_registry_reinvokes_private_registration_hook() -> None:
    source = (Path(__file__).resolve().parents[1] / "SlicerBoneImagingToolboxLib" / "batch_backends.py").read_text(
        encoding="utf-8"
    )

    assert 'getattr(module, "register_private_batch_backends", None)' in source
    assert "registration()" in source


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


def test_remote_batch_config_maps_selected_mac_tmp_dataset_root_even_when_configured_root_differs() -> None:
    config = RemoteBatchConfig(
        name="arc",
        host="arc.ucalgary.ca",
        remote_root="/home/mwalle/bone-batch-smoke",
        local_root="/tmp/bone-batch-smoke-local",
        python="/home/mwalle/miniforge3/envs/bone/bin/python",
        work_dir="/home/mwalle/bone-batch",
    )

    args = config.remote_args(
        [
            "-m",
            "parosol_py.cli",
            "/private/tmp/bone-batch-smoke-local/sub-001/ses-001/xct/input.AIM",
            "--dataset-root",
            "/private/tmp/bone-batch-smoke-local",
        ],
        dataset_root="/private/tmp/bone-batch-smoke-local",
    )

    assert args == [
        "-m",
        "parosol_py.cli",
        "/home/mwalle/bone-batch-smoke/sub-001/ses-001/xct/input.AIM",
        "--dataset-root",
        "/home/mwalle/bone-batch-smoke",
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


def test_slurm_backend_builds_async_sbatch_submission_without_partition() -> None:
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

    assert argv[0] == "ssh"
    assert "arc.ucalgary.ca" in argv
    assert "ConnectTimeout=8" in argv
    script = argv[-1]
    assert "sbatch --parsable" in script
    assert "--wait" not in script
    assert "--partition" not in script
    assert "/env/bin/python -m bone_microarchitecture.cli run-batch /remote/data" in script
    assert "cat /remote/work/jobs/micro_001/slurm.log" not in script


def test_slurm_backend_parses_submitted_job_id() -> None:
    assert SshSlurmBatchBackend.parse_job_id("47742177\n") == "47742177"
    assert SshSlurmBatchBackend.parse_job_id("47742177;cluster\n") == "47742177"
    assert SshSlurmBatchBackend.parse_job_id("Submitted batch job 47742177\n") == "47742177"


def test_slurm_backend_builds_status_cancel_and_log_commands() -> None:
    backend = SshSlurmBatchBackend(
        RemoteBatchConfig(
            name="arc",
            host="arc.ucalgary.ca",
            remote_root="/remote/data",
            python="/env/bin/python",
            work_dir="/remote/work",
        )
    )

    status = backend.status_argv("47742177")
    statuses = backend.statuses_argv(["47742177", "47742178"])
    cancel = backend.cancel_argv("47742177")
    log = backend.log_argv("bone-job")
    shell = backend.shell_argv("echo ok")

    assert status[0] == "ssh"
    assert "-o" in status
    assert "BatchMode=yes" in status
    assert "ConnectTimeout=8" in status
    assert "ConnectionAttempts=1" in status
    assert "arc.ucalgary.ca" in status
    assert "squeue" in status[-1]
    assert "sacct" in status[-1]
    assert statuses[0] == "ssh"
    assert "arc.ucalgary.ca" in statuses
    assert "for job in 47742177 47742178" in statuses[-1]
    assert 'printf "%s|%s|' in statuses[-1]
    assert cancel[0] == "ssh"
    assert "arc.ucalgary.ca" in cancel
    assert cancel[-1] == "scancel 47742177"
    assert "tail -n 80 /remote/work/jobs/bone-job/slurm.log" in log[-1]
    assert shell[0] == "ssh"
    assert "ConnectTimeout=8" in shell
    assert shell[-1] == "bash -lc 'echo ok'"


def test_slurm_backend_uses_fast_ssh_probe_and_configurable_connect_timeout() -> None:
    backend = SshSlurmBatchBackend(
        RemoteBatchConfig(
            name="arc",
            host="arc.ucalgary.ca",
            remote_root="/remote/data",
            python="/env/bin/python",
            work_dir="/remote/work",
            connect_timeout_seconds=3,
        )
    )

    argv = backend.availability_argv()

    assert argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
        "-o",
        "ConnectionAttempts=1",
        "arc.ucalgary.ca",
        "true",
    ]


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
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1",
        "-az",
        "arc.ucalgary.ca:/remote/data/derivatives/Microarchitecture/",
        "/local/data/derivatives/Microarchitecture/",
    ]
