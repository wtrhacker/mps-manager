"""Build and bundle this repository's native manager; wheels need no Rust tools."""
import os
from pathlib import Path
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class BuildClient(build_py):
    def run(self):
        super().run()
        configured = os.environ.get("MPS_MANAGER_BINARY")
        if configured:
            binary = Path(configured).resolve()
        else:
            repository = Path(__file__).resolve().parent
            environment = dict(os.environ)
            target = Path(environment.get("CARGO_TARGET_DIR", repository / ".mps-manager" / "build" / "target"))
            environment["CARGO_TARGET_DIR"] = str(target)
            subprocess.run(["cargo", "build", "--locked", "--release", "--manifest-path", "native/Cargo.toml"],
                           cwd=repository, env=environment, check=True)
            binary = target / "release" / "mps-manager-native"
        if not binary.is_file():
            raise RuntimeError(f"native manager is missing: {binary}")
        destination = Path(self.build_lib) / "mps_runner" / "_bin"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, destination / "mps-manager-native")


class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"build_py": BuildClient, "bdist_wheel": NativeWheel})
