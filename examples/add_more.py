"""A second independent Python program can find the same queue by human title."""
from mps_runner import connect_mps_manager

manager = connect_mps_manager(title="Polymer stiffness queue")
manager.submit("another_simulation.py", args=["--seed", "7"], title="Extra experiment")
