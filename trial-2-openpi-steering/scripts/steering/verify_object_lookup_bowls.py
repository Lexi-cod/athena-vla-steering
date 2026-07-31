import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_NAME = "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate"

bm_dict = benchmark.get_benchmark_dict()
bm = bm_dict["libero_90"]()

task_id = None
for i in range(bm.get_num_tasks()):
    if bm.get_task(i).name == TASK_NAME:
        task_id = i
        break
assert task_id is not None, f"task {TASK_NAME} not found in libero_90"

task = bm.get_task(task_id)
task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
print("bddl file:", task_bddl_file)
print("language:", task.language)

env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": 128, "camera_widths": 128}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()

print("\n=== obs dict keys (objects only) ===")
for k in obs.keys():
    if "bowl" in k or "plate" in k or "eef" in k:
        print(" ", k)

print("\n=== obj_body_id map ===")
for name, bid in env.env.obj_body_id.items():
    print(f"  {name!r} -> body_id {bid}")

names = ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3", "plate_1"]
print("\n=== root_body strings ===")
for name in names:
    obj = env.env.objects_dict[name]
    print(f"  {name}.root_body = {obj.root_body!r}")

print("\n=== queried 3D positions (settle + a few steps) ===")
import numpy as np

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
for step in range(10):
    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

for name in names:
    print(f"  {name}_pos = {obs[f'{name}_pos']}")
print(f"  robot0_eef_pos = {obs['robot0_eef_pos']}")

env.close()
