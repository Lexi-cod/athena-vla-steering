import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_NAME = "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate"

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

env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": 128, "camera_widths": 128}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()

print("\n=== obs dict keys ===")
for k in obs.keys():
    print(" ", k)

print("\n=== obj_body_id map ===")
for name, bid in env.env.obj_body_id.items():
    print(f"  {name!r} -> body_id {bid}")

print("\n=== root_body strings ===")
for name in ["red_coffee_mug_1", "porcelain_mug_1", "plate_1"]:
    obj = env.env.objects_dict[name]
    print(f"  {name}.root_body = {obj.root_body!r}")

print("\n=== queried 3D positions ===")
for name in ["red_coffee_mug_1", "porcelain_mug_1", "plate_1"]:
    pos_via_obj_body_id = env.env.sim.data.body_xpos[env.env.obj_body_id[name]]
    root_body_name = env.env.objects_dict[name].root_body
    pos_via_get_body_xpos = env.env.sim.data.get_body_xpos(root_body_name)
    print(f"  {name}:")
    print(f"    sim.data.body_xpos[obj_body_id[name]]      = {pos_via_obj_body_id}")
    print(f"    sim.data.get_body_xpos(root_body_name)     = {pos_via_get_body_xpos}")

print("\n=== stepping with random actions to confirm obs updates ===")
import numpy as np
rng = np.random.default_rng(0)
for step in range(5):
    action = rng.uniform(-1, 1, size=7)
    obs, reward, done, info = env.step(action)
    red_pos = obs["red_coffee_mug_1_pos"]
    porc_pos = obs["porcelain_mug_1_pos"]
    eef_pos = obs["robot0_eef_pos"]
    print(f"  step {step}: eef_pos={eef_pos}  red_mug_pos={red_pos}  porcelain_mug_pos={porc_pos}")
    manual_red_to_eef = red_pos - eef_pos
    print(f"           manual (red_pos - eef_pos)      = {manual_red_to_eef}")
    print(f"           obs['red_coffee_mug_1_to_robot0_eef_pos'] (gripper-frame) = {obs['red_coffee_mug_1_to_robot0_eef_pos']}")

env.close()
