// The REAL PushT scene, vendored verbatim from
// curriculum/common/envs/pusht/pusht.xml (the authoritative reference shared by
// pusht_env.py, the grader, and chapters 0.4/1.1/1.2). Kept byte-for-byte in
// sync with that file — it defines the obs/action semantics the browser must
// match. If the curriculum MJCF changes, re-vendor this string in the same PR.
//
// Self-contained MJCF: the only asset is a builtin checker texture, so it loads
// into MuJoCo-WASM (MjModel.from_xml_string) with no external files.
export const PUSHT_XML = `
<mujoco model="pusht">
  <option timestep="0.01" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global offwidth="640" offheight="640"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.94" rgb2="0.83 0.83 0.87"
             width="256" height="256"/>
    <material name="table_mat" texture="grid" texrepeat="8 8" reflectance="0"/>
  </asset>

  <worldbody>
    <camera name="top" pos="0 0 1.0" quat="1 0 0 0" fovy="50"/>

    <geom name="table" type="plane" size="0.45 0.45 0.1" material="table_mat"
          contype="0" conaffinity="0"/>

    <!-- workspace walls -->
    <geom name="wall_n" type="box" pos="0  0.41 0.03" size="0.43 0.02 0.03" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall_s" type="box" pos="0 -0.41 0.03" size="0.43 0.02 0.03" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall_e" type="box" pos=" 0.41 0 0.03" size="0.02 0.43 0.03" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall_w" type="box" pos="-0.41 0 0.03" size="0.02 0.43 0.03" rgba="0.6 0.6 0.6 1"/>

    <!-- fixed goal pose, visual only -->
    <body name="target" pos="0 0 0.0005">
      <site name="target_site" pos="0 0 0" size="0.005" rgba="0 0 0 0"/>
      <geom name="target_bar"  type="box" size="0.06 0.015 0.0005" pos="0  0.00 0"
            rgba="0.35 0.8 0.4 0.5" contype="0" conaffinity="0"/>
      <geom name="target_stem" type="box" size="0.015 0.045 0.0005" pos="0 -0.06 0"
            rgba="0.35 0.8 0.4 0.5" contype="0" conaffinity="0"/>
    </body>

    <!-- T-shaped block: bar 0.12 x 0.03, stem 0.03 x 0.09, welded -->
    <body name="tee" pos="0 0 0.0152">
      <joint name="tee_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0  0.00 0"
            rgba="0.45 0.5 0.95 1" mass="0.06"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 -0.06 0"
            rgba="0.45 0.5 0.95 1" mass="0.045"/>
    </body>

    <!-- cylindrical pusher on two actuated slides -->
    <body name="pusher" pos="0 0 0.02">
      <joint name="pusher_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" damping="0.5"/>
      <geom name="pusher_tip" type="cylinder" size="0.015 0.02"
            rgba="0.9 0.4 0.35 1" mass="0.2"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="pusher_vx" joint="pusher_x" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
    <velocity name="pusher_vy" joint="pusher_y" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
  </actuator>
</mujoco>
`;

// The REAL cartpole scene: the <mujoco> MODEL BODY vendored verbatim (the source
// file's leading comment header is omitted; the model itself is byte-identical) from
// curriculum/common/envs/cartpole/cartpole.xml — the authoritative reference
// shared by cartpole_env.py and ch2.1's PPO. Kept in exact sync with that file:
// it defines the joints (slider, hinge) and the single force actuator the
// browser env drives, so the WASM dynamics match the Python training env. If
// the curriculum MJCF changes, re-vendor this string in the same PR.
//
// Self-contained MJCF: the only asset is a builtin checker texture, so it loads
// into MuJoCo-WASM (MjModel.from_xml_string) with no external files.
export const CARTPOLE_XML = `
<mujoco model="cartpole">
  <option timestep="0.01" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.94" rgb2="0.83 0.83 0.87"
             width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="10 4" reflectance="0"/>
  </asset>

  <worldbody>
    <!-- side camera looking along +y at the x-z plane the cartpole lives in -->
    <camera name="side" pos="0 -4.0 0.6" xyaxes="1 0 0 0 0 1"/>

    <!-- visual-only floor and rail (nothing collides) -->
    <geom name="floor" type="plane" pos="0 0 -0.6" size="3.5 1.0 0.1" material="floor_mat"
          contype="0" conaffinity="0"/>
    <geom name="rail" type="capsule" fromto="-2.9 0 0 2.9 0 0" size="0.02"
          rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>

    <!-- cart on the rail: one slide joint, driven by the motor below -->
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.9 2.9"/>
      <geom name="cart_geom" type="box" size="0.1 0.05 0.05" rgba="0.30 0.35 0.60 1"
            mass="1.0" contype="0" conaffinity="0"/>

      <!-- pole hinged at the cart center; qpos 0 = straight up (+z) -->
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 1.0" size="0.02"
              rgba="0.80 0.35 0.30 1" mass="0.1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- single force actuator on the cart: ctrl in [-1, 1] => +-10 N -->
    <motor name="slide_force" joint="slider" gear="10" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
`;
