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

// The REAL pusher-reach scene: the <mujoco> MODEL BODY vendored verbatim (the
// source file's leading comment header is omitted; the model itself is
// byte-identical) from curriculum/common/envs/pusher_reach/pusher_reach.xml —
// the authoritative reference shared by pusher_reach_env.py and ch2.2's SAC (and
// ch4's offline/serl chapters). Kept in exact sync with that file: it defines the
// two hinge joints (shoulder, elbow) and the two torque actuators the browser env
// drives, so the WASM dynamics match the Python training env. If the curriculum
// MJCF changes, re-vendor this string in the same PR.
//
// A planar 2-link arm whose `fingertip` site must reach a seeded target. The
// target is a MOCAP body (no dynamics, no collision), positioned each reset via
// data.mocap_pos — so the browser env manages the target purely in JS (obs is
// built from the fingertip and the env's stored target; see pusher_reach_obs.ts).
//
// Self-contained MJCF: the only asset is a builtin checker texture, so it loads
// into MuJoCo-WASM (MjModel.from_xml_string) with no external files.
export const PUSHER_REACH_XML = `
<mujoco model="pusher_reach">
  <option timestep="0.01" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global offwidth="640" offheight="640"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.94" rgb2="0.83 0.83 0.87"
             width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0"/>
  </asset>

  <default>
    <!-- shared link look; no collisions (contype/conaffinity 0) -->
    <geom type="capsule" size="0.01" rgba="0.30 0.35 0.60 1" contype="0" conaffinity="0"/>
    <joint type="hinge" axis="0 0 1" damping="0.1" armature="0.01"/>
  </default>

  <worldbody>
    <!-- top-down camera looking along -z at the x-y plane the arm lives in -->
    <camera name="top" pos="0 0 0.5" quat="1 0 0 0" fovy="50"/>

    <!-- visual-only floor (nothing collides) -->
    <geom name="floor" type="plane" pos="0 0 -0.02" size="0.3 0.3 0.1" material="floor_mat"
          contype="0" conaffinity="0"/>

    <!-- fixed shoulder anchor at the origin (visual hub) -->
    <geom name="anchor" type="cylinder" size="0.012 0.005" pos="0 0 0"
          rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>

    <!-- link 1: shoulder hinge at origin, capsule out to (0.1, 0, 0) -->
    <body name="link1" pos="0 0 0">
      <joint name="shoulder"/>
      <geom name="link1_geom" fromto="0 0 0 0.1 0 0"/>

      <!-- link 2: elbow hinge at the end of link 1 -->
      <body name="link2" pos="0.1 0 0">
        <joint name="elbow"/>
        <geom name="link2_geom" fromto="0 0 0 0.1 0 0" rgba="0.55 0.35 0.30 1"/>
        <!-- end-effector marker at the far tip of link 2 -->
        <site name="fingertip" pos="0.1 0 0" size="0.012" rgba="0.9 0.4 0.35 1"/>
      </body>
    </body>

    <!-- seeded random target: a mocap body, positioned each reset via mocap_pos -->
    <body name="target" mocap="true" pos="0.15 0 0">
      <geom name="target_geom" type="sphere" size="0.012" rgba="0.35 0.8 0.4 0.6"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <!-- one torque motor per joint: ctrl in [-1, 1] => +-0.5 N*m -->
    <motor name="shoulder_torque" joint="shoulder" gear="0.5" ctrlrange="-1 1"/>
    <motor name="elbow_torque"    joint="elbow"    gear="0.5" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
`;

// The REAL quadruped scene: the <mujoco> MODEL BODY vendored verbatim (the source
// file's leading comment header is omitted; the model itself is byte-identical)
// from curriculum/common/envs/quadruped/quadruped.xml — the authoritative reference
// shared by quadruped_env.py and ch2.4/2.5/2.7. Kept in exact sync with that file:
// it defines the free-joint floating base (`root`), the 8 leg hinges, the 8 PD
// position servos, and the four collidable feet — so the WASM dynamics match the
// Python training env. If the curriculum MJCF changes, re-vendor this string in the
// same PR.
//
// A minimal from-scratch quadruped: a box torso on a 6-DOF free joint carrying four
// two-joint legs (hip + knee). obs[23] = 8 joint angles, 8 joint vels, torso height,
// torso up-vector (from the torso body's world rotation matrix), torso linear
// velocity (see quadruped_obs.ts). Only the four foot↔floor pairs collide.
//
// Self-contained MJCF: the only asset is a builtin checker texture, so it loads into
// MuJoCo-WASM (MjModel.from_xml_string) with no external files.
export const QUADRUPED_XML = `
<mujoco model="quadruped">
  <option timestep="0.005" integrator="implicitfast" solver="Newton"
          iterations="50" ls_iterations="20" cone="pyramidal"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.94" rgb2="0.83 0.83 0.87"
             width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="12 12" reflectance="0"/>
  </asset>

  <default>
    <default class="leg">
      <geom type="capsule" size="0.018" rgba="0.30 0.35 0.60 1" contype="0" conaffinity="0"
            mass="0.25"/>
      <joint type="hinge" axis="0 1 0" damping="0.5" armature="0.01" frictionloss="0.02"/>
    </default>
    <default class="foot">
      <geom type="sphere" size="0.022" rgba="0.80 0.35 0.30 1" contype="1" conaffinity="1"
            mass="0.05" friction="1.0 0.02 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
  </default>

  <worldbody>
    <camera name="side" pos="0 -1.6 0.5" xyaxes="1 0 0 0 0.4 1"/>
    <light name="top" pos="0 0 2" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>

    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"
          contype="1" conaffinity="1" friction="1.0 0.02 0.001"/>

    <body name="torso" pos="0 0 0.30">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.18 0.09 0.035" rgba="0.20 0.24 0.42 1"
            mass="3.0" contype="0" conaffinity="0"/>

      <body name="FL_thigh" pos="0.15 0.08 -0.02">
        <joint name="FL_hip" class="leg"/>
        <geom name="FL_thigh_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
        <body name="FL_shin" pos="0 0 -0.13">
          <joint name="FL_knee" class="leg"/>
          <geom name="FL_shin_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
          <geom name="FL_foot" class="foot" pos="0 0 -0.13"/>
        </body>
      </body>

      <body name="FR_thigh" pos="0.15 -0.08 -0.02">
        <joint name="FR_hip" class="leg"/>
        <geom name="FR_thigh_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
        <body name="FR_shin" pos="0 0 -0.13">
          <joint name="FR_knee" class="leg"/>
          <geom name="FR_shin_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
          <geom name="FR_foot" class="foot" pos="0 0 -0.13"/>
        </body>
      </body>

      <body name="HL_thigh" pos="-0.15 0.08 -0.02">
        <joint name="HL_hip" class="leg"/>
        <geom name="HL_thigh_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
        <body name="HL_shin" pos="0 0 -0.13">
          <joint name="HL_knee" class="leg"/>
          <geom name="HL_shin_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
          <geom name="HL_foot" class="foot" pos="0 0 -0.13"/>
        </body>
      </body>

      <body name="HR_thigh" pos="-0.15 -0.08 -0.02">
        <joint name="HR_hip" class="leg"/>
        <geom name="HR_thigh_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
        <body name="HR_shin" pos="0 0 -0.13">
          <joint name="HR_knee" class="leg"/>
          <geom name="HR_shin_geom" class="leg" fromto="0 0 0 0 0 -0.13"/>
          <geom name="HR_foot" class="foot" pos="0 0 -0.13"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="FL_hip_act"  joint="FL_hip"  kp="20" kv="0.5" ctrlrange="-1.2 1.2" forcerange="-12 12"/>
    <position name="FL_knee_act" joint="FL_knee" kp="20" kv="0.5" ctrlrange="-2.4 0.2" forcerange="-12 12"/>
    <position name="FR_hip_act"  joint="FR_hip"  kp="20" kv="0.5" ctrlrange="-1.2 1.2" forcerange="-12 12"/>
    <position name="FR_knee_act" joint="FR_knee" kp="20" kv="0.5" ctrlrange="-2.4 0.2" forcerange="-12 12"/>
    <position name="HL_hip_act"  joint="HL_hip"  kp="20" kv="0.5" ctrlrange="-1.2 1.2" forcerange="-12 12"/>
    <position name="HL_knee_act" joint="HL_knee" kp="20" kv="0.5" ctrlrange="-2.4 0.2" forcerange="-12 12"/>
    <position name="HR_hip_act"  joint="HR_hip"  kp="20" kv="0.5" ctrlrange="-1.2 1.2" forcerange="-12 12"/>
    <position name="HR_knee_act" joint="HR_knee" kp="20" kv="0.5" ctrlrange="-2.4 0.2" forcerange="-12 12"/>
  </actuator>
</mujoco>
`;
