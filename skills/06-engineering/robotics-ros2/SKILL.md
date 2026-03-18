---
name: robotics-ros2
description: "ROS2 robotics development with rclpy: nodes, topics, services, actions, tf2 transforms, and Nav2 navigation."
tags:
  - robotics
  - ros2
  - navigation
  - python
  - embedded
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - "rclpy>=1.0 (via ROS2 Humble/Iron/Jazzy install)"
  - "pandas>=2.0"
  - "numpy>=1.24"
  - "matplotlib>=3.7"
last_updated: "2026-03-17"
status: stable
---

# ROS2 Robotics Development with rclpy

A comprehensive guide to building robot applications with ROS2 (Robot Operating System 2) using
Python. This skill covers the full stack: node lifecycle, topic publish/subscribe, service and
action patterns, tf2 coordinate transforms, URDF loading, and autonomous navigation with Nav2.

---

## When to Use This Skill

Use this skill when you need to:

- Build ROS2 nodes in Python (rclpy) for robot control, sensing, or data processing
- Implement publish/subscribe communication between robot components
- Create service servers/clients for synchronous request-response interactions
- Implement action servers/clients for long-running tasks with feedback (e.g., move to goal)
- Transform coordinates between reference frames using tf2
- Load and parse URDF robot description files
- Send navigation goals to Nav2 and monitor their execution
- Integrate sensor data (LiDAR, camera, IMU) with ROS2 message types
- Debug and introspect a running ROS2 system

**Do NOT use this skill for:**

- Non-ROS robotics frameworks (e.g., raw serial, pure OpenCV pipelines without ROS)
- ROS1 (rospy) — the APIs differ significantly; see a dedicated ROS1 skill
- Simulation setup (Gazebo/Isaac Sim configuration) — those have their own workflows

---

## Background & Key Concepts

### ROS2 Architecture Overview

ROS2 is built on DDS (Data Distribution Service) middleware, providing a distributed,
real-time communication backbone. The key abstractions are:

| Concept | Description |
|---|---|
| **Node** | A process that participates in the ROS2 graph; the fundamental unit of computation |
| **Topic** | Anonymous publish/subscribe channel; one-to-many, fire-and-forget |
| **Service** | Synchronous request-response; blocks the caller until the server responds |
| **Action** | Asynchronous goal-feedback-result pattern for long-running tasks |
| **Parameter** | Runtime-configurable key-value store per node |
| **tf2** | Coordinate frame transform library; maintains a tree of frame relationships over time |
| **Nav2** | Navigation stack providing path planning, obstacle avoidance, and recovery behaviors |

### ROS2 Message Types

ROS2 uses `.msg`, `.srv`, and `.action` IDL files to define typed interfaces. Common types:

- `std_msgs/msg/String`, `std_msgs/msg/Float64`
- `geometry_msgs/msg/Twist` (velocity commands), `geometry_msgs/msg/PoseStamped`
- `sensor_msgs/msg/LaserScan`, `sensor_msgs/msg/Image`, `sensor_msgs/msg/Imu`
- `nav_msgs/msg/Odometry`, `nav_msgs/msg/OccupancyGrid`
- `tf2_msgs/msg/TFMessage`

### Node Lifecycle

A `rclpy` node follows this lifecycle:
1. `rclpy.init()` — initialize the client library
2. Node constructor — create publishers, subscribers, services, timers
3. `rclpy.spin(node)` — enter event loop (blocks)
4. Shutdown via Ctrl-C or `rclpy.shutdown()`

### tf2 Transform Tree

The tf2 library maintains a directed tree of coordinate frames. Each edge is a `TransformStamped`
message. Common frames: `map` -> `odom` -> `base_link` -> `base_laser`, etc.

### Nav2 Simple Commander

`nav2_simple_commander` provides a Python API to interact with the Nav2 stack without writing
low-level action clients. It handles lifecycle management of Nav2 servers internally.

---

## Environment Setup

### 1. Install ROS2 (Humble recommended for LTS)

```bash
# Ubuntu 22.04 — ROS2 Humble
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-nav2-bringup \
  ros-humble-nav2-simple-commander ros-humble-tf2-tools
```

### 2. Source ROS2 and Install Python Dependencies

```bash
# Add to ~/.bashrc for persistence
source /opt/ros/humble/setup.bash

# Python packages used alongside ROS2
pip install pandas>=2.0 numpy>=1.24 matplotlib>=3.7
```

### 3. Create a ROS2 Package

```bash
# Create a colcon workspace
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# Create a Python package
ros2 pkg create --build-type ament_python my_robot_pkg \
  --dependencies rclpy std_msgs geometry_msgs sensor_msgs nav_msgs tf2_ros

cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 4. Verify Installation

```bash
# Check ROS2 is sourced
ros2 --version

# List available message types
ros2 interface list | grep geometry_msgs

# Run a demo node
ros2 run demo_nodes_py talker &
ros2 run demo_nodes_py listener
```

---

## Core Workflow

### Step 1: Create and Spin a Basic Node

The minimal skeleton for any ROS2 Python node:

```python
#!/usr/bin/env python3
"""minimal_node.py — bare-minimum rclpy node."""

import rclpy
from rclpy.node import Node


class MinimalNode(Node):
    """A node that logs a greeting on startup."""

    def __init__(self) -> None:
        super().__init__("minimal_node")
        self.get_logger().info("MinimalNode has started!")

        # Create a wall timer: callback fires every 1.0 second
        self.timer = self.create_timer(1.0, self.timer_callback)
        self._count = 0

    def timer_callback(self) -> None:
        self._count += 1
        self.get_logger().info(f"Tick #{self._count}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MinimalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### Step 2: Publish and Subscribe to Topics

```python
#!/usr/bin/env python3
"""pubsub_demo.py — demonstrates topic publisher and subscriber in one node."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np


class VelocityController(Node):
    """
    Publishes velocity commands on /cmd_vel and
    subscribes to odometry on /odom to track position.
    """

    def __init__(self) -> None:
        super().__init__("velocity_controller")

        # Publisher: send Twist messages at 10 Hz
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", qos_profile=10)

        # Subscriber: receive odometry updates
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            qos_profile=10,
        )

        # Timer drives control loop at 10 Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._target_x = 2.0  # drive toward x=2 m
        self.get_logger().info("VelocityController ready.")

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry) -> None:
        """Update internal pose estimate from odometry."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        self._x = pos.x
        self._y = pos.y

        # Quaternion -> yaw (rotation about Z)
        siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1.0 - 2.0 * (orient.y ** 2 + orient.z ** 2)
        self._yaw = np.arctan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def control_loop(self) -> None:
        """Simple proportional controller: drive toward target_x."""
        error = self._target_x - self._x
        cmd = Twist()

        if abs(error) > 0.05:
            cmd.linear.x = float(np.clip(0.5 * error, -0.5, 0.5))
        else:
            self.get_logger().info("Target reached!")

        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### Step 3: Implement a Service Server and Client

```python
#!/usr/bin/env python3
"""service_demo.py — service server + client for a simple math operation."""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


# ──────────────────────────────────────────────────────────────────────────────
# Service SERVER
# ──────────────────────────────────────────────────────────────────────────────
class AddTwoIntsServer(Node):
    """Provides an AddTwoInts service on /add_two_ints."""

    def __init__(self) -> None:
        super().__init__("add_two_ints_server")
        self.srv = self.create_service(
            AddTwoInts,
            "/add_two_ints",
            self.handle_request,
        )
        self.get_logger().info("AddTwoInts service is ready.")

    def handle_request(
        self,
        request: AddTwoInts.Request,
        response: AddTwoInts.Response,
    ) -> AddTwoInts.Response:
        response.sum = request.a + request.b
        self.get_logger().info(
            f"Request: {request.a} + {request.b} = {response.sum}"
        )
        return response


# ──────────────────────────────────────────────────────────────────────────────
# Service CLIENT
# ──────────────────────────────────────────────────────────────────────────────
class AddTwoIntsClient(Node):
    """Calls the AddTwoInts service synchronously."""

    def __init__(self) -> None:
        super().__init__("add_two_ints_client")
        self.client = self.create_client(AddTwoInts, "/add_two_ints")

        # Wait for the server to come online (timeout 5 s)
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /add_two_ints not available!")

    def call(self, a: int, b: int) -> int:
        """Send a blocking service call and return the sum."""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result().sum
        else:
            raise RuntimeError("Service call failed")


def main(args=None) -> None:
    rclpy.init(args=args)

    server = AddTwoIntsServer()
    client = AddTwoIntsClient()

    # In a real application, server and client run in separate processes.
    # Here we use a MultiThreadedExecutor to run both in one process.
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(client)

    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    result = client.call(3, 7)
    client.get_logger().info(f"3 + 7 = {result}")

    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### Step 4: Action Server and Client (Long-Running Tasks)

```python
#!/usr/bin/env python3
"""action_demo.py — Fibonacci action server with feedback streaming."""

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):
    """Computes Fibonacci sequence up to requested order, streaming partial results."""

    def __init__(self) -> None:
        super().__init__("fibonacci_action_server")
        self._action_server = ActionServer(
            self,
            Fibonacci,
            "fibonacci",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def goal_callback(self, goal_request) -> GoalResponse:
        self.get_logger().info(f"Received goal: order={goal_request.order}")
        if goal_request.order <= 0:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("Cancellation requested.")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle) -> Fibonacci.Result:
        self.get_logger().info("Executing Fibonacci goal...")
        feedback_msg = Fibonacci.Feedback()
        sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Goal was cancelled.")
                return Fibonacci.Result()

            sequence.append(sequence[-1] + sequence[-2])
            feedback_msg.partial_sequence = sequence
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.1)  # simulate computation time

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence
        self.get_logger().info(f"Result: {sequence}")
        return result


class FibonacciActionClient(Node):
    """Sends a Fibonacci goal and prints feedback as it arrives."""

    def __init__(self) -> None:
        super().__init__("fibonacci_action_client")
        self._client = ActionClient(self, Fibonacci, "fibonacci")

    def send_goal(self, order: int) -> None:
        self._client.wait_for_server()
        goal = Fibonacci.Goal()
        goal.order = order

        self.get_logger().info(f"Sending goal: order={order}")
        send_goal_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected.")
            return
        self.get_logger().info("Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg) -> None:
        partial = feedback_msg.feedback.partial_sequence
        self.get_logger().info(f"Feedback: {partial}")

    def result_callback(self, future) -> None:
        result = future.result().result
        self.get_logger().info(f"Final sequence: {result.sequence}")
        rclpy.shutdown()
```

### Step 5: tf2 Coordinate Transforms

```python
#!/usr/bin/env python3
"""tf2_demo.py — broadcast and lookup coordinate transforms."""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import math


class TF2Demo(Node):
    """
    Broadcasts a rotating transform from 'world' to 'robot_base',
    then looks up the transform and logs it.
    """

    def __init__(self) -> None:
        super().__init__("tf2_demo")

        # Broadcaster sends our custom transforms
        self.broadcaster = TransformBroadcaster(self)

        # Buffer + Listener receive and cache all transforms in the system
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._angle = 0.0
        self.timer = self.create_timer(0.05, self.update)  # 20 Hz

    def update(self) -> None:
        now = self.get_clock().now().to_msg()
        self._angle += 0.02  # radians per tick

        # ------------------------------------------------------------------
        # Broadcast world -> robot_base
        # ------------------------------------------------------------------
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "world"
        t.child_frame_id = "robot_base"

        t.transform.translation.x = math.cos(self._angle) * 1.0
        t.transform.translation.y = math.sin(self._angle) * 1.0
        t.transform.translation.z = 0.0

        # Quaternion for rotation about Z by self._angle
        t.transform.rotation.z = math.sin(self._angle / 2.0)
        t.transform.rotation.w = math.cos(self._angle / 2.0)

        self.broadcaster.sendTransform(t)

        # ------------------------------------------------------------------
        # Broadcast robot_base -> sensor_frame (static offset)
        # ------------------------------------------------------------------
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = "robot_base"
        t2.child_frame_id = "sensor_frame"
        t2.transform.translation.x = 0.3   # 30 cm in front of base
        t2.transform.translation.z = 0.15  # 15 cm above base
        t2.transform.rotation.w = 1.0      # no rotation
        self.broadcaster.sendTransform(t2)

        # ------------------------------------------------------------------
        # Lookup: where is sensor_frame in world coordinates?
        # ------------------------------------------------------------------
        try:
            tf = self.tf_buffer.lookup_transform(
                "world",
                "sensor_frame",
                rclpy.time.Time(),  # latest available
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z
            self.get_logger().debug(
                f"sensor_frame in world: ({tx:.3f}, {ty:.3f}, {tz:.3f})"
            )
        except Exception as e:
            self.get_logger().warning(f"TF lookup failed: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TF2Demo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

---

## Advanced Usage

### Nav2 Simple Commander — Send Navigation Goals

```python
#!/usr/bin/env python3
"""nav2_navigate.py — send waypoints to Nav2 and monitor execution."""

import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration


def make_pose(navigator: BasicNavigator, x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    """Helper: build a PoseStamped in the 'map' frame."""
    import math
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def main() -> None:
    rclpy.init()
    navigator = BasicNavigator()

    # Set initial pose (must match the robot's actual starting position)
    initial_pose = make_pose(navigator, x=0.0, y=0.0, yaw=0.0)
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2 to fully activate
    navigator.waitUntilNav2Active()

    # ── Navigate to a single goal ─────────────────────────────────────────
    goal = make_pose(navigator, x=3.0, y=1.5, yaw=1.57)
    navigator.goToPose(goal)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback:
            remaining = Duration.from_msg(feedback.estimated_time_remaining)
            print(f"ETA: {remaining.sec:.1f}s remaining")

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print("Navigation succeeded!")
    elif result == TaskResult.CANCELED:
        print("Navigation was canceled.")
    elif result == TaskResult.FAILED:
        print("Navigation failed — check costmap or planner logs.")

    # ── Follow a sequence of waypoints ───────────────────────────────────
    waypoints = [
        make_pose(navigator, 1.0, 0.0),
        make_pose(navigator, 2.0, 1.0),
        make_pose(navigator, 3.0, 0.0),
        make_pose(navigator, 0.0, 0.0),  # return to origin
    ]
    navigator.followWaypoints(waypoints)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback:
            idx = feedback.current_waypoint
            print(f"Visiting waypoint {idx + 1}/{len(waypoints)}")

    print("Waypoint following complete.")
    navigator.lifecycleShutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### URDF Loading and Joint State Publishing

```python
#!/usr/bin/env python3
"""urdf_joints.py — parse URDF and publish joint states."""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.parameter import Parameter


EXAMPLE_URDF = """<?xml version="1.0"?>
<robot name="two_dof_arm">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
  </joint>
</robot>
"""


class JointStatePublisher(Node):
    """Publishes sinusoidal joint states for a 2-DOF arm."""

    JOINT_NAMES = ["joint1", "joint2"]

    def __init__(self) -> None:
        super().__init__("joint_state_publisher")

        # Declare the URDF as a string parameter (in production, load from file)
        self.declare_parameter("robot_description", EXAMPLE_URDF)

        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.create_timer(0.02, self.publish_joints)  # 50 Hz
        self._t = 0.0

    def publish_joints(self) -> None:
        self._t += 0.02
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.JOINT_NAMES
        msg.position = [
            math.sin(self._t * 0.5) * 1.0,          # joint1: ±1 rad, 0.5 Hz
            math.sin(self._t * 0.5 + math.pi) * 0.8, # joint2: out of phase
        ]
        msg.velocity = [
            0.5 * math.cos(self._t * 0.5),
            0.4 * math.cos(self._t * 0.5 + math.pi),
        ]
        msg.effort = [0.0, 0.0]
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### Sensor Data Analysis with pandas and matplotlib

```python
#!/usr/bin/env python3
"""sensor_analysis.py — subscribe to LaserScan, buffer readings, plot with matplotlib."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


class LaserScanAnalyzer(Node):
    """Buffers 200 LaserScan messages and produces a range heatmap."""

    BUFFER_SIZE = 200

    def __init__(self) -> None:
        super().__init__("laser_scan_analyzer")
        self.sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self.get_logger().info("Collecting scan data...")

    def scan_callback(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
        self._buffer.append(ranges)

        if len(self._buffer) == self.BUFFER_SIZE:
            self.analyze_and_plot(msg)
            self.get_logger().info("Analysis complete — shutting down.")
            rclpy.shutdown()

    def analyze_and_plot(self, last_msg: LaserScan) -> None:
        data = np.array(self._buffer)  # shape: (200, N_beams)
        n_beams = data.shape[1]
        angles = np.linspace(
            last_msg.angle_min,
            last_msg.angle_max,
            n_beams,
        )

        df = pd.DataFrame(data, columns=[f"beam_{i}" for i in range(n_beams)])
        print(df.describe().T[["mean", "std", "min", "max"]].head(10))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap: time x beam angle
        im = axes[0].imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[np.degrees(angles[0]), np.degrees(angles[-1]), 0, self.BUFFER_SIZE],
            cmap="viridis",
        )
        axes[0].set_xlabel("Beam angle (deg)")
        axes[0].set_ylabel("Scan index (time)")
        axes[0].set_title("Range Heatmap")
        plt.colorbar(im, ax=axes[0], label="Range (m)")

        # Mean range profile
        axes[1].plot(np.degrees(angles), data.mean(axis=0), label="Mean range")
        axes[1].fill_between(
            np.degrees(angles),
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=0.3,
            label="±1 std",
        )
        axes[1].set_xlabel("Beam angle (deg)")
        axes[1].set_ylabel("Range (m)")
        axes[1].set_title("Mean Range Profile")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("/tmp/laser_analysis.png", dpi=150)
        self.get_logger().info("Plot saved to /tmp/laser_analysis.png")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaserScanAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### Issue: `rclpy.init()` raises `RuntimeError: rcl_init failed`

**Cause:** ROS2 environment is not sourced, or another ROS2 instance is running on the same domain.

```bash
# Check that ROS2 is sourced
echo $ROS_DISTRO  # should print 'humble', 'iron', etc.
source /opt/ros/humble/setup.bash

# Isolate from other ROS2 systems on the network
export ROS_DOMAIN_ID=42  # pick any 0–101; default is 0
```

### Issue: Subscriber never receives messages

**Cause:** QoS mismatch between publisher and subscriber, or wrong topic name.

```bash
# List active topics
ros2 topic list

# Inspect QoS profile of a topic
ros2 topic info /cmd_vel --verbose

# Echo messages to verify data is flowing
ros2 topic echo /cmd_vel
```

```python
# Use a compatible QoS profile (reliable, transient_local for latched topics)
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)
self.sub = self.create_subscription(msg_type, topic, callback, qos)
```

### Issue: tf2 lookup raises `LookupException`

**Cause:** The requested transform has not been broadcast yet, or the parent/child frame names are wrong.

```python
from rclpy.duration import Duration

try:
    tf = self.tf_buffer.lookup_transform(
        "world", "robot_base",
        rclpy.time.Time(),
        timeout=Duration(seconds=1.0),  # wait up to 1 s
    )
except Exception as e:
    self.get_logger().error(f"TF lookup error: {e}")
```

```bash
# Visualize the tf tree
ros2 run tf2_tools view_frames
evince frames.pdf
```

### Issue: Nav2 action server not available

```bash
# Check Nav2 nodes are running
ros2 node list | grep nav

# Restart Nav2 lifecycle
ros2 lifecycle set /bt_navigator configure
ros2 lifecycle set /bt_navigator activate
```

### Issue: High CPU usage in spin loop

```python
# Use MultiThreadedExecutor for I/O-bound nodes
from rclpy.executors import MultiThreadedExecutor

executor = MultiThreadedExecutor(num_threads=4)
executor.add_node(node_a)
executor.add_node(node_b)
executor.spin()
```

---

## External Resources

- ROS2 Documentation: https://docs.ros.org/en/humble/
- rclpy API Reference: https://docs.ros2.org/latest/api/rclpy/
- Nav2 Documentation: https://navigation.ros.org/
- tf2 Python Tutorial: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html
- ROS2 Design Patterns (REP-2004): https://ros.org/reps/rep-2004.html
- Awesome ROS2: https://github.com/fkromer/awesome-ros2
- ROS2 QoS Guide: https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html

---

## Examples

### Example 1: Differential Drive Robot Controller

A complete node that subscribes to a target pose and drives a differential robot toward it
using a proportional heading + velocity controller.

```python
#!/usr/bin/env python3
"""diff_drive_controller.py — full differential drive controller example."""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np


class DiffDriveController(Node):
    """
    Subscribes to /goal_pose (PoseStamped) and /odom (Odometry).
    Publishes velocity commands to /cmd_vel to steer toward the goal.

    Control law:
      1. Compute heading error (angle to goal vs current yaw)
      2. If |heading error| > threshold, rotate in place
      3. Else drive forward with proportional speed
    """

    LINEAR_KP = 0.5     # m/s per meter of distance error
    ANGULAR_KP = 1.5    # rad/s per radian of heading error
    GOAL_TOLERANCE = 0.1  # meters
    MAX_LINEAR = 0.4    # m/s
    MAX_ANGULAR = 1.0   # rad/s

    def __init__(self) -> None:
        super().__init__("diff_drive_controller")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self._x = self._y = self._yaw = 0.0
        self._goal_x = self._goal_y = None
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("DiffDriveController ready — send /goal_pose")

    def goal_callback(self, msg: PoseStamped) -> None:
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        self.get_logger().info(
            f"New goal: ({self._goal_x:.2f}, {self._goal_y:.2f})"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self._yaw = math.atan2(siny, cosy)

    def control_loop(self) -> None:
        if self._goal_x is None:
            return

        dx = self._goal_x - self._x
        dy = self._goal_y - self._y
        distance = math.hypot(dx, dy)

        cmd = Twist()

        if distance < self.GOAL_TOLERANCE:
            self.get_logger().info("Goal reached!")
            self._goal_x = None
            self.cmd_pub.publish(cmd)  # stop
            return

        desired_yaw = math.atan2(dy, dx)
        heading_error = desired_yaw - self._yaw

        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        if abs(heading_error) > 0.3:
            # Rotate in place
            cmd.angular.z = float(
                np.clip(self.ANGULAR_KP * heading_error, -self.MAX_ANGULAR, self.MAX_ANGULAR)
            )
        else:
            # Drive forward
            cmd.linear.x = float(
                np.clip(self.LINEAR_KP * distance, 0.0, self.MAX_LINEAR)
            )
            cmd.angular.z = float(
                np.clip(self.ANGULAR_KP * heading_error, -self.MAX_ANGULAR, self.MAX_ANGULAR)
            )

        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiffDriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Terminal 1: run the controller
ros2 run my_robot_pkg diff_drive_controller

# Terminal 2: send a goal pose
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 3.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}'
```

---

### Example 2: Sensor Fusion Node (IMU + Odometry)

Fuses IMU angular velocity with odometry using a complementary filter to produce a
smooth yaw estimate, published as a custom TF transform.

```python
#!/usr/bin/env python3
"""sensor_fusion.py — complementary filter fusing IMU + odometry yaw."""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class ComplementaryFilter(Node):
    """
    Fuses IMU gyroscope (high-frequency, no drift in short term) with
    odometry yaw (low-frequency, long-term stable) using a simple
    complementary filter:

        yaw_fused = alpha * (yaw_imu_integrated) + (1 - alpha) * yaw_odom
    """

    ALPHA = 0.98      # weight for IMU integration
    DT = 0.02         # assume ~50 Hz IMU

    def __init__(self) -> None:
        super().__init__("complementary_filter")

        self.imu_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 50)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.broadcaster = TransformBroadcaster(self)

        self._yaw_fused = 0.0
        self._yaw_odom = 0.0
        self._last_imu_stamp = None

    def imu_cb(self, msg: Imu) -> None:
        gz = msg.angular_velocity.z   # rad/s about Z

        # Integrate gyro
        if self._last_imu_stamp is not None:
            stamp_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            last_ns = self._last_imu_stamp
            dt = (stamp_ns - last_ns) * 1e-9
            dt = max(0.0, min(dt, 0.1))   # clamp to [0, 100 ms]
        else:
            dt = self.DT

        self._last_imu_stamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec

        yaw_imu_integrated = self._yaw_fused + gz * dt

        # Complementary filter
        self._yaw_fused = (
            self.ALPHA * yaw_imu_integrated
            + (1 - self.ALPHA) * self._yaw_odom
        )

        self._broadcast(msg.header.stamp)

    def odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self._yaw_odom = math.atan2(siny, cosy)

    def _broadcast(self, stamp) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link_fused"
        t.transform.rotation.z = math.sin(self._yaw_fused / 2.0)
        t.transform.rotation.w = math.cos(self._yaw_fused / 2.0)
        self.broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ComplementaryFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

**Verification:**

```bash
# Check the fused frame appears in the tf tree
ros2 run tf2_tools view_frames
# Confirm base_link_fused appears under odom

# Monitor the fused yaw in real time
ros2 run tf2_ros tf2_echo odom base_link_fused
```
