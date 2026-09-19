#!/usr/bin/env python3

# Copyright 2024 Husarion sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Start/stop move_group on demand over ROS services.

MoveIt's move_group costs roughly a core and a half on the Orin, but the arm only
needs to plan for two brief windows (home on startup, dock on shutdown). This node
lets a remote orchestrator -- running on the host, reaching the robot purely over
DDS -- bring move_group up only for those windows and tear it down otherwise, so
its cost is not paid during navigation/transit.

It owns move_group as a child process (a `ros2 launch open_manipulator_x_moveit
move_group.launch.py`) and exposes two std_srvs/Trigger services:

    ~/start  -> spawn move_group if not already running (idempotent)
    ~/stop   -> terminate move_group if running (idempotent)

The child runs in its own session (process group) so the whole launch tree is
signalled as a unit, and it is killed on node shutdown so move_group never
outlives the supervisor. This node is launched in place of move_group when the
robot is brought up with start_moveit:=False.
"""

import os
import signal
import subprocess

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class MoveItSupervisor(Node):

    def __init__(self):
        super().__init__('moveit_supervisor')
        # move_group launch is parameterised so a different arm/moveit config can
        # reuse this supervisor without code changes.
        self.declare_parameter('launch_package', 'open_manipulator_x_moveit')
        self.declare_parameter('launch_file', 'move_group.launch.py')
        # SIGINT lets move_group shut its lifecycle down cleanly; if it overruns
        # this budget we escalate to SIGKILL so ~/stop always returns promptly.
        self.declare_parameter('stop_grace_period', 10.0)

        self._proc = None
        self._start_srv = self.create_service(Trigger, '~/start', self._on_start)
        self._stop_srv = self.create_service(Trigger, '~/stop', self._on_stop)
        self.get_logger().info(
            'moveit_supervisor ready (services: ~/start, ~/stop). '
            'move_group is not running until ~/start is called.')

    def _running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _on_start(self, request, response):
        if self._running():
            response.success = True
            response.message = f'move_group already running (pid {self._proc.pid})'
            return response
        pkg = self.get_parameter('launch_package').value
        launch_file = self.get_parameter('launch_file').value
        cmd = ['ros2', 'launch', pkg, launch_file]
        try:
            # start_new_session=True puts ros2 launch and its move_group child in
            # a fresh process group we can signal together in _terminate().
            self._proc = subprocess.Popen(cmd, start_new_session=True)
        except OSError as exc:
            self._proc = None
            response.success = False
            response.message = f'failed to start move_group: {exc}'
            self.get_logger().error(response.message)
            return response
        response.success = True
        response.message = f'started move_group (pid {self._proc.pid})'
        self.get_logger().info(response.message)
        return response

    def _on_stop(self, request, response):
        if not self._running():
            self._proc = None
            response.success = True
            response.message = 'move_group not running'
            return response
        response.message = self._terminate()
        response.success = True
        self.get_logger().info(response.message)
        return response

    def _terminate(self) -> str:
        pid = self._proc.pid
        grace = float(self.get_parameter('stop_grace_period').value)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGINT)  # graceful shutdown of the launch tree
            try:
                self._proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._proc.wait(timeout=5.0)
            msg = f'stopped move_group (pid {pid})'
        except ProcessLookupError:
            msg = f'move_group (pid {pid}) already gone'
        except Exception as exc:  # noqa: BLE001 - stop must always return
            msg = f'error stopping move_group (pid {pid}): {exc}'
        finally:
            self._proc = None
        return msg

    def destroy_node(self):
        # Never let move_group outlive the supervisor.
        if self._running():
            self._terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MoveItSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
