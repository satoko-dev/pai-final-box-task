#!/usr/bin/env python3
"""Gazeboで箱をつかみ、移動して置く。

実行: python3 pick_place_box.py --speed 2
各動作前にEnterを待つ: --step
"""
import argparse
import itertools
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import time

# 関節名・目標角度などの設定
ARM_NAMES = ['crane_plus_joint1', 'crane_plus_joint2',
             'crane_plus_joint4', 'crane_plus_joint3']
HAND_NAME = 'crane_plus_joint_hand'
ARM_TOPIC = '/crane_plus_arm_controller/joint_trajectory'
HAND_TOPIC = '/crane_plus_gripper_controller/joint_trajectory'
PLACE_YAW = 0.0
TABLE_Z = 1.015
# 現在のtable.sdfの配置を前提とする。worldとbaseの水平軸は一致、joint1=0は+x方向。
# 土台を移動・回転した場合は、この座標変換の設定も変更する。
BASE_X = 0.0
BASE_Y = 0.0
BASE_YAW = 0.0
# 箱中心から手首までの水平補正。半径方向は外向きが正、接線方向は反時計回りが正。
# 初期値0は「手首を箱中心の真上に配置」。必要性を確認した場合だけ調整する（単位m）。
GRASP_RADIAL_OFFSET = 0.0
GRASP_TANGENTIAL_OFFSET = 0.0
# 固定指に合わせる角度補正。箱が-90度方向なら、目標は-95.2度。
GRASP_YAW_OFFSET = math.radians(-5.2)#補正分
OPEN = -0.15  # 開く幅を縮小。箱に当たる場合は-0.20、必要なら-0.25へ。
CLOSE = 0.25
#各関節の可動域
LIMITS = [(-2.61799, 2.61799), (-1.82474, 1.78372),
          (-1.84009, 1.76836), (-2.55655, 2.45427)]

# 実行
def main():
    # 1. 実行オプションを読み取る。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', action='store_true', help='各動作前にEnterを待つ。初回の確認用')
    parser.add_argument('--speed', type=float, default=1., help='動作時間の倍率。2で指定時間が半分（0.5〜2）')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not .5 <= args.speed <= 2.:
        parser.error('--speedは0.5〜2の範囲で指定してください。')
    
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    # 2. ROS2の通信を開始する。
    rclpy.init()
    node = Node(
        'pick_place_demo',
        parameter_overrides=[Parameter('use_sim_time', value=True)]
    )

    try:
        robot = Robot(node, args.step, args.speed)
        robot.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0

class Robot:
    """ロボットの状態と操作をまとめる。親クラスはありません。"""

    # 通信の準備
    def __init__(self, node, step, speed=1.):
        from sensor_msgs.msg import JointState
        from rosgraph_msgs.msg import Clock
        from trajectory_msgs.msg import JointTrajectory
        self.node, self.step = node, step
        self.speed = speed
        self.state, self.received = {}, {}
        self.sim_time = None
        self.last_clock = 0.
        self.arm = node.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.hand = node.create_publisher(JointTrajectory, HAND_TOPIC, 10)
        self.sub = node.create_subscription(JointState, '/joint_states', self.on_state, 10)
        self.clock_sub = node.create_subscription(Clock, '/clock', self.on_clock, 10)
        self.height = None
        self.radius = None
        self.pick_yaw = None
        self.pick_center = None

    # ロボットの動作順（最初に読む）
    def run(self):
        self.prepare()
        print(f'動作時間の倍率: {self.speed:g}（指定時間を倍率で割ります）', flush=True)
        self.move_arm('1. 直立のまま箱中心の方向へ回転', [self.pick_yaw, 0., 0., 0.], 8)
        self.gripper('2. 箱の方向を向いてからハンドを開く', OPEN)
        self.move_arm('3. 箱へ近づく準備姿勢', [self.pick_yaw, .3, .1, -.2])
        self.wrist('4. 箱を挟む位置へ下げる', self.pick_yaw, .15)
        self.gripper('5. 一度の動作で箱をつかむ（目標0.25）', CLOSE, seconds=4, contact=True)
        self.wrist('6. 傾きを保ち、一度に約2.8cm持ち上げる', self.pick_yaw, .178, seconds=4)
        lowest, _, _ = cube_geometry(read_cube())
        if lowest-TABLE_Z < .008:
            raise RuntimeError('箱の浮上を確認できません。手順を中止しました。指は自動で開きません。')
        # 関節2,4,3は直前と同じ値。関節1だけを0へ戻す。
        self.wrist('7. 高さと傾きを保ち、初期方向へ回転',
                   PLACE_YAW, .178, seconds=10)
        if cube_geometry(read_cube())[0]-TABLE_Z < .008:
            raise RuntimeError('運搬中の保持を確認できません。以降の動作を中止しました。')
        # 持ち上げ経路の逆向き。最低点を測って接地直前まで一度に下降。
        # 誤差・滑りがあった場合のみ追加の下降を行う。
        self.land('8. 箱を着地直前まで下ろす')
        self.gripper('9. 指を必要な幅だけ開き、机の上に箱を置く', OPEN, seconds=3)
        self.advance(1.)
        bottom, center, _ = cube_geometry(read_cube())
        if abs(bottom-TABLE_Z) > .004 or math.hypot(center[0]-self.pick_center[0], center[1]-self.pick_center[1]) < .03:
            raise RuntimeError('移動先での接地を確認できません。腕を退避せず停止します。')
        self.wrist('10. 指を開いたまま上へ退避', PLACE_YAW, .175)
        self.gripper('11. ハンドの開き幅を確認', OPEN)
        self.move_arm('12. 腕を起こす準備姿勢', [PLACE_YAW, .3, .1, -.2], 8)
        self.move_arm('13. 全関節0の初期直立姿勢へ戻す', [0., 0., 0., 0.])
        self.gripper('14. ハンドも初期角度0へ戻す', 0.)
        bottom, center, _ = cube_geometry(read_cube())
        if abs(bottom-TABLE_Z) > .004:
            raise RuntimeError('最終確認で接地を確認できませんでした。')
        print(f'\n完了: 箱中心 x={center[0]:.3f}, y={center[1]:.3f} m。腕・指は初期角度0です。', flush=True)

    # 関節角度で腕を動かす
    def move_arm(self, title, values, seconds=6):
        self.move(title, self.arm, ARM_NAMES, values, seconds)

    # 手首の高さを指定して腕を動かす
    def wrist(self, title, yaw, height, pitch=math.pi, seconds=6):
        self.move_arm(title, arm_pose(yaw, height, pitch, radius=self.radius), seconds)
        self.height = height

    # 指を開閉する
    def gripper(self, title, value, seconds=5, contact=False):
        self.move(title, self.hand, [HAND_NAME], [value], seconds, contact)

    # ROS2へ指令を送り、動作完了を待つ
    def move(self, title, publisher, names, values, seconds, contact=False):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        print('\n' + title, flush=True)
        self.wait(lambda: self.fresh(names) and time.monotonic()-self.last_clock < 2., 10.)
        if all(abs(self.state[n]-v) < .005 for n, v in zip(names, values)):
            print('既に目標位置のため、この動作を省略します。', flush=True)
            return
        if self.step:
            input('Enterでこの動作を実行（Ctrl+Cで中断）: ')
        self.check_publishers()
        self.wait(lambda: self.fresh(names) and time.monotonic()-self.last_clock < 2., 10.)
        self.wait(lambda: publisher.get_subscription_count() > 0, 10.)
        trajectory = JointTrajectory()
        trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        duration_ns = round(seconds / self.speed * 1_000_000_000)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(duration_ns, 1_000_000_000)
        duration = duration_ns / 1_000_000_000
        trajectory.points = [point]
        sent_at = time.monotonic()
        publisher.publish(trajectory)
        self.advance(duration + (.25 if contact else 0.))
        # 把持中は箱に阻まれて目標角度へ届かないことが正常に起こる。
        if not contact:
            self.wait(lambda: self.fresh(names)
                      and all(self.received[n] > sent_at for n in names)
                      and all(abs(self.state[n]-v) < .015 for n, v in zip(names, values)), 20.)
        elif not self.fresh(names):
            raise RuntimeError('ハンドの最新状態が取得できません。')
        print(f'この動作の実時間: {time.monotonic()-sent_at:.1f}秒', flush=True)
        print('実際の角度: ' + ', '.join(f'{self.state[n]:.4f}' for n in names), flush=True)

    # 箱の高さを測りながら下ろす
    def land(self,title):
        # 箱が滑ると高さが変わるので、下降ごとに実際の最低点を再測定する。
        for attempt in range(4):
            bottom, _, _ = cube_geometry(read_cube())
            gap = bottom-TABLE_Z
            print(f'箱の最低点から机まで: {gap*1000:.1f} mm', flush=True)
            if -.002 <= gap <= .002:
                return
            if gap < -.002 or gap > .07:
                raise RuntimeError('接地計算の前提外です。下降を中止しました。')
            distance = gap-.001
            height = self.height-distance
            if not .135 <= height <= .18:
                raise RuntimeError('下降高さが確認範囲外です。')
            self.wrist(    f'{title}（{attempt+1}回目）',    PLACE_YAW, height, seconds=4)
        bottom, _, _ = cube_geometry(read_cube())
        if not -.002 <= bottom-TABLE_Z <= .002:
            raise RuntimeError('着地直前の高さを確認できません。指は閉じたまま停止します。')

    # 開始条件を確認する
    def prepare(self):
        self.wait(lambda: self.sim_time is not None and self.fresh(ARM_NAMES+[HAND_NAME])
                  and self.arm.get_subscription_count() > 0
                  and self.hand.get_subscription_count() > 0, 20.)
        start = self.sim_time
        self.wait(lambda: self.sim_time > start, 10.)
        if any(abs(self.state[n]) > .03 for n in ARM_NAMES):
            raise RuntimeError('開始時の腕が全関節0ではありません。Gazeboを再起動してください。')
        self.check_publishers()
        pose = read_cube()
        p, _ = pose
        bottom, center, tilt = cube_geometry(pose)
        if math.hypot(p[0], p[1]+.1) > .004 or abs(bottom-TABLE_Z) > .002 or tilt > .035:
            raise RuntimeError('箱が初期位置・水平ではありません。箱を初期状態へ戻してください。')
        self.pick_center = center
        self.radius, self.pick_yaw = grasp_target(center)
        # 移動を始める前に、使用する高さの到達可能性を確認する。
        for yaw in (self.pick_yaw, PLACE_YAW):
            for height in (.135, .15, .175, .178):
                arm_pose(yaw, height, radius=self.radius)
        print(f'箱中心 world: x={center[0]:.4f}, y={center[1]:.4f} m', flush=True)
        print(f'手首目標: 水平距離={self.radius:.4f} m, '
              f'方向={math.degrees(self.pick_yaw):.2f}度（補正込み）', flush=True)
        # ロボット記述の両指の設定を確認。物理エンジンでの有効値の証明ではない。
        import xml.etree.ElementTree as ET
        result = subprocess.run(['ros2', 'param', 'get', '/robot_state_publisher',
                                 'robot_description'], capture_output=True, text=True, timeout=15)
        xml_start = result.stdout.find('<?xml')
        if result.returncode or xml_start < 0:
            raise RuntimeError('robot_descriptionが取得できません。')
        root = ET.fromstring(result.stdout[xml_start:])
        for link in ('crane_plus_link4', 'crane_plus_link_hand'):
            element = root.find("gazebo[@reference='" + link + "']")
            if element is None or any(float(element.findtext(tag, '0')) != 0.2
                                      for tag in ('mu1', 'mu2')):
                raise RuntimeError('両指のmu1/mu2=0.2を確認できません。')
        print('初期状態と両指の設定0.2を確認。木箱は元の設定へ復元・Gazebo再起動済みであることが前提です。', flush=True)

    # ほかの操作プログラムとの競合を確認する
    def check_publishers(self):
        if any(self.node.count_publishers(topic) > 1 for topic in (ARM_TOPIC, HAND_TOPIC)):
            raise RuntimeError('別の指令送信元があります。teleop等を終了してください。')

    # 状態が条件を満たすまで通信を受け取る
    def wait(self, condition, timeout=120.):
        import rclpy
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=.1)
            if condition():
                return
        raise RuntimeError('待機がタイムアウトしました。Gazebo・一時停止・接続を確認してください。')

    # 関節情報が新しいか調べる
    def fresh(self, names):
        now = time.monotonic()
        return all(n in self.state and now-self.received[n] < 2. for n in names)

    # Gazeboの時刻で待つ
    def advance(self, seconds):
        start = self.sim_time
        self.wait(lambda: self.sim_time is not None and self.sim_time >= start + seconds)

    # 届いた関節角度を保存する
    def on_state(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.state[name] = position
            self.received[name] = time.monotonic()

    # 届いたシミュレーション時刻を保存する
    def on_clock(self, msg):
        value = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.sim_time is not None and value < self.sim_time - .1:
            raise RuntimeError('シミュレーション時刻が戻りました。再起動後に実行してください。')
        self.sim_time, self.last_clock = value, time.monotonic()


# 計算と箱の情報取得（必要になったときに読む）
def grasp_target(center):
    """Gazeboの箱中心(world)から、土台基準の手首目標radius,yawを計算。"""
    dx = center[0] - BASE_X
    dy = center[1] - BASE_Y
    box_radius = math.hypot(dx, dy)
    if box_radius < 1e-6:
        raise RuntimeError('箱が土台の回転軸上にあり、接近方向を決められません。')
    box_yaw = math.atan2(dy, dx) - BASE_YAW
    radial = box_radius + GRASP_RADIAL_OFFSET
    if radial <= 0:
        raise RuntimeError('把持位置の半径方向補正が大きすぎます。')
    radius = math.hypot(radial, GRASP_TANGENTIAL_OFFSET)
    yaw = box_yaw + math.atan2(GRASP_TANGENTIAL_OFFSET, radial) + GRASP_YAW_OFFSET
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return radius, yaw


def arm_pose(yaw, height, pitch=math.pi, *, radius):
    """高さはロボットbaseからの手首高さ。順序は1,2,4,3。"""
    z = height - 0.0714
    cosine = (radius**2 + z**2 - 0.083**2 - 0.0935**2) / (2 * 0.083 * 0.0935)
    if not -1 <= cosine <= 1:
        raise RuntimeError('手首目標が到達範囲外です。')
    elbow = math.acos(cosine)
    shoulder = math.atan2(radius, z) - math.atan2(
        0.0935 * math.sin(elbow), 0.083 + 0.0935 * math.cos(elbow))
    pose = [yaw, shoulder, shoulder + elbow - pitch, -elbow]
    if not all(lo <= q <= hi for q, (lo, hi) in zip(pose, LIMITS)):
        raise RuntimeError('関節可動範囲外です。')
    return pose

def read_cube():
    return capture_cube(['ign', 'topic', '-e', '-t', '/world/default/pose/info'])

def capture_cube(command):
    # 固定2秒間の収集をやめ、箱の完全な姿勢が1回届いたら直ちに返す。
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 3.
            buffer = b''
            while time.monotonic() < deadline:
                if not selector.select(timeout=.1):
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                buffer = (buffer + chunk)[-1000000:]
                try:
                    return parse_cube(buffer.decode('utf-8', errors='replace'))
                except RuntimeError:
                    # 最後のブロックがまだ受信途中なら、続きを待つ。
                    continue
        raise RuntimeError('箱の姿勢を3秒以内に取得できませんでした。')
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=.5)
        process.stdout.close()

def parse_cube(text):
    # protobuf TextFormatのposeブロックを、波括弧の深さで抽出。
    for match in re.finditer(r'\bpose\s*\{', text):
        start = match.end()
        depth = 1
        end = start
        while end < len(text) and depth:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        if depth:
            continue
        block = text[start:end - 1]
        if not re.search(r'\bname:\s*"wood_cube_5cm"', block):
            continue

        def fields(label, names):
            section = re.search(r'\b' + label + r'\s*\{([^{}]*)\}', block)
            if section is None:
                raise RuntimeError('箱の' + label + 'がありません。')
            result = []
            for name in names:
                number = re.search(r'\b' + name + r':\s*([-+\d.eE]+)', section.group(1))
                result.append(float(number.group(1)) if number else 0.0)
            return result

        position = fields('position', 'xyz')
        quaternion = fields('orientation', 'xyzw')
        if not all(math.isfinite(v) for v in position + quaternion):
            raise RuntimeError('箱の姿勢に非数値があります。')
        return position, quaternion
    raise RuntimeError('wood_cube_5cmの姿勢を取得できませんでした。')

def cube_geometry(pose):
    """箱モデル原点は底面中心。8頂点の最低高さと中心、傾きを返す。"""
    p, quat = pose
    n = math.sqrt(sum(v*v for v in quat))
    if n < 1e-6:
        raise RuntimeError('箱の回転情報が不正です。')
    x, y, z, w = [v / n for v in quat]
    rotation = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    lowest = min(p[2] + sum(rotation[2][i]*v[i] for i in range(3))
                 for v in itertools.product([-.025, .025], [-.025, .025], [0., .05]))
    center = [p[i] + .025 * rotation[i][2] for i in range(3)]
    tilt = math.acos(max(-1., min(1., rotation[2][2])))
    return lowest, center, tilt

def check():
    # 箱中心の座標変換と、全動作高さでのFK/IK整合性を確認する。
    target_radius, target_yaw = grasp_target([0., -.1, TABLE_Z+.025])
    assert abs(target_radius-.1) < 1e-9 and abs(target_yaw-(-math.pi/2+GRASP_YAW_OFFSET)) < 1e-9
    for center in ([0., -.1, 1.04], [.001, -.099, 1.04], [-.001, -.101, 1.04]):
        target_radius, target_yaw = grasp_target(center)
        for yaw in (target_yaw, PLACE_YAW):
            # land()の下降範囲も含める。
            for i in range(44):
                height = .135 + i*.001
                _, q2, q4, q3 = arm_pose(yaw, height, radius=target_radius)
                actual_r = .083*math.sin(q2) + .0935*math.sin(q2-q3)
                actual_h = .0714 + .083*math.cos(q2) + .0935*math.cos(q2-q3)
                assert abs(actual_r-target_radius) < 1e-9
                assert abs(actual_h-height) < 1e-9
                assert abs(q2-q3-q4-math.pi) < 1e-9
    fixture = 'pose {name: "wood_cube_5cm" position {y: -0.1 z: 1.015} orientation {w: 1}}'
    p = parse_cube(fixture)
    assert abs(cube_geometry(p)[0]-TABLE_Z) < 1e-9
    # 実験で使った箱姿勢: 最低点から机まで約32.985mm。
    p = ([.09379692, -.1346834, 1.0625223894171971],
         [-.19025954350206223, -.13572109891009287, .26330350461332008, .93597668447209037])
    assert abs(cube_geometry(p)[0]-TABLE_Z-.03298488355) < 1e-8
    try:
        arm_pose(target_yaw, .20, radius=target_radius)  # pitch=piでは手首制限を超える
    except RuntimeError:
        pass
    else:
        raise AssertionError('関節制限の検証が動作していません。')
    print('計算チェック成功（ROS/Gazeboでの動作確認は別途必要）。')

if __name__ == '__main__':
    raise SystemExit(main())
