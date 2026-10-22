import numpy as np
from collections import deque


class PoseFilter(object):
    """关键点去抖动滤波器（视频时序后处理）

    对每个关键点独立维护历史轨迹，主要处理：
    1. 中值预筛：先用中值滤波剔除单帧孤立跳点，再做平滑，避免误检点污染历史；
    2. 失效保护：检测失败的关键点（(0,0)/NaN/inf）不参与平滑，短期输出上一次的有效结果，
       连续丢失超过 lost_frames 帧后清空该点历史，避免长期输出陈旧坐标、恢复时跳变；
    3. 跳变重捕获：位移超过 max_step 时先暂停更新（保持上一次输出），连续 recover_frames 帧
       同方向超限才判定为真实快速运动并重新捕获，兼顾抗漂移与快速跟踪；
    4. 指数加权平滑：对最近 wsize 帧的观测按 decay 指数衰减加权平均，越久远的帧权重越小；
    5. 速度自适应：用平滑后的速度调整平滑力度，快速运动时减弱平滑（减小拖影/延迟），
       静止或微抖时加强平滑；速度按帧间隔 dt 归一化，跳帧时不会误判。
    """

    def __init__(self, kid: list = None, wsize: int = 10, decay: float = 0.1, vTh: float = 15.0,
                 max_step: float = None, median_size: int = 3, recover_frames: int = 3,
                 lost_frames: int = 15):
        """
        max_step 建议设为略大于目标真实最大单帧位移。设得过小（小于真实速度）会出现“保持→跳一步”的阶梯跟踪；不确定时保持 None。
        静止场景抖动仍明显：减小 decay（如 0.05）或增大 wsize。
        快速运动拖影：增大 decay、降低 vTh。
        手部等有快速局部运动的场景，可把 median_size 设为 1 关闭中值预筛以减小 1 帧延迟。
        :param kid:   需要跟踪的关键点ID，可指定多个,如[0,1,2,3]；为None或空列表时处理全部关键点
        :param wsize: 平滑窗口，即参与平滑的历史帧数；越大越平滑，但延迟也越大
        :param decay: 权重衰减系数，值越大，历史影响衰减的越快，平滑力度越小，取值范围[0,1)
        :param vTh:   速度阈值（像素/帧），位移超过该值时减弱平滑以降低延迟；
                      <=0 表示关闭速度自适应，始终使用固定的 decay
        :param max_step: 单帧最大允许位移（像素/帧），用于区分"偶发跳变"和"真实快速运动"；
                         建议设为略大于目标真实最大单帧位移。设得过小（小于真实速度）会出现“保持→跳一步”的阶梯跟踪
                         None 表示不启用跳变检测
        :param median_size: 中值预筛窗口大小，用于剔除孤立跳点；1 表示关闭预筛
        :param recover_frames: 连续 N 帧同方向超过 max_step 时，判定为真实运动并重新捕获
        :param lost_frames: 关键点连续丢失帧数超过该值时清空其历史状态
        """
        self.kid = kid
        self.wsize = max(int(wsize), 1)
        self.decay = float(np.clip(decay, 0.0, 0.999))
        self.vTh = float(vTh)
        self.max_step = max_step
        self.median_size = max(int(median_size), 1)
        self.recover_frames = max(int(recover_frames), 1)
        self.lost_frames = max(int(lost_frames), 1)
        self.reset()

    def reset(self):
        """清空滤波状态（切换视频、目标ID变化或场景切换时调用）"""
        self.state = {}  # 关键点ID -> 状态字典，见 _get_state

    def update(self, points: np.ndarray, dt: float = 1.0):
        """
        :param points: (N,2)关键点
        :param dt: 帧间隔（以帧为单位，默认1.0；跳帧/变帧率时传入实际间隔，用于速度归一化）
        :return: (N,2)平滑后的关键点；当 points 为 numpy 数组时会原地修改，同时返回结果
        """
        if points is None or len(points) == 0:
            return points
        points = np.asarray(points)
        dt = 1.0 if dt is None or dt <= 0 else float(dt)
        # kid 为空时平滑全部关键点
        kids = self.kid if self.kid else range(len(points))
        for k in kids:
            k = int(k)
            if k < 0 or k >= len(points):
                continue
            point = np.array(points[k, :2], dtype=np.float32)  # copy，避免写入结果时污染历史
            if not self._is_valid(point):
                # 无效点：短期保持上一次有效结果，长时间丢失则清空历史，避免输出陈旧坐标
                state = self.state.get(k)
                if state is None or state["last"] is None:
                    continue
                state["miss"] += 1
                if state["miss"] > self.lost_frames:
                    self.state.pop(k, None)
                else:
                    points[k, :2] = state["last"]
                continue
            state = self._get_state(k)
            state["miss"] = 0
            points[k, :2] = self._smooth(state, point, dt)
        return points

    def _get_state(self, k: int):
        """获取（或初始化）关键点 k 的滤波状态"""
        state = self.state.get(k)
        if state is None:
            state = {
                "raw": deque(maxlen=self.median_size),  # 中值预筛缓存
                "hist": deque(maxlen=self.wsize),  # 参与加权平均的观测
                "last": None,  # 上一次输出的平滑坐标
                "speed": 0.0,  # 平滑后的运动速度（像素/帧）
                "jump_cnt": 0,  # 连续超限帧数
                "jump_dir": None,  # 上一次超限位移的方向
                "miss": 0,  # 连续丢失帧数
            }
            self.state[k] = state
        return state

    def _smooth(self, state: dict, point: np.ndarray, dt: float):
        """对单个关键点做时序平滑"""
        # 1. 中值预筛：剔除单帧孤立跳点，避免误检污染平滑历史
        state["raw"].append(point)
        if self.median_size > 1:
            obs = np.median(np.asarray(state["raw"], dtype=np.float32), axis=0)
        else:
            obs = point
        last = state["last"]
        if last is None:
            state["hist"].append(obs)
            state["last"] = obs
            return obs
        # 2. 跳变检测：超限时先保持上一次输出，连续同方向超限才认为是真实快速运动
        disp = obs - last
        dist = float(np.linalg.norm(disp))
        if self.max_step and dist > self.max_step * dt:
            direction = disp / (dist + 1e-6)
            if state["jump_dir"] is not None and float(np.dot(state["jump_dir"], direction)) > 0.8:
                state["jump_cnt"] += 1
            else:
                state["jump_cnt"] = 1
            state["jump_dir"] = direction
            if state["jump_cnt"] < self.recover_frames:
                return last  # 视为偶发跳变，保持上一次输出
            # 确认为真实运动：清空历史，直接重新捕获当前观测
            state["raw"].clear()
            state["raw"].append(obs)
            state["hist"].clear()
            state["hist"].append(obs)
            state["speed"] = 0.0
            state["jump_cnt"], state["jump_dir"] = 0, None
            state["last"] = obs
            return obs
        state["jump_cnt"], state["jump_dir"] = 0, None
        # 3. 速度估计：按 dt 归一化，并做 EMA 平滑，避免单帧噪声导致平滑力度剧烈变化
        speed = 0.5 * (dist / dt) + 0.5 * state["speed"]
        state["speed"] = speed
        # 4. 速度自适应：运动越快，decay 越接近1，历史权重越小（平滑越弱，延迟越小）
        decay = self.decay
        if self.vTh > 0:
            decay = decay + (1.0 - decay) * min(speed / self.vTh, 1.0)
        # 帧间隔越大，历史帧越"过期"，按 dt 次方加速衰减
        decay = 1.0 - (1.0 - decay) ** dt
        state["hist"].append(obs)
        # 5. 指数加权滑动平均：越新的观测权重越大，age 为距当前帧的帧数
        age = np.arange(len(state["hist"]) - 1, -1, -1)
        weights = (1.0 - decay) ** age
        smooth = np.average(np.asarray(state["hist"], dtype=np.float32), axis=0, weights=weights)
        state["last"] = smooth
        return smooth

    @staticmethod
    def _is_valid(point: np.ndarray):
        """判断关键点是否有效：排除 NaN/inf 以及检测失败时被置为(0,0)的点"""
        return bool(np.all(np.isfinite(point))) and not (point[0] == 0 and point[1] == 0)
