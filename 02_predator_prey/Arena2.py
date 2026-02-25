"""
Arena2.py  —  Obstacle Course Arena
────────────────────────────────────────────────────────────────────────────────
A grid with randomly scattered rock obstacles.  Agents cannot walk through them.
Each episode regenerates the layout, so agents learn to navigate ANY map.

Drop-in replacement for Arena1 — identical interface for the controller.

═══════════════════════════════════════════════════════════════════════════════
OBSERVATION VECTOR  (19 values)
───────────────────────────────────────────────────────────────────────────────
  Idx  Sensor                                              Range
  ───  ──────────────────────────────────────────────────  ─────────
  0-14 Same 15 sensors as Arena1 (compass, dist, pos, walls, velocity, time)
  15   Is cell North blocked? (wall or obstacle)           [ 0,  1]
  16   Is cell East  blocked?                              [ 0,  1]
  17   Is cell South blocked?                              [ 0,  1]
  18   Is cell West  blocked?                              [ 0,  1]

Indices 15-18 tell each agent which moves are physically available right now.

═══════════════════════════════════════════════════════════════════════════════
OBSTACLE GENERATION
───────────────────────────────────────────────────────────────────────────────
  • ~14% of grid cells become obstacles each episode.
  • Spawn thirds (top-left for predator, bottom-right for prey) are kept clear.
  • BFS flood-fill validates that a path exists before accepting the layout.
  • Up to 25 random attempts; if none are valid the layout is thinned until one is.

═══════════════════════════════════════════════════════════════════════════════
REWARDS  (same balance philosophy as Arena1)
───────────────────────────────────────────────────────────────────────────────
PREDATOR
  Per-axis gap-reduction × 2.5          Diagonal approach (not L-shape)
  -0.15/step                            Time pressure
  Tail-position penalty  -1.8           Break exact tail-follow configuration
  Stagnation penalty  -0.5              For genuine lock-in (looser: obstacles cause pauses)
  +250 catch  /  -80 timeout

PREY
  Per-axis gap-opening  × 3.5           Running reward per axis
  ×2.0 danger-zone multiplier           Double reward when predator ≤ 3 cells
  +0.4/step survival trickle
  -200 caught  /  +300 timeout
────────────────────────────────────────────────────────────────────────────────
"""

import math
import random
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


OBSTACLE_DENSITY = 0.14  # fraction of cells to fill with obstacles


class ArenaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    _COL_BG        = (240, 240, 235)
    _COL_GRID      = (210, 210, 205)
    _COL_OBSTACLE  = ( 80,  72,  64)
    _COL_OBS_EDGE  = (115, 104,  90)
    _COL_PRED      = (231,  76,  60)
    _COL_PREY      = ( 52, 152, 219)
    _COL_CAUGHT    = (243, 156,  18)
    _COL_SURVIVED  = ( 46, 204, 113)
    _COL_TRAIL_P   = (252, 200, 195)
    _COL_TRAIL_PR  = (190, 220, 255)

    # ------------------------------------------------------------------

    def __init__(self, role: str = "predator", grid_size: int = 20,
                 opponent_model=None, render_mode=None):
        super().__init__()

        assert role in ("predator", "prey")
        self.role           = role
        self.grid_size      = grid_size
        self.opponent_model = opponent_model
        self.render_mode    = render_mode

        self.window_size = 600
        self.window      = None
        self.clock       = None
        self.replay_rect = None
        self.close_rect  = None

        self.action_space = spaces.Discrete(4)

        # 19-dim observation
        low  = np.full(19, -1.0, dtype=np.float32)
        high = np.full(19,  1.0, dtype=np.float32)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18]:
            low[i], high[i] = 0.0, 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        # Episode state (reset fills these properly)
        self.predator_pos        = np.array([0, 0])
        self.prey_pos            = np.array([grid_size - 1, grid_size - 1])
        self.obstacles: set      = set()
        self.step_count          = 0
        self.max_steps           = grid_size * 22   # slightly longer — obstacles slow things
        self.pred_trail: list    = []
        self.prey_trail: list    = []
        self.last_outcome        = None
        self._pred_last_move     = np.array([0.0, 0.0], dtype=np.float32)
        self._prey_last_move     = np.array([0.0, 0.0], dtype=np.float32)
        self._dist_window: deque = deque(maxlen=15)
        self._prev_euclid        = 0.0

    # ------------------------------------------------------------------

    def set_opponent_model(self, model):
        self.opponent_model = model

    def _euclid(self) -> float:
        return float(np.linalg.norm(
            self.predator_pos.astype(float) - self.prey_pos.astype(float)))

    # ── Obstacle helpers ────────────────────────────────────────────────────

    def _is_blocked(self, x: int, y: int) -> bool:
        return (x < 0 or x >= self.grid_size or
                y < 0 or y >= self.grid_size or
                (x, y) in self.obstacles)

    def _bfs_reachable(self, start, goal, obstacles: set) -> bool:
        g = self.grid_size
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]),  int(goal[1])
        if (sx, sy) in obstacles or (gx, gy) in obstacles:
            return False
        visited = {(sx, sy)}
        q = deque([(sx, sy)])
        while q:
            x, y = q.popleft()
            if x == gx and y == gy:
                return True
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < g and 0 <= ny < g
                        and (nx, ny) not in obstacles
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _generate_obstacles(self, pred_start, prey_start) -> set:
        g     = self.grid_size
        third = max(2, g // 3)

        # Keep spawn zones clear
        safe = ({(x, y) for x in range(third)        for y in range(third)} |
                {(x, y) for x in range(g-third, g)   for y in range(g-third, g)})

        candidates = [(x, y) for x in range(g) for y in range(g)
                      if (x, y) not in safe]

        n_obs = int(len([(x, y) for x in range(g) for y in range(g)])
                    * OBSTACLE_DENSITY)
        n_obs = min(n_obs, len(candidates))

        last_good = set()
        for _ in range(25):
            chosen = set(random.sample(candidates, n_obs))
            if self._bfs_reachable(pred_start, prey_start, chosen):
                return chosen
            if not last_good:
                last_good = chosen

        # Thin down the last attempt until the path is clear
        obs_list = list(last_good)
        random.shuffle(obs_list)
        result = set(obs_list)
        while obs_list and not self._bfs_reachable(pred_start, prey_start, result):
            result.discard(obs_list.pop())
        return result

    # ── Observation ─────────────────────────────────────────────────────────

    def _obs_for(self, role: str) -> np.ndarray:
        own      = self.predator_pos if role == "predator" else self.prey_pos
        opp      = self.prey_pos     if role == "predator" else self.predator_pos
        opp_last = (self._prey_last_move  if role == "predator"
                    else self._pred_last_move)

        max_d   = float(self.grid_size - 1)
        max_eu  = max_d * math.sqrt(2)
        max_man = max_d * 2.0

        dx, dy = float(opp[0] - own[0]), float(opp[1] - own[1])
        angle  = math.atan2(dy, dx)

        euclidean = math.sqrt(dx * dx + dy * dy) / max_eu
        manhattan = (abs(int(opp[0]) - int(own[0])) +
                     abs(int(opp[1]) - int(own[1]))) / max_man

        own_x, own_y = own[0] / max_d, own[1] / max_d
        opp_x, opp_y = opp[0] / max_d, opp[1] / max_d

        wall_n = own[1] / max_d
        wall_e = (max_d - own[0]) / max_d
        wall_s = (max_d - own[1]) / max_d
        wall_w = own[0] / max_d

        steps_left = max(0.0, float(self.max_steps - self.step_count)) / self.max_steps

        # Per-direction block sensors
        ox, oy = int(own[0]), int(own[1])
        blk_n = 1.0 if self._is_blocked(ox,     oy - 1) else 0.0
        blk_e = 1.0 if self._is_blocked(ox + 1, oy    ) else 0.0
        blk_s = 1.0 if self._is_blocked(ox,     oy + 1) else 0.0
        blk_w = 1.0 if self._is_blocked(ox - 1, oy    ) else 0.0

        return np.array([
            math.sin(angle), math.cos(angle),
            euclidean,
            own_x, own_y,
            opp_x, opp_y,
            wall_n, wall_e, wall_s, wall_w,
            steps_left,
            manhattan,
            float(opp_last[0]),
            float(opp_last[1]),
            blk_n, blk_e, blk_s, blk_w,
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._obs_for(self.role)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        third = max(2, self.grid_size // 3)
        self.predator_pos = np.array([
            np.random.randint(0, third),
            np.random.randint(0, third),
        ])
        self.prey_pos = np.array([
            np.random.randint(self.grid_size - third, self.grid_size),
            np.random.randint(self.grid_size - third, self.grid_size),
        ])

        # Fresh obstacle layout each episode → generalises to any map
        self.obstacles = self._generate_obstacles(
            self.predator_pos, self.prey_pos)

        self.step_count         = 0
        self.last_outcome       = None
        self.pred_trail         = [tuple(self.predator_pos)]
        self.prey_trail         = [tuple(self.prey_pos)]
        self._pred_last_move[:] = 0
        self._prey_last_move[:] = 0
        self._prev_euclid       = self._euclid()
        self._dist_window.clear()
        self._dist_window.append(self._prev_euclid)

        return self._get_obs(), {}

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, action):
        if isinstance(action, np.ndarray): action = int(action.item())
        elif not isinstance(action, int):  action = int(action)

        self.step_count += 1

        # Opponent action
        opp_role = "prey" if self.role == "predator" else "predator"
        opp_obs  = self._obs_for(opp_role)
        if self.opponent_model is not None:
            opp_action, _ = self.opponent_model.predict(
                opp_obs, deterministic=False)
            opp_action = int(opp_action)
        else:
            opp_action = self.action_space.sample()

        pred_action = action     if self.role == "predator" else opp_action
        prey_action = opp_action if self.role == "predator" else action

        # Pre-move snapshot for per-axis delta
        pred_pos_before = self.predator_pos.copy()
        prey_pos_before = self.prey_pos.copy()

        pred_move = np.array(self._moves[pred_action], dtype=np.float32)
        prey_move = np.array(self._moves[prey_action], dtype=np.float32)
        self._pred_last_move = pred_move.copy()
        self._prey_last_move = prey_move.copy()

        # Move with obstacle collision — stay in place if blocked
        def try_move(pos, move):
            nx = int(pos[0]) + int(move[0])
            ny = int(pos[1]) + int(move[1])
            return np.array([nx, ny]) if not self._is_blocked(nx, ny) else pos.copy()

        self.predator_pos = try_move(self.predator_pos, pred_move)
        self.prey_pos     = try_move(self.prey_pos,     prey_move)

        self.pred_trail.append(tuple(self.predator_pos))
        self.prey_trail.append(tuple(self.prey_pos))
        if len(self.pred_trail) > 40: self.pred_trail.pop(0)
        if len(self.prey_trail) > 40: self.prey_trail.pop(0)

        # Outcomes
        curr_euclid = self._euclid()
        caught      = np.array_equal(self.predator_pos, self.prey_pos)
        timed_out   = self.step_count >= self.max_steps
        terminated  = caught or timed_out

        self._dist_window.append(curr_euclid)

        # Per-axis gap deltas
        px_b, py_b = int(pred_pos_before[0]), int(pred_pos_before[1])
        bx_b, by_b = int(prey_pos_before[0]), int(prey_pos_before[1])
        px,   py   = int(self.predator_pos[0]), int(self.predator_pos[1])
        bx,   by   = int(self.prey_pos[0]),     int(self.prey_pos[1])

        dx_before = abs(px_b - bx_b);  dy_before = abs(py_b - by_b)
        dx_after  = abs(px   - bx);    dy_after  = abs(py   - by)

        pred_dx_closed = dx_before - dx_after
        pred_dy_closed = dy_before - dy_after
        prey_dx_opened = dx_after  - dx_before
        prey_dy_opened = dy_after  - dy_before

        # ── PREDATOR REWARD ────────────────────────────────────────────────
        if self.role == "predator":
            if caught:
                reward = 250.0
                self.last_outcome = {"result": "caught", "steps": self.step_count}
            elif timed_out:
                reward = -80.0
                self.last_outcome = {"result": "survived", "steps": self.step_count}
            else:
                reward = (pred_dx_closed + pred_dy_closed) * 2.5 - 0.15

                # Tail-position penalty
                pmx = int(self._prey_last_move[0])
                pmy = int(self._prey_last_move[1])
                if abs(pmx) + abs(pmy) > 0 and curr_euclid <= 2.2:
                    if px == bx - pmx and py == by - pmy:
                        reward -= 1.8

                # Stagnation — looser threshold (obstacles legitimately cause pauses)
                if len(self._dist_window) >= 12:
                    spread = max(self._dist_window) - min(self._dist_window)
                    if spread < 0.5:
                        reward -= 0.5

        # ── PREY REWARD ────────────────────────────────────────────────────
        else:
            if caught:
                reward = -200.0
                self.last_outcome = {"result": "caught", "steps": self.step_count}
            elif timed_out:
                reward = 300.0
                self.last_outcome = {"result": "survived", "steps": self.step_count}
            else:
                axis_reward = (prey_dx_opened + prey_dy_opened) * 3.5
                if curr_euclid < 3.0:
                    axis_reward *= 2.0
                reward = axis_reward + 0.4

        info = {}
        if terminated and self.last_outcome:
            info["outcome"] = self.last_outcome

        return self._get_obs(), reward, terminated, False, info

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption(
                f"Arena 2 — Obstacle Course  [{self.grid_size}×{self.grid_size}]")
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        cell = max(1, self.window_size // self.grid_size)
        self.window.fill(self._COL_BG)

        # Grid lines
        for x in range(0, self.window_size, cell):
            pygame.draw.line(self.window, self._COL_GRID,
                             (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, cell):
            pygame.draw.line(self.window, self._COL_GRID,
                             (0, y), (self.window_size, y))

        # Obstacles — solid blocks with a lighter top/left edge for depth
        for (ox, oy) in self.obstacles:
            rx, ry = ox * cell, oy * cell
            pygame.draw.rect(self.window, self._COL_OBSTACLE,
                             (rx, ry, cell, cell))
            pygame.draw.line(self.window, self._COL_OBS_EDGE,
                             (rx, ry), (rx + cell - 1, ry), 1)
            pygame.draw.line(self.window, self._COL_OBS_EDGE,
                             (rx, ry), (rx, ry + cell - 1), 1)

        # Trails
        n_pred = max(len(self.pred_trail), 1)
        for i, (tx, ty) in enumerate(self.pred_trail[:-1]):
            alpha = int(200 * (i / n_pred))
            r = min(255, self._COL_TRAIL_P[0])
            g = max(0,   self._COL_TRAIL_P[1] - (200 - alpha) // 2)
            b = max(0,   self._COL_TRAIL_P[2] - (200 - alpha) // 2)
            pygame.draw.rect(self.window, (r, g, b),
                             (tx * cell, ty * cell, cell, cell))

        n_prey = max(len(self.prey_trail), 1)
        for i, (tx, ty) in enumerate(self.prey_trail[:-1]):
            alpha = int(200 * (i / n_prey))
            r = max(0,   self._COL_TRAIL_PR[0] - (200 - alpha) // 2)
            g = min(255, self._COL_TRAIL_PR[1])
            b = min(255, self._COL_TRAIL_PR[2])
            pygame.draw.rect(self.window, (r, g, b),
                             (tx * cell, ty * cell, cell, cell))

        # Predator
        px, py = self.predator_pos
        pygame.draw.rect(self.window, self._COL_PRED,
                         (px * cell, py * cell, cell, cell))
        cx = px * cell + cell // 2
        cy = py * cell + cell // 2
        rv = max(2, cell // 3)
        pygame.draw.polygon(self.window, (160, 30, 20),
                             [(cx, cy - rv), (cx + rv, cy + rv), (cx - rv, cy + rv)])

        # Prey
        bx, by = self.prey_pos
        pygame.draw.rect(self.window, self._COL_PREY,
                         (bx * cell, by * cell, cell, cell))
        pygame.draw.circle(self.window, (25, 90, 170),
                           (bx * cell + cell // 2, by * cell + cell // 2),
                           max(2, cell // 3))

        # Bearing arrow
        self._draw_bearing_arrow(cell)

        # HUD
        font = pygame.font.SysFont("Arial", 13, bold=True)
        self.window.blit(font.render("▲ Predator", True, self._COL_PRED), (8,  8))
        self.window.blit(font.render("● Prey",     True, self._COL_PREY), (8, 26))
        self.window.blit(font.render("■ Obstacle", True, self._COL_OBS_EDGE), (8, 44))
        step_surf = font.render(
            f"Step {self.step_count} / {self.max_steps}", True, (80, 80, 80))
        self.window.blit(step_surf,
                         (self.window_size - step_surf.get_width() - 8, 8))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_bearing_arrow(self, cell):
        px, py = self.predator_pos
        bx, by = self.prey_pos
        dx = float(bx - px)
        dy = float(by - py)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return
        s   = cell * 1.6
        pcx = px * cell + cell // 2
        pcy = py * cell + cell // 2
        pygame.draw.line(self.window, (200, 200, 40),
                         (pcx, pcy),
                         (int(pcx + dx / dist * s),
                          int(pcy + dy / dist * s)), 2)

    def draw_game_over(self, message: str):
        if not pygame.font.get_init():
            pygame.font.init()

        font     = pygame.font.SysFont("Arial", 36, bold=True)
        btn_font = pygame.font.SysFont("Arial", 18, bold=True)

        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(165)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, (0, 0))

        if   "PREDATOR WINS" in message.upper(): color = self._COL_CAUGHT
        elif "PREY WINS"     in message.upper(): color = self._COL_SURVIVED
        else:                                     color = (210, 210, 210)

        text = font.render(message, True, color)
        self.window.blit(text,
                         (self.window_size // 2 - text.get_width() // 2, 220))

        self.replay_rect = pygame.Rect(170, 310, 110, 42)
        self.close_rect  = pygame.Rect(320, 310, 110, 42)
        pygame.draw.rect(self.window, (52, 152, 219), self.replay_rect,
                         border_radius=6)
        pygame.draw.rect(self.window, (231, 76, 60),  self.close_rect,
                         border_radius=6)
        self.window.blit(btn_font.render("Replay", True, (255, 255, 255)),
                         (self.replay_rect.centerx - 30,
                          self.replay_rect.centery - 10))
        self.window.blit(btn_font.render("Close",  True, (255, 255, 255)),
                         (self.close_rect.centerx  - 24,
                          self.close_rect.centery  - 10))
        pygame.display.flip()