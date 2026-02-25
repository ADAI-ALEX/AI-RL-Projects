"""
Arena1.py  —  Open Chase Arena
────────────────────────────────────────────────────────────────────────────────
Pure pursuit and evasion on a flat grid.  No obstacles.

Environment only — the controller (predprey_controller.py) drives training.
Add arenas by creating Arena2.py, Arena3.py … with the same ArenaEnv interface.

═══════════════════════════════════════════════════════════════════════════════
OBSERVATION VECTOR  (15 values)
───────────────────────────────────────────────────────────────────────────────
  Idx  Sensor                                              Range
  ───  ──────────────────────────────────────────────────  ─────────
   0   sin(compass bearing to opponent)                    [-1,  1]
   1   cos(compass bearing to opponent)                    [-1,  1]
   2   Euclidean distance to opponent (normalised)         [ 0,  1]
   3   Own X position (normalised)                         [ 0,  1]
   4   Own Y position (normalised)                         [ 0,  1]
   5   Opponent X position (normalised)                    [ 0,  1]
   6   Opponent Y position (normalised)                    [ 0,  1]
   7   Wall clearance — North                              [ 0,  1]
   8   Wall clearance — East                               [ 0,  1]
   9   Wall clearance — South                              [ 0,  1]
  10   Wall clearance — West                               [ 0,  1]
  11   Steps remaining (normalised)                        [ 0,  1]
  12   Manhattan distance to opponent (normalised)         [ 0,  1]
  13   Opponent last move — X component (-1 / 0 / +1)     [-1,  1]
  14   Opponent last move — Y component (-1 / 0 / +1)     [-1,  1]

═══════════════════════════════════════════════════════════════════════════════
ACTIONS:  0=North  1=East  2=South  3=West

═══════════════════════════════════════════════════════════════════════════════
REWARD DESIGN
───────────────────────────────────────────────────────────────────────────────
PREDATOR (targets ~50% long-run catch rate against trained prey)
  Per-axis reduction reward  — rewards closing EACH axis gap independently.
    With Euclidean delta, the predator learns to close the LARGER gap first
    (L-shape). With per-axis delta, both directions give equal reward, so
    the stochastic policy alternates → natural staircase/diagonal pursuit.
  Time pressure              — step cost keeps matches short.
  Tail-position penalty      — fires when predator is exactly one step
    behind the prey in its movement direction: the classic tail-follow lock.
  Stagnation penalty         — distance variance over 15 steps; catches
    situations where the tail penalty alone isn't enough.
  +250 catch  /  -80 timeout

PREY (targets ~50% long-run survival against trained predator)
  Per-axis opening reward    — symmetric mirror of predator axis reward.
    Prey is rewarded for increasing EACH axis gap, not just overall distance.
  Survival trickle           — +0.4/step so the prey always wants to live.
  Danger-zone multiplier     — when predator is ≤3 cells away, the per-axis
    reward is doubled. This creates strong escape urgency when cornered.
  No corner penalty          — previously backfired; removed.
  -200 caught  /  +300 timeout  (asymmetric: prey gets bigger timeout reward
    than predator's catch reward, giving prey a slight structural advantage
    to compensate for the harder task of surviving a finite grid forever)
────────────────────────────────────────────────────────────────────────────────
"""

import math
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ArenaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    _COL_BG       = (245, 245, 245)
    _COL_GRID     = (215, 215, 215)
    _COL_PRED     = (231,  76,  60)
    _COL_PREY     = ( 52, 152, 219)
    _COL_CAUGHT   = (243, 156,  18)
    _COL_SURVIVED = ( 46, 204, 113)
    _COL_TRAIL_P  = (252, 200, 195)
    _COL_TRAIL_PR = (190, 220, 255)

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

        # 15-dim observation
        low  = np.full(15, -1.0, dtype=np.float32)
        high = np.full(15,  1.0, dtype=np.float32)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            low[i], high[i] = 0.0, 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        # Episode state — populated fully in reset()
        self.predator_pos        = np.array([0, 0])
        self.prey_pos            = np.array([grid_size - 1, grid_size - 1])
        self.step_count          = 0
        self.max_steps           = grid_size * 20
        self.pred_trail: list    = []
        self.prey_trail: list    = []
        self.last_outcome        = None
        self._pred_last_move     = np.array([0.0, 0.0], dtype=np.float32)
        self._prey_last_move     = np.array([0.0, 0.0], dtype=np.float32)
        self._dist_window: deque = deque(maxlen=15)   # longer window for stagnation
        self._prev_euclid        = 0.0

    # ------------------------------------------------------------------

    def set_opponent_model(self, model):
        """Inject frozen opponent. Pass None for random opponent."""
        self.opponent_model = model

    def _euclid(self, a=None, b=None) -> float:
        if a is None: a = self.predator_pos
        if b is None: b = self.prey_pos
        return float(np.linalg.norm(a.astype(float) - b.astype(float)))

    # ------------------------------------------------------------------

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
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._obs_for(self.role)

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

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

        # Save pre-move positions for per-axis delta computation
        pred_pos_before = self.predator_pos.copy()
        prey_pos_before = self.prey_pos.copy()

        # Velocity recording
        pred_move = np.array(self._moves[pred_action], dtype=np.float32)
        prey_move = np.array(self._moves[prey_action], dtype=np.float32)
        self._pred_last_move = pred_move.copy()
        self._prey_last_move = prey_move.copy()

        # Apply moves
        self.predator_pos = np.clip(
            self.predator_pos + pred_move.astype(int), 0, self.grid_size - 1)
        self.prey_pos = np.clip(
            self.prey_pos + prey_move.astype(int), 0, self.grid_size - 1)

        self.pred_trail.append(tuple(self.predator_pos))
        self.prey_trail.append(tuple(self.prey_pos))
        if len(self.pred_trail) > 40: self.pred_trail.pop(0)
        if len(self.prey_trail) > 40: self.prey_trail.pop(0)

        # ── Outcome detection ──────────────────────────────────────────────────
        curr_euclid = self._euclid()
        caught      = np.array_equal(self.predator_pos, self.prey_pos)
        timed_out   = self.step_count >= self.max_steps
        terminated  = caught or timed_out

        self._dist_window.append(curr_euclid)

        # ── Per-axis gap deltas (the core of the diagonal-pursuit fix) ────────
        # Computed from position BEFORE this step's moves.
        px_b, py_b = int(pred_pos_before[0]), int(pred_pos_before[1])
        bx_b, by_b = int(prey_pos_before[0]), int(prey_pos_before[1])
        px,   py   = int(self.predator_pos[0]), int(self.predator_pos[1])
        bx,   by   = int(self.prey_pos[0]),     int(self.prey_pos[1])

        # Unsigned axis gap before and after
        dx_before = abs(px_b - bx_b)
        dy_before = abs(py_b - by_b)
        dx_after  = abs(px   - bx)
        dy_after  = abs(py   - by)

        # For predator: positive when it reduced the gap on each axis
        pred_dx_closed = dx_before - dx_after   # +1 = closed x-gap by 1
        pred_dy_closed = dy_before - dy_after   # +1 = closed y-gap by 1

        # For prey: positive when it opened the gap on each axis
        prey_dx_opened = dx_after  - dx_before
        prey_dy_opened = dy_after  - dy_before

        # ── PREDATOR REWARD ────────────────────────────────────────────────────
        if self.role == "predator":

            if caught:
                reward = 250.0
                self.last_outcome = {"result": "caught", "steps": self.step_count}

            elif timed_out:
                reward = -80.0
                self.last_outcome = {"result": "survived", "steps": self.step_count}

            else:
                # 1. Per-axis reduction reward  ─────────────────────────────────
                # Rewards closing EACH axis gap independently.  Since both the X
                # and Y moves give the same reward (1 cell closed = +2.5), the
                # policy is indifferent between horizontal and vertical progress →
                # during training the stochastic policy alternates naturally,
                # producing a staircase/diagonal path.
                reward = (pred_dx_closed + pred_dy_closed) * 2.5 - 0.1

                # 2. Tail-position penalty  ─────────────────────────────────────
                # The tail-follow lock occurs when predator sits exactly one step
                # behind prey in prey's movement direction.  Detect and penalise.
                pmx = int(self._prey_last_move[0])
                pmy = int(self._prey_last_move[1])
                if abs(pmx) + abs(pmy) > 0 and curr_euclid <= 2.2:
                    # Tail position: predator would be at (prey_pos - prey_move)
                    tail_x = bx - pmx
                    tail_y = by - pmy
                    if px == tail_x and py == tail_y:
                        reward -= 1.8   # strong nudge to break the lock

                # 3. Stagnation penalty  ────────────────────────────────────────
                # If distance has barely varied over 15 steps, something is wrong.
                if len(self._dist_window) >= 12:
                    spread = max(self._dist_window) - min(self._dist_window)
                    if spread < 0.8:
                        reward -= 0.7

        # ── PREY REWARD ────────────────────────────────────────────────────────
        else:
            if caught:
                reward = -200.0
                self.last_outcome = {"result": "caught", "steps": self.step_count}

            elif timed_out:
                reward = 300.0   # higher than predator's catch reward → structural advantage
                self.last_outcome = {"result": "survived", "steps": self.step_count}

            else:
                # 1. Per-axis opening reward  ────────────────────────────────────
                # Mirror of predator: reward increasing EACH axis gap.
                axis_reward = (prey_dx_opened + prey_dy_opened) * 3.5

                # 2. Danger-zone multiplier  ─────────────────────────────────────
                # When predator is close, urgency matters more.  Double the axis
                # reward so the prey prioritises escape above all else.
                if curr_euclid < 3.0:
                    axis_reward *= 2.0

                reward = axis_reward + 0.4   # +0.4/step just for surviving

        info = {}
        if terminated and self.last_outcome:
            info["outcome"] = self.last_outcome

        return self._get_obs(), reward, terminated, False, info

    # ── Rendering ──────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption(
                f"Arena 1 — Open Chase  [{self.grid_size}×{self.grid_size}]")
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        cell = max(1, self.window_size // self.grid_size)
        self.window.fill(self._COL_BG)

        for x in range(0, self.window_size, cell):
            pygame.draw.line(self.window, self._COL_GRID,
                             (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, cell):
            pygame.draw.line(self.window, self._COL_GRID,
                             (0, y), (self.window_size, y))

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
        r  = max(2, cell // 3)
        pygame.draw.polygon(self.window, (160, 30, 20),
                             [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)])

        # Prey
        bx, by = self.prey_pos
        pygame.draw.rect(self.window, self._COL_PREY,
                         (bx * cell, by * cell, cell, cell))
        pygame.draw.circle(self.window, (25, 90, 170),
                           (bx * cell + cell // 2, by * cell + cell // 2),
                           max(2, cell // 3))

        # Intercept arrows from predator
        self._draw_arrows(cell)

        # HUD
        font = pygame.font.SysFont("Arial", 14, bold=True)
        self.window.blit(font.render("▲ Predator",  True, self._COL_PRED), (8,  8))
        self.window.blit(font.render("● Prey",      True, self._COL_PREY), (8, 26))
        self.window.blit(font.render("→ Direct",    True, (180, 180, 30)), (8, 44))
        self.window.blit(font.render("⇒ Intercept", True, (255, 140,  0)), (8, 60))
        step_surf = font.render(
            f"Step {self.step_count} / {self.max_steps}", True, (80, 80, 80))
        self.window.blit(step_surf,
                         (self.window_size - step_surf.get_width() - 8, 8))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_arrows(self, cell):
        """Yellow = direct bearing to prey.  Orange = predicted intercept point."""
        px, py = self.predator_pos
        bx, by = self.prey_pos
        pcx = px * cell + cell // 2
        pcy = py * cell + cell // 2

        dx, dy = float(bx - px), float(by - py)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0.5:
            s = cell * 1.6
            pygame.draw.line(self.window, (180, 180, 30),
                             (pcx, pcy),
                             (int(pcx + dx / dist * s),
                              int(pcy + dy / dist * s)), 2)

        pmx, pmy = float(self._prey_last_move[0]), float(self._prey_last_move[1])
        if abs(pmx) + abs(pmy) > 0:
            ix, iy = bx + pmx, by + pmy
            dix, diy = float(ix - px), float(iy - py)
            di = math.sqrt(dix * dix + diy * diy)
            if di > 0.5:
                s = cell * 2.2
                pygame.draw.line(self.window, (255, 140, 0),
                                 (pcx, pcy),
                                 (int(pcx + dix / di * s),
                                  int(pcy + diy / di * s)), 2)

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