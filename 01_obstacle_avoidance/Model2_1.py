import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ObstacleGridEnv(gym.Env):
    """
    Model 2.1 — ObstacleGridEnv
    ────────────────────────────────────────────────────────────────────────────
    Environment is identical to Model 2.  The key training improvements in this
    version live in the Controller (chunked training + best-model rollback).

    One small environment tweak: the observation space bounds are tightened to
    exactly [-1, 1] / [0, 1] per-sensor instead of a blanket [-1, 1] for every
    channel.  This gives the PPO normaliser cleaner statistics and reduces the
    gradient noise that contributes to policy collapse.
    ────────────────────────────────────────────────────────────────────────────
    Observation vector (13 sensors):
      [0-1]   Continuous normalised goal direction (dx, dy)  ∈ [-1, 1]
      [2-9]   Ray-cast distances, 8 dirs N/NE/E/SE/S/SW/W/NW ∈ [0, 1]
      [10-11] Normalised agent position (x, y)               ∈ [0, 1]
      [12]    Revisit signal (saturates at 5 visits → 1.0)   ∈ [0, 1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=20):
        super().__init__()
        self.grid_size = grid_size

        self.obstacle_density = 0.12
        self.num_obstacles = int((self.grid_size ** 2) * self.obstacle_density)

        self.render_mode = render_mode
        self.window_size = 600
        self.window = None
        self.clock = None
        self.replay_rect = None
        self.close_rect = None

        self.action_space = spaces.Discrete(4)

        # Tighter per-sensor bounds — helps PPO's internal obs normaliser
        low  = np.array([-1, -1] + [0]*8 + [0, 0, 0], dtype=np.float32)
        high = np.array([ 1,  1] + [1]*8 + [1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 8-direction ray vectors: N NE E SE S SW W NW
        self._ray_dirs = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

        # Internal state
        self.obstacle_set: set  = set()
        self.obstacles:    list = []
        self.visited_counts: dict = {}
        self.agent_pos = np.array([0, 0])
        self.goal_pos  = np.array([grid_size - 1, grid_size - 1])
        self.step_count = 0
        self.max_steps  = grid_size * 15

    # ── SENSORS ────────────────────────────────────────────────────────────────

    def _cast_ray(self, ax: int, ay: int, ddx: int, ddy: int) -> float:
        for dist in range(1, self.grid_size):
            nx = ax + ddx * dist
            ny = ay + ddy * dist
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                return (dist - 1) / (self.grid_size - 1)
            if (nx, ny) in self.obstacle_set:
                return (dist - 1) / (self.grid_size - 1)
        return 1.0

    def _get_obs(self) -> np.ndarray:
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        gx, gy = int(self.goal_pos[0]),  int(self.goal_pos[1])
        max_d  = self.grid_size - 1

        norm_dx = (gx - ax) / max_d
        norm_dy = (gy - ay) / max_d

        rays = [self._cast_ray(ax, ay, ddx, ddy) for ddx, ddy in self._ray_dirs]

        norm_x = ax / max_d
        norm_y = ay / max_d

        visits = self.visited_counts.get((ax, ay), 0)
        revisit_signal = min(visits / 5.0, 1.0)

        return np.array([norm_dx, norm_dy] + rays + [norm_x, norm_y, revisit_signal],
                        dtype=np.float32)

    # ── LIFECYCLE ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0])
        self.goal_pos  = np.array([self.grid_size - 1, self.grid_size - 1])
        self.visited_counts = {}
        self.step_count = 0
        self.max_steps  = self.grid_size * 15

        obstacle_set: set = set()
        while len(obstacle_set) < self.num_obstacles:
            x = np.random.randint(self.grid_size)
            y = np.random.randint(self.grid_size)
            if (x < 3 and y < 3) or (x > self.grid_size - 4 and y > self.grid_size - 4):
                continue
            obstacle_set.add((x, y))

        self.obstacle_set = obstacle_set
        self.obstacles    = [list(o) for o in obstacle_set]

        return self._get_obs(), {}

    def step(self, action):
        if isinstance(action, np.ndarray): action = action.item()
        elif not isinstance(action, int):  action = int(action)

        prev_dist = (abs(self.agent_pos[0] - self.goal_pos[0]) +
                     abs(self.agent_pos[1] - self.goal_pos[1]))
        self.step_count += 1

        move    = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[action]
        new_pos = self.agent_pos + np.array(move)

        terminated, reward, msg = False, 0.0, ""

        if (new_pos[0] < 0 or new_pos[0] >= self.grid_size or
                new_pos[1] < 0 or new_pos[1] >= self.grid_size):
            reward, terminated, msg = -10.0, True, "Hit Outer Wall!"
            self.last_outcome = 0

        elif (int(new_pos[0]), int(new_pos[1])) in self.obstacle_set:
            reward, terminated, msg = -10.0, True, "Hit an Obstacle!"
            self.last_outcome = 0

        elif np.array_equal(new_pos, self.goal_pos):
            reward, terminated, msg = 500.0, True, "AI WON!"
            self.last_outcome = 1

        else:
            self.agent_pos = new_pos
            pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))

            visit_count = self.visited_counts.get(pos_tuple, 0) + 1
            self.visited_counts[pos_tuple] = visit_count

            curr_dist = (abs(self.agent_pos[0] - self.goal_pos[0]) +
                         abs(self.agent_pos[1] - self.goal_pos[1]))

            reward = -0.1
            if curr_dist < prev_dist:
                reward += 0.3
            elif curr_dist > prev_dist:
                reward -= 0.1

            if visit_count > 1:
                reward -= 0.3 * (visit_count - 1)

            if visit_count > 8:
                reward, terminated, msg = -10.0, True, "Loop Error!"
                self.last_outcome = 0
            elif self.step_count > self.max_steps:
                reward, terminated, msg = -10.0, True, "Timeout!"
                self.last_outcome = 0

        info = {"msg": msg} if msg else {}
        if terminated:
            info["outcome"] = getattr(self, "last_outcome", 0)

        return self._get_obs(), reward, terminated, False, info

    # ── RENDERING ──────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.set_caption(f"AI Navigator v2.1 - {self.grid_size}x{self.grid_size}")
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock  = pygame.time.Clock()

            cell_size = max(1, self.window_size // self.grid_size)
            self.window.fill((230, 230, 230))

            if self.grid_size <= 40:
                for x in range(0, self.window_size, cell_size):
                    pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, self.window_size))
                for y in range(0, self.window_size, cell_size):
                    pygame.draw.line(self.window, (200, 200, 200), (0, y), (self.window_size, y))

            for obs in self.obstacles:
                pygame.draw.rect(self.window, (50, 50, 50),
                                 (obs[0]*cell_size, obs[1]*cell_size, cell_size, cell_size))
            pygame.draw.rect(self.window, (46, 204, 113),
                             (self.goal_pos[0]*cell_size, self.goal_pos[1]*cell_size,
                              cell_size, cell_size))
            pygame.draw.rect(self.window, (52, 152, 219),
                             (self.agent_pos[0]*cell_size, self.agent_pos[1]*cell_size,
                              cell_size, cell_size))

            pygame.display.flip()
            self.clock.tick(30)

    def draw_game_over(self, message):
        if not pygame.font.get_init(): pygame.font.init()
        font     = pygame.font.SysFont("Arial", 40, bold=True)
        btn_font = pygame.font.SysFont("Arial", 20, bold=True)

        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, (0, 0))

        if   "WON"  in message: color = (46,  204, 113)
        elif "Loop" in message: color = (243, 156,  18)
        else:                   color = (231,  76,  60)

        text = font.render(message, True, color)
        self.window.blit(text, (self.window_size//2 - text.get_width()//2, 200))

        self.replay_rect = pygame.Rect(180, 300, 100, 40)
        self.close_rect  = pygame.Rect(320, 300, 100, 40)

        pygame.draw.rect(self.window, (52, 152, 219), self.replay_rect, border_radius=5)
        pygame.draw.rect(self.window, (231,  76,  60), self.close_rect,  border_radius=5)

        r_text = btn_font.render("Replay", True, (255, 255, 255))
        c_text = btn_font.render("Close",  True, (255, 255, 255))
        self.window.blit(r_text, (self.replay_rect.centerx - r_text.get_width()//2,
                                  self.replay_rect.centery - r_text.get_height()//2))
        self.window.blit(c_text, (self.close_rect.centerx  - c_text.get_width()//2,
                                  self.close_rect.centery  - c_text.get_height()//2))

        pygame.display.flip()