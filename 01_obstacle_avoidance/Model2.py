import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ObstacleGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=20):
        super().__init__()
        self.grid_size = grid_size

        # 12% obstacle density (unchanged)
        self.obstacle_density = 0.12
        self.num_obstacles = int((self.grid_size ** 2) * self.obstacle_density)

        self.render_mode = render_mode
        self.window_size = 600
        self.window = None
        self.clock = None

        self.replay_rect = None
        self.close_rect = None

        self.action_space = spaces.Discrete(4)

        # ── EXPANDED OBSERVATION SPACE (13 sensors) ────────────────────────────
        #
        #  [0-1]  Continuous normalised goal direction (dx, dy)
        #         Range -1..1.  Replaces the old binary compass.
        #         The agent now knows *how far* the goal is, not just "left or right".
        #
        #  [2-9]  Ray-cast distances in 8 directions: N NE E SE S SW W NW
        #         Value = fraction of grid_size that is clear (0 = immediately
        #         blocked, 1 = full grid-length clear).
        #         The agent can now "see ahead" through open corridors instead of
        #         being surprised by a wall one step away.
        #
        #  [10-11] Normalised agent position (x/max, y/max)
        #          Gives the agent map-awareness so it can learn edge/corner behaviour.
        #
        #  [12]   Revisit signal  –  how many times the agent has visited the
        #         current cell, capped and normalised to 0..1 (saturates at 5).
        #         This is the key anti-loop sensor: the agent can now OBSERVE
        #         that it is going in circles and learn to escape.
        #
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)

        # 8-direction ray vectors: N NE E SE S SW W NW
        self._ray_dirs = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

        # Internal state (populated in reset)
        self.obstacle_set: set = set()
        self.obstacles: list = []
        self.visited_counts: dict = {}
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])
        self.step_count = 0
        self.max_steps = grid_size * 15

    # ── SENSORS ────────────────────────────────────────────────────────────────

    def _cast_ray(self, ax: int, ay: int, ddx: int, ddy: int) -> float:
        """Return the normalised free distance along a direction.
           0.0  = the very next cell is blocked/wall
           1.0  = entire grid length is clear
        """
        for dist in range(1, self.grid_size):
            nx = ax + ddx * dist
            ny = ay + ddy * dist
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                return (dist - 1) / (self.grid_size - 1)
            if (nx, ny) in self.obstacle_set:   # O(1) lookup via set
                return (dist - 1) / (self.grid_size - 1)
        return 1.0

    def _get_obs(self) -> np.ndarray:
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        gx, gy = int(self.goal_pos[0]),  int(self.goal_pos[1])
        max_d  = self.grid_size - 1

        # Continuous normalised goal direction
        norm_dx = (gx - ax) / max_d
        norm_dy = (gy - ay) / max_d

        # 8 ray-cast readings
        rays = [self._cast_ray(ax, ay, ddx, ddy) for ddx, ddy in self._ray_dirs]

        # Normalised position
        norm_x = ax / max_d
        norm_y = ay / max_d

        # Revisit signal
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
        self.max_steps = self.grid_size * 15

        # Build obstacles – use a set for O(1) collision checks everywhere
        obstacle_set: set = set()
        while len(obstacle_set) < self.num_obstacles:
            x = np.random.randint(self.grid_size)
            y = np.random.randint(self.grid_size)
            # Protect the start corner (top-left) and goal corner (bottom-right)
            if (x < 3 and y < 3) or (x > self.grid_size - 4 and y > self.grid_size - 4):
                continue
            obstacle_set.add((x, y))

        self.obstacle_set = obstacle_set
        self.obstacles    = [list(o) for o in obstacle_set]  # kept for renderer

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

        # ── COLLISION / TERMINAL CHECKS ────────────────────────────────────────
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
            # ── VALID MOVE ─────────────────────────────────────────────────────
            self.agent_pos = new_pos
            pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))

            visit_count = self.visited_counts.get(pos_tuple, 0) + 1
            self.visited_counts[pos_tuple] = visit_count

            curr_dist = (abs(self.agent_pos[0] - self.goal_pos[0]) +
                         abs(self.agent_pos[1] - self.goal_pos[1]))

            # ── REWARD SHAPING ─────────────────────────────────────────────────
            #
            #  1. Small step tax encourages finishing faster.
            #  2. Progress bonus rewards genuine movement toward the goal.
            #  3. Small regression penalty nudges against needless backtracking.
            #  4. Escalating revisit penalty – this is the core loop-breaker.
            #     Each extra visit to a tile costs more, creating a "heat map"
            #     of recently visited cells that the agent learns to avoid.
            #
            reward = -0.1                                   # 1. step tax

            if curr_dist < prev_dist:
                reward += 0.3                               # 2. progress bonus
            elif curr_dist > prev_dist:
                reward -= 0.1                               # 3. regression nudge

            if visit_count > 1:                             # 4. revisit penalty
                reward -= 0.3 * (visit_count - 1)

            # ── SOFT LOOP TERMINATION (much tighter than Model1's 15) ─────────
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

    # ── RENDERING (identical to Model1) ────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.set_caption(f"AI Navigator v2 - {self.grid_size}x{self.grid_size}")
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