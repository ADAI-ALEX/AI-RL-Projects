import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ObstacleGridEnv(gym.Env):
    """
    Model 3 — "Spatial Awareness"
    ────────────────────────────────────────────────────────────────────────────
    Built on Model 2.1, with four major new tools:

      1. DISTANCE MAGNITUDE   — A single 0→1 value for how far the goal is.
                                The agent can now weigh up whether to go around
                                a large obstacle or push straight through.

      2. ENHANCED LIDAR       — 8-direction ray-casts (inherited from Model 2.1),
                                plus 4 short-range wall-proximity flags for the
                                cardinal directions so the agent never "forgets"
                                what's immediately next to it.

      3. 7×7 MINI-MAP         — A 49-value snapshot of the world centred on the
                                agent.  Each cell encodes: 0 = free, 1 = obstacle
                                or out-of-bounds wall.  Lets the agent see corners,
                                U-traps, and corridor gaps several steps ahead.

      4. BREADCRUMB NEIGHBOURS — The visit-count of the 4 cardinal neighbours,
                                normalised 0→1 (saturates at 5 visits).  The agent
                                can see which tiles have already been explored and
                                actively prefer unvisited paths, eliminating loops
                                far more effectively than a single revisit scalar.

    ────────────────────────────────────────────────────────────────────────────
    Full observation vector  (67 sensors total):

      [0-1]    Continuous normalised goal direction (dx, dy)      ∈ [-1,  1]
      [2]      Normalised distance magnitude to goal              ∈ [ 0,  1]
      [3-10]   Ray-cast free distances, 8 dirs N/NE/E/SE/S/SW/W/NW ∈ [ 0, 1]
      [11-12]  Normalised agent position (x/max, y/max)           ∈ [ 0,  1]
      [13]     Current-cell revisit signal (sat. at 5)            ∈ [ 0,  1]
      [14-17]  Neighbour breadcrumb counts N/E/S/W (sat. at 5)    ∈ [ 0,  1]
      [18-66]  7×7 local mini-map (row-major, 0=free, 1=blocked)  ∈ [ 0,  1]
    ────────────────────────────────────────────────────────────────────────────
    """

    MINIMAP_SIZE = 7          # must be odd
    OBS_DIM      = 67

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=20):
        super().__init__()
        self.grid_size = grid_size

        self.obstacle_density = 0.12
        self.num_obstacles = int((self.grid_size ** 2) * self.obstacle_density)

        self.render_mode  = render_mode
        self.window_size  = 600
        self.window       = None
        self.clock        = None
        self.replay_rect  = None
        self.close_rect   = None

        self.action_space = spaces.Discrete(4)

        # Per-sensor bounds
        low  = np.array(
            [-1, -1]           # goal direction
            + [0]              # distance magnitude
            + [0] * 8          # LIDAR rays
            + [0, 0]           # position
            + [0]              # current-cell revisit
            + [0] * 4          # neighbour breadcrumbs
            + [0] * 49,        # 7×7 mini-map
            dtype=np.float32
        )
        high = np.array(
            [1,  1]
            + [1]
            + [1] * 8
            + [1, 1]
            + [1]
            + [1] * 4
            + [1] * 49,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 8-direction ray vectors: N NE E SE S SW W NW
        self._ray_dirs = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

        # Internal state
        self.obstacle_set:    set  = set()
        self.obstacles:       list = []
        self.visited_counts:  dict = {}
        self.agent_pos  = np.array([0, 0])
        self.goal_pos   = np.array([grid_size - 1, grid_size - 1])
        self.step_count = 0
        self.max_steps  = grid_size * 15

    # ── SENSORS ────────────────────────────────────────────────────────────────

    def _cast_ray(self, ax: int, ay: int, ddx: int, ddy: int) -> float:
        """Normalised free distance in one direction (0 = blocked next step, 1 = full clear)."""
        for dist in range(1, self.grid_size):
            nx, ny = ax + ddx * dist, ay + ddy * dist
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                return (dist - 1) / (self.grid_size - 1)
            if (nx, ny) in self.obstacle_set:
                return (dist - 1) / (self.grid_size - 1)
        return 1.0

    def _build_minimap(self, ax: int, ay: int) -> np.ndarray:
        """
        Return a flattened 7×7 grid centred on the agent.
        1 = obstacle or out-of-bounds wall, 0 = free.
        """
        half  = self.MINIMAP_SIZE // 2          # 3
        cells = []
        for dy in range(-half, half + 1):       # rows: north → south
            for dx in range(-half, half + 1):   # cols: west  → east
                wx, wy = ax + dx, ay + dy
                if wx < 0 or wx >= self.grid_size or wy < 0 or wy >= self.grid_size:
                    cells.append(1.0)           # treat out-of-bounds as wall
                elif (wx, wy) in self.obstacle_set:
                    cells.append(1.0)
                else:
                    cells.append(0.0)
        return np.array(cells, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        gx, gy = int(self.goal_pos[0]),  int(self.goal_pos[1])
        max_d  = (self.grid_size - 1) * np.sqrt(2)   # max possible Euclidean dist

        # 1. Goal direction (normalised)
        md     = self.grid_size - 1
        norm_dx = (gx - ax) / md
        norm_dy = (gy - ay) / md

        # 2. Distance magnitude (Euclidean, normalised)
        dist_mag = np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2) / max_d

        # 3. LIDAR — 8 ray-cast readings
        rays = [self._cast_ray(ax, ay, ddx, ddy) for ddx, ddy in self._ray_dirs]

        # 4. Normalised position
        norm_x = ax / md
        norm_y = ay / md

        # 5. Current-cell revisit signal
        curr_visits     = self.visited_counts.get((ax, ay), 0)
        revisit_signal  = min(curr_visits / 5.0, 1.0)

        # 6. Neighbour breadcrumbs — N, E, S, W
        neighbour_dirs  = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        breadcrumbs     = []
        for ddx, ddy in neighbour_dirs:
            nx, ny = ax + ddx, ay + ddy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                v = self.visited_counts.get((nx, ny), 0)
            else:
                v = 5   # treat wall as "fully explored" to deter walking into it
            breadcrumbs.append(min(v / 5.0, 1.0))

        # 7. 7×7 mini-map
        minimap = self._build_minimap(ax, ay)

        return np.concatenate([
            [norm_dx, norm_dy, dist_mag],
            rays,
            [norm_x, norm_y, revisit_signal],
            breadcrumbs,
            minimap
        ]).astype(np.float32)

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

    # ── STEP ───────────────────────────────────────────────────────────────────

    def step(self, action):
        if isinstance(action, np.ndarray): action = action.item()
        elif not isinstance(action, int):  action = int(action)

        prev_dist   = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        self.step_count += 1

        move    = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[action]
        new_pos = self.agent_pos + np.array(move)

        terminated, reward, msg = False, 0.0, ""

        # ── Terminal collision checks ──────────────────────────────────────────
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
            # ── Valid move ─────────────────────────────────────────────────────
            self.agent_pos = new_pos
            pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))

            visit_count = self.visited_counts.get(pos_tuple, 0) + 1
            self.visited_counts[pos_tuple] = visit_count

            curr_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

            # ── Reward shaping ─────────────────────────────────────────────────
            #
            #  Step tax        — encourages finishing faster.
            #  Progress bonus  — rewards genuine movement toward the goal.
            #                    Larger bonus when the agent is far away (it
            #                    matters more to make progress from distance).
            #  Backtrack nudge — small penalty for moving away from goal.
            #  Revisit penalty — escalating cost per extra visit to same tile.
            #                    Works alongside the breadcrumb sensor so the
            #                    agent can both observe and be punished for loops.
            #
            reward = -0.1                                           # step tax

            if curr_dist < prev_dist:
                # Scale progress bonus by how far away the goal still is —
                # early progress is cheap, late-game precision is rewarded more.
                progress_scale = 1.0 + (curr_dist / (self.grid_size * 2))
                reward += 0.3 * progress_scale
            elif curr_dist > prev_dist:
                reward -= 0.15                                      # backtrack nudge

            if visit_count > 1:
                reward -= 0.4 * (visit_count - 1)                  # escalating revisit

            # ── Loop / timeout termination ─────────────────────────────────────
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
                pygame.display.set_caption(
                    f"AI Navigator v3 — Spatial Awareness  [{self.grid_size}×{self.grid_size}]")
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock  = pygame.time.Clock()

            cell_size = max(1, self.window_size // self.grid_size)
            self.window.fill((230, 230, 230))

            # Grid lines
            if self.grid_size <= 40:
                for x in range(0, self.window_size, cell_size):
                    pygame.draw.line(self.window, (210, 210, 210), (x, 0), (x, self.window_size))
                for y in range(0, self.window_size, cell_size):
                    pygame.draw.line(self.window, (210, 210, 210), (0, y), (self.window_size, y))

            # Breadcrumb heat-map — pale yellow tint for visited tiles
            max_heat = 8
            for (bx, by), count in self.visited_counts.items():
                if count > 0:
                    alpha   = min(count / max_heat, 1.0)
                    r = int(255)
                    g = int(255 - alpha * 120)
                    b = int(200 - alpha * 200)
                    pygame.draw.rect(self.window, (r, g, b),
                                     (bx * cell_size, by * cell_size, cell_size, cell_size))

            # Obstacles
            for obs in self.obstacles:
                pygame.draw.rect(self.window, (50, 50, 50),
                                 (obs[0]*cell_size, obs[1]*cell_size, cell_size, cell_size))

            # 7×7 mini-map outline (visual debug aid)
            if self.grid_size <= 60:
                half = self.MINIMAP_SIZE // 2
                ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
                rx = (ax - half) * cell_size
                ry = (ay - half) * cell_size
                rw = self.MINIMAP_SIZE * cell_size
                pygame.draw.rect(self.window, (180, 100, 200), (rx, ry, rw, rw), 1)

            # Goal
            pygame.draw.rect(self.window, (46, 204, 113),
                             (self.goal_pos[0]*cell_size, self.goal_pos[1]*cell_size,
                              cell_size, cell_size))

            # Agent
            pygame.draw.rect(self.window, (52, 152, 219),
                             (self.agent_pos[0]*cell_size, self.agent_pos[1]*cell_size,
                              cell_size, cell_size))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def draw_game_over(self, message):
        if not pygame.font.get_init(): pygame.font.init()
        font     = pygame.font.SysFont("Arial", 40, bold=True)
        btn_font = pygame.font.SysFont("Arial", 20, bold=True)

        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(160)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, (0, 0))

        if   "WON"     in message: color = (46,  204, 113)
        elif "Loop"    in message: color = (243, 156,  18)
        elif "Timeout" in message: color = (155,  89, 182)
        else:                      color = (231,  76,  60)

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