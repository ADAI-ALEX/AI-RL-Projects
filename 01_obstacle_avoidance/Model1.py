# Model1.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ObstacleGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        
        # 12% obstacle density
        self.obstacle_density = 0.12 
        self.num_obstacles = int((self.grid_size ** 2) * self.obstacle_density)
        
        self.render_mode = render_mode
        self.window_size = 600
        self.window = None
        self.clock = None
        
        self.replay_rect = None
        self.close_rect = None

        self.action_space = spaces.Discrete(4)
        
        # 8-Sensors: 4 for the Compass (Goal Dir) + 4 for the Radar (Obstacle Dir)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        dx, dy = gx - ax, gy - ay
        
        compass_up = 1 if dy < 0 else 0
        compass_right = 1 if dx > 0 else 0
        compass_down = 1 if dy > 0 else 0
        compass_left = 1 if dx < 0 else 0
        compass = [compass_up, compass_right, compass_down, compass_left]

        radar = [0, 0, 0, 0]
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)] 
        for i, move in enumerate(moves):
            nx, ny = ax + move[0], ay + move[1]
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                radar[i] = 1 
            elif [nx, ny] in self.obstacles:
                radar[i] = 1 

        return np.array(compass + radar, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        
        self.visited_counts = {}
        self.step_count = 0
        # Give it plenty of time to figure out a path
        self.max_steps = self.grid_size * 10 
        
        obstacle_set = set()
        while len(obstacle_set) < self.num_obstacles:
            x = np.random.randint(self.grid_size)
            y = np.random.randint(self.grid_size)
            if (x < 3 and y < 3) or (x > self.grid_size - 4 and y > self.grid_size - 4):
                continue
            obstacle_set.add((x, y))
            
        self.obstacles = [list(obs) for obs in obstacle_set]
        
        return self._get_obs(), {}

    def step(self, action):
        if isinstance(action, np.ndarray): action = action.item() 
        elif not isinstance(action, int): action = int(action)

        prev_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        self.step_count += 1

        move = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[action]
        new_pos = self.agent_pos + np.array(move)

        terminated, reward, msg = False, 0, ""

        pos_tuple = tuple(new_pos)
        self.visited_counts[pos_tuple] = self.visited_counts.get(pos_tuple, 0) + 1

        # We only kill it for looping if it gets TRULY stuck (15 visits to the same tile)
        if self.visited_counts[pos_tuple] > 15 or self.step_count > self.max_steps:
            reward, terminated, msg = -10, True, "Loop Error!" 
            self.last_outcome = 0
        elif new_pos[0] < 0 or new_pos[0] >= self.grid_size or new_pos[1] < 0 or new_pos[1] >= self.grid_size:
            reward, terminated, msg = -10, True, "Hit Outer Wall!"
            self.last_outcome = 0
        elif list(new_pos) in self.obstacles:
            reward, terminated, msg = -10, True, "Hit an Obstacle!"
            self.last_outcome = 0
        elif np.array_equal(new_pos, self.goal_pos):
            reward, terminated, msg = 500, True, "AI WON!"
            self.last_outcome = 1
        else:
            self.agent_pos = new_pos
            curr_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            
            # --- THE "BACK TO BASICS" REWARD ---
            reward = -0.1 # Just a tiny tax for taking a step. Hitting the goal faster = better score.
            
            # Give a small cookie for walking the right way, but ZERO punishment for backtracking
            if curr_dist < prev_dist: 
                reward += 0.2 

        info = {"msg": msg} if msg else {}
        if terminated: info["outcome"] = getattr(self, "last_outcome", 0)
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.set_caption(f"AI Navigator - {self.grid_size}x{self.grid_size}")
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()

            cell_size = max(1, self.window_size // self.grid_size)
            self.window.fill((230, 230, 230))

            if self.grid_size <= 40:
                for x in range(0, self.window_size, cell_size):
                    pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, self.window_size))
                for y in range(0, self.window_size, cell_size):
                    pygame.draw.line(self.window, (200, 200, 200), (0, y), (self.window_size, y))

            for obs in self.obstacles:
                pygame.draw.rect(self.window, (50, 50, 50), (obs[0]*cell_size, obs[1]*cell_size, cell_size, cell_size))
            pygame.draw.rect(self.window, (46, 204, 113), (self.goal_pos[0]*cell_size, self.goal_pos[1]*cell_size, cell_size, cell_size))
            pygame.draw.rect(self.window, (52, 152, 219), (self.agent_pos[0]*cell_size, self.agent_pos[1]*cell_size, cell_size, cell_size))

            pygame.display.flip()
            self.clock.tick(30)

    def draw_game_over(self, message):
        if not pygame.font.get_init(): pygame.font.init()
        font = pygame.font.SysFont("Arial", 40, bold=True)
        btn_font = pygame.font.SysFont("Arial", 20, bold=True)
        
        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, (0,0))
        
        if "WON" in message: color = (46, 204, 113)
        elif "Loop" in message: color = (243, 156, 18) 
        else: color = (231, 76, 60) 
            
        text = font.render(message, True, color)
        self.window.blit(text, (self.window_size//2 - text.get_width()//2, 200))
        
        self.replay_rect = pygame.Rect(180, 300, 100, 40)
        self.close_rect = pygame.Rect(320, 300, 100, 40)
        
        pygame.draw.rect(self.window, (52, 152, 219), self.replay_rect, border_radius=5)
        pygame.draw.rect(self.window, (231, 76, 60), self.close_rect, border_radius=5)
        
        r_text = btn_font.render("Replay", True, (255,255,255))
        c_text = btn_font.render("Close", True, (255,255,255))
        self.window.blit(r_text, (self.replay_rect.centerx - r_text.get_width()//2, self.replay_rect.centery - r_text.get_height()//2))
        self.window.blit(c_text, (self.close_rect.centerx - c_text.get_width()//2, self.close_rect.centery - c_text.get_height()//2))
        
        pygame.display.flip()