import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import csv
import matplotlib.pyplot as plt
import os
import glob
import shutil
import importlib
import pygame
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EVAL_CHUNK_STEPS   = 10_000   # Evaluate accuracy every N steps
ROLLBACK_THRESHOLD = 5.0      # Revert to best model if accuracy drops this many %
CONFIG_FILE        = "ai_lab_config.txt"  # Persists last-used model selection
MIN_OUTCOMES_FOR_EVAL = 30    # Don't evaluate until we have this many episode results
PPO_LEARNING_RATE  = 0.0003   # PPO default (was 0.001 — much more stable)


# ==========================================
# 1. THE CALLBACK
# ==========================================
class TrainingCallback(BaseCallback):
    def __init__(self, chunk_steps, app, steps_before_chunk, total_steps_to_train):
        super().__init__(verbose=0)
        self.chunk_steps          = chunk_steps
        self.app                  = app
        self.steps_before_chunk   = steps_before_chunk    # steps already done in this run before this chunk
        self.total_steps_to_train = total_steps_to_train  # the full run target (e.g. 100k)
        self.progress_freq        = max(100, self.chunk_steps // 200)
        # Local counter — avoids using self.num_timesteps which accumulates across
        # the model's entire lifetime and would inflate progress on subsequent runs.
        self._local_steps = 0

    def _on_step(self) -> bool:
        if self.app.stop_requested:
            return False

        self._local_steps += 1
        if self._local_steps % self.progress_freq == 0:
            overall_steps_done = self.steps_before_chunk + self._local_steps
            percent = min(100.0, (overall_steps_done / self.total_steps_to_train) * 100)
            self.app.root.after(0, self.app._update_progress_ui,
                                percent, overall_steps_done, self.total_steps_to_train)

        for info in self.locals.get('infos', []):
            if 'outcome' in info:
                self.app.current_outcomes.append(info['outcome'])

        return True


# ==========================================
# 2. THE MAIN CONTROLLER GUI
# ==========================================
class AITrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Obstacle Avoidance — ADAI Studios")
        self.root.geometry("500x870")

        self.base_dir = "ai_models"
        os.makedirs(self.base_dir, exist_ok=True)

        self.stop_requested = False
        self.current_outcomes = deque(maxlen=200)   # Wider window for stability

        self.available_models = {
            "Model 1 (8-Sensor Radar)":    "Model1",
            "Model 2 (Ray-Casting)":       "Model2",
            "Model 2.1 (Stable Train)":    "Model2_1",
            "Model 3 (Spatial Awareness)": "Model3",
        }

        # --- TRAINING UI SECTION ---
        tk.Label(root, text="Master Training Controller",
                 font=("Arial", 16, "bold")).pack(pady=10)

        tk.Label(root, text="Select AI Architecture:",
                 font=("Arial", 10, "bold")).pack()
        self.model_combo = ttk.Combobox(root, state="readonly",
                                        values=list(self.available_models.keys()), width=30)
        # Default to last-used model, falling back to the first entry
        last_model = self._load_last_model()
        model_names = list(self.available_models.keys())
        default_index = model_names.index(last_model) if last_model in model_names else 0
        self.model_combo.current(default_index)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        self.model_combo.pack(pady=5)

        tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", pady=10, padx=20)

        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var, fg="#555").pack()

        self.acc_var = tk.StringVar()
        tk.Label(root, textvariable=self.acc_var,
                 font=("Arial", 11, "bold"), fg="#1565c0").pack(pady=2)

        # Best accuracy display
        self.best_acc_var = tk.StringVar(value="Best Accuracy: N/A")
        tk.Label(root, textvariable=self.best_acc_var,
                 font=("Arial", 10), fg="#2e7d32").pack(pady=2)

        # Rollback event label
        self.rollback_var = tk.StringVar(value="")
        self.rollback_label = tk.Label(root, textvariable=self.rollback_var,
                                       font=("Arial", 9, "italic"), fg="#e65100")
        self.rollback_label.pack(pady=2)

        tk.Label(root, text="Steps per Training Run:").pack(pady=(5, 0))
        self.steps_var = tk.StringVar(value="100000")
        ttk.Combobox(root, textvariable=self.steps_var,
                     values=["10000", "50000", "100000", "250000", "1000000"],
                     width=15).pack(pady=5)

        tk.Label(root, text="Training Grid Size (NxN):").pack(pady=(5, 0))
        self.train_grid_size_var = tk.StringVar(value="20")
        ttk.Combobox(root, textvariable=self.train_grid_size_var,
                     values=["10", "20", "30", "40", "50"], width=15).pack(pady=5)

        # Rollback threshold control
        tk.Label(root, text="Rollback Threshold (%):").pack(pady=(5, 0))
        self.rollback_thresh_var = tk.StringVar(value=str(ROLLBACK_THRESHOLD))
        ttk.Combobox(root, textvariable=self.rollback_thresh_var,
                     values=["3", "5", "8", "10", "15"],
                     width=15).pack(pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.train_btn = tk.Button(btn_frame, text="Start Training",
                                   command=self.start_training, height=2, width=15,
                                   bg="#e1f5fe", font=("Arial", 10, "bold"))
        self.train_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop Early",
                                  command=self.stop_training, state=tk.DISABLED,
                                  height=2, width=15, bg="#ffebee",
                                  font=("Arial", 10, "bold"))
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.progress = ttk.Progressbar(root, orient="horizontal",
                                        length=350, mode="determinate")

        self.separator = tk.Frame(root, height=2, bd=1, relief="sunken")
        self.separator.pack(fill="x", pady=15, padx=20)

        # --- SIMULATION UI SECTION ---
        self.graph_btn = tk.Button(root, text="📈 Show Accuracy Graph",
                                   command=self.show_graph, width=30,
                                   font=("Arial", 10))
        self.graph_btn.pack(pady=5)

        tk.Label(root, text="Select Checkpoint to Visualize:").pack(pady=(10, 0))
        self.checkpoint_combo = ttk.Combobox(root, state="readonly", width=30)
        self.checkpoint_combo.pack(pady=5)

        tk.Label(root, text="Simulation Grid Size (NxN):").pack(pady=(5, 0))
        self.sim_grid_size_var = tk.StringVar(value="100")
        ttk.Combobox(root, textvariable=self.sim_grid_size_var,
                     values=["20", "30", "40", "50", "100", "150", "200"],
                     width=15).pack(pady=5)

        self.sim_btn = tk.Button(root, text="▶ Watch AI Play",
                                 command=self.simulate_model, height=2, width=30,
                                 bg="#c8e6c9", font=("Arial", 10, "bold"))
        self.sim_btn.pack(pady=10)

        self.reset_btn = tk.Button(root, text="⚠ Wipe Current Model Logs",
                                   command=self.clear_models,
                                   bg="#ffcdd2", fg="#c62828",
                                   font=("Arial", 9, "bold"))
        self.reset_btn.pack(pady=10)

        self.on_model_change()

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def get_current_setup(self):
        ui_name     = self.model_combo.get()
        module_name = self.available_models[ui_name]
        model_dir   = os.path.join(self.base_dir, module_name)
        log_file    = os.path.join(model_dir, "accuracy_log.csv")
        return module_name, model_dir, log_file

    def on_model_change(self, event=None):
        module_name, self.active_dir, self.log_file = self.get_current_setup()
        os.makedirs(self.active_dir, exist_ok=True)
        self._save_last_model(self.model_combo.get())  # persist selection

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["Timestep", "Accuracy", "Event"])

        self.current_outcomes.clear()
        self.rollback_var.set("")
        self.best_acc_var.set("Best Accuracy: N/A")
        self.update_ui_state()

    def _save_last_model(self, ui_name: str):
        try:
            with open(CONFIG_FILE, "w") as f:
                f.write(ui_name)
        except OSError:
            pass

    def _load_last_model(self) -> str:
        try:
            with open(CONFIG_FILE, "r") as f:
                return f.read().strip()
        except OSError:
            return ""

    def update_ui_state(self):
        files = glob.glob(f"{self.active_dir}/model_*.zip")
        # Only count regular step checkpoints (not 'best')
        step_files = [f for f in files
                      if os.path.basename(f).replace("model_", "").replace(".zip", "").isdigit()]

        if not step_files:
            self.total_timesteps = 0
            steps = []
        else:
            steps = sorted([int(os.path.basename(f).replace("model_", "").replace(".zip", ""))
                            for f in step_files])
            self.total_timesteps = steps[-1] if steps else 0

        self.status_var.set(
            f"Total Training ({self.model_combo.get()}): {self.total_timesteps} steps")
        self.acc_var.set("Current Accuracy: N/A")

        self.checkpoint_combo['values'] = [f"Model Step {s}" for s in steps]
        if steps:
            self.checkpoint_combo.current(len(steps) - 1)
        else:
            self.checkpoint_combo.set('')

    def clear_models(self):
        if messagebox.askyesno("Confirm",
                               f"Delete all models and logs for {self.model_combo.get()}?"):
            shutil.rmtree(self.active_dir)
            self.on_model_change()

    # ── TRAINING ───────────────────────────────────────────────────────────────

    def start_training(self):
        try:
            steps = int(self.steps_var.get())
        except ValueError:
            return messagebox.showerror("Error", "Invalid steps.")

        self.stop_requested = False
        self.train_btn.config(state=tk.DISABLED, text="TRAINING...")
        self.stop_btn.config(state=tk.NORMAL)
        self.model_combo.config(state=tk.DISABLED)
        self.rollback_var.set("")

        self.progress['value'] = 0
        self.progress.pack(pady=5, before=self.separator)

        self.total_timesteps_before_run = self.total_timesteps
        threading.Thread(target=self._train_worker, args=(steps,), daemon=True).start()

    def stop_training(self):
        self.stop_requested = True
        self.stop_btn.config(state=tk.DISABLED, text="Stopping...")

    def _train_worker(self, total_steps_to_train: int):
        """
        Chunked training loop with best-model rollback.

        Instead of one big model.learn() call, we train in EVAL_CHUNK_STEPS
        increments.  After each chunk we measure win-rate accuracy.  If it has
        dropped more than rollback_threshold% below the all-time best, we reload
        the best checkpoint and continue training from there — discarding the
        bad update batch entirely.
        """
        module_name, _, _ = self.get_current_setup()

        try:
            mod       = importlib.import_module(module_name)
            grid_size = int(self.train_grid_size_var.get())
            env       = mod.ObstacleGridEnv(grid_size=grid_size)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Import Error", f"Failed to load {module_name}.py:\n{e}"))
            self.root.after(0, self._finish_ui)
            return

        rollback_threshold = float(self.rollback_thresh_var.get())
        best_model_path    = os.path.join(self.active_dir, "model_best")
        start_model_path   = os.path.join(self.active_dir, f"model_{self.total_timesteps}")

        # ── Load or create PPO model ───────────────────────────────────────────
        if os.path.exists(start_model_path + ".zip"):
            model = PPO.load(start_model_path, env=env, device="auto")
        else:
            model = PPO("MlpPolicy", env, verbose=0,
                        learning_rate=PPO_LEARNING_RATE,   # 0.0003 (stable default)
                        n_steps=2048,
                        batch_size=64,
                        device="auto")

        # Load best-accuracy memory from log (survives across training runs)
        best_accuracy        = self._read_best_accuracy_from_log()
        steps_trained_so_far = 0
        rollback_count       = 0

        # ── Chunked training loop ─────────────────────────────────────────────
        while steps_trained_so_far < total_steps_to_train and not self.stop_requested:

            chunk = min(EVAL_CHUNK_STEPS, total_steps_to_train - steps_trained_so_far)
            self.current_outcomes.clear()

            callback = TrainingCallback(chunk, self,
                                        steps_before_chunk=steps_trained_so_far,
                                        total_steps_to_train=total_steps_to_train)
            model.learn(total_timesteps=chunk,
                        callback=callback,
                        reset_num_timesteps=False)

            steps_trained_so_far    += chunk
            self.total_timesteps    += chunk
            abs_steps                = self.total_timesteps_before_run + steps_trained_so_far

            # ── Evaluate this chunk ───────────────────────────────────────────
            if len(self.current_outcomes) >= MIN_OUTCOMES_FOR_EVAL:
                accuracy = (sum(self.current_outcomes) /
                            len(self.current_outcomes)) * 100

                self.root.after(0, self.acc_var.set,
                                f"Current Accuracy: {accuracy:.1f}%")

                event_label = ""

                if accuracy > best_accuracy:
                    # ── New best — save checkpoint ─────────────────────────────
                    best_accuracy = accuracy
                    model.save(best_model_path)
                    self.root.after(0, self.best_acc_var.set,
                                    f"Best Accuracy: {best_accuracy:.1f}%  ✓")
                    self.root.after(0, self.rollback_var.set, "")
                    event_label = "best"

                elif accuracy < best_accuracy - rollback_threshold:
                    # ── Regression — roll back to best ────────────────────────
                    rollback_count += 1
                    event_label     = f"rollback#{rollback_count}"

                    msg = (f"⚠ Rollback #{rollback_count}  |  "
                           f"{accuracy:.1f}% < best {best_accuracy:.1f}% "
                           f"(−{best_accuracy - accuracy:.1f}%)")
                    self.root.after(0, self.rollback_var.set, msg)

                    if os.path.exists(best_model_path + ".zip"):
                        model = PPO.load(best_model_path, env=env, device="auto")

                # Log every chunk
                with open(self.log_file, "a", newline="") as f:
                    csv.writer(f).writerow([abs_steps, f"{accuracy:.2f}", event_label])

            # Save a regular step checkpoint for the graph / visualiser
            step_path = os.path.join(self.active_dir, f"model_{self.total_timesteps}")
            model.save(step_path)

        self.root.after(0, self._finish_ui)

    def _read_best_accuracy_from_log(self) -> float:
        """Scan the accuracy log for the highest value seen — persists across runs."""
        if not os.path.exists(self.log_file):
            return 0.0
        best = 0.0
        try:
            with open(self.log_file, "r") as f:
                reader = csv.reader(f)
                next(reader)   # skip header
                for row in reader:
                    if len(row) >= 2:
                        try:
                            val = float(row[1])
                            if val > best:
                                best = val
                        except ValueError:
                            pass
        except StopIteration:
            pass
        return best

    # ── UI CALLBACKS ───────────────────────────────────────────────────────────

    def _update_progress_ui(self, percent, overall_steps_done, total_steps):
        self.progress['value'] = percent
        self.status_var.set(f"Training... {percent:.1f}%  ({overall_steps_done:,} / {total_steps:,} steps)")

    def _update_accuracy_ui(self, accuracy):
        self.acc_var.set(f"Current Accuracy: {accuracy:.1f}%")

    def _finish_ui(self):
        self.train_btn.config(state=tk.NORMAL, text="Start Training")
        self.stop_btn.config(state=tk.DISABLED, text="Stop Early")
        self.model_combo.config(state="readonly")
        self.progress.pack_forget()
        self.update_ui_state()
        if not self.stop_requested:
            messagebox.showinfo("Done", "Training Run Finished!")

    # ── GRAPH ──────────────────────────────────────────────────────────────────

    def show_graph(self):
        if not os.path.exists(self.log_file):
            return messagebox.showinfo("No Data", "No training data yet.")

        steps, accs, events = [], [], []
        with open(self.log_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    try:
                        steps.append(int(row[0]))
                        accs.append(float(row[1]))
                        events.append(row[2] if len(row) > 2 else "")
                    except ValueError:
                        pass

        if not steps:
            return messagebox.showinfo("No Data", "No training data yet.")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, accs, color='#1565c0', linewidth=2, label="Win Rate %")

        # Mark best checkpoints
        best_x = [steps[i] for i, e in enumerate(events) if e == "best"]
        best_y = [accs[i]  for i, e in enumerate(events) if e == "best"]
        if best_x:
            ax.scatter(best_x, best_y, color='#2e7d32', zorder=5,
                       s=60, label="New Best", marker="^")

        # Mark rollbacks
        rb_x = [steps[i] for i, e in enumerate(events) if e.startswith("rollback")]
        rb_y = [accs[i]  for i, e in enumerate(events) if e.startswith("rollback")]
        if rb_x:
            ax.scatter(rb_x, rb_y, color='#e65100', zorder=5,
                       s=60, label="Rollback", marker="v")

        ax.set_xscale('log')
        ax.set_title(f"Performance: {self.model_combo.get()}", fontsize=13)
        ax.set_xlabel("Total Timesteps Trained (Log Scale)")
        ax.set_ylabel("Win Rate % (Accuracy)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # ── SIMULATION ─────────────────────────────────────────────────────────────

    def simulate_model(self):
        selection = self.checkpoint_combo.get()
        if not selection:
            return
        step = selection.split(" ")[-1]
        self.sim_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_sim_worker, args=(step,), daemon=True).start()

    def _run_sim_worker(self, step):
        module_name, _, _ = self.get_current_setup()
        mod       = importlib.import_module(module_name)
        grid_size = int(self.sim_grid_size_var.get())
        env       = mod.ObstacleGridEnv(render_mode="human", grid_size=grid_size)

        model_path = os.path.join(self.active_dir, f"model_{step}")
        model      = PPO.load(model_path)

        obs, _    = env.reset()
        env.render()
        sim_state = "PLAYING"

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.window = None
                    self.root.after(0, lambda: self.sim_btn.config(state=tk.NORMAL))
                    return

                if (sim_state == "GAME_OVER" and
                        event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                    if env.replay_rect and env.replay_rect.collidepoint(event.pos):
                        obs, _ = env.reset()
                        sim_state = "PLAYING"
                    elif env.close_rect and env.close_rect.collidepoint(event.pos):
                        pygame.quit()
                        env.window = None
                        self.root.after(0, lambda: self.sim_btn.config(state=tk.NORMAL))
                        return

            if sim_state == "PLAYING":
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if terminated or truncated:
                    sim_state = "GAME_OVER"
                    env.draw_game_over(info.get('msg', 'Finished'))

            time.sleep(0.01)


if __name__ == "__main__":
    root = tk.Tk()
    app  = AITrainerApp(root)
    root.mainloop()