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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EVAL_CHUNK_STEPS        = 10_000
EVAL_EPISODES           = 10
PPO_LEARNING_RATE       = 0.0003
CONFIG_FILE             = "predprey_config.txt"

EMA_ALPHA               = 0.4   # weight of newest eval in EMA
ROLLBACK_PATIENCE       = 2     # consecutive drops before predator rollback

PREY_RATIO              = 1.5   # prey trains 50% more per cycle than predator
PREY_RECOVERY_THRESHOLD = 20.0  # if prey EMA below this, insert recovery burst
PREY_RECOVERY_STEPS     = 5_000


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class TrainingCallback(BaseCallback):
    def __init__(self, chunk_steps, app, steps_before, total_steps, label):
        super().__init__(verbose=0)
        self.chunk_steps    = chunk_steps
        self.app            = app
        self.steps_before   = steps_before
        self.total_steps    = total_steps
        self.label          = label
        self.progress_freq  = max(50, chunk_steps // 200)
        self._local_steps   = 0

    def _on_step(self) -> bool:
        if self.app.stop_requested:
            return False
        self._local_steps += 1
        if self._local_steps % self.progress_freq == 0:
            overall = self.steps_before + self._local_steps
            pct     = min(100.0, overall / self.total_steps * 100)
            self.app.root.after(
                0, self.app._update_progress_ui,
                pct, overall, self.total_steps, self.label)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
class AITrainerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Predator vs Prey — ADAI Studios")
        self.root.geometry("510x700")
        self.root.resizable(False, False)

        self.base_dir = "marl_models"
        os.makedirs(self.base_dir, exist_ok=True)

        self.stop_requested   = False
        self.pred_total_steps = 0
        self.prey_total_steps = 0

        self.available_arenas = {
            "Arena 1 — Open Chase":          "Arena1",
            "Arena 2 — Scattered Obstacles": "Arena2",
        }

        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        hdr = tk.Frame(root, bg="#1a237e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Predator vs Prey — Controller",
                 font=("Arial", 13, "bold"), bg="#1a237e",
                 fg="white").pack(side="left", padx=12, pady=7)
        tk.Label(hdr, text="Arena:", bg="#1a237e",
                 fg="#cfd8dc", font=("Arial", 9)).pack(side="left", padx=(20, 4))
        self.arena_combo = ttk.Combobox(
            hdr, state="readonly",
            values=list(self.available_arenas.keys()), width=22)
        saved = self._load_config()
        names = list(self.available_arenas.keys())
        self.arena_combo.current(names.index(saved) if saved in names else 0)
        self.arena_combo.bind("<<ComboboxSelected>>", self.on_arena_change)
        self.arena_combo.pack(side="left", pady=7)

        # ═══════════════════════════════════════════════════════════════════
        # METRICS PANEL
        # ═══════════════════════════════════════════════════════════════════
        met = tk.Frame(root, relief="groove", bd=1)
        met.pack(fill="x", padx=8, pady=(6, 2))

        # Predator column
        tk.Label(met, text="PREDATOR", font=("Arial", 9, "bold"),
                 fg="#c0392b").grid(row=0, column=0, padx=24, pady=(4, 0))
        self.pred_rate_var = tk.StringVar(value="Capture: —")
        self.pred_ema_var  = tk.StringVar(value="EMA: —")
        self.pred_best_var = tk.StringVar(value="Best EMA: —")
        tk.Label(met, textvariable=self.pred_rate_var,
                 font=("Arial", 10, "bold"), fg="#c0392b").grid(row=1, column=0, padx=24)
        tk.Label(met, textvariable=self.pred_ema_var,
                 font=("Arial", 9), fg="#6d1616").grid(row=2, column=0, padx=24)
        tk.Label(met, textvariable=self.pred_best_var,
                 font=("Arial", 8), fg="#27ae60").grid(row=3, column=0, padx=24,
                                                        pady=(0, 4))

        tk.Frame(met, width=1, bg="#ccc").grid(
            row=0, column=1, rowspan=5, sticky="ns", pady=4)

        # Prey column
        tk.Label(met, text="PREY", font=("Arial", 9, "bold"),
                 fg="#2980b9").grid(row=0, column=2, padx=24, pady=(4, 0))
        self.prey_rate_var = tk.StringVar(value="Survival: —")
        self.prey_ema_var  = tk.StringVar(value="EMA: —")
        self.prey_best_var = tk.StringVar(value="Best EMA: —")
        tk.Label(met, textvariable=self.prey_rate_var,
                 font=("Arial", 10, "bold"), fg="#2980b9").grid(row=1, column=2, padx=24)
        tk.Label(met, textvariable=self.prey_ema_var,
                 font=("Arial", 9), fg="#174f8a").grid(row=2, column=2, padx=24)
        tk.Label(met, textvariable=self.prey_best_var,
                 font=("Arial", 8), fg="#27ae60").grid(row=3, column=2, padx=24,
                                                        pady=(0, 4))

        self.phase_var = tk.StringVar(value="")
        tk.Label(met, textvariable=self.phase_var,
                 font=("Arial", 8, "italic"), fg="#7b1fa2").grid(
            row=4, column=0, columnspan=3, pady=(0, 3))

        self.rollback_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.rollback_var,
                 font=("Arial", 8, "italic"), fg="#e65100").pack()

        # ═══════════════════════════════════════════════════════════════════
        # PROGRESS BAR
        # ═══════════════════════════════════════════════════════════════════
        prog_frame = tk.Frame(root)
        prog_frame.pack(fill="x", padx=8, pady=2)
        self.status_var = tk.StringVar(value="Ready — no training yet.")
        tk.Label(prog_frame, textvariable=self.status_var,
                 font=("Arial", 8), fg="#555", anchor="w").pack(fill="x")
        self.progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                        length=494, mode="determinate")
        self.progress.pack(fill="x")

        # ═══════════════════════════════════════════════════════════════════
        # NOTEBOOK  (Training | Simulation)
        # ═══════════════════════════════════════════════════════════════════
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=8, pady=6)

        train_tab = tk.Frame(nb)
        sim_tab   = tk.Frame(nb)
        nb.add(train_tab, text="  Training  ")
        nb.add(sim_tab,   text="  Simulation  ")

        # ── Training tab ──────────────────────────────────────────────────
        def add_row(parent, label, var, values, default, r):
            tk.Label(parent, text=label, anchor="w",
                     font=("Arial", 9)).grid(row=r, column=0,
                                             sticky="w", padx=14, pady=4)
            var.set(default)
            ttk.Combobox(parent, textvariable=var, values=values,
                         width=14, state="readonly").grid(
                row=r, column=1, sticky="w", padx=6, pady=4)

        self.steps_var           = tk.StringVar()
        self.train_grid_var      = tk.StringVar()
        self.rollback_thresh_var = tk.StringVar()

        add_row(train_tab, "Total Steps per Run:",
                self.steps_var,
                ["10000", "50000", "100000", "250000", "500000"],
                "100000", 0)
        add_row(train_tab, "Training Grid Size (NxN):",
                self.train_grid_var,
                ["10", "15", "20", "30", "40"],
                "15", 1)
        add_row(train_tab, "Predator Rollback Threshold (%):",
                self.rollback_thresh_var,
                ["5", "8", "10", "15", "20"],
                "10", 2)

        tk.Label(train_tab,
                 text="Prey trains 50% more steps than predator per cycle.\n"
                      "Prey NEVER rolls back — keeps learning forward always.\n"
                      "Only predator rolls back (needs 2 consecutive drops).",
                 font=("Arial", 8), fg="#555", justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 4))

        btn_frame = tk.Frame(train_tab)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=6)

        self.train_btn = tk.Button(
            btn_frame, text="Start Training",
            command=self.start_training, height=2, width=16,
            bg="#e3f2fd", font=("Arial", 10, "bold"))
        self.train_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = tk.Button(
            btn_frame, text="Stop Early",
            command=self.stop_training, state=tk.DISABLED,
            height=2, width=16, bg="#ffebee", font=("Arial", 10, "bold"))
        self.stop_btn.grid(row=0, column=1, padx=6)

        # Wipe and Graph buttons side-by-side
        action_frame = tk.Frame(train_tab)
        action_frame.grid(row=5, column=0, columnspan=2, pady=(4, 0))

        self.reset_btn = tk.Button(
            action_frame, text="Wipe Models & Logs",
            command=self.clear_arena,
            bg="#ffcdd2", fg="#c62828", font=("Arial", 8, "bold"), width=18)
        self.reset_btn.grid(row=0, column=0, padx=6)

        self.train_graph_btn = tk.Button(
            action_frame, text="Show Metrics Graph",
            command=self.show_graph,
            bg="#e8eaf6", font=("Arial", 8, "bold"), width=18)
        self.train_graph_btn.grid(row=0, column=1, padx=6)

        # ── Simulation tab ────────────────────────────────────────────────
        tk.Label(sim_tab, text="Predator Checkpoint:", anchor="w",
                 font=("Arial", 9)).grid(row=0, column=0, sticky="w",
                                          padx=14, pady=5)
        self.pred_ckpt_var   = tk.StringVar()
        self.pred_ckpt_combo = ttk.Combobox(
            sim_tab, textvariable=self.pred_ckpt_var, width=22, state="readonly")
        self.pred_ckpt_combo.grid(row=0, column=1, sticky="w", padx=6, pady=5)

        tk.Label(sim_tab, text="Prey Checkpoint:", anchor="w",
                 font=("Arial", 9)).grid(row=1, column=0, sticky="w",
                                          padx=14, pady=5)
        self.prey_ckpt_var   = tk.StringVar()
        self.prey_ckpt_combo = ttk.Combobox(
            sim_tab, textvariable=self.prey_ckpt_var, width=22, state="readonly")
        self.prey_ckpt_combo.grid(row=1, column=1, sticky="w", padx=6, pady=5)

        tk.Label(sim_tab, text="Simulation Grid (NxN):", anchor="w",
                 font=("Arial", 9)).grid(row=2, column=0, sticky="w",
                                          padx=14, pady=5)
        self.sim_grid_var = tk.StringVar(value="20")
        ttk.Combobox(sim_tab, textvariable=self.sim_grid_var,
                     values=["15", "20", "30", "40", "50"],
                     width=22, state="readonly").grid(
            row=2, column=1, sticky="w", padx=6, pady=5)

        self.sim_btn = tk.Button(
            sim_tab, text="Watch Match",
            command=self.watch_match, height=2, width=24,
            bg="#c8e6c9", font=("Arial", 10, "bold"))
        self.sim_btn.grid(row=3, column=0, columnspan=2, pady=10)

        self.on_arena_change()

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _get_paths(self):
        ui_name     = self.arena_combo.get()
        module_name = self.available_arenas[ui_name]
        arena_dir   = os.path.join(self.base_dir, module_name)
        pred_dir    = os.path.join(arena_dir, "predator")
        prey_dir    = os.path.join(arena_dir, "prey")
        log_file    = os.path.join(arena_dir, "metrics_log.csv")
        return module_name, arena_dir, pred_dir, prey_dir, log_file

    def on_arena_change(self, event=None):
        _, _, pred_dir, prey_dir, log_file = self._get_paths()
        for d in (pred_dir, prey_dir):
            os.makedirs(d, exist_ok=True)
        self._save_config(self.arena_combo.get())

        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["PredSteps", "CaptureRate", "SurvivalRate",
                     "CapEMA", "SurvEMA", "Event"])

        self.rollback_var.set("")
        for v in (self.pred_best_var, self.prey_best_var):
            v.set("Best EMA: —")
        for v in (self.pred_ema_var, self.prey_ema_var):
            v.set("EMA: —")
        self.update_ui_state()

    def update_ui_state(self):
        _, _, pred_dir, prey_dir, _ = self._get_paths()

        def latest_steps(d):
            files = glob.glob(os.path.join(d, "model_*.zip"))
            valid = sorted([
                int(os.path.basename(f).replace("model_", "").replace(".zip", ""))
                for f in files
                if os.path.basename(f).replace("model_", "").replace(".zip", "").isdigit()
            ])
            return valid, (valid[-1] if valid else 0)

        pred_steps, pred_total = latest_steps(pred_dir)
        prey_steps, prey_total = latest_steps(prey_dir)
        self.pred_total_steps  = pred_total
        self.prey_total_steps  = prey_total

        self.status_var.set(
            f"Predator: {pred_total:,} steps  |  Prey: {prey_total:,} steps trained")

        self.pred_ckpt_combo["values"] = [f"Step {s}" for s in pred_steps]
        self.prey_ckpt_combo["values"] = [f"Step {s}" for s in prey_steps]
        if pred_steps: self.pred_ckpt_combo.current(len(pred_steps) - 1)
        else:          self.pred_ckpt_combo.set("")
        if prey_steps: self.prey_ckpt_combo.current(len(prey_steps) - 1)
        else:          self.prey_ckpt_combo.set("")

    def _save_config(self, name):
        try:
            with open(CONFIG_FILE, "w") as f: f.write(name)
        except OSError: pass

    def _load_config(self):
        try:
            with open(CONFIG_FILE) as f: return f.read().strip()
        except OSError: return ""

    def clear_arena(self):
        _, arena_dir, _, _, _ = self._get_paths()
        if messagebox.askyesno(
                "Confirm",
                f"Delete ALL models and logs for {self.arena_combo.get()}?"):
            shutil.rmtree(arena_dir, ignore_errors=True)
            self.on_arena_change()

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════════

    def start_training(self):
        try:
            steps = int(self.steps_var.get())
        except ValueError:
            return messagebox.showerror("Error", "Invalid step count.")

        self.stop_requested = False
        self.train_btn.config(state=tk.DISABLED, text="TRAINING…")
        self.stop_btn.config(state=tk.NORMAL)
        self.arena_combo.config(state=tk.DISABLED)
        self.rollback_var.set("")
        self.progress["value"] = 0

        threading.Thread(target=self._train_worker,
                         args=(steps,), daemon=True).start()

    def stop_training(self):
        self.stop_requested = True
        self.stop_btn.config(state=tk.DISABLED, text="Stopping…")

    def _train_worker(self, total_steps):
        """
        Alternating-phase loop:
          Phase A — Predator trains for half_chunk steps
          Phase B — Prey trains for half_chunk × PREY_RATIO steps  (50% more)
          Recovery — Extra prey burst if its EMA is below PREY_RECOVERY_THRESHOLD
          Evaluate — 10 silent episodes, EMA update
          Rollback — PREDATOR ONLY, after 2 consecutive drops below threshold
                     Prey never rolls back; it always keeps its latest policy.
        """
        module_name, _, pred_dir, prey_dir, log_file = self._get_paths()
        try:
            mod       = importlib.import_module(module_name)
            grid_size = int(self.train_grid_var.get())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Import Error", f"Could not load {module_name}.py:\n{e}"))
            self.root.after(0, self._finish_ui)
            return

        rollback_thresh = float(self.rollback_thresh_var.get())
        half_chunk      = EVAL_CHUNK_STEPS // 2
        prey_chunk      = int(half_chunk * PREY_RATIO)

        pred_env = mod.ArenaEnv(role="predator", grid_size=grid_size)
        prey_env = mod.ArenaEnv(role="prey",     grid_size=grid_size)

        def load_or_new(env, agent_dir, existing_steps):
            path = os.path.join(agent_dir, f"model_{existing_steps}")
            if os.path.exists(path + ".zip"):
                return PPO.load(path, env=env, device="auto")
            return PPO("MlpPolicy", env, verbose=0,
                       learning_rate=PPO_LEARNING_RATE,
                       n_steps=2048, batch_size=64, device="auto")

        pred_model = load_or_new(pred_env, pred_dir, self.pred_total_steps)
        prey_model = load_or_new(prey_env, prey_dir, self.prey_total_steps)

        best_cap_ema, best_surv_ema, cap_ema, surv_ema = \
            self._read_ema_from_log(log_file)

        pred_drop_streak = 0
        rollback_count   = 0
        steps_done       = 0

        while steps_done < total_steps and not self.stop_requested:

            # ── Phase A: Predator ─────────────────────────────────────────
            self.root.after(0, self.phase_var.set,
                            "Phase A — Training Predator…")
            pred_env.set_opponent_model(prey_model)
            cb = TrainingCallback(half_chunk, self, steps_done,
                                  total_steps, "Predator")
            pred_model.learn(half_chunk, callback=cb,
                             reset_num_timesteps=False)
            steps_done            += half_chunk
            self.pred_total_steps += half_chunk
            pred_model.save(
                os.path.join(pred_dir, f"model_{self.pred_total_steps}"))
            if self.stop_requested: break

            # ── Phase B: Prey (skewed ratio) ──────────────────────────────
            self.root.after(0, self.phase_var.set,
                            f"Phase B — Training Prey ({prey_chunk:,} steps)…")
            prey_env.set_opponent_model(pred_model)
            cb = TrainingCallback(prey_chunk, self, steps_done,
                                  total_steps, "Prey")
            prey_model.learn(prey_chunk, callback=cb,
                             reset_num_timesteps=False)
            steps_done            += prey_chunk
            self.prey_total_steps += prey_chunk
            prey_model.save(
                os.path.join(prey_dir, f"model_{self.prey_total_steps}"))
            if self.stop_requested: break

            # ── Recovery burst: extra prey training when struggling ────────
            if 0 < surv_ema < PREY_RECOVERY_THRESHOLD:
                self.root.after(0, self.phase_var.set,
                                f"Recovery — Boosting Prey "
                                f"(survival EMA {surv_ema:.1f}%)…")
                prey_env.set_opponent_model(pred_model)
                cb = TrainingCallback(PREY_RECOVERY_STEPS, self,
                                      steps_done, total_steps, "Prey Recovery")
                prey_model.learn(PREY_RECOVERY_STEPS, callback=cb,
                                 reset_num_timesteps=False)
                steps_done            += PREY_RECOVERY_STEPS
                self.prey_total_steps += PREY_RECOVERY_STEPS
                prey_model.save(
                    os.path.join(prey_dir, f"model_{self.prey_total_steps}"))
                if self.stop_requested: break

            # ── Evaluate ──────────────────────────────────────────────────
            self.root.after(0, self.phase_var.set, "Evaluating…")
            capture_rate, _ = self._evaluate(mod, pred_model, prey_model,
                                             grid_size)
            survival_rate   = 100.0 - capture_rate

            # Seed EMA on first real evaluation
            if cap_ema == 0.0 and surv_ema == 0.0:
                cap_ema, surv_ema = capture_rate, survival_rate
            else:
                cap_ema  = EMA_ALPHA * capture_rate  + (1 - EMA_ALPHA) * cap_ema
                surv_ema = EMA_ALPHA * survival_rate + (1 - EMA_ALPHA) * surv_ema

            self.root.after(0, self.pred_rate_var.set,
                            f"Capture: {capture_rate:.1f}%")
            self.root.after(0, self.prey_rate_var.set,
                            f"Survival: {survival_rate:.1f}%")
            self.root.after(0, self.pred_ema_var.set,
                            f"EMA: {cap_ema:.1f}%")
            self.root.after(0, self.prey_ema_var.set,
                            f"EMA: {surv_ema:.1f}%")

            event_label = ""

            # ── Predator rollback (only predator — prey never rolls back) ──
            if cap_ema > best_cap_ema:
                best_cap_ema     = cap_ema
                pred_drop_streak = 0
                pred_model.save(os.path.join(pred_dir, "model_best"))
                self.root.after(0, self.pred_best_var.set,
                                f"Best EMA: {best_cap_ema:.1f}%  ✓")
                event_label = "pred_best"
            elif cap_ema < best_cap_ema - rollback_thresh:
                pred_drop_streak += 1
                if pred_drop_streak >= ROLLBACK_PATIENCE:
                    pred_drop_streak = 0
                    rollback_count  += 1
                    bp = os.path.join(pred_dir, "model_best")
                    if os.path.exists(bp + ".zip"):
                        pred_model = PPO.load(bp, env=pred_env, device="auto")
                    msg = (f"Predator rollback #{rollback_count}  "
                           f"(EMA {cap_ema:.1f}% vs best {best_cap_ema:.1f}%)")
                    self.root.after(0, self.rollback_var.set, msg)
                    event_label = f"pred_rb#{rollback_count}"
            else:
                pred_drop_streak = 0

            # ── Prey best tracker (no rollback) ───────────────────────────
            if surv_ema > best_surv_ema:
                best_surv_ema = surv_ema
                prey_model.save(os.path.join(prey_dir, "model_best"))
                self.root.after(0, self.prey_best_var.set,
                                f"Best EMA: {best_surv_ema:.1f}%  ✓")
                if not event_label:
                    event_label = "prey_best"

            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.pred_total_steps,
                    f"{capture_rate:.2f}",
                    f"{survival_rate:.2f}",
                    f"{cap_ema:.2f}",
                    f"{surv_ema:.2f}",
                    event_label,
                ])

        self.root.after(0, self.phase_var.set, "")
        self.root.after(0, self._finish_ui)

    # ── Evaluation ──────────────────────────────────────────────────────────

    def _evaluate(self, mod, pred_model, prey_model, grid_size,
                  n_episodes=EVAL_EPISODES):
        eval_grid = min(grid_size, 15)
        env = mod.ArenaEnv(role="predator", grid_size=eval_grid,
                           opponent_model=prey_model)
        catches = total_steps = 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done:
                action, _ = pred_model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                if done:
                    outcome = info.get("outcome", {})
                    if outcome.get("result") == "caught":
                        catches += 1
                    total_steps += outcome.get("steps", env.step_count)
        return (catches / n_episodes) * 100, total_steps / n_episodes

    def _read_ema_from_log(self, log_file):
        best_cap = best_surv = cap_ema = surv_ema = 0.0
        if not os.path.exists(log_file):
            return best_cap, best_surv, cap_ema, surv_ema
        try:
            with open(log_file) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) < 5: continue
                    try:
                        ce, se   = float(row[3]), float(row[4])
                        cap_ema  = ce
                        surv_ema = se
                        if ce > best_cap:  best_cap  = ce
                        if se > best_surv: best_surv = se
                    except ValueError:
                        pass
        except StopIteration:
            pass
        return best_cap, best_surv, cap_ema, surv_ema

    # ── UI helpers ───────────────────────────────────────────────────────────

    def _update_progress_ui(self, percent, overall_done, total, phase):
        self.progress["value"] = percent
        self.status_var.set(
            f"[{phase}]  {percent:.1f}%  ({overall_done:,} / {total:,} steps)")

    def _finish_ui(self):
        self.train_btn.config(state=tk.NORMAL, text="Start Training")
        self.stop_btn.config(state=tk.DISABLED, text="Stop Early")
        self.arena_combo.config(state="readonly")
        self.update_ui_state()
        if not self.stop_requested:
            messagebox.showinfo("Done", "Training Run Finished!")

    # ═══════════════════════════════════════════════════════════════════════
    # GRAPH  (shared by training tab button and old sim tab location)
    # ═══════════════════════════════════════════════════════════════════════

    def show_graph(self):
        _, _, _, _, log_file = self._get_paths()
        if not os.path.exists(log_file):
            return messagebox.showinfo("No Data", "No training data yet.")

        steps, caps, survs, cap_emas, surv_emas, events = [], [], [], [], [], []
        with open(log_file) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3: continue
                try:
                    steps.append(int(row[0]))
                    caps.append(float(row[1]))
                    survs.append(float(row[2]))
                    cap_emas.append(float(row[3])  if len(row) > 3 else caps[-1])
                    surv_emas.append(float(row[4]) if len(row) > 4 else survs[-1])
                    events.append(row[5]           if len(row) > 5 else "")
                except ValueError:
                    pass

        if not steps:
            return messagebox.showinfo("No Data", "No training data yet.")

        fig, ax = plt.subplots(figsize=(11, 6))

        # Raw values (faint)
        ax.plot(steps, caps,  color="#c0392b", alpha=0.2, linewidth=1,
                label="Capture %  (raw)")
        ax.plot(steps, survs, color="#2980b9", alpha=0.2, linewidth=1,
                linestyle="--", label="Survival %  (raw)")

        # EMA values (prominent)
        ax.plot(steps, cap_emas,  color="#c0392b", linewidth=2.5,
                label="Predator Capture EMA")
        ax.plot(steps, surv_emas, color="#2980b9", linewidth=2.5,
                linestyle="--", label="Prey Survival EMA")

        # Event markers
        for i, ev in enumerate(events):
            if "pred_best" in ev:
                ax.axvline(steps[i], color="#c0392b", alpha=0.2, linewidth=1)
            if "prey_best" in ev:
                ax.axvline(steps[i], color="#2980b9", alpha=0.2, linewidth=1)
            if "_rb" in ev:
                ax.axvline(steps[i], color="#e65100", alpha=0.35,
                           linewidth=1, linestyle=":")

        ax.axhline(50, color="#aaa", linewidth=1, linestyle=":", alpha=0.6,
                   label="50% balance line")
        ax.set_xlabel("Predator Steps Trained", fontsize=11)
        ax.set_ylabel("Rate  %", fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_title(
            f"Co-Evolution Metrics — {self.arena_combo.get()}",
            fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.show()

    # ═══════════════════════════════════════════════════════════════════════
    # SIMULATION
    # ═══════════════════════════════════════════════════════════════════════

    def watch_match(self):
        pred_sel = self.pred_ckpt_combo.get()
        prey_sel = self.prey_ckpt_combo.get()
        if not pred_sel or not prey_sel:
            return messagebox.showwarning(
                "No Models",
                "Train both agents first, then select checkpoints.")
        self.sim_btn.config(state=tk.DISABLED)
        threading.Thread(
            target=self._run_sim_worker,
            args=(pred_sel.split()[-1], prey_sel.split()[-1]),
            daemon=True).start()

    def _run_sim_worker(self, pred_step, prey_step):
        module_name, _, pred_dir, prey_dir, _ = self._get_paths()
        mod       = importlib.import_module(module_name)
        grid_size = int(self.sim_grid_var.get())

        pred_model = PPO.load(os.path.join(pred_dir, f"model_{pred_step}"))
        prey_model = PPO.load(os.path.join(prey_dir, f"model_{prey_step}"))

        env = mod.ArenaEnv(role="predator", grid_size=grid_size,
                           opponent_model=prey_model, render_mode="human")
        obs, _ = env.reset()
        env.render()
        sim_state = "PLAYING"

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.window = None
                    self.root.after(
                        0, lambda: self.sim_btn.config(state=tk.NORMAL))
                    return
                if (sim_state == "GAME_OVER" and
                        event.type == pygame.MOUSEBUTTONDOWN and
                        event.button == 1):
                    if env.replay_rect and env.replay_rect.collidepoint(event.pos):
                        obs, _    = env.reset()
                        sim_state = "PLAYING"
                    elif env.close_rect and env.close_rect.collidepoint(event.pos):
                        pygame.quit()
                        env.window = None
                        self.root.after(
                            0, lambda: self.sim_btn.config(state=tk.NORMAL))
                        return

            if sim_state == "PLAYING":
                action, _ = pred_model.predict(obs, deterministic=False)
                obs, _, term, trunc, info = env.step(action)
                env.render()
                if term or trunc:
                    sim_state = "GAME_OVER"
                    outcome   = info.get("outcome", {})
                    result    = outcome.get("result", "")
                    n_steps   = outcome.get("steps", env.step_count)
                    if result == "caught":
                        msg = f"Predator Wins!  ({n_steps} steps)"
                    elif result in ("survived", "timeout"):
                        msg = f"Prey Wins!  ({n_steps} steps)"
                    else:
                        msg = f"Prey Wins!  ({n_steps} steps)"  # safe fallback
                    env.draw_game_over(msg)

            time.sleep(0.05)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AITrainerApp(root)
    root.mainloop()