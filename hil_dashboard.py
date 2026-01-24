#!/usr/bin/env python3
"""
🧠 PRISM-Zero v3.1 HIL DASHBOARD
Interactive terminal UI for monitoring and controlling neuromorphic training.
Works over SSH - no GUI required!

Usage:
    python3 hil_dashboard.py [output_dir]

Controls:
    S - Spike exploration (10 episodes)
    E - Set epsilon manually
    L - Adjust learning rate multiplier
    P - Pause/Resume training
    C - Force checkpoint save
    R - Refresh display
    Q - Quit dashboard
"""

import json
import os
import sys
import time
import select
import termios
import tty
from datetime import datetime
from pathlib import Path

# ANSI Colors
class C:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

# Block characters for graphs
BLOCKS = ['░', '▒', '▓', '█']
BARS = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']

def clear_screen():
    print('\033[2J\033[H', end='')

def move_cursor(row, col):
    print(f'\033[{row};{col}H', end='')

def draw_box(row, col, width, height, title=""):
    """Draw a box with optional title"""
    move_cursor(row, col)
    print(f"╔{'═' * (width-2)}╗")
    if title:
        title_pos = (width - len(title) - 2) // 2
        move_cursor(row, col + title_pos)
        print(f" {C.BOLD}{title}{C.RESET} ")
    for i in range(1, height-1):
        move_cursor(row + i, col)
        print(f"║{' ' * (width-2)}║")
    move_cursor(row + height - 1, col)
    print(f"╚{'═' * (width-2)}╝")

def progress_bar(value, max_val, width=20, fill_char='█', empty_char='░', color=C.GREEN):
    """Create a progress bar with color"""
    if max_val == 0:
        pct = 0
    else:
        pct = min(1.0, max(0.0, value / max_val))
    filled = int(pct * width)
    return f"{color}{fill_char * filled}{C.DIM}{empty_char * (width - filled)}{C.RESET}"

def pressure_gauge(pressure, width=30):
    """
    Pressure gauge that shows when HIL intervention is recommended.
    pressure: 0-100 scale
    """
    # Color based on pressure level
    if pressure < 30:
        color = C.GREEN
        status = "NOMINAL"
        icon = "🟢"
    elif pressure < 60:
        color = C.YELLOW
        status = "ELEVATED"
        icon = "🟡"
    elif pressure < 80:
        color = C.RED
        status = "HIGH"
        icon = "🟠"
    else:
        color = C.BG_RED + C.WHITE
        status = "CRITICAL"
        icon = "🔴"

    # Animated pressure bar with gradient
    filled = int(pressure / 100 * width)
    bar = ""
    for i in range(width):
        if i < filled:
            if i < width * 0.3:
                bar += f"{C.GREEN}█"
            elif i < width * 0.6:
                bar += f"{C.YELLOW}█"
            elif i < width * 0.8:
                bar += f"{C.RED}█"
            else:
                bar += f"{C.MAGENTA}█"
        else:
            bar += f"{C.DIM}░"

    return bar + C.RESET, status, icon, color

def breathing_bar(phase, width=20):
    """Create a breathing/pulsing animation for activity indicator"""
    # Phase goes from 0 to 1 and back
    intensity = abs(phase * 2 - 1)  # 0->1->0
    filled = int(intensity * width / 2)
    center = width // 2
    bar = ""
    for i in range(width):
        dist = abs(i - center)
        if dist <= filled:
            bar += f"{C.CYAN}█"
        else:
            bar += f"{C.DIM}░"
    return bar + C.RESET

def calculate_pressure(status):
    """
    Calculate intervention pressure based on multiple factors.
    Returns 0-100 scale.
    """
    pressure = 0
    reasons = []

    # Factor 1: Stuck episodes (0-40 points)
    stuck_ratio = status.get('episodes_without_improvement', 0) / status.get('patience', 50)
    stuck_pressure = stuck_ratio * 40
    pressure += stuck_pressure
    if stuck_ratio > 0.5:
        reasons.append(f"Stuck {status['episodes_without_improvement']}/{status['patience']} eps")

    # Factor 2: Reward trend (0-25 points)
    reward_trend = status.get('learning_monitor', {}).get('reward_trend', 0)
    if reward_trend < -0.001:
        trend_pressure = min(25, abs(reward_trend) * 2500)
        pressure += trend_pressure
        reasons.append(f"Declining rewards ({reward_trend:+.4f})")

    # Factor 3: Error trend (0-20 points)
    error_trend = status.get('learning_monitor', {}).get('error_trend', 0)
    if error_trend > 100:
        error_pressure = min(20, error_trend / 500 * 20)
        pressure += error_pressure
        reasons.append(f"Rising error ({error_trend:+.0f})")

    # Factor 4: Epsilon too low without progress (0-15 points)
    epsilon = status.get('epsilon', 1.0)
    if epsilon < 0.2 and stuck_ratio > 0.3:
        low_eps_pressure = (0.2 - epsilon) * 75
        pressure += low_eps_pressure
        reasons.append(f"Low exploration (ε={epsilon:.2f})")

    return min(100, pressure), reasons

def get_recommendation(pressure, status):
    """Get HIL recommendation based on pressure and status"""
    if pressure < 30:
        return "✅ Training nominal - no intervention needed", None

    stuck = status.get('episodes_without_improvement', 0)
    epsilon = status.get('epsilon', 1.0)
    reward_trend = status.get('learning_monitor', {}).get('reward_trend', 0)
    error_trend = status.get('learning_monitor', {}).get('error_trend', 0)

    if stuck > 30 and epsilon < 0.3:
        return "🔥 SPIKE EXPLORATION recommended (press S)", {"spike_exploration": 15}
    elif reward_trend < -0.005:
        return "📉 SLOW LEARNING recommended (press L, set 0.5)", {"learning_rate_multiplier": 0.5}
    elif error_trend > 500 and stuck > 20:
        return "🎚️ BOOST EPSILON recommended (press E, set 0.6)", {"set_epsilon": 0.6}
    elif pressure > 60:
        return "🔥 SPIKE EXPLORATION suggested (press S)", {"spike_exploration": 10}
    else:
        return "⏳ Monitor closely - intervention may be needed soon", None

def send_hil_command(output_dir, command_dict):
    """Send a command to the HIL control file"""
    control_path = os.path.join(output_dir, "hil_control.json")
    try:
        with open(control_path, 'w') as f:
            json.dump(command_dict, f, indent=2)
        return True
    except Exception as e:
        return False

def get_key_nonblocking():
    """Get a keypress without blocking (returns None if no key pressed)"""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None

def load_status(output_dir):
    """Load the current HIL status"""
    status_path = os.path.join(output_dir, "hil_status.json")
    try:
        with open(status_path) as f:
            return json.load(f)
    except:
        return None

def main():
    # Get output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "/home/diddy/Desktop/PRISM4D-bio/training_output_neuro_full"

    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    # Check if we have a proper terminal
    if not sys.stdin.isatty():
        print("Error: This dashboard requires an interactive terminal.")
        print("Run it directly in your terminal, not through a pipe.")
        print(f"\nUsage: python3 {sys.argv[0]} [{output_dir}]")
        sys.exit(1)

    # Setup terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        phase = 0
        message = ""
        message_time = 0

        while True:
            clear_screen()
            status = load_status(output_dir)

            if status is None:
                print(f"{C.RED}⚠️  Cannot read status file. Is training running?{C.RESET}")
                time.sleep(1)
                continue

            # Calculate pressure
            pressure, reasons = calculate_pressure(status)
            pressure_bar, pressure_status, pressure_icon, pressure_color = pressure_gauge(pressure)
            recommendation, rec_command = get_recommendation(pressure, status)

            # Header
            print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}║{C.RESET}  🧠 {C.BOLD}PRISM-Zero v3.1 HIL DASHBOARD{C.RESET}                    {breathing_bar(phase, 12)}  {C.CYAN}║{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C.RESET}")
            print()

            # Current Target Section
            target = status.get('current_target', 'Unknown')
            family = status.get('current_family', 'Unknown')
            print(f"  {C.BOLD}🎯 TARGET:{C.RESET} {C.YELLOW}{target}{C.RESET} ({C.DIM}{family}{C.RESET})")
            print(f"  {C.BOLD}📊 PROGRESS:{C.RESET} Target {status['target_idx']}/{status['total_targets']} │ Episode {status['episode']}/{status['max_episodes']}")
            print()

            # Pressure Gauge (the key feature!)
            print(f"  {C.BOLD}⚡ INTERVENTION PRESSURE{C.RESET}")
            print(f"  {pressure_bar} {pressure_icon} {pressure_color}{C.BOLD}{pressure_status}{C.RESET} ({pressure:.0f}%)")
            if reasons:
                print(f"  {C.DIM}Factors: {', '.join(reasons)}{C.RESET}")
            print()

            # Recommendation
            print(f"  {C.BOLD}💡 RECOMMENDATION:{C.RESET}")
            print(f"     {recommendation}")
            print()

            # Stats Grid
            print(f"  ╭─────────────────────────────────╮  ╭─────────────────────────────────╮")

            # Reward info
            reward = status.get('episode_reward', 0)
            best = status.get('best_reward', 0)
            reward_color = C.GREEN if reward > 0 else C.RED if reward < -0.01 else C.YELLOW
            print(f"  │ {C.BOLD}💰 REWARD{C.RESET}                       │  │ {C.BOLD}🧬 LEARNING{C.RESET}                     │")
            print(f"  │   Current: {reward_color}{reward:+.6f}{C.RESET}           │  │   Epsilon: {status['epsilon']:.2%}               │")
            print(f"  │   Best:    {C.GREEN}{best:+.6f}{C.RESET}           │  │   {progress_bar(status['epsilon'], 1, 20)}  │")

            reward_trend = status.get('learning_monitor', {}).get('reward_trend', 0)
            trend_icon = "📈" if reward_trend > 0.001 else "📉" if reward_trend < -0.001 else "➡️"
            error_trend = status.get('learning_monitor', {}).get('error_trend', 0)
            error_icon = "✅" if error_trend < -10 else "⚠️" if error_trend > 10 else "➖"
            print(f"  │   Trend:   {trend_icon} {reward_trend:+.4f}           │  │   Error:   {error_icon} {status.get('rls_error', 0):.0f}              │")
            print(f"  ╰─────────────────────────────────╯  ╰─────────────────────────────────╯")
            print()

            # Patience meter (building pressure visualization)
            stuck = status.get('episodes_without_improvement', 0)
            patience = status.get('patience', 50)
            patience_pct = stuck / patience

            print(f"  {C.BOLD}⏳ PATIENCE METER{C.RESET} (early stopping pressure)")
            patience_bar = ""
            for i in range(40):
                pos = i / 40
                if pos < patience_pct:
                    if pos < 0.5:
                        patience_bar += f"{C.GREEN}█"
                    elif pos < 0.7:
                        patience_bar += f"{C.YELLOW}█"
                    elif pos < 0.9:
                        patience_bar += f"{C.RED}█"
                    else:
                        patience_bar += f"{C.MAGENTA}█"
                else:
                    patience_bar += f"{C.DIM}░"
            print(f"  [{patience_bar}{C.RESET}] {stuck}/{patience} episodes stuck")
            print()

            # Family Performance
            print(f"  {C.BOLD}👨‍👩‍👧‍👦 FAMILY PERFORMANCE{C.RESET}")
            fam_perf = status.get('learning_monitor', {}).get('family_performance', {})
            for fam, stats in sorted(fam_perf.items()):
                done = stats.get('targets_completed', 0)
                total = stats.get('targets_count', 1)
                avg = stats.get('avg_best_reward', 0)
                pct = done / total if total > 0 else 0
                icon = "✅" if done == total else "🔄" if done > 0 else "⏳"
                bar = progress_bar(done, total, 10)
                print(f"     {icon} {fam:18s} {bar} {done}/{total}  avg: {avg:+.5f}")
            print()

            # Time info
            elapsed = status.get('total_time_secs', 0) / 60
            eta = status.get('eta_secs', 0) / 60
            print(f"  {C.BOLD}⏱️ TIME:{C.RESET} Elapsed: {elapsed:.1f}m │ ETA: {eta:.0f}m ({eta/60:.1f}h)")
            print()

            # HIL Controls
            print(f"  {C.BOLD}{C.CYAN}╭────────────────────── HIL CONTROLS ──────────────────────╮{C.RESET}")
            print(f"  {C.CYAN}│{C.RESET}  {C.BOLD}S{C.RESET} Spike Exploration   {C.BOLD}E{C.RESET} Set Epsilon   {C.BOLD}L{C.RESET} Learning Rate  {C.CYAN}│{C.RESET}")
            print(f"  {C.CYAN}│{C.RESET}  {C.BOLD}P{C.RESET} Pause/Resume        {C.BOLD}C{C.RESET} Checkpoint    {C.BOLD}Q{C.RESET} Quit Dashboard {C.CYAN}│{C.RESET}")
            print(f"  {C.BOLD}{C.CYAN}╰──────────────────────────────────────────────────────────╯{C.RESET}")

            # Message area
            if message and time.time() - message_time < 3:
                print()
                print(f"  {C.BG_GREEN}{C.WHITE} {message} {C.RESET}")

            # Status line
            paused = status.get('paused', False)
            paused_str = f"{C.RED}⏸️  PAUSED{C.RESET}" if paused else f"{C.GREEN}▶️  RUNNING{C.RESET}"
            lr_mult = status.get('learning_rate_multiplier', 1.0)
            print()
            print(f"  {paused_str} │ LR: {lr_mult:.1f}x │ Updated: {status.get('timestamp', 'N/A')[:19]}")

            # Handle input
            key = get_key_nonblocking()
            if key:
                key = key.lower()
                if key == 'q':
                    break
                elif key == 's':
                    send_hil_command(output_dir, {"spike_exploration": 10})
                    message = "🔥 Spike exploration activated for 10 episodes!"
                    message_time = time.time()
                elif key == 'p':
                    new_paused = not paused
                    send_hil_command(output_dir, {"pause": new_paused})
                    message = f"{'⏸️ Training PAUSED' if new_paused else '▶️ Training RESUMED'}!"
                    message_time = time.time()
                elif key == 'c':
                    send_hil_command(output_dir, {"save_checkpoint": True})
                    message = "💾 Checkpoint save requested!"
                    message_time = time.time()
                elif key == 'e':
                    # Quick epsilon presets
                    print(f"\n  {C.BOLD}Set Epsilon:{C.RESET} 1=0.8  2=0.5  3=0.3  4=0.1")
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    choice = input("  Choice (1-4): ").strip()
                    tty.setcbreak(sys.stdin.fileno())
                    eps_map = {'1': 0.8, '2': 0.5, '3': 0.3, '4': 0.1}
                    if choice in eps_map:
                        send_hil_command(output_dir, {"set_epsilon": eps_map[choice]})
                        message = f"🎚️ Epsilon set to {eps_map[choice]}"
                        message_time = time.time()
                elif key == 'l':
                    # Learning rate presets
                    print(f"\n  {C.BOLD}Learning Rate Multiplier:{C.RESET} 1=0.3x  2=0.5x  3=1.0x  4=2.0x  5=3.0x")
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    choice = input("  Choice (1-5): ").strip()
                    tty.setcbreak(sys.stdin.fileno())
                    lr_map = {'1': 0.3, '2': 0.5, '3': 1.0, '4': 2.0, '5': 3.0}
                    if choice in lr_map:
                        send_hil_command(output_dir, {"learning_rate_multiplier": lr_map[choice]})
                        message = f"📈 Learning rate set to {lr_map[choice]}x"
                        message_time = time.time()

            # Update animation phase
            phase = (phase + 0.05) % 1.0

            # Refresh rate
            time.sleep(0.5)

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        clear_screen()
        print("Dashboard closed.")

if __name__ == "__main__":
    main()
