#!/usr/bin/env python3
"""
🧠 PRISM-Zero v3.1 WEB HIL DASHBOARD
Mobile-friendly web interface for monitoring and controlling neuromorphic training.

Usage:
    pip install flask
    python3 hil_web_dashboard.py [--port 5000] [--host 0.0.0.0]

Access from phone:
    1. SSH tunnel: ssh -L 5000:localhost:5000 user@server
       Then open http://localhost:5000 on phone
    2. Or direct: http://<server-ip>:5000 (if on same network)
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, render_template_string, jsonify, request
except ImportError:
    print("=" * 60)
    print("Flask not installed. Install it with:")
    print("    pip install flask")
    print("=" * 60)
    sys.exit(1)

app = Flask(__name__)

# Enable CORS for all routes (needed for ngrok/external access)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Configuration - Default to training_output_ultra (override with --output)
OUTPUT_DIR = "/home/diddy/Desktop/PRISM4D-bio/training_output_ultra"

# Alert thresholds (user can toggle these)
ALERT_CONFIG = {
    "stuck_threshold": 35,      # Alert when stuck > this many episodes
    "pressure_threshold": 60,   # Alert when pressure > this %
    "reward_decline": -0.005,   # Alert when reward trend < this
    "error_spike": 1000,        # Alert when error trend > this
    "target_complete": True,    # Alert on target completion
    "training_paused": True,    # Alert if training pauses
}

# HTML Template with embedded CSS and JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>🧠 PRISM-Zero HIL Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a25;
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --accent-cyan: #00d4ff;
            --accent-green: #00ff88;
            --accent-yellow: #ffcc00;
            --accent-red: #ff4444;
            --accent-purple: #aa55ff;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 10px;
            padding-bottom: 80px;
        }

        .header {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 15px;
            margin-bottom: 15px;
            border: 1px solid var(--accent-cyan);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }

        .header h1 {
            font-size: 1.4em;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header .subtitle {
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-top: 10px;
            animation: pulse 2s infinite;
        }

        .status-running { background: var(--accent-green); color: #000; }
        .status-paused { background: var(--accent-red); color: #fff; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 12px;
            border: 1px solid #2a2a35;
        }

        .card-title {
            font-size: 0.9em;
            color: var(--accent-cyan);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .target-info {
            text-align: center;
        }

        .target-name {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--accent-yellow);
        }

        .target-family {
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .progress-info {
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
            text-align: center;
        }

        .progress-item {
            flex: 1;
        }

        .progress-value {
            font-size: 1.3em;
            font-weight: bold;
        }

        .progress-label {
            font-size: 0.75em;
            color: var(--text-secondary);
        }

        /* Pressure Gauge */
        .pressure-container {
            text-align: center;
        }

        .pressure-gauge {
            width: 100%;
            height: 30px;
            background: var(--bg-secondary);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            margin: 10px 0;
        }

        .pressure-fill {
            height: 100%;
            border-radius: 15px;
            transition: width 0.5s ease, background 0.5s ease;
        }

        .pressure-nominal { background: linear-gradient(90deg, #00ff88, #00cc66); }
        .pressure-elevated { background: linear-gradient(90deg, #00cc66, #ffcc00); }
        .pressure-high { background: linear-gradient(90deg, #ffcc00, #ff6600); }
        .pressure-critical { background: linear-gradient(90deg, #ff6600, #ff0000); animation: critical-pulse 0.5s infinite; }

        @keyframes critical-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .pressure-status {
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 5px;
        }

        .pressure-reasons {
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        /* Recommended Action Panel */
        .recommendation-panel {
            background: linear-gradient(135deg, #1a2a35, #1a1a25);
            border: 2px solid var(--accent-purple);
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
        }

        .recommendation-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }

        .recommendation-header .icon {
            font-size: 1.3em;
        }

        .recommendation-header .title {
            font-size: 0.9em;
            color: var(--accent-purple);
            font-weight: bold;
        }

        .recommendation-action {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }

        .recommendation-action.primary {
            border-left: 4px solid var(--accent-green);
        }

        .recommendation-action.secondary {
            border-left: 4px solid var(--accent-yellow);
            opacity: 0.8;
        }

        .recommendation-action.wait {
            border-left: 4px solid var(--accent-cyan);
            opacity: 0.7;
        }

        .action-name {
            font-weight: bold;
            font-size: 1em;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .action-reason {
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 5px;
            line-height: 1.4;
        }

        .action-confidence {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            margin-left: auto;
        }

        .confidence-high { background: var(--accent-green); color: #000; }
        .confidence-medium { background: var(--accent-yellow); color: #000; }
        .confidence-low { background: var(--accent-cyan); color: #000; }

        .quick-action-btn {
            background: var(--accent-purple);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 0.85em;
            cursor: pointer;
            margin-top: 8px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.2s;
        }

        .quick-action-btn:hover {
            background: #cc66ff;
            transform: scale(0.98);
        }

        .no-action-needed {
            text-align: center;
            padding: 15px;
            color: var(--accent-green);
            font-size: 0.95em;
        }

        .no-action-needed .icon {
            font-size: 2em;
            display: block;
            margin-bottom: 8px;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .stat-box {
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 10px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
        }

        .stat-value.positive { color: var(--accent-green); }
        .stat-value.negative { color: var(--accent-red); }
        .stat-value.neutral { color: var(--accent-yellow); }

        .stat-label {
            font-size: 0.7em;
            color: var(--text-secondary);
            margin-top: 3px;
        }

        /* Patience Meter */
        .patience-bar {
            width: 100%;
            height: 20px;
            background: var(--bg-secondary);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }

        .patience-fill {
            height: 100%;
            transition: width 0.5s ease;
            background: linear-gradient(90deg,
                var(--accent-green) 0%,
                var(--accent-green) 50%,
                var(--accent-yellow) 70%,
                var(--accent-red) 90%,
                var(--accent-purple) 100%
            );
        }

        /* Family Performance */
        .family-item {
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #2a2a35;
        }

        .family-item:last-child {
            border-bottom: none;
        }

        .family-icon {
            font-size: 1.2em;
            width: 30px;
        }

        .family-name {
            flex: 1;
            font-size: 0.85em;
        }

        .family-progress {
            width: 60px;
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin: 0 10px;
        }

        .family-progress-fill {
            height: 100%;
            background: var(--accent-green);
        }

        .family-stats {
            font-size: 0.75em;
            color: var(--text-secondary);
            width: 80px;
            text-align: right;
        }

        /* HIL Controls */
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }

        .control-btn {
            background: var(--bg-secondary);
            border: 2px solid #3a3a45;
            border-radius: 12px;
            padding: 15px 10px;
            color: var(--text-primary);
            font-size: 0.85em;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }

        .control-btn:hover, .control-btn:active {
            background: var(--bg-card);
            border-color: var(--accent-cyan);
            transform: scale(0.98);
        }

        .control-btn.active {
            border-color: var(--accent-green);
            background: rgba(0, 255, 136, 0.1);
        }

        .control-btn .icon {
            font-size: 1.5em;
        }

        .control-btn.spike { border-color: #ff6600; }
        .control-btn.spike:hover { background: rgba(255, 102, 0, 0.2); }

        .control-btn.pause { border-color: var(--accent-red); }
        .control-btn.pause:hover { background: rgba(255, 68, 68, 0.2); }

        .control-btn.resume { border-color: var(--accent-green); background: rgba(0, 255, 136, 0.15); }
        .control-btn.resume:hover { background: rgba(0, 255, 136, 0.3); }

        /* Learning Chart */
        .chart-container {
            position: relative;
            height: 200px;
            width: 100%;
            margin: 10px 0;
        }

        .chart-legend {
            display: flex;
            justify-content: center;
            gap: 15px;
            font-size: 0.75em;
            margin-top: 8px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }

        /* Neural Network Visualization */
        .neural-viz-container {
            background: linear-gradient(135deg, #0a1520, #1a1a25);
            border: 2px solid var(--accent-purple);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .neural-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
        }

        .neural-header h2 {
            font-size: 1.1em;
            color: var(--accent-purple);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .neural-header .arch-badge {
            background: var(--bg-secondary);
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 0.75em;
            color: var(--accent-cyan);
            font-family: monospace;
        }

        /* Heatmap canvas */
        .heatmap-container {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 12px;
        }

        .heatmap-title {
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .heatmap-stats {
            font-size: 0.7em;
            color: var(--accent-cyan);
        }

        .heatmap-canvas {
            width: 100%;
            height: 80px;
            border-radius: 5px;
            image-rendering: pixelated;
        }

        /* Network flow diagram */
        .network-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 10px;
            background: var(--bg-secondary);
            border-radius: 10px;
            margin-bottom: 12px;
            overflow-x: auto;
        }

        .flow-layer {
            text-align: center;
            min-width: 70px;
        }

        .flow-layer .neurons {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
            margin-bottom: 5px;
        }

        .flow-layer .neuron {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transition: all 0.3s;
        }

        .flow-layer .neuron.active {
            box-shadow: 0 0 8px currentColor;
        }

        .flow-layer .label {
            font-size: 0.65em;
            color: var(--text-secondary);
        }

        .flow-layer .count {
            font-size: 0.75em;
            font-weight: bold;
            color: var(--text-primary);
        }

        .flow-arrow {
            color: var(--accent-cyan);
            font-size: 1.2em;
            opacity: 0.5;
        }

        /* Q-value bars */
        .qvalue-section {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }

        .qvalue-param {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 10px;
        }

        .qvalue-param .param-name {
            font-size: 0.75em;
            color: var(--accent-yellow);
            margin-bottom: 6px;
            text-transform: uppercase;
        }

        .qvalue-bars {
            display: flex;
            align-items: flex-end;
            gap: 3px;
            height: 40px;
        }

        .qvalue-bar {
            flex: 1;
            min-width: 0;
            background: var(--accent-cyan);
            border-radius: 2px 2px 0 0;
            transition: height 0.3s, background 0.3s;
            position: relative;
        }

        .qvalue-bar.selected {
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
        }

        .qvalue-bar .value {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.6em;
            color: var(--text-secondary);
            white-space: nowrap;
            display: none;
        }

        .qvalue-bar:hover .value {
            display: block;
        }

        /* Feature input display */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            gap: 5px;
        }

        .feature-item {
            background: var(--bg-secondary);
            padding: 6px;
            border-radius: 5px;
            text-align: center;
        }

        .feature-item .name {
            font-size: 0.6em;
            color: var(--text-secondary);
            text-overflow: ellipsis;
            overflow: hidden;
        }

        .feature-item .value {
            font-size: 0.8em;
            font-weight: bold;
            color: var(--accent-cyan);
        }

        .feature-item .bar {
            height: 3px;
            background: var(--bg-card);
            border-radius: 2px;
            margin-top: 3px;
            overflow: hidden;
        }

        .feature-item .bar-fill {
            height: 100%;
            background: var(--accent-cyan);
            transition: width 0.3s;
        }

        /* Weight matrix mini heatmap */
        .weight-matrix-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 5px;
        }

        .weight-head {
            background: var(--bg-secondary);
            border-radius: 5px;
            padding: 5px;
            text-align: center;
        }

        .weight-head .head-label {
            font-size: 0.6em;
            color: var(--text-secondary);
            margin-bottom: 3px;
        }

        .weight-head canvas {
            width: 100%;
            height: 30px;
            border-radius: 3px;
        }

        .weight-head .stats {
            font-size: 0.55em;
            color: var(--accent-cyan);
            margin-top: 2px;
        }

        /* AI Assistant Panel */
        .ai-assistant-panel {
            position: fixed;
            bottom: 80px;
            right: 15px;
            width: 320px;
            max-height: 450px;
            background: linear-gradient(135deg, #1a1a25, #0a1520);
            border: 2px solid var(--accent-purple);
            border-radius: 15px;
            display: none;
            flex-direction: column;
            z-index: 1000;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        .ai-assistant-panel.open {
            display: flex;
        }

        .ai-header {
            padding: 12px 15px;
            background: linear-gradient(135deg, var(--accent-purple), #6633cc);
            border-radius: 13px 13px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .ai-header h3 {
            font-size: 0.95em;
            color: white;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .ai-close {
            background: none;
            border: none;
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            opacity: 0.8;
        }

        .ai-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            max-height: 280px;
        }

        .ai-message {
            margin-bottom: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            line-height: 1.4;
        }

        .ai-message.assistant {
            background: var(--bg-secondary);
            border-left: 3px solid var(--accent-purple);
        }

        .ai-message.user {
            background: rgba(0, 212, 255, 0.15);
            border-left: 3px solid var(--accent-cyan);
        }

        .ai-message.action {
            background: rgba(255, 204, 0, 0.15);
            border-left: 3px solid var(--accent-yellow);
        }

        .ai-action-btn {
            display: inline-block;
            background: var(--accent-green);
            color: #000;
            padding: 6px 12px;
            border-radius: 6px;
            margin-top: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.85em;
        }

        .ai-action-btn:hover {
            background: #00cc6a;
        }

        .ai-action-btn.deny {
            background: var(--accent-red);
            color: white;
            margin-left: 8px;
        }

        .ai-input-area {
            padding: 10px;
            border-top: 1px solid var(--bg-secondary);
            display: flex;
            gap: 8px;
        }

        .ai-input {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
            color: var(--text-primary);
            font-size: 0.85em;
        }

        .ai-send {
            background: var(--accent-purple);
            border: none;
            border-radius: 8px;
            padding: 10px 15px;
            color: white;
            cursor: pointer;
            font-weight: bold;
        }

        .ai-fab {
            position: fixed;
            bottom: 15px;
            right: 15px;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-purple), #6633cc);
            border: none;
            color: white;
            font-size: 1.5em;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(170, 85, 255, 0.4);
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .ai-fab:hover {
            transform: scale(1.1);
        }

        .ai-thinking {
            display: flex;
            align-items: center;
            gap: 5px;
            color: var(--text-secondary);
            font-style: italic;
        }

        .ai-thinking .dot {
            width: 6px;
            height: 6px;
            background: var(--accent-purple);
            border-radius: 50%;
            animation: pulse 1s infinite;
        }

        .ai-thinking .dot:nth-child(2) { animation-delay: 0.2s; }
        .ai-thinking .dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }

        /* Collapsible sections */
        .section-toggle {
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0;
        }

        .section-toggle .toggle-icon {
            transition: transform 0.3s;
        }

        .section-toggle.collapsed .toggle-icon {
            transform: rotate(-90deg);
        }

        .section-content {
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .section-content.collapsed {
            max-height: 0;
        }

        /* External Access Banner */
        .external-access-banner {
            background: linear-gradient(135deg, #1a2a35, #0a1520);
            border: 1px solid var(--accent-purple);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            font-size: 0.85em;
        }

        .external-access-banner .url {
            background: var(--bg-secondary);
            padding: 8px 12px;
            border-radius: 6px;
            margin-top: 8px;
            font-family: monospace;
            word-break: break-all;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .external-access-banner .copy-btn {
            background: var(--accent-cyan);
            border: none;
            color: #000;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }

        /* Alerts Panel */
        .alerts-panel {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--bg-card);
            border-top: 2px solid var(--accent-cyan);
            padding: 10px;
            z-index: 100;
        }

        .alerts-toggle {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }

        .alerts-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .alerts-content.open {
            max-height: 300px;
            margin-top: 10px;
        }

        .alert-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 8px;
        }

        .alert-item .alert-text {
            font-size: 0.85em;
        }

        .alert-toggle {
            width: 50px;
            height: 26px;
            background: #444;
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }

        .alert-toggle.on {
            background: var(--accent-green);
        }

        .alert-toggle::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: left 0.3s;
        }

        .alert-toggle.on::after {
            left: 26px;
        }

        /* Active Alert Notification */
        .active-alert {
            background: var(--accent-red);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: alert-flash 1s infinite;
        }

        @keyframes alert-flash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .active-alert .dismiss {
            margin-left: auto;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
        }

        /* Time info */
        .time-info {
            display: flex;
            justify-content: space-around;
            text-align: center;
            margin-top: 10px;
        }

        .time-item {
            flex: 1;
        }

        .time-value {
            font-size: 1.1em;
            font-weight: bold;
            color: var(--accent-cyan);
        }

        .time-label {
            font-size: 0.7em;
            color: var(--text-secondary);
        }

        /* Modal for inputs */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 200;
            align-items: center;
            justify-content: center;
        }

        .modal.open {
            display: flex;
        }

        .modal-content {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 15px;
            width: 90%;
            max-width: 350px;
        }

        .modal-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            text-align: center;
        }

        .modal-options {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .modal-option {
            background: var(--bg-secondary);
            border: 2px solid #3a3a45;
            border-radius: 10px;
            padding: 15px;
            color: var(--text-primary);
            font-size: 1em;
            cursor: pointer;
            text-align: center;
        }

        .modal-option:hover {
            border-color: var(--accent-cyan);
        }

        .modal-close {
            margin-top: 15px;
            width: 100%;
            padding: 12px;
            background: #444;
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1em;
            cursor: pointer;
        }

        /* Responsive */
        @media (min-width: 600px) {
            body {
                max-width: 500px;
                margin: 0 auto;
            }
        }

        .last-update {
            text-align: center;
            font-size: 0.7em;
            color: var(--text-secondary);
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <!-- Active Alerts Area -->
    <div id="active-alerts"></div>

    <!-- External Access Banner (shown when tunnel active) -->
    <div class="external-access-banner" id="external-banner" style="display: none;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>🌐</span>
            <strong>External Access Active</strong>
        </div>
        <div class="url">
            <span id="external-url">--</span>
            <button class="copy-btn" onclick="copyExternalUrl()">Copy</button>
        </div>
    </div>

    <!-- Header -->
    <div class="header">
        <h1>🧠 PRISM-Zero v3.1</h1>
        <div class="subtitle">HIL Neuromorphic Training Dashboard</div>
        <div id="status-badge" class="status-badge status-running">▶ RUNNING</div>
    </div>

    <!-- Target Info -->
    <div class="card">
        <div class="target-info">
            <div class="target-name" id="target-name">Loading...</div>
            <div class="target-family" id="target-family">--</div>
        </div>
        <div class="progress-info">
            <div class="progress-item">
                <div class="progress-value" id="target-progress">-/-</div>
                <div class="progress-label">TARGETS</div>
            </div>
            <div class="progress-item">
                <div class="progress-value" id="episode-progress">-/-</div>
                <div class="progress-label">EPISODES</div>
            </div>
            <div class="progress-item">
                <div class="progress-value" id="epsilon-value">--%</div>
                <div class="progress-label">EXPLORE</div>
            </div>
        </div>
    </div>

    <!-- Pressure Gauge -->
    <div class="card">
        <div class="card-title">⚡ INTERVENTION PRESSURE</div>
        <div class="pressure-container">
            <div class="pressure-gauge">
                <div class="pressure-fill pressure-nominal" id="pressure-fill" style="width: 0%"></div>
            </div>
            <div class="pressure-status" id="pressure-status">NOMINAL</div>
            <div class="pressure-reasons" id="pressure-reasons">Calculating...</div>
        </div>

        <!-- Recommended Action Panel -->
        <div class="recommendation-panel">
            <div class="recommendation-header">
                <span class="icon">🎯</span>
                <span class="title">RECOMMENDED ACTION</span>
            </div>
            <div id="recommendation-content">
                <div class="no-action-needed">
                    <span class="icon">✨</span>
                    Analyzing training status...
                </div>
            </div>
        </div>
    </div>

    <!-- Stats Grid -->
    <div class="card">
        <div class="card-title">📊 METRICS</div>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value neutral" id="reward-current">--</div>
                <div class="stat-label">CURRENT REWARD</div>
            </div>
            <div class="stat-box">
                <div class="stat-value positive" id="reward-best">--</div>
                <div class="stat-label">BEST REWARD</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="reward-trend">--</div>
                <div class="stat-label">TREND</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="rls-error">--</div>
                <div class="stat-label">RLS ERROR</div>
            </div>
        </div>
    </div>

    <!-- Learning Effectiveness Chart -->
    <div class="card">
        <div class="card-title">📈 LEARNING PROGRESS (Live)</div>
        <div class="chart-container">
            <canvas id="learningChart"></canvas>
        </div>
        <div class="chart-legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #00ff88;"></div>
                <span>Reward</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ff4444;"></div>
                <span>RLS Error (÷1000)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #00d4ff;"></div>
                <span>Epsilon ×100</span>
            </div>
        </div>
    </div>

    <!-- Neural Network Visualization (TensorFlow Playground Style) -->
    <div class="neural-viz-container">
        <div class="neural-header">
            <h2>🧠 Neural Network Playground</h2>
            <span class="arch-badge" id="nn-architecture">--</span>
        </div>

        <!-- Network Flow Diagram -->
        <div class="network-flow" id="network-flow">
            <div class="flow-layer" id="input-layer">
                <div class="neurons" id="input-neurons"></div>
                <div class="count">23</div>
                <div class="label">Features</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-layer" id="reservoir-layer">
                <div class="neurons" id="reservoir-neurons"></div>
                <div class="count" id="reservoir-count">512</div>
                <div class="label">SNN Reservoir</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-layer" id="output-layer">
                <div class="neurons" id="output-neurons"></div>
                <div class="count">20</div>
                <div class="label">Q-Values</div>
            </div>
        </div>

        <!-- Reservoir Activation Heatmap -->
        <div class="heatmap-container">
            <div class="heatmap-title">
                <span>🔥 Reservoir Activations (Live)</span>
                <span class="heatmap-stats" id="reservoir-stats">μ=0.00 σ=0.00 sparse=0%</span>
            </div>
            <canvas id="reservoir-heatmap" class="heatmap-canvas" width="256" height="16"></canvas>
        </div>

        <!-- Q-Value Visualization -->
        <div class="section-toggle" onclick="toggleSection('qvalues')">
            <span class="card-title" style="margin:0;">📊 Q-Values by Parameter</span>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="section-content" id="qvalues-section">
            <div class="qvalue-section" id="qvalue-display">
                <!-- Q-value bars rendered by JS -->
            </div>
        </div>

        <!-- Weight Matrix Visualization -->
        <div class="section-toggle" onclick="toggleSection('weights')">
            <span class="card-title" style="margin:0;">⚖️ RLS Weight Heads</span>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="section-content" id="weights-section">
            <div class="heatmap-container" style="margin-bottom:0;">
                <div class="heatmap-title">
                    <span>Weight Distribution (20 heads × 64 samples)</span>
                </div>
                <canvas id="weight-heatmap" class="heatmap-canvas" width="64" height="20"></canvas>
            </div>
            <div class="weight-matrix-grid" id="weight-stats-grid" style="margin-top:10px;">
                <!-- Weight stats rendered by JS -->
            </div>
        </div>

        <!-- Feature Input Visualization -->
        <div class="section-toggle collapsed" onclick="toggleSection('features')">
            <span class="card-title" style="margin:0;">📥 Input Features</span>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="section-content collapsed" id="features-section">
            <div class="features-grid" id="features-display">
                <!-- Features rendered by JS -->
            </div>
        </div>
    </div>

    <!-- Patience Meter -->
    <div class="card">
        <div class="card-title">⏳ PATIENCE METER</div>
        <div class="patience-bar">
            <div class="patience-fill" id="patience-fill" style="width: 0%"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: var(--text-secondary);">
            <span id="patience-text">0/50 stuck</span>
            <span id="patience-action">Early stop in --</span>
        </div>
    </div>

    <!-- Family Performance -->
    <div class="card">
        <div class="card-title">👨‍👩‍👧‍👦 FAMILY PERFORMANCE</div>
        <div id="family-list">Loading...</div>
    </div>

    <!-- Time Info -->
    <div class="card">
        <div class="card-title">⏱️ TIME</div>
        <div class="time-info">
            <div class="time-item">
                <div class="time-value" id="time-elapsed">--</div>
                <div class="time-label">ELAPSED</div>
            </div>
            <div class="time-item">
                <div class="time-value" id="time-eta">--</div>
                <div class="time-label">ETA</div>
            </div>
            <div class="time-item">
                <div class="time-value" id="lr-mult">1.0x</div>
                <div class="time-label">LR MULT</div>
            </div>
        </div>
    </div>

    <!-- HIL Controls -->
    <div class="card">
        <div class="card-title">🎛️ HIL CONTROLS</div>
        <div class="controls-grid">
            <button class="control-btn spike" onclick="sendCommand('spike')">
                <span class="icon">🔥</span>
                <span>SPIKE</span>
            </button>
            <button class="control-btn" onclick="openModal('epsilon')">
                <span class="icon">🎚️</span>
                <span>EPSILON</span>
            </button>
            <button class="control-btn" onclick="openModal('lr')">
                <span class="icon">📈</span>
                <span>LR RATE</span>
            </button>
            <button class="control-btn pause" onclick="sendCommand('pause')" id="pause-btn">
                <span class="icon">⏸️</span>
                <span>PAUSE</span>
            </button>
            <button class="control-btn resume" onclick="sendCommand('resume')" id="resume-btn">
                <span class="icon">▶️</span>
                <span>RESUME</span>
            </button>
            <button class="control-btn" onclick="sendCommand('checkpoint')">
                <span class="icon">💾</span>
                <span>SAVE</span>
            </button>
            <button class="control-btn" onclick="openModal('alerts')">
                <span class="icon">🔔</span>
                <span>ALERTS</span>
            </button>
            <button class="control-btn" onclick="openModal('tunnel')">
                <span class="icon">🌐</span>
                <span>TUNNEL</span>
            </button>
        </div>
    </div>

    <div class="last-update">Last update: <span id="last-update">--</span></div>

    <!-- AI Assistant FAB -->
    <button class="ai-fab" onclick="toggleAIPanel()" title="AI Training Assistant">
        🤖
    </button>

    <!-- AI Assistant Panel -->
    <div class="ai-assistant-panel" id="ai-panel">
        <div class="ai-header">
            <h3>🤖 AI Training Assistant</h3>
            <button class="ai-close" onclick="toggleAIPanel()">×</button>
        </div>
        <div class="ai-messages" id="ai-messages">
            <div class="ai-message assistant">
                👋 Hi! I'm your AI training assistant powered by Google Gemini. I can analyze your training telemetry and suggest HIL interventions.
                <br><br>
                Ask me things like:
                <br>• "How is training going?"
                <br>• "Should I spike exploration?"
                <br>• "What do you recommend?"
            </div>
        </div>
        <div class="ai-input-area">
            <input type="text" class="ai-input" id="ai-input" placeholder="Ask the AI assistant..." onkeypress="if(event.key==='Enter')sendAIMessage()">
            <button class="ai-send" onclick="sendAIMessage()">Send</button>
        </div>
    </div>

    <!-- Epsilon Modal -->
    <div class="modal" id="epsilon-modal">
        <div class="modal-content">
            <div class="modal-title">🎚️ Set Epsilon</div>
            <div class="modal-options">
                <button class="modal-option" onclick="sendCommand('epsilon', 0.8)">0.8 (High Explore)</button>
                <button class="modal-option" onclick="sendCommand('epsilon', 0.5)">0.5 (Balanced)</button>
                <button class="modal-option" onclick="sendCommand('epsilon', 0.3)">0.3 (More Exploit)</button>
                <button class="modal-option" onclick="sendCommand('epsilon', 0.1)">0.1 (Low Explore)</button>
            </div>
            <button class="modal-close" onclick="closeModal('epsilon')">Cancel</button>
        </div>
    </div>

    <!-- LR Modal -->
    <div class="modal" id="lr-modal">
        <div class="modal-content">
            <div class="modal-title">📈 Learning Rate Multiplier</div>
            <div class="modal-options">
                <button class="modal-option" onclick="sendCommand('lr', 0.3)">0.3x (Slow)</button>
                <button class="modal-option" onclick="sendCommand('lr', 0.5)">0.5x (Careful)</button>
                <button class="modal-option" onclick="sendCommand('lr', 1.0)">1.0x (Normal)</button>
                <button class="modal-option" onclick="sendCommand('lr', 2.0)">2.0x (Fast)</button>
                <button class="modal-option" onclick="sendCommand('lr', 3.0)">3.0x (Aggressive)</button>
            </div>
            <button class="modal-close" onclick="closeModal('lr')">Cancel</button>
        </div>
    </div>

    <!-- Alerts Modal -->
    <div class="modal" id="alerts-modal">
        <div class="modal-content">
            <div class="modal-title">🔔 Alert Settings</div>
            <div id="alert-settings">
                <div class="alert-item">
                    <span class="alert-text">High Pressure (>60%)</span>
                    <div class="alert-toggle on" data-alert="pressure" onclick="toggleAlert(this)"></div>
                </div>
                <div class="alert-item">
                    <span class="alert-text">Stuck Episodes (>35)</span>
                    <div class="alert-toggle on" data-alert="stuck" onclick="toggleAlert(this)"></div>
                </div>
                <div class="alert-item">
                    <span class="alert-text">Reward Declining</span>
                    <div class="alert-toggle on" data-alert="reward" onclick="toggleAlert(this)"></div>
                </div>
                <div class="alert-item">
                    <span class="alert-text">Target Complete</span>
                    <div class="alert-toggle on" data-alert="complete" onclick="toggleAlert(this)"></div>
                </div>
                <div class="alert-item">
                    <span class="alert-text">Training Paused</span>
                    <div class="alert-toggle on" data-alert="paused" onclick="toggleAlert(this)"></div>
                </div>
            </div>
            <button class="modal-close" onclick="closeModal('alerts')">Done</button>
        </div>
    </div>

    <!-- Tunnel Modal -->
    <div class="modal" id="tunnel-modal">
        <div class="modal-content">
            <div class="modal-title">🌐 External Access Setup</div>
            <p style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 15px;">
                Access this dashboard from anywhere outside your local network.
            </p>
            <div class="modal-options">
                <button class="modal-option" onclick="startTunnel('ngrok')">
                    <strong>ngrok</strong><br>
                    <span style="font-size: 0.8em; color: var(--text-secondary);">Fast setup, free tier available</span>
                </button>
                <button class="modal-option" onclick="startTunnel('cloudflare')">
                    <strong>Cloudflare Tunnel</strong><br>
                    <span style="font-size: 0.8em; color: var(--text-secondary);">More secure, requires account</span>
                </button>
                <button class="modal-option" onclick="startTunnel('localtunnel')">
                    <strong>LocalTunnel</strong><br>
                    <span style="font-size: 0.8em; color: var(--text-secondary);">No signup required</span>
                </button>
            </div>
            <div id="tunnel-status" style="margin-top: 15px; padding: 10px; background: var(--bg-secondary); border-radius: 8px; display: none;">
                <div id="tunnel-status-text">Starting tunnel...</div>
            </div>
            <button class="modal-close" onclick="closeModal('tunnel')">Close</button>
        </div>
    </div>

    <script>
        // Learning Chart Setup
        let learningChart = null;
        const chartData = {
            labels: [],
            rewards: [],
            errors: [],
            epsilons: []
        };
        const MAX_CHART_POINTS = 50;

        function initChart() {
            const ctx = document.getElementById('learningChart').getContext('2d');
            learningChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [
                        {
                            label: 'Reward',
                            data: chartData.rewards,
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            tension: 0.3,
                            fill: true,
                            yAxisID: 'y'
                        },
                        {
                            label: 'RLS Error (÷1000)',
                            data: chartData.errors,
                            borderColor: '#ff4444',
                            backgroundColor: 'rgba(255, 68, 68, 0.1)',
                            tension: 0.3,
                            fill: false,
                            yAxisID: 'y1'
                        },
                        {
                            label: 'Epsilon ×100',
                            data: chartData.epsilons,
                            borderColor: '#00d4ff',
                            backgroundColor: 'rgba(0, 212, 255, 0.1)',
                            tension: 0.3,
                            fill: false,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#888',
                                maxTicksLimit: 6
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#00ff88'
                            },
                            title: {
                                display: true,
                                text: 'Reward',
                                color: '#00ff88'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false,
                            },
                            ticks: {
                                color: '#888'
                            },
                            title: {
                                display: true,
                                text: 'Error/Epsilon',
                                color: '#888'
                            }
                        }
                    }
                }
            });
        }

        function updateChart(data) {
            if (!learningChart) return;

            const episode = data.episode || 0;
            const reward = data.episode_reward || 0;
            const error = (data.rls_error || 0) / 1000; // Scale down for visibility
            const epsilon = (data.epsilon || 0) * 100; // Scale up for visibility

            // Add new data point
            chartData.labels.push(`E${episode}`);
            chartData.rewards.push(reward);
            chartData.errors.push(error);
            chartData.epsilons.push(epsilon);

            // Keep only last N points
            if (chartData.labels.length > MAX_CHART_POINTS) {
                chartData.labels.shift();
                chartData.rewards.shift();
                chartData.errors.shift();
                chartData.epsilons.shift();
            }

            learningChart.update('none');
        }

        // External tunnel state
        let tunnelUrl = null;

        function copyExternalUrl() {
            if (tunnelUrl) {
                navigator.clipboard.writeText(tunnelUrl);
                showNotification('📋 URL copied!');
            }
        }

        async function startTunnel(type) {
            const statusDiv = document.getElementById('tunnel-status');
            const statusText = document.getElementById('tunnel-status-text');
            statusDiv.style.display = 'block';
            statusText.innerHTML = `<span style="color: var(--accent-yellow);">⏳ Starting ${type} tunnel...</span>`;

            try {
                const response = await fetch(`/api/tunnel/${type}`, { method: 'POST' });
                const result = await response.json();

                if (result.success && result.url) {
                    tunnelUrl = result.url;
                    statusText.innerHTML = `<span style="color: var(--accent-green);">✅ Tunnel active!</span><br><code>${result.url}</code>`;

                    // Show external banner
                    document.getElementById('external-banner').style.display = 'block';
                    document.getElementById('external-url').textContent = result.url;

                    showNotification('🌐 External access enabled!');
                } else {
                    statusText.innerHTML = `<span style="color: var(--accent-red);">❌ ${result.error || 'Failed to start tunnel'}</span><br>
                        <span style="font-size: 0.85em; color: var(--text-secondary);">
                            Install with: <code>${getInstallCmd(type)}</code>
                        </span>`;
                }
            } catch (e) {
                statusText.innerHTML = `<span style="color: var(--accent-red);">❌ Connection error</span>`;
            }
        }

        function getInstallCmd(type) {
            switch(type) {
                case 'ngrok': return 'snap install ngrok';
                case 'cloudflare': return 'sudo apt install cloudflared';
                case 'localtunnel': return 'npm install -g localtunnel';
                default: return '';
            }
        }

        // Check for existing tunnel on load
        async function checkTunnel() {
            try {
                const response = await fetch('/api/tunnel/status');
                const result = await response.json();
                if (result.active && result.url) {
                    tunnelUrl = result.url;
                    document.getElementById('external-banner').style.display = 'block';
                    document.getElementById('external-url').textContent = result.url;
                }
            } catch (e) {}
        }

        // ============================================================
        // NEURAL NETWORK VISUALIZATION
        // ============================================================

        // Toggle collapsible sections
        function toggleSection(name) {
            const section = document.getElementById(`${name}-section`);
            const toggle = section.previousElementSibling;
            section.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
        }

        // Color scale for heatmaps (blue -> white -> red)
        function valueToColor(value, min, max) {
            // Normalize to [-1, 1]
            const range = Math.max(Math.abs(min), Math.abs(max)) || 1;
            const norm = Math.max(-1, Math.min(1, value / range));

            let r, g, b;
            if (norm < 0) {
                // Blue for negative
                r = Math.floor(128 + 127 * (1 + norm));
                g = Math.floor(128 + 127 * (1 + norm));
                b = 255;
            } else {
                // Red for positive
                r = 255;
                g = Math.floor(128 + 127 * (1 - norm));
                b = Math.floor(128 + 127 * (1 - norm));
            }
            return `rgb(${r},${g},${b})`;
        }

        // Activation color (dark -> bright cyan/green)
        function activationToColor(value) {
            const v = Math.max(0, Math.min(1, value));
            const r = Math.floor(v * 50);
            const g = Math.floor(100 + v * 155);
            const b = Math.floor(150 + v * 105);
            return `rgb(${r},${g},${b})`;
        }

        // Draw reservoir heatmap
        function drawReservoirHeatmap(data) {
            const canvas = document.getElementById('reservoir-heatmap');
            if (!canvas || !data.reservoir_heatmap) return;

            const ctx = canvas.getContext('2d');
            const values = data.reservoir_heatmap;
            const width = Math.min(256, values.length);
            const height = 16;

            // Find min/max for normalization
            const max = Math.max(...values.map(Math.abs)) || 1;

            // Clear canvas
            ctx.fillStyle = '#12121a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw each cell
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height;
            for (let i = 0; i < width; i++) {
                const v = values[i] || 0;
                ctx.fillStyle = activationToColor((v + max) / (2 * max));
                ctx.fillRect(i * cellWidth, 0, cellWidth + 1, cellHeight);
            }

            // Update stats
            const stats = data.reservoir_stats;
            if (stats) {
                document.getElementById('reservoir-stats').textContent =
                    `μ=${stats.mean.toFixed(3)} σ=${stats.std.toFixed(3)} sparse=${(stats.sparsity*100).toFixed(0)}%`;
            }
        }

        // Draw weight heatmap
        function drawWeightHeatmap(data) {
            const canvas = document.getElementById('weight-heatmap');
            if (!canvas || !data.weight_heatmap) return;

            const ctx = canvas.getContext('2d');
            const weights = data.weight_heatmap;
            const numHeads = weights.length;
            const numWeights = weights[0]?.length || 64;

            // Find global min/max
            let globalMax = 0;
            for (const row of weights) {
                for (const w of row) {
                    globalMax = Math.max(globalMax, Math.abs(w));
                }
            }
            globalMax = globalMax || 1;

            // Clear canvas
            ctx.fillStyle = '#12121a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw each cell
            const cellWidth = canvas.width / numWeights;
            const cellHeight = canvas.height / numHeads;
            for (let h = 0; h < numHeads; h++) {
                for (let w = 0; w < numWeights; w++) {
                    const value = weights[h]?.[w] || 0;
                    ctx.fillStyle = valueToColor(value, -globalMax, globalMax);
                    ctx.fillRect(w * cellWidth, h * cellHeight, cellWidth + 1, cellHeight + 1);
                }
            }
        }

        // Render Q-value bars
        function renderQValues(data) {
            const container = document.getElementById('qvalue-display');
            if (!container || !data.q_values_by_param) return;

            const params = data.param_names || ['temp', 'friction', 'spring_k', 'bias'];
            let html = '';

            for (let p = 0; p < data.q_values_by_param.length; p++) {
                const qvals = data.q_values_by_param[p];
                const maxIdx = qvals.indexOf(Math.max(...qvals));
                const maxVal = Math.max(...qvals.map(Math.abs)) || 1;

                html += `<div class="qvalue-param">
                    <div class="param-name">${params[p] || `Param ${p}`}</div>
                    <div class="qvalue-bars">`;

                for (let i = 0; i < qvals.length; i++) {
                    const height = (Math.abs(qvals[i]) / maxVal) * 100;
                    const isSelected = i === maxIdx;
                    html += `<div class="qvalue-bar ${isSelected ? 'selected' : ''}"
                        style="height: ${Math.max(5, height)}%;">
                        <span class="value">${qvals[i].toFixed(3)}</span>
                    </div>`;
                }

                html += `</div></div>`;
            }

            container.innerHTML = html;
        }

        // Render weight stats grid
        function renderWeightStats(data) {
            const container = document.getElementById('weight-stats-grid');
            if (!container || !data.weight_stats) return;

            // Show first 8 heads (4 params × 2)
            const stats = data.weight_stats.slice(0, 8);
            const params = ['T', 'F', 'K', 'B'];

            let html = '';
            for (let i = 0; i < stats.length; i++) {
                const s = stats[i];
                const paramIdx = Math.floor(i / 2);
                const binIdx = i % 2;
                html += `<div class="weight-head">
                    <div class="head-label">${params[paramIdx]}${binIdx}</div>
                    <div class="stats">μ=${s.mean.toFixed(3)} L2=${s.l2_norm.toFixed(2)}</div>
                </div>`;
            }
            container.innerHTML = html;
        }

        // Render feature inputs
        function renderFeatures(data) {
            const container = document.getElementById('features-display');
            if (!container || !data.feature_values) return;

            const names = data.feature_names || [];
            const values = data.feature_values;
            const maxVal = Math.max(...values.map(Math.abs)) || 1;

            let html = '';
            for (let i = 0; i < values.length; i++) {
                const pct = (Math.abs(values[i]) / maxVal) * 100;
                html += `<div class="feature-item">
                    <div class="name">${names[i] || `f${i}`}</div>
                    <div class="value">${values[i].toFixed(2)}</div>
                    <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
                </div>`;
            }
            container.innerHTML = html;
        }

        // Render network flow neurons
        function renderNetworkFlow(data) {
            // Input neurons (show 5 representative)
            const inputNeurons = document.getElementById('input-neurons');
            if (inputNeurons) {
                const feats = data.feature_values || [];
                const maxFeat = Math.max(...feats.map(Math.abs)) || 1;
                let html = '';
                for (let i = 0; i < 5; i++) {
                    const v = Math.abs(feats[i*4] || 0) / maxFeat;
                    const color = activationToColor(v);
                    html += `<div class="neuron ${v > 0.3 ? 'active' : ''}" style="background:${color}"></div>`;
                }
                inputNeurons.innerHTML = html;
            }

            // Reservoir neurons (show 5 representative)
            const resNeurons = document.getElementById('reservoir-neurons');
            if (resNeurons && data.reservoir_heatmap) {
                const vals = data.reservoir_heatmap;
                const maxRes = Math.max(...vals.map(Math.abs)) || 1;
                let html = '';
                const step = Math.floor(vals.length / 5);
                for (let i = 0; i < 5; i++) {
                    const v = (vals[i * step] + maxRes) / (2 * maxRes);
                    const color = activationToColor(v);
                    html += `<div class="neuron ${v > 0.6 ? 'active' : ''}" style="background:${color}"></div>`;
                }
                resNeurons.innerHTML = html;
            }

            // Output neurons (show 5 representative Q-values)
            const outNeurons = document.getElementById('output-neurons');
            if (outNeurons && data.q_values) {
                const qv = data.q_values;
                const maxQ = Math.max(...qv.map(Math.abs)) || 1;
                let html = '';
                for (let i = 0; i < 5; i++) {
                    const v = (qv[i*4] + maxQ) / (2 * maxQ);
                    const color = valueToColor(qv[i*4], -maxQ, maxQ);
                    html += `<div class="neuron ${Math.abs(qv[i*4]) > maxQ*0.5 ? 'active' : ''}" style="background:${color}"></div>`;
                }
                outNeurons.innerHTML = html;
            }

            // Update architecture badge
            if (data.architecture) {
                document.getElementById('nn-architecture').textContent = data.architecture;
            }
            if (data.reservoir_size) {
                document.getElementById('reservoir-count').textContent = data.reservoir_size;
            }
        }

        // Fetch and update neural state
        async function updateNeuralViz() {
            try {
                const response = await fetch('/api/neural_state');
                const data = await response.json();

                if (data.error) {
                    console.log('Neural state not available:', data.error);
                    return;
                }

                drawReservoirHeatmap(data);
                drawWeightHeatmap(data);
                renderQValues(data);
                renderWeightStats(data);
                renderFeatures(data);
                renderNetworkFlow(data);

            } catch (e) {
                console.log('Neural viz update failed:', e);
            }
        }

        // ============================================================
        // AI ASSISTANT (Google Gemini)
        // ============================================================

        let aiPanelOpen = false;
        let pendingAction = null;
        let lastTrainingData = null;

        function toggleAIPanel() {
            aiPanelOpen = !aiPanelOpen;
            document.getElementById('ai-panel').classList.toggle('open', aiPanelOpen);
        }

        function addAIMessage(content, type = 'assistant') {
            const messagesDiv = document.getElementById('ai-messages');
            const msg = document.createElement('div');
            msg.className = `ai-message ${type}`;
            msg.innerHTML = content;
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function addAIActionMessage(content, action) {
            const messagesDiv = document.getElementById('ai-messages');
            const msg = document.createElement('div');
            msg.className = 'ai-message action';
            msg.innerHTML = `
                ${content}
                <br>
                <span class="ai-action-btn" onclick="executeAIAction('${action}')">✓ Approve</span>
                <span class="ai-action-btn deny" onclick="denyAIAction()">✗ Deny</span>
            `;
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            pendingAction = action;
        }

        async function executeAIAction(action) {
            if (!pendingAction) return;

            try {
                const actionMap = {
                    // Main actions
                    'spike_10': { spike_exploration: 10 },
                    'spike_20': { spike_exploration: 20 },
                    'epsilon_high': { set_epsilon: 0.7 },
                    'epsilon_low': { set_epsilon: 0.2 },
                    'lr_high': { learning_rate_multiplier: 3.0 },
                    'lr_low': { learning_rate_multiplier: 0.5 },
                    'lr_normal': { learning_rate_multiplier: 1.0 },
                    'pause': { pause: true },
                    'resume': { pause: false },
                    'checkpoint': { save_checkpoint: true },
                    // Aliases (AI might return these)
                    'spike': { spike_exploration: 10 },
                    'spike10': { spike_exploration: 10 },
                    'spike20': { spike_exploration: 20 },
                    'explore': { spike_exploration: 10 },
                    'exploration': { spike_exploration: 10 },
                    'high_epsilon': { set_epsilon: 0.7 },
                    'low_epsilon': { set_epsilon: 0.2 },
                    'high_lr': { learning_rate_multiplier: 3.0 },
                    'low_lr': { learning_rate_multiplier: 0.5 },
                    'normal_lr': { learning_rate_multiplier: 1.0 },
                    'reset_lr': { learning_rate_multiplier: 1.0 },
                    'save': { save_checkpoint: true },
                    'save_checkpoint': { save_checkpoint: true },
                    'unpause': { pause: false }
                };

                // Normalize action name (lowercase, trim)
                const normalizedAction = action.toLowerCase().trim();
                const command = actionMap[normalizedAction] || actionMap[action];

                console.log('AI Action:', action, 'Normalized:', normalizedAction, 'Command:', command);

                if (command) {
                    const resp = await fetch('/api/command', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(command)
                    });
                    const result = await resp.json();
                    console.log('Command result:', result);
                    addAIMessage(`✅ Action executed: <strong>${action}</strong>`, 'assistant');
                    showNotification(`🤖 AI action: ${action}`);
                } else {
                    console.error('Unknown action:', action);
                    addAIMessage(`⚠️ Unknown action: <strong>${action}</strong>. Available: spike_10, spike_20, epsilon_high, epsilon_low, lr_high, lr_low, pause, resume, checkpoint`, 'assistant');
                }
            } catch (e) {
                addAIMessage(`❌ Failed to execute action: ${e.message}`, 'assistant');
            }
            pendingAction = null;
        }

        function denyAIAction() {
            addAIMessage('❌ Action cancelled by user.', 'assistant');
            pendingAction = null;
        }

        async function sendAIMessage() {
            const input = document.getElementById('ai-input');
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            addAIMessage(message, 'user');

            // Show thinking indicator
            const thinkingId = 'thinking-' + Date.now();
            addAIMessage(`<div class="ai-thinking" id="${thinkingId}"><span class="dot"></span><span class="dot"></span><span class="dot"></span> Analyzing...</div>`, 'assistant');

            try {
                // Get current training status
                const statusResp = await fetch('/api/status');
                lastTrainingData = await statusResp.json();

                // Call AI endpoint
                const response = await fetch('/api/ai/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        training_data: lastTrainingData
                    })
                });

                const result = await response.json();

                // Remove thinking indicator
                document.getElementById(thinkingId)?.parentElement?.remove();

                if (result.error) {
                    addAIMessage(`⚠️ ${result.error}`, 'assistant');
                } else if (result.action) {
                    addAIActionMessage(result.response, result.action);
                } else {
                    addAIMessage(result.response, 'assistant');
                }

            } catch (e) {
                document.getElementById(thinkingId)?.parentElement?.remove();
                addAIMessage(`⚠️ Error: ${e.message}`, 'assistant');
            }
        }

        // Alert settings state
        let alertSettings = {
            pressure: true,
            stuck: true,
            reward: true,
            complete: true,
            paused: true
        };

        // Track previous state for alerts
        let prevState = {
            targetsCompleted: 0,
            paused: false
        };

        // Active alerts
        let activeAlerts = [];

        function toggleAlert(el) {
            const alert = el.dataset.alert;
            alertSettings[alert] = !alertSettings[alert];
            el.classList.toggle('on', alertSettings[alert]);
            localStorage.setItem('alertSettings', JSON.stringify(alertSettings));
        }

        // Load saved alert settings
        const savedAlerts = localStorage.getItem('alertSettings');
        if (savedAlerts) {
            alertSettings = JSON.parse(savedAlerts);
            document.querySelectorAll('.alert-toggle').forEach(el => {
                el.classList.toggle('on', alertSettings[el.dataset.alert]);
            });
        }

        function openModal(type) {
            document.getElementById(type + '-modal').classList.add('open');
        }

        function closeModal(type) {
            document.getElementById(type + '-modal').classList.remove('open');
        }

        function showAlert(message, type = 'warning') {
            const alertId = Date.now();
            activeAlerts.push({ id: alertId, message, type });
            renderAlerts();

            // Vibrate on mobile if supported
            if (navigator.vibrate) {
                navigator.vibrate([200, 100, 200]);
            }
        }

        function dismissAlert(id) {
            activeAlerts = activeAlerts.filter(a => a.id !== id);
            renderAlerts();
        }

        function renderAlerts() {
            const container = document.getElementById('active-alerts');
            container.innerHTML = activeAlerts.map(alert => `
                <div class="active-alert">
                    <span>⚠️ ${alert.message}</span>
                    <button class="dismiss" onclick="dismissAlert(${alert.id})">✕</button>
                </div>
            `).join('');
        }

        async function sendCommand(type, value = null) {
            let command = {};

            switch(type) {
                case 'spike':
                    command = { spike_exploration: 10 };
                    showNotification('🔥 Spike exploration activated!');
                    break;
                case 'epsilon':
                    command = { set_epsilon: value };
                    closeModal('epsilon');
                    showNotification(`🎚️ Epsilon set to ${value}`);
                    break;
                case 'lr':
                    command = { learning_rate_multiplier: value };
                    closeModal('lr');
                    showNotification(`📈 LR multiplier set to ${value}x`);
                    break;
                case 'pause':
                    command = { pause: true };
                    showNotification('⏸️ Training paused!');
                    break;
                case 'resume':
                    command = { pause: false };
                    showNotification('▶️ Training resumed!');
                    break;
                case 'checkpoint':
                    command = { save_checkpoint: true };
                    showNotification('💾 Checkpoint saved!');
                    break;
            }

            try {
                await fetch('/api/command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(command)
                });
            } catch (e) {
                console.error('Command failed:', e);
            }
        }

        function showNotification(message) {
            // Create temporary notification
            const notif = document.createElement('div');
            notif.style.cssText = `
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: var(--accent-green);
                color: black;
                padding: 10px 20px;
                border-radius: 10px;
                font-weight: bold;
                z-index: 300;
                animation: fadeOut 2s forwards;
            `;
            notif.textContent = message;
            document.body.appendChild(notif);
            setTimeout(() => notif.remove(), 2000);
        }

        // Recommendation Engine - analyzes live data and suggests HIL actions
        function calculateRecommendations(data, pressure) {
            const recommendations = [];

            const stuckRatio = data.episodes_without_improvement / data.patience;
            const rewardTrend = data.learning_monitor?.reward_trend || 0;
            const errorTrend = data.learning_monitor?.error_trend || 0;
            const epsilon = data.epsilon || 0;
            const lrMult = data.learning_rate_multiplier || 1.0;
            const recentRewards = data.learning_monitor?.recent_rewards || [];

            // Check if rewards are completely flat (stuck in local minimum)
            const rewardsFlat = recentRewards.length > 5 &&
                new Set(recentRewards.slice(-5).map(r => r.toFixed(6))).size === 1;

            // CRITICAL: Training completely stuck with flat rewards
            if (rewardsFlat && stuckRatio > 0.4) {
                recommendations.push({
                    priority: 'primary',
                    icon: '🔥',
                    name: 'Spike Exploration',
                    reason: `Rewards completely flat for ${recentRewards.length} episodes. Agent is stuck in local minimum - needs random exploration burst to escape.`,
                    confidence: 'high',
                    action: () => sendCommand('spike')
                });
            }

            // HIGH: Stuck for many episodes but not flat
            else if (stuckRatio > 0.6 && !rewardsFlat) {
                recommendations.push({
                    priority: 'primary',
                    icon: '🎚️',
                    name: 'Increase Epsilon to 0.5',
                    reason: `Stuck ${data.episodes_without_improvement}/${data.patience} episodes. Gradual exploration increase may help find better policy without disrupting learned behavior.`,
                    confidence: 'high',
                    action: () => sendCommand('epsilon', 0.5)
                });
            }

            // MEDIUM: Low epsilon while stuck
            else if (epsilon < 0.15 && stuckRatio > 0.3) {
                recommendations.push({
                    priority: 'primary',
                    icon: '🎚️',
                    name: 'Raise Epsilon to 0.3',
                    reason: `Exploration too low (${(epsilon*100).toFixed(0)}%) while stuck. Agent can't discover new strategies - needs more random actions.`,
                    confidence: 'medium',
                    action: () => sendCommand('epsilon', 0.3)
                });
            }

            // Learning rate too high causing instability
            if (errorTrend > 500 && lrMult > 1.0) {
                recommendations.push({
                    priority: 'primary',
                    icon: '📉',
                    name: 'Reduce Learning Rate',
                    reason: `RLS error rising rapidly (+${errorTrend.toFixed(0)}) with LR at ${lrMult.toFixed(1)}x. High learning rate causing weight instability.`,
                    confidence: 'high',
                    action: () => sendCommand('lr', 0.5)
                });
            }

            // Good progress - could accelerate
            if (rewardTrend > 0.001 && errorTrend < 0 && lrMult < 2.0 && stuckRatio < 0.2) {
                recommendations.push({
                    priority: 'secondary',
                    icon: '📈',
                    name: 'Boost Learning Rate',
                    reason: `Training progressing well (reward trend +${rewardTrend.toFixed(4)}, error falling). Could accelerate learning safely.`,
                    confidence: 'medium',
                    action: () => sendCommand('lr', 2.0)
                });
            }

            // Near patience limit - save checkpoint
            if (stuckRatio > 0.8 && pressure > 60) {
                recommendations.push({
                    priority: 'secondary',
                    icon: '💾',
                    name: 'Save Checkpoint Now',
                    reason: `Only ${data.patience - data.episodes_without_improvement} episodes before early stop. Save current weights in case intervention fails.`,
                    confidence: 'high',
                    action: () => sendCommand('checkpoint')
                });
            }

            // Very high epsilon wasting compute
            if (epsilon > 0.6 && stuckRatio < 0.2 && rewardTrend >= 0) {
                recommendations.push({
                    priority: 'secondary',
                    icon: '🎚️',
                    name: 'Lower Epsilon to 0.3',
                    reason: `High exploration (${(epsilon*100).toFixed(0)}%) while making progress. Reduce random actions to exploit learned policy.`,
                    confidence: 'medium',
                    action: () => sendCommand('epsilon', 0.3)
                });
            }

            // Training going well - just observe
            if (recommendations.length === 0) {
                if (pressure < 20) {
                    recommendations.push({
                        priority: 'wait',
                        icon: '✅',
                        name: 'No Action Needed',
                        reason: 'Training is progressing normally. Agent is learning effectively - no intervention required.',
                        confidence: 'low',
                        action: null
                    });
                } else if (pressure < 40) {
                    recommendations.push({
                        priority: 'wait',
                        icon: '👀',
                        name: 'Monitor Closely',
                        reason: 'Some early warning signs. Continue observing - may need intervention if pressure increases.',
                        confidence: 'low',
                        action: null
                    });
                }
            }

            return recommendations;
        }

        function renderRecommendations(recommendations) {
            const container = document.getElementById('recommendation-content');

            if (!recommendations || recommendations.length === 0) {
                container.innerHTML = `
                    <div class="no-action-needed">
                        <span class="icon">✨</span>
                        All systems nominal
                    </div>
                `;
                return;
            }

            let html = '';
            for (const rec of recommendations.slice(0, 2)) { // Show top 2 recommendations
                const confidenceClass = rec.confidence === 'high' ? 'confidence-high' :
                                       rec.confidence === 'medium' ? 'confidence-medium' : 'confidence-low';

                html += `
                    <div class="recommendation-action ${rec.priority}">
                        <div class="action-name">
                            ${rec.icon} ${rec.name}
                            <span class="action-confidence ${confidenceClass}">${rec.confidence.toUpperCase()}</span>
                        </div>
                        <div class="action-reason">${rec.reason}</div>
                        ${rec.action ? `<button class="quick-action-btn" onclick="executeRecommendation(${recommendations.indexOf(rec)})">⚡ Apply Now</button>` : ''}
                    </div>
                `;
            }

            container.innerHTML = html;
        }

        // Store current recommendations for quick action buttons
        let currentRecommendations = [];

        function executeRecommendation(index) {
            if (currentRecommendations[index] && currentRecommendations[index].action) {
                currentRecommendations[index].action();
            }
        }

        function calculatePressure(data) {
            let pressure = 0;
            let reasons = [];

            const stuckRatio = data.episodes_without_improvement / data.patience;
            pressure += stuckRatio * 40;
            if (stuckRatio > 0.5) reasons.push(`Stuck ${data.episodes_without_improvement}/${data.patience}`);

            const rewardTrend = data.learning_monitor?.reward_trend || 0;
            if (rewardTrend < -0.001) {
                pressure += Math.min(25, Math.abs(rewardTrend) * 2500);
                reasons.push('Declining rewards');
            }

            const errorTrend = data.learning_monitor?.error_trend || 0;
            if (errorTrend > 100) {
                pressure += Math.min(20, errorTrend / 500 * 20);
                reasons.push('Rising error');
            }

            if (data.epsilon < 0.2 && stuckRatio > 0.3) {
                pressure += (0.2 - data.epsilon) * 75;
                reasons.push('Low exploration');
            }

            return { pressure: Math.min(100, pressure), reasons };
        }

        async function updateDashboard() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                if (!data || data.error) {
                    document.getElementById('target-name').textContent = 'No data';
                    return;
                }

                // Update target info
                document.getElementById('target-name').textContent = data.current_target || '--';
                document.getElementById('target-family').textContent = data.current_family || '--';
                document.getElementById('target-progress').textContent = `${data.target_idx}/${data.total_targets}`;
                document.getElementById('episode-progress').textContent = `${data.episode}/${data.max_episodes}`;
                document.getElementById('epsilon-value').textContent = `${(data.epsilon * 100).toFixed(0)}%`;

                // Update status badge
                const badge = document.getElementById('status-badge');
                const pauseBtn = document.getElementById('pause-btn');
                if (data.paused) {
                    badge.textContent = '⏸ PAUSED';
                    badge.className = 'status-badge status-paused';
                    pauseBtn.innerHTML = '<span class="icon">▶️</span><span>RESUME</span>';
                } else {
                    badge.textContent = '▶ RUNNING';
                    badge.className = 'status-badge status-running';
                    pauseBtn.innerHTML = '<span class="icon">⏸️</span><span>PAUSE</span>';
                }

                // Calculate and update pressure
                const { pressure, reasons } = calculatePressure(data);
                const pressureFill = document.getElementById('pressure-fill');
                pressureFill.style.width = `${pressure}%`;

                let pressureClass = 'pressure-nominal';
                let statusText = '🟢 NOMINAL';
                if (pressure >= 80) {
                    pressureClass = 'pressure-critical';
                    statusText = '🔴 CRITICAL';
                } else if (pressure >= 60) {
                    pressureClass = 'pressure-high';
                    statusText = '🟠 HIGH';
                } else if (pressure >= 30) {
                    pressureClass = 'pressure-elevated';
                    statusText = '🟡 ELEVATED';
                }

                pressureFill.className = `pressure-fill ${pressureClass}`;
                document.getElementById('pressure-status').textContent = statusText;
                document.getElementById('pressure-reasons').textContent = reasons.length ? reasons.join(' • ') : 'All systems nominal';

                // Calculate and render recommendations
                currentRecommendations = calculateRecommendations(data, pressure);
                renderRecommendations(currentRecommendations);

                // Check for alerts
                if (alertSettings.pressure && pressure >= 60 && !activeAlerts.some(a => a.message.includes('Pressure'))) {
                    showAlert(`High pressure (${pressure.toFixed(0)}%) - Consider intervention`);
                }

                if (alertSettings.stuck && data.episodes_without_improvement >= 35 && !activeAlerts.some(a => a.message.includes('stuck'))) {
                    showAlert(`${data.episodes_without_improvement} episodes stuck - Spike exploration?`);
                }

                if (alertSettings.complete && data.stats.targets_completed > prevState.targetsCompleted) {
                    showAlert(`Target completed! Now on ${data.current_target}`);
                }

                if (alertSettings.paused && data.paused && !prevState.paused) {
                    showAlert('Training has been paused');
                }

                prevState.targetsCompleted = data.stats.targets_completed;
                prevState.paused = data.paused;

                // Update stats
                const reward = data.episode_reward || 0;
                const rewardEl = document.getElementById('reward-current');
                rewardEl.textContent = reward >= 0 ? `+${reward.toFixed(5)}` : reward.toFixed(5);
                rewardEl.className = `stat-value ${reward > 0 ? 'positive' : reward < -0.01 ? 'negative' : 'neutral'}`;

                document.getElementById('reward-best').textContent = `+${(data.best_reward || 0).toFixed(5)}`;

                const trend = data.learning_monitor?.reward_trend || 0;
                const trendEl = document.getElementById('reward-trend');
                trendEl.textContent = `${trend >= 0 ? '📈' : '📉'} ${trend >= 0 ? '+' : ''}${trend.toFixed(4)}`;
                trendEl.className = `stat-value ${trend > 0.001 ? 'positive' : trend < -0.001 ? 'negative' : 'neutral'}`;

                const errorTrend = data.learning_monitor?.error_trend || 0;
                const errorEl = document.getElementById('rls-error');
                errorEl.textContent = `${errorTrend < 0 ? '✅' : '⚠️'} ${(data.rls_error || 0).toFixed(0)}`;
                errorEl.className = `stat-value ${errorTrend < -10 ? 'positive' : errorTrend > 100 ? 'negative' : 'neutral'}`;

                // Update patience
                const stuck = data.episodes_without_improvement || 0;
                const patience = data.patience || 50;
                document.getElementById('patience-fill').style.width = `${(stuck / patience) * 100}%`;
                document.getElementById('patience-text').textContent = `${stuck}/${patience} stuck`;
                document.getElementById('patience-action').textContent = `Early stop in ${patience - stuck}`;

                // Update family performance
                const families = data.learning_monitor?.family_performance || {};
                let familyHtml = '';
                for (const [name, stats] of Object.entries(families).sort()) {
                    const done = stats.targets_completed || 0;
                    const total = stats.targets_count || 1;
                    const pct = (done / total) * 100;
                    const icon = done === total ? '✅' : done > 0 ? '🔄' : '⏳';
                    const avg = stats.avg_best_reward || 0;
                    familyHtml += `
                        <div class="family-item">
                            <span class="family-icon">${icon}</span>
                            <span class="family-name">${name}</span>
                            <div class="family-progress">
                                <div class="family-progress-fill" style="width: ${pct}%"></div>
                            </div>
                            <span class="family-stats">${done}/${total} | ${avg >= 0 ? '+' : ''}${avg.toFixed(4)}</span>
                        </div>
                    `;
                }
                document.getElementById('family-list').innerHTML = familyHtml;

                // Update time
                const elapsed = (data.total_time_secs || 0) / 60;
                const eta = (data.eta_secs || 0) / 60;
                document.getElementById('time-elapsed').textContent = elapsed < 60 ? `${elapsed.toFixed(0)}m` : `${(elapsed/60).toFixed(1)}h`;
                document.getElementById('time-eta').textContent = eta < 60 ? `${eta.toFixed(0)}m` : `${(eta/60).toFixed(1)}h`;
                document.getElementById('lr-mult').textContent = `${(data.learning_rate_multiplier || 1.0).toFixed(1)}x`;

                // Update timestamp
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

                // Update learning chart
                updateChart(data);

                // Update pause/resume button visibility based on state
                const pauseBtnEl = document.getElementById('pause-btn');
                const resumeBtnEl = document.getElementById('resume-btn');
                if (data.paused) {
                    pauseBtnEl.style.opacity = '0.5';
                    resumeBtnEl.style.opacity = '1';
                } else {
                    pauseBtnEl.style.opacity = '1';
                    resumeBtnEl.style.opacity = '0.5';
                }

            } catch (e) {
                console.error('Update failed:', e);
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            checkTunnel();
            updateNeuralViz();
        });

        // Initial update and refresh every 2 seconds
        updateDashboard();
        setInterval(updateDashboard, 2000);

        // Neural viz updates slightly less frequently (every 3 seconds)
        setInterval(updateNeuralViz, 3000);

        // Add CSS animation for notifications
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeOut {
                0% { opacity: 1; }
                70% { opacity: 1; }
                100% { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
"""

def load_status():
    """Load the current HIL status"""
    status_path = os.path.join(OUTPUT_DIR, "hil_status.json")
    try:
        with open(status_path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def send_command(command_dict):
    """Send a command to the HIL control file (merges with existing, preserves all values)"""
    control_path = os.path.join(OUTPUT_DIR, "hil_control.json")
    try:
        # Read existing control file - start with safe defaults
        existing = {}
        if os.path.exists(control_path):
            try:
                with open(control_path) as f:
                    existing = json.load(f)
            except:
                pass

        # Ensure all required fields exist with sensible defaults
        # Also fix null values for fields that shouldn't be null
        if "spike_exploration" not in existing:
            existing["spike_exploration"] = 0
        if "set_epsilon" not in existing:
            existing["set_epsilon"] = None
        # learning_rate_multiplier should NEVER be null - default to 1.0
        if "learning_rate_multiplier" not in existing or existing.get("learning_rate_multiplier") is None:
            existing["learning_rate_multiplier"] = 1.0
        if "save_checkpoint" not in existing:
            existing["save_checkpoint"] = False
        if "pause" not in existing:
            existing["pause"] = False
        if "ack" not in existing:
            existing["ack"] = 0

        # Reset one-shot commands ONLY (these are consumed by trainer)
        # Don't reset learning_rate_multiplier - it's persistent!
        existing["spike_exploration"] = 0
        existing["save_checkpoint"] = False
        existing["set_epsilon"] = None  # This is also one-shot

        # Merge new commands - only update fields that are explicitly provided
        for key, value in command_dict.items():
            # Only update if the value is not None (allows explicit null to clear)
            if value is not None or key in ['set_epsilon']:  # set_epsilon can be None to clear
                existing[key] = value

        # Write merged control file
        with open(control_path, 'w') as f:
            json.dump(existing, f, indent=2)
        return True
    except Exception as e:
        print(f"Error sending command: {e}")
        return False

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    return jsonify(load_status())

@app.route('/api/neural_state')
def api_neural_state():
    """Load neural network internal state for visualization"""
    neural_path = os.path.join(OUTPUT_DIR, "neural_state.json")
    try:
        with open(neural_path) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/command', methods=['POST'])
def api_command():
    command = request.json
    print(f"[HIL] Received command: {command}")
    success = send_command(command)
    # Return the current state after command
    control_path = os.path.join(OUTPUT_DIR, "hil_control.json")
    try:
        with open(control_path) as f:
            current_state = json.load(f)
    except:
        current_state = {}
    print(f"[HIL] Control file after: {current_state}")
    return jsonify({"success": success, "current_state": current_state})

@app.route('/api/control', methods=['GET'])
def api_control():
    """Read current HIL control state"""
    control_path = os.path.join(OUTPUT_DIR, "hil_control.json")
    try:
        with open(control_path) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})

# Google Gemini AI Configuration
GEMINI_API_KEY = "AIzaSyBI6ceNtmg67UK0V38Pz5LW7pMNgawMbug"
GEMINI_MODEL = "gemini-2.0-flash"

@app.route('/api/ai/chat', methods=['POST'])
def api_ai_chat():
    """AI assistant endpoint using Google Gemini"""
    import urllib.request
    import urllib.error

    data = request.json
    user_message = data.get('message', '')
    training_data = data.get('training_data', {})

    # Build context from training data
    context = f"""You are an AI assistant for a neuromorphic machine learning training system called PRISM-Zero.
You help the user monitor and control their training via Human-in-the-Loop (HIL) commands.

Current Training Status:
- Target: {training_data.get('current_target', 'Unknown')} ({training_data.get('current_family', 'Unknown')})
- Progress: Target {training_data.get('target_idx', '?')}/{training_data.get('total_targets', '?')}, Episode {training_data.get('episode', '?')}/{training_data.get('max_episodes', '?')}
- Epsilon (exploration): {training_data.get('epsilon', 0):.4f} ({training_data.get('epsilon', 0)*100:.1f}%)
- Current Reward: {training_data.get('episode_reward', 0):.4f}
- Best Reward: {training_data.get('best_reward', 0):.4f}
- Episodes without improvement: {training_data.get('episodes_without_improvement', 0)}/{training_data.get('patience', 50)}
- Learning Rate Multiplier: {training_data.get('learning_rate_multiplier', 1.0)}x
- Paused: {training_data.get('paused', False)}

Learning Monitor:
- Reward Trend: {training_data.get('learning_monitor', {}).get('reward_trend', 0):.6f}
- Error Trend: {training_data.get('learning_monitor', {}).get('error_trend', 0):.2f}
- RLS Error: {training_data.get('learning_monitor', {}).get('current_rls_error', 0):.2f}

Available HIL Actions you can suggest (user must approve):
- spike_10: Spike exploration for 10 episodes (epsilon -> 0.8)
- spike_20: Spike exploration for 20 episodes
- epsilon_high: Set epsilon to 0.7 (more exploration)
- epsilon_low: Set epsilon to 0.2 (more exploitation)
- lr_high: Set learning rate to 3.0x (faster learning)
- lr_low: Set learning rate to 0.5x (slower, more stable)
- lr_normal: Reset learning rate to 1.0x
- pause: Pause training
- resume: Resume training
- checkpoint: Save a checkpoint

When suggesting an action, respond with JSON in this EXACT format:
{{"response": "Your explanation here", "action": "action_name"}}

For regular responses without actions, just respond with:
{{"response": "Your response here"}}

Be concise, helpful, and proactive in suggesting interventions when the data indicates problems."""

    prompt = f"{context}\n\nUser: {user_message}\n\nAssistant:"

    try:
        # Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500
            }
        }).encode('utf-8')

        req = urllib.request.Request(url, data=payload, headers={
            'Content-Type': 'application/json'
        })

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))

        # Extract response text
        response_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        # Try to parse as JSON (for actions)
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return jsonify(parsed)
        except:
            pass

        # Return as plain response
        return jsonify({"response": response_text})

    except urllib.error.HTTPError as e:
        return jsonify({"error": f"API error: {e.code} - {e.reason}"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Tunnel state
tunnel_process = None
tunnel_url = None

@app.route('/api/tunnel/status')
def api_tunnel_status():
    global tunnel_url, tunnel_process
    active = tunnel_process is not None and tunnel_process.poll() is None
    return jsonify({"active": active, "url": tunnel_url if active else None})

@app.route('/api/tunnel/<tunnel_type>', methods=['POST'])
def api_start_tunnel(tunnel_type):
    global tunnel_process, tunnel_url
    import subprocess
    import re

    # Kill existing tunnel if any
    if tunnel_process:
        tunnel_process.terminate()
        tunnel_process = None
        tunnel_url = None

    try:
        if tunnel_type == 'ngrok':
            # Check if ngrok is installed
            result = subprocess.run(['which', 'ngrok'], capture_output=True)
            if result.returncode != 0:
                return jsonify({"success": False, "error": "ngrok not installed. Run: snap install ngrok"})

            # Start ngrok
            tunnel_process = subprocess.Popen(
                ['ngrok', 'http', '5000', '--log=stdout'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait a moment then get the URL from ngrok API
            time.sleep(3)
            try:
                import urllib.request
                with urllib.request.urlopen('http://localhost:4040/api/tunnels') as response:
                    data = json.loads(response.read())
                    for t in data.get('tunnels', []):
                        if 'https' in t.get('public_url', ''):
                            tunnel_url = t['public_url']
                            return jsonify({"success": True, "url": tunnel_url})
                    # Fallback to http
                    if data.get('tunnels'):
                        tunnel_url = data['tunnels'][0].get('public_url')
                        return jsonify({"success": True, "url": tunnel_url})
            except:
                pass
            return jsonify({"success": False, "error": "Could not get ngrok URL. Check ngrok status."})

        elif tunnel_type == 'cloudflare':
            result = subprocess.run(['which', 'cloudflared'], capture_output=True)
            if result.returncode != 0:
                return jsonify({"success": False, "error": "cloudflared not installed. Run: sudo apt install cloudflared"})

            tunnel_process = subprocess.Popen(
                ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Cloudflare outputs URL to stderr
            time.sleep(5)
            try:
                # Read stderr non-blocking
                import select
                if select.select([tunnel_process.stderr], [], [], 0.1)[0]:
                    output = tunnel_process.stderr.read(4096).decode()
                    match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', output)
                    if match:
                        tunnel_url = match.group(0)
                        return jsonify({"success": True, "url": tunnel_url})
            except:
                pass
            return jsonify({"success": False, "error": "Could not get cloudflare URL. Check output."})

        elif tunnel_type == 'localtunnel':
            result = subprocess.run(['which', 'lt'], capture_output=True)
            if result.returncode != 0:
                return jsonify({"success": False, "error": "localtunnel not installed. Run: npm install -g localtunnel"})

            tunnel_process = subprocess.Popen(
                ['lt', '--port', '5000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(3)
            try:
                output = tunnel_process.stdout.readline().decode()
                match = re.search(r'https://[a-z0-9-]+\.loca\.lt', output)
                if match:
                    tunnel_url = match.group(0)
                    return jsonify({"success": True, "url": tunnel_url})
            except:
                pass
            return jsonify({"success": False, "error": "Could not get localtunnel URL"})

        else:
            return jsonify({"success": False, "error": f"Unknown tunnel type: {tunnel_type}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description='PRISM-Zero HIL Web Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Training output directory')
    args = parser.parse_args()

    OUTPUT_DIR = args.output

    print("=" * 60)
    print("🧠 PRISM-Zero v3.1 WEB HIL DASHBOARD")
    print("=" * 60)
    print(f"📁 Monitoring: {OUTPUT_DIR}")
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print()
    print("📱 ACCESS FROM YOUR PHONE:")
    print(f"   Local network: http://<your-server-ip>:{args.port}")
    print(f"   SSH tunnel:    ssh -L {args.port}:localhost:{args.port} user@server")
    print(f"                  Then open http://localhost:{args.port}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
