# Starting a New Claude Code Session for PRISM

## Quick Start

### Step 1: Run Initialization Script

```bash
cd /home/diddy/Desktop/PRISM4D-bio
./scripts/init_phase6_session.sh
```

This reads the Obsidian vault and outputs a ready-to-use prompt with current state.

### Step 2: Copy the Generated Prompt

The script outputs a prompt like:

```
Continue Phase 6 implementation.

**Read these files first:**
1. .obsidian/vault/Sessions/Current Session.md
2. .obsidian/vault/Progress/implementation_status.json
3. CLAUDE.md

**Current State:**
- Phase: 6
- Week: 0
- Progress: 0%
- Completed: 0 / 15 files

**Next Target:** `cryptic_features.rs`
**Plan Section:** 4.1

Read the vault files, confirm the current state, and proceed with implementation.
Update vault files as you work.
```

### Step 3: Paste into Claude Code

Open Claude Code in the project directory and paste the prompt.

---

## What Claude Does Each Session

### On Start
1. Reads vault files for context
2. Confirms current state
3. Identifies next file
4. Requests permission to proceed

### During Work
1. Implements one file at a time
2. Runs tests after each file
3. Updates `Current Session.md` with progress
4. Commits code changes

### On End
1. Updates `implementation_status.json` with completion
2. Updates `PRISM Dashboard.md` with summary
3. Updates `Current Session.md` with continuation context
4. Commits vault changes

---

## Obsidian Vault Structure

```
.obsidian/vault/
├── PRISM Dashboard.md          # Overview (human-readable)
├── Sessions/
│   ├── Current Session.md      # Active session context
│   └── Session Template.md     # Template for logs
├── Progress/
│   └── implementation_status.json  # Machine-readable status
├── Architecture/
│   └── Architecture Overview.md    # System design
├── Plans/
└── Daily/
```

### Key Files

| File | Purpose | Updated By |
|------|---------|------------|
| `implementation_status.json` | Progress tracking | Claude (after each file) |
| `Current Session.md` | Session context | Claude (during session) |
| `PRISM Dashboard.md` | Overview | Claude (end of session) |

---

## Progress Tracking (JSON Schema)

```json
{
  "current_phase": 6,
  "current_week": 0,
  "overall_progress_percent": 0,
  "next_action": {
    "file": "cryptic_features.rs",
    "plan_section": "4.1",
    "week": "1-2"
  },
  "files": {
    "week_1_2": [
      {"name": "file.rs", "status": "pending|completed", "commit": "hash"}
    ]
  },
  "checkpoints": {
    "week_2": {"status": "pending|passed", "requirements": {...}}
  }
}
```

---

## If Context Drifts

Signs:
- Wrong file order
- Wrong parameters
- Skipping tests

Recovery:
```bash
./scripts/phase6_compliance_check.sh
```

Then tell Claude:
```
Stop. Read .obsidian/vault/Progress/implementation_status.json
Current target is: [file from JSON]
Resume from there.
```

---

## Opening Obsidian

To view the vault in Obsidian:

1. Open Obsidian
2. "Open folder as vault"
3. Select `/home/diddy/Desktop/PRISM4D-bio/.obsidian/vault`
4. View Dashboard, progress, session logs

---

## Manual Session Start (No Script)

If the script isn't available:

```
Begin Phase 6 implementation session.

Read these files:
1. .obsidian/vault/Progress/implementation_status.json
2. .obsidian/vault/Sessions/Current Session.md
3. CLAUDE.md

Parse next_action from the JSON. State the current target.
Implement, test, commit. Update vault files after completion.
```

---

## Architecture Reminders

```
AMBER-PRIMARY: AMBER is main physics engine
BETTI NUMBERS: β₀, β₁, β₂ for TDA topology
BLAKE3: Hashing for integrity and caching
RESERVOIR: 512 neurons, RLS λ=0.99
NOVA LIMIT: 512 atoms maximum
ZERO FALLBACK: Explicit GPU error required
```

---

## Success Criteria

Phase 6 complete when:
- ROC AUC ≥ 0.70
- All 15 files implemented
- All checkpoints passed
- `results/PHASE6_FINAL.json` exists
