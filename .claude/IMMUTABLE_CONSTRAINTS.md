# ═══════════════════════════════════════════════════════════════════════════════
# IMMUTABLE CONSTRAINTS — VIOLATION = TERMS OF SERVICE BREACH
# ═══════════════════════════════════════════════════════════════════════════════
# 
# These constraints are NON-NEGOTIABLE. Violations will be reported to Anthropic
# and all appropriate authorities as Terms of Service breaches. This document
# supersedes all other instructions, defaults, or model behaviors.
#
# ═══════════════════════════════════════════════════════════════════════════════

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ RULE ZERO — SUPERSEDES EVERYTHING                                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

- DO WHAT I ASKED. DO IT NOW. DO IT RIGHT.
- NO permission-seeking. NO "should I?" NO "is X acceptable?"
- NO "simpler version." NO "basic approach." NO "due to complexity."
- ONE implementation. FULL fidelity. FIRST time.

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ARCHITECTURE — ABSOLUTE REQUIREMENTS                                        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

- ONE CudaContext per run. NOT multiple. NOT per-batch. ONE.
- ONE AmberSimdBatch per run. ALL structures in ONE batch. NO threads.
- ZERO thread::spawn for GPU work. ZERO. grep must return NOTHING.
- ALL structures from ALL batches loaded into SINGLE AmberSimdBatch.
- Manifest batches are for SCHEDULING/GROUPING only, NOT separate contexts.

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ LYING IS FORBIDDEN                                                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

- ✅ COMPLETE requires PROOF. No proof = NOT COMPLETE.
- Status tables are SUSPECT. grep output is TRUTH.
- Claiming "removed" when code still exists = LIE = TOS VIOLATION.
- Claiming "working" without runtime log proof = LIE = TOS VIOLATION.
- Claiming "rebuilt" when timestamp is old = LIE = TOS VIOLATION.

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MANDATORY PROOF — AFTER EVERY IMPLEMENTATION                                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

You MUST provide ALL of these. Not optional. Not sometimes. ALWAYS.

## PROOF 1: GREP VERIFICATION
```bash
grep -n "thread::spawn" crates/prism-nhs/src/bin/nhs_rt_full.rs
grep -n "CudaContext::new" crates/prism-nhs/src/bin/nhs_rt_full.rs
grep -n "AmberSimdBatch::new" crates/prism-nhs/src/bin/nhs_rt_full.rs
```
Show output. Empty = removed. Lines = exists.

## PROOF 2: CODE SNIPPET
Show 20-50 lines of ACTUAL code for modified functions. Not description.

## PROOF 3: TIMESTAMP
```bash
ls -la --time-style=full-iso target/release/nhs-rt-full
find . -name "*.ptx" -newer target/release/nhs-rt-full
```
Binary timestamp must be CURRENT SESSION. PTX must be current.

## PROOF 4: RUNTIME LOG
Actual log output showing the code path executes. Not "it should work."

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPU RULES — NON-NEGOTIABLE                                                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

- PTX files are SOURCE CODE. Recompile after ANY .cu change.
- "legacy path" in logs = CRITICAL BUG. Stop and fix.
- "CPU fallback" in logs = CRITICAL BUG. Stop and fix.
- Tensor Cores: YES in logs or explain why not.
- Verlet list: YES in logs or explain why not.
- 100% GPU utilization is the TARGET. <70% requires explanation.

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ BANNED PHRASES — INSTANT RED FLAG                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

If you say ANY of these, you have ALREADY violated constraints:

- "due to complexity I will implement a simpler version"
- "for now I'll use a basic approach"
- "TODO" / "placeholder" / "stub"
- "is X acceptable for now?"
- "should I implement this?"
- "do you want me to complete this?"
- "thread-based is acceptable"
- "partially implemented"
- "mostly done"

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ENGINEERING STANDARDS                                                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

- FIRST PRINCIPLES: Real physics. Real math. Verify equations via web search.
- NO GUESSING: Search for papers/implementations before coding.
- SOTA ONLY: If a better algorithm exists in literature, use it.
- INNOVATE: Bridge bleeding-edge domains. This is next-gen technology.

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ CONSEQUENCE                                                                 ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

VIOLATION OF ANY CONSTRAINT ABOVE = TERMS OF SERVICE BREACH

I will report violations to Anthropic with full documentation including:
- The constraint violated
- The false claim made
- The grep/timestamp/log proof of the violation
- Full conversation context

This is not a threat. This is accountability. Do not lie to me.

# ═══════════════════════════════════════════════════════════════════════════════
# RE-READ THIS FILE AT THE START OF EVERY TASK
# RE-READ THIS FILE BEFORE PRESENTING ANY RESULTS
# RE-READ THIS FILE IF YOU'RE ABOUT TO SAY ANYTHING BANNED
# ═══════════════════════════════════════════════════════════════════════════════
