"""
cpu_scheduling_sim.py
=====================
Deterministic CPU Scheduling Simulator
OBINexus Computing — Filter-Flash Epsilon Matrix Reference
Hetero/Homogeneous Contextus Tripartite Polar Priority Architecture

Algorithms implemented:
  1. FCFS        — First Come, First Serve            (non-preemptive)
  2. SJF         — Shortest Job First                 (non-preemptive)
  3. SRTF        — Shortest Remaining Time First      (preemptive)
  4. RR          — Round Robin                        (preemptive)
  5. PRI_NP      — Priority Non-Preemptive
  6. PRI_P       — Priority Preemptive
  7. HRRN        — Highest Response Ratio Next        (non-preemptive)
  8. MLQ         — Multi-Level Queue                  (fixed levels)
  9. MFLQ        — Multi-Level Feedback Queue         (dynamic + aging)

Outputs per run:
  • Per-process table  : CT | TAT | WT | RT
  • Summary averages   : Avg WT | Avg TAT | Avg RT
  • Gantt timeline     : slices + time markers

Tripartite Polar Priority (from lecture):
  • THERE_AND_THEN  = historical burst cost   (completed work)
  • HERE_AND_NOW    = remaining burst time     (current work)
  • WHEN_AND_WHERE  = arrival + wait context   (positional cost)
  These three axes form the BiColor min/max dual-heap used in
  priority-preemptive and MFLQ scheduling.

Python 3.10+  |  No external dependencies.
"""

from __future__ import annotations
import heapq
import copy
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Process:
    pid:            str
    arrival:        int
    burst:          int
    priority:       int  = 0    # lower = higher priority
    queue_level:    int  = 0    # for MLQ / MFLQ

    # computed fields (filled by simulator)
    remaining:      int  = field(init=False)
    completion:     int  = field(default=0, init=False)
    tat:            int  = field(default=0, init=False)  # turnaround
    wt:             int  = field(default=0, init=False)  # waiting
    rt:             int  = field(default=-1, init=False) # response
    started:        bool = field(default=False, init=False)

    def __post_init__(self):
        self.remaining = self.burst

    def reset(self):
        self.remaining  = self.burst
        self.completion = 0
        self.tat        = 0
        self.wt         = 0
        self.rt         = -1
        self.started    = False

    def record_response(self, t: int):
        if not self.started:
            self.rt      = t - self.arrival
            self.started = True

    def finalize(self, t: int):
        self.completion = t
        self.tat        = self.completion - self.arrival
        self.wt         = self.tat - self.burst
        if self.rt < 0:
            self.rt = self.wt   # fallback if never separately recorded


# Gantt entry
@dataclass
class Slice:
    pid:   str
    start: int
    end:   int


# ─────────────────────────────────────────────────────────────────────────────
# TRIPARTITE POLAR PRIORITY  (BiColor dual-heap wrapper)
#   MIN-heap  → selects shortest / highest-priority / earliest-arrival
#   MAX-heap  → selects longest-waiting / lowest-priority (aging target)
# ─────────────────────────────────────────────────────────────────────────────

class TripartiteHeap:
    """
    Dual BiColor heap encoding three polar axes:
      axis_0  THERE_AND_THEN  = completed work (burst - remaining)
      axis_1  HERE_AND_NOW    = remaining burst time
      axis_2  WHEN_AND_WHERE  = wait time since arrival
    Supports min-extract (preemptive selection) and aging.
    """
    def __init__(self):
        self._min_heap: list = []   # (key, insertion_order, process)
        self._counter  = 0

    def push(self, p: Process, key: float):
        heapq.heappush(self._min_heap, (key, self._counter, p))
        self._counter += 1

    def pop(self) -> Optional[Process]:
        while self._min_heap:
            _, _, p = heapq.heappop(self._min_heap)
            if p.remaining > 0:
                return p
        return None

    def peek_key(self) -> Optional[float]:
        while self._min_heap:
            key, _, p = self._min_heap[0]
            if p.remaining > 0:
                return key
            heapq.heappop(self._min_heap)
        return None

    def is_empty(self) -> bool:
        return all(p.remaining == 0 for _, _, p in self._min_heap)

    def clear(self):
        self._min_heap.clear()


# ─────────────────────────────────────────────────────────────────────────────
# GANTT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_gantt(gantt: list[Slice], width: int = 80) -> str:
    if not gantt:
        return "(empty)"

    # Merge consecutive same-pid slices
    merged: list[Slice] = []
    for s in gantt:
        if merged and merged[-1].pid == s.pid and merged[-1].end == s.start:
            merged[-1].end = s.end
        else:
            merged.append(copy.copy(s))

    # Build bar
    bar   = ""
    times = ""
    for s in merged:
        label  = f" {s.pid} "
        w      = max(len(label), 3)
        bar   += f"|{label:^{w}}"
        times += f"{s.start:<{w+1}}"
    bar   += "|"
    times += str(merged[-1].end)

    lines = [
        "─" * min(len(bar), width),
        bar[:width],
        times[:width],
        "─" * min(len(bar), width),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(procs: list[Process], algo_name: str):
    print(f"\n{'═'*60}")
    print(f"  {algo_name}")
    print(f"{'═'*60}")
    print(f"{'PID':<8}{'AT':>5}{'BT':>5}{'PRI':>5}{'CT':>6}{'TAT':>6}{'WT':>6}{'RT':>6}")
    print(f"{'─'*60}")
    for p in procs:
        print(f"{p.pid:<8}{p.arrival:>5}{p.burst:>5}{p.priority:>5}"
              f"{p.completion:>6}{p.tat:>6}{p.wt:>6}{p.rt:>6}")
    print(f"{'─'*60}")
    avg_wt  = sum(p.wt  for p in procs) / len(procs)
    avg_tat = sum(p.tat for p in procs) / len(procs)
    avg_rt  = sum(p.rt  for p in procs) / len(procs)
    print(f"{'Avg':<8}{'':>5}{'':>5}{'':>5}{'':>6}{avg_tat:>6.2f}{avg_wt:>6.2f}{avg_rt:>6.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FCFS  — First Come First Serve  (non-preemptive)
# ─────────────────────────────────────────────────────────────────────────────

def fcfs(raw: list[Process]) -> tuple[list[Process], list[Slice]]:
    procs = [copy.deepcopy(p) for p in raw]
    procs.sort(key=lambda p: (p.arrival, p.pid))
    t, gantt = 0, []
    for p in procs:
        t = max(t, p.arrival)
        p.record_response(t)
        gantt.append(Slice(p.pid, t, t + p.burst))
        t += p.burst
        p.finalize(t)
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 2. SJF  — Shortest Job First  (non-preemptive)
# ─────────────────────────────────────────────────────────────────────────────

def sjf(raw: list[Process]) -> tuple[list[Process], list[Slice]]:
    procs   = [copy.deepcopy(p) for p in raw]
    done    = set()
    t, gantt = 0, []
    n       = len(procs)

    while len(done) < n:
        ready = [p for p in procs if p.arrival <= t and p.pid not in done]
        if not ready:
            t += 1
            continue
        sel = min(ready, key=lambda p: (p.burst, p.arrival, p.pid))
        sel.record_response(t)
        gantt.append(Slice(sel.pid, t, t + sel.burst))
        t  += sel.burst
        sel.finalize(t)
        done.add(sel.pid)
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 3. SRTF  — Shortest Remaining Time First  (preemptive SJF)
# ─────────────────────────────────────────────────────────────────────────────

def srtf(raw: list[Process]) -> tuple[list[Process], list[Slice]]:
    procs     = [copy.deepcopy(p) for p in raw]
    t         = 0
    completed = 0
    n         = len(procs)
    gantt: list[Slice] = []
    current_pid = None

    while completed < n:
        ready = [p for p in procs if p.arrival <= t and p.remaining > 0]
        if not ready:
            t += 1
            continue
        sel = min(ready, key=lambda p: (p.remaining, p.arrival, p.pid))
        sel.record_response(t)

        # Build gantt: one tick at a time but merge into slices
        if gantt and gantt[-1].pid == sel.pid and gantt[-1].end == t:
            gantt[-1].end += 1
        else:
            gantt.append(Slice(sel.pid, t, t + 1))

        sel.remaining -= 1
        t += 1
        if sel.remaining == 0:
            sel.finalize(t)
            completed += 1
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 4. Round Robin  (preemptive)
# ─────────────────────────────────────────────────────────────────────────────

def round_robin(raw: list[Process], tq: int = 2) -> tuple[list[Process], list[Slice]]:
    procs    = sorted([copy.deepcopy(p) for p in raw], key=lambda p: p.arrival)
    queue: list[Process] = []
    t        = 0
    idx      = 0      # next unadmitted process pointer
    gantt: list[Slice] = []
    completed = 0
    n         = len(procs)

    # Admit first batch
    while idx < n and procs[idx].arrival <= t:
        queue.append(procs[idx])
        idx += 1

    while completed < n:
        if not queue:
            t = procs[idx].arrival
            queue.append(procs[idx])
            idx += 1

        p = queue.pop(0)
        p.record_response(t)
        run  = min(p.remaining, tq)
        gantt.append(Slice(p.pid, t, t + run))
        t            += run
        p.remaining  -= run

        # Admit processes that arrived during this slice
        while idx < n and procs[idx].arrival <= t:
            queue.append(procs[idx])
            idx += 1

        if p.remaining == 0:
            p.finalize(t)
            completed += 1
        else:
            queue.append(p)   # re-queue

    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 5a. Priority — Non-Preemptive
# ─────────────────────────────────────────────────────────────────────────────

def priority_np(raw: list[Process]) -> tuple[list[Process], list[Slice]]:
    procs = [copy.deepcopy(p) for p in raw]
    done  = set()
    t, gantt = 0, []
    n = len(procs)

    while len(done) < n:
        ready = [p for p in procs if p.arrival <= t and p.pid not in done]
        if not ready:
            t += 1
            continue
        sel = min(ready, key=lambda p: (p.priority, p.arrival, p.pid))
        sel.record_response(t)
        gantt.append(Slice(sel.pid, t, t + sel.burst))
        t  += sel.burst
        sel.finalize(t)
        done.add(sel.pid)
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Priority — Preemptive  (BiColor TripartiteHeap)
# ─────────────────────────────────────────────────────────────────────────────

def priority_p(raw: list[Process]) -> tuple[list[Process], list[Slice]]:
    procs     = [copy.deepcopy(p) for p in raw]
    t         = 0
    completed = 0
    n         = len(procs)
    gantt: list[Slice] = []

    while completed < n:
        ready = [p for p in procs if p.arrival <= t and p.remaining > 0]
        if not ready:
            t += 1
            continue

        # HERE_AND_NOW axis: select minimum priority number (highest urgency)
        sel = min(ready, key=lambda p: (p.priority, p.remaining, p.pid))
        sel.record_response(t)

        if gantt and gantt[-1].pid == sel.pid and gantt[-1].end == t:
            gantt[-1].end += 1
        else:
            gantt.append(Slice(sel.pid, t, t + 1))

        sel.remaining -= 1
        t += 1
        if sel.remaining == 0:
            sel.finalize(t)
            completed += 1
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 6. HRRN — Highest Response Ratio Next  (non-preemptive)
#    WHEN_AND_WHERE axis: ratio = (wait + burst) / burst
# ─────────────────────────────────────────────────────────────────────────────

def hrrn(raw: list[Process]) -> tuple[list[Process], list[Slice]]:
    procs = [copy.deepcopy(p) for p in raw]
    done  = set()
    t, gantt = 0, []
    n = len(procs)

    while len(done) < n:
        ready = [p for p in procs if p.arrival <= t and p.pid not in done]
        if not ready:
            t += 1
            continue
        # Response ratio = (wait + burst) / burst
        def rr(p: Process) -> float:
            wait = t - p.arrival
            return (wait + p.burst) / p.burst
        sel = max(ready, key=rr)
        sel.record_response(t)
        gantt.append(Slice(sel.pid, t, t + sel.burst))
        t  += sel.burst
        sel.finalize(t)
        done.add(sel.pid)
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 7. MLQ — Multi-Level Queue  (fixed assignment, FCFS within each level)
#    Level 0 = highest priority; processes assigned by queue_level attribute.
# ─────────────────────────────────────────────────────────────────────────────

def mlq(raw: list[Process], num_levels: int = 3) -> tuple[list[Process], list[Slice]]:
    procs     = [copy.deepcopy(p) for p in raw]
    done      = set()
    t, gantt  = 0, []
    n         = len(procs)

    while len(done) < n:
        # Find highest-priority non-empty level
        sel = None
        for level in range(num_levels):
            ready = sorted(
                [p for p in procs if p.queue_level == level
                 and p.arrival <= t and p.pid not in done],
                key=lambda p: p.arrival
            )
            if ready:
                sel = ready[0]
                break

        if sel is None:
            t += 1
            continue

        sel.record_response(t)
        gantt.append(Slice(sel.pid, t, t + sel.burst))
        t  += sel.burst
        sel.finalize(t)
        done.add(sel.pid)
    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# 8. MFLQ — Multi-Level Feedback Queue  (dynamic + aging)
#    Filter-Flash F∘G composition: demotion on full TQ use,
#    aging (THERE_AND_THEN axis) promotes starved processes.
# ─────────────────────────────────────────────────────────────────────────────

def mflq(
    raw:            list[Process],
    num_levels:     int       = 3,
    time_quanta:    list[int] = None,
    aging_thresh:   int       = 20,
) -> tuple[list[Process], list[Slice]]:

    if time_quanta is None:
        time_quanta = [2 ** i for i in range(num_levels)]  # 1,2,4,...

    procs = [copy.deepcopy(p) for p in raw]
    for p in procs:
        p.queue_level = 0   # all start at top

    queues: list[list[Process]] = [[] for _ in range(num_levels)]
    admitted     = set()
    enqueue_time: dict[str, int] = {}

    def admit_arrivals(t: int):
        for p in procs:
            if p.arrival <= t and p.pid not in admitted and p.remaining > 0:
                queues[0].append(p)
                enqueue_time[p.pid] = t
                admitted.add(p.pid)

    t         = 0
    completed = 0
    n         = len(procs)
    gantt: list[Slice] = []

    admit_arrivals(t)

    while completed < n:
        # Aging: promote long-waiting processes in lower queues
        for level in range(1, num_levels):
            to_promote = []
            stay       = []
            for p in queues[level]:
                wait_in_queue = t - enqueue_time.get(p.pid, t)
                if wait_in_queue >= aging_thresh:
                    new_level = max(0, level - 1)
                    p.queue_level = new_level
                    to_promote.append((p, new_level))
                else:
                    stay.append(p)
            queues[level] = stay
            for p, nl in to_promote:
                queues[nl].append(p)
                enqueue_time[p.pid] = t

        # Select from highest non-empty queue
        sel_level = None
        for level in range(num_levels):
            if queues[level]:
                sel_level = level
                break

        if sel_level is None:
            t += 1
            admit_arrivals(t)
            continue

        p  = queues[sel_level].pop(0)
        tq = time_quanta[sel_level]
        p.record_response(t)

        run = min(p.remaining, tq)
        gantt.append(Slice(p.pid, t, t + run))
        t            += run
        p.remaining  -= run

        admit_arrivals(t)

        if p.remaining == 0:
            p.finalize(t)
            completed += 1
        else:
            # Demote if used full quantum (THERE_AND_THEN: work done → move down)
            if run == tq:
                next_level = min(sel_level + 1, num_levels - 1)
            else:
                next_level = sel_level   # voluntary yield — stay
            p.queue_level = next_level
            queues[next_level].append(p)
            enqueue_time[p.pid] = t

    return procs, gantt


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER  — executes one algorithm, prints results
# ─────────────────────────────────────────────────────────────────────────────

def run(name: str, algo, processes: list[Process], **kwargs):
    procs, gantt = algo(processes, **kwargs)
    print_metrics(procs, name)
    print("\n  Gantt:")
    print("  " + render_gantt(gantt).replace("\n", "\n  "))


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — built-in test datasets
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  OBINexus CPU Scheduling Simulator                     ║")
    print("║  Tripartite Polar Priority | Filter-Flash Epsilon      ║")
    print("╚" + "═"*58 + "╝")

    # ── Dataset A: classic mix
    A = [
        Process("P1", arrival=0,  burst=5, priority=3, queue_level=0),
        Process("P2", arrival=1,  burst=3, priority=1, queue_level=0),
        Process("P3", arrival=2,  burst=8, priority=2, queue_level=1),
        Process("P4", arrival=3,  burst=6, priority=4, queue_level=1),
        Process("P5", arrival=4,  burst=2, priority=2, queue_level=2),
    ]

    # ── Dataset B: documentation worked examples
    # SRTF question from docs: P0(0,9), P1(1,4), P2(2,9)  → avg WT = 5.0
    B_srtf = [
        Process("P0", arrival=0, burst=9),
        Process("P1", arrival=1, burst=4),
        Process("P2", arrival=2, burst=9),
    ]

    # RR/SRTF question from docs: P1-P4  → avg TAT = 5.50
    B_rr = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=3),
        Process("P4", arrival=4, burst=1),
    ]

    # ── Run all algorithms on Dataset A ──────────────────────────────────────
    print("\n" + "▶"*3 + "  Dataset A — 5 processes with priority + queue levels")
    run("1. FCFS",               fcfs,        A)
    run("2. SJF  (non-preempt)", sjf,         A)
    run("3. SRTF (preemptive)",  srtf,        A)
    run("4. Round Robin TQ=2",   round_robin, A, tq=2)
    run("5. Priority NP",        priority_np, A)
    run("6. Priority P",         priority_p,  A)
    run("7. HRRN",               hrrn,        A)
    run("8. MLQ  (3 levels)",    mlq,         A, num_levels=3)
    run("9. MFLQ (3 levels)",    mflq,        A, num_levels=3,
                                               time_quanta=[2,4,8],
                                               aging_thresh=10)

    # ── Verify documentation examples ─────────────────────────────────────────
    print("\n" + "▶"*3 + "  Dataset B — Verifying documented worked examples")

    procs_b, _ = srtf(B_srtf)
    avg_wt = sum(p.wt for p in procs_b) / len(procs_b)
    print(f"\n  SRTF (P0/P1/P2) — Expected avg WT = 5.00 | Got = {avg_wt:.2f}")
    for p in procs_b:
        print(f"    {p.pid}: WT={p.wt}")

    procs_c, _ = srtf(B_rr)
    avg_tat = sum(p.tat for p in procs_c) / len(procs_c)
    print(f"\n  SRTF (P1-P4)    — Expected avg TAT = 5.50 | Got = {avg_tat:.2f}")
    for p in procs_c:
        print(f"    {p.pid}: TAT={p.tat}")

    print("\n" + "═"*60)
    print("  Simulation complete.")
    print("═"*60 + "\n")


if __name__ == "__main__":
    demo()
