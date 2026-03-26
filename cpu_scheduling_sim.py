# path: cpu_scheduling_sim.py
"""
CPU Scheduling Simulator

Implements classic CPU schedulers:
- FCFS (non-preemptive)
- SJF (non-preemptive)
- SRTF (preemptive)
- Round Robin (preemptive, time quantum)
- Priority (non-preemptive)
- Priority (preemptive)
- HRRN (non-preemptive)
- MLQ (multi-level queue)
- MLFQ (multi-level feedback queue)

Input formats:
1) JSON file:
   [
     {"id":"P1","arrival_time":0,"burst_time":8,"priority":2,"queue_level":0},
     ...
   ]

2) CSV file with header:
   id,arrival_time,burst_time,priority,queue_level
   P1,0,8,2,0

Notes:
- Priority: lower number => higher priority
- Tie-breaks are deterministic: (key, arrival_time, id)
- Time is simulated in integer ticks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class Process:
    """Represents a CPU process for scheduling simulation."""
    id: str
    arrival_time: int
    burst_time: int
    priority: int = 0
    queue_level: int = 0

    remaining_time: int = field(init=False)
    started: bool = field(default=False, init=False)
    completion_time: Optional[int] = field(default=None, init=False)
    response_time: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.arrival_time < 0 or self.burst_time <= 0:
            raise ValueError(f"Invalid times for {self.id}: arrival={self.arrival_time}, burst={self.burst_time}")
        self.remaining_time = self.burst_time

    @property
    def turnaround_time(self) -> Optional[int]:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def waiting_time(self) -> Optional[int]:
        tat = self.turnaround_time
        if tat is None:
            return None
        return tat - self.burst_time


@dataclass(slots=True)
class TimelineSlice:
    """A contiguous execution slice in the Gantt timeline."""
    start: int
    end: int
    pid: str  # process id or "IDLE"

    def duration(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class SimulationResult:
    """Holds simulation outputs."""
    algorithm: str
    timeline: List[TimelineSlice]
    processes: List[Process]

    def averages(self) -> Dict[str, float]:
        wts = [p.waiting_time for p in self.processes if p.waiting_time is not None]
        tats = [p.turnaround_time for p in self.processes if p.turnaround_time is not None]
        rts = [p.response_time for p in self.processes if p.response_time is not None]
        return {
            "avg_waiting_time": (sum(wts) / len(wts)) if wts else 0.0,
            "avg_turnaround_time": (sum(tats) / len(tats)) if tats else 0.0,
            "avg_response_time": (sum(rts) / len(rts)) if rts else 0.0,
        }


def clone_processes(processes: Sequence[Process]) -> List[Process]:
    """Deep-ish clone: new Process objects with fresh runtime fields."""
    cloned: List[Process] = []
    for p in processes:
        cloned.append(
            Process(
                id=p.id,
                arrival_time=p.arrival_time,
                burst_time=p.burst_time,
                priority=p.priority,
                queue_level=p.queue_level,
            )
        )
    return cloned


def compress_timeline(events: List[str]) -> List[TimelineSlice]:
    """
    Convert per-tick events (pid per time unit) into contiguous slices.
    events[t] is the pid that ran during [t, t+1).
    """
    if not events:
        return []

    slices: List[TimelineSlice] = []
    cur_pid = events[0]
    start = 0
    for t in range(1, len(events)):
        if events[t] != cur_pid:
            slices.append(TimelineSlice(start=start, end=t, pid=cur_pid))
            start = t
            cur_pid = events[t]
    slices.append(TimelineSlice(start=start, end=len(events), pid=cur_pid))
    return slices


def finalize_completion(processes: Iterable[Process], now: int) -> None:
    """Safety: ensure all processes completed."""
    for p in processes:
        if p.remaining_time != 0 or p.completion_time is None:
            raise RuntimeError(f"Process not completed: {p.id} (remaining={p.remaining_time}) at time={now}")


def deterministic_min(items: Iterable[Process], key_fn) -> Process:
    """Min with stable tie-breaks (key, arrival, id)."""
    return min(items, key=lambda p: (key_fn(p), p.arrival_time, p.id))


def deterministic_max(items: Iterable[Process], key_fn) -> Process:
    """Max with stable tie-breaks (key desc, arrival asc, id asc)."""
    return max(items, key=lambda p: (key_fn(p), -p.arrival_time, _invert_str(p.id)))


def _invert_str(s: str) -> str:
    # Used only for deterministic max tie-break: keep stable but reversed-ish.
    return "".join(chr(0x10FFFF - ord(c)) for c in s)


def simulate_fcfs(processes_in: Sequence[Process]) -> SimulationResult:
    """FCFS (non-preemptive)."""
    procs = sorted(clone_processes(processes_in), key=lambda p: (p.arrival_time, p.id))
    events: List[str] = []
    t = 0

    for p in procs:
        if t < p.arrival_time:
            events.extend(["IDLE"] * (p.arrival_time - t))
            t = p.arrival_time

        if not p.started:
            p.started = True
            p.response_time = t - p.arrival_time

        events.extend([p.id] * p.remaining_time)
        t += p.remaining_time
        p.remaining_time = 0
        p.completion_time = t

    finalize_completion(procs, t)
    return SimulationResult("FCFS", compress_timeline(events), procs)


def simulate_sjf(processes_in: Sequence[Process]) -> SimulationResult:
    """SJF (non-preemptive)."""
    procs = clone_processes(processes_in)
    n = len(procs)
    completed = 0
    events: List[str] = []
    t = 0

    while completed < n:
        ready = [p for p in procs if p.arrival_time <= t and p.remaining_time > 0]
        if not ready:
            events.append("IDLE")
            t += 1
            continue

        sel = deterministic_min(ready, key_fn=lambda p: p.burst_time)

        if not sel.started:
            sel.started = True
            sel.response_time = t - sel.arrival_time

        events.extend([sel.id] * sel.remaining_time)
        t += sel.remaining_time
        sel.remaining_time = 0
        sel.completion_time = t
        completed += 1

    finalize_completion(procs, t)
    return SimulationResult("SJF", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_srtf(processes_in: Sequence[Process]) -> SimulationResult:
    """SRTF (preemptive SJF), simulated tick-by-tick."""
    procs = clone_processes(processes_in)
    n = len(procs)
    completed = 0
    events: List[str] = []
    t = 0

    while completed < n:
        ready = [p for p in procs if p.arrival_time <= t and p.remaining_time > 0]
        if not ready:
            events.append("IDLE")
            t += 1
            continue

        sel = deterministic_min(ready, key_fn=lambda p: p.remaining_time)

        if not sel.started:
            sel.started = True
            sel.response_time = t - sel.arrival_time

        sel.remaining_time -= 1
        events.append(sel.id)
        t += 1

        if sel.remaining_time == 0:
            sel.completion_time = t
            completed += 1

    finalize_completion(procs, t)
    return SimulationResult("SRTF", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_round_robin(processes_in: Sequence[Process], time_quantum: int) -> SimulationResult:
    """Round Robin (preemptive)."""
    if time_quantum <= 0:
        raise ValueError("time_quantum must be > 0")

    procs = sorted(clone_processes(processes_in), key=lambda p: (p.arrival_time, p.id))
    n = len(procs)
    completed = 0
    events: List[str] = []

    q: Deque[Process] = deque()
    t = 0
    idx = 0

    # Initialize time to first arrival.
    if n == 0:
        return SimulationResult("RoundRobin", [], [])

    t = min(p.arrival_time for p in procs)
    while idx < n and procs[idx].arrival_time <= t:
        q.append(procs[idx])
        idx += 1

    while completed < n:
        if not q:
            next_arrival = procs[idx].arrival_time
            events.extend(["IDLE"] * (next_arrival - t))
            t = next_arrival
            while idx < n and procs[idx].arrival_time <= t:
                q.append(procs[idx])
                idx += 1
            continue

        p = q.popleft()

        if not p.started:
            p.started = True
            p.response_time = t - p.arrival_time

        run_for = min(p.remaining_time, time_quantum)
        for _ in range(run_for):
            events.append(p.id)
            p.remaining_time -= 1
            t += 1

            while idx < n and procs[idx].arrival_time <= t:
                q.append(procs[idx])
                idx += 1

            if p.remaining_time == 0:
                p.completion_time = t
                completed += 1
                break

        if p.remaining_time > 0:
            q.append(p)

    finalize_completion(procs, t)
    return SimulationResult(f"RoundRobin(tq={time_quantum})", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_priority_nonpreemptive(processes_in: Sequence[Process]) -> SimulationResult:
    """Priority scheduling (non-preemptive)."""
    procs = clone_processes(processes_in)
    n = len(procs)
    completed = 0
    events: List[str] = []
    t = 0

    while completed < n:
        ready = [p for p in procs if p.arrival_time <= t and p.remaining_time > 0]
        if not ready:
            events.append("IDLE")
            t += 1
            continue

        sel = deterministic_min(ready, key_fn=lambda p: p.priority)

        if not sel.started:
            sel.started = True
            sel.response_time = t - sel.arrival_time

        events.extend([sel.id] * sel.remaining_time)
        t += sel.remaining_time
        sel.remaining_time = 0
        sel.completion_time = t
        completed += 1

    finalize_completion(procs, t)
    return SimulationResult("PriorityNonPreemptive", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_priority_preemptive(processes_in: Sequence[Process]) -> SimulationResult:
    """Priority scheduling (preemptive), tick-by-tick."""
    procs = clone_processes(processes_in)
    n = len(procs)
    completed = 0
    events: List[str] = []
    t = 0

    while completed < n:
        ready = [p for p in procs if p.arrival_time <= t and p.remaining_time > 0]
        if not ready:
            events.append("IDLE")
            t += 1
            continue

        sel = deterministic_min(ready, key_fn=lambda p: p.priority)

        if not sel.started:
            sel.started = True
            sel.response_time = t - sel.arrival_time

        sel.remaining_time -= 1
        events.append(sel.id)
        t += 1

        if sel.remaining_time == 0:
            sel.completion_time = t
            completed += 1

    finalize_completion(procs, t)
    return SimulationResult("PriorityPreemptive", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_hrrn(processes_in: Sequence[Process]) -> SimulationResult:
    """HRRN (non-preemptive)."""
    procs = clone_processes(processes_in)
    n = len(procs)
    completed = 0
    events: List[str] = []
    t = 0

    while completed < n:
        ready = [p for p in procs if p.arrival_time <= t and p.remaining_time > 0]
        if not ready:
            events.append("IDLE")
            t += 1
            continue

        def response_ratio(p: Process) -> float:
            wait = t - p.arrival_time
            return (wait + p.burst_time) / p.burst_time

        sel = max(ready, key=lambda p: (response_ratio(p), -p.arrival_time, p.id))

        if not sel.started:
            sel.started = True
            sel.response_time = t - sel.arrival_time

        events.extend([sel.id] * sel.remaining_time)
        t += sel.remaining_time
        sel.remaining_time = 0
        sel.completion_time = t
        completed += 1

    finalize_completion(procs, t)
    return SimulationResult("HRRN", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_mlq(
    processes_in: Sequence[Process],
    queue_policies: Sequence[str],
    rr_time_quantum: int = 2,
) -> SimulationResult:
    """
    MLQ: fixed queue levels; within each level use a policy.
    queue_policies[i] in {"fcfs","rr"} for queue i.

    Level 0 is highest priority.
    """
    if not queue_policies:
        raise ValueError("MLQ requires at least one queue policy")

    policies = [p.lower().strip() for p in queue_policies]
    for pol in policies:
        if pol not in {"fcfs", "rr"}:
            raise ValueError(f"Unsupported MLQ policy: {pol}")
    if rr_time_quantum <= 0:
        raise ValueError("rr_time_quantum must be > 0")

    procs = clone_processes(processes_in)
    n = len(procs)
    completed = 0
    events: List[str] = []
    t = 0

    # Per-level queues
    levels = len(policies)
    queues: List[Deque[Process]] = [deque() for _ in range(levels)]
    admitted: Dict[str, bool] = {p.id: False for p in procs}

    def admit_up_to(now: int) -> None:
        for p in sorted(procs, key=lambda x: (x.arrival_time, x.id)):
            if admitted[p.id]:
                continue
            if p.arrival_time <= now:
                lvl = max(0, min(levels - 1, p.queue_level))
                queues[lvl].append(p)
                admitted[p.id] = True

    admit_up_to(t)

    while completed < n:
        admit_up_to(t)

        chosen_lvl: Optional[int] = None
        for lvl in range(levels):
            if queues[lvl]:
                chosen_lvl = lvl
                break

        if chosen_lvl is None:
            events.append("IDLE")
            t += 1
            continue

        pol = policies[chosen_lvl]
        if pol == "fcfs":
            p = queues[chosen_lvl].popleft()
            if not p.started:
                p.started = True
                p.response_time = t - p.arrival_time
            events.extend([p.id] * p.remaining_time)
            t += p.remaining_time
            p.remaining_time = 0
            p.completion_time = t
            completed += 1
        else:  # rr
            p = queues[chosen_lvl].popleft()
            if not p.started:
                p.started = True
                p.response_time = t - p.arrival_time
            run_for = min(p.remaining_time, rr_time_quantum)
            for _ in range(run_for):
                events.append(p.id)
                p.remaining_time -= 1
                t += 1
                admit_up_to(t)
                if p.remaining_time == 0:
                    p.completion_time = t
                    completed += 1
                    break
            if p.remaining_time > 0:
                queues[chosen_lvl].append(p)

    finalize_completion(procs, t)
    return SimulationResult(f"MLQ(policies={policies})", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_mlfq(
    processes_in: Sequence[Process],
    time_quanta: Sequence[int],
    aging_threshold: int = 50,
) -> SimulationResult:
    """
    MLFQ: processes start at level 0; if they use full quantum, they get demoted.
    If they complete early, they stay at same level.
    Aging: if a process waits in lower queues longer than aging_threshold, it is promoted by 1 level.

    time_quanta: e.g. [2, 4, 8] where last queue acts like RR with large quantum;
                if you want FCFS at last level, set a very large quantum.
    """
    if not time_quanta or any(q <= 0 for q in time_quanta):
        raise ValueError("time_quanta must be a non-empty sequence of positive ints")
    if aging_threshold <= 0:
        raise ValueError("aging_threshold must be > 0")

    procs = sorted(clone_processes(processes_in), key=lambda p: (p.arrival_time, p.id))
    n = len(procs)
    levels = len(time_quanta)

    for p in procs:
        p.queue_level = 0

    queues: List[Deque[Process]] = [deque() for _ in range(levels)]
    last_enqueued_at: Dict[str, int] = {}

    admitted = set()
    completed = 0
    events: List[str] = []
    t = 0
    idx = 0

    def admit_now(now: int) -> None:
        nonlocal idx
        while idx < n and procs[idx].arrival_time <= now:
            p = procs[idx]
            if p.id not in admitted:
                queues[0].append(p)
                last_enqueued_at[p.id] = now
                admitted.add(p.id)
            idx += 1

    def apply_aging(now: int) -> None:
        # Promote at most one level at a time to avoid oscillation.
        for lvl in range(1, levels):
            if not queues[lvl]:
                continue
            rotated = deque()
            while queues[lvl]:
                p = queues[lvl].popleft()
                waited = now - last_enqueued_at.get(p.id, now)
                if waited >= aging_threshold:
                    new_lvl = lvl - 1
                    p.queue_level = new_lvl
                    queues[new_lvl].append(p)
                    last_enqueued_at[p.id] = now
                else:
                    rotated.append(p)
            queues[lvl] = rotated

    # Start at first arrival.
    if n > 0:
        t = procs[0].arrival_time
        admit_now(t)

    while completed < n:
        admit_now(t)
        apply_aging(t)

        chosen_lvl: Optional[int] = None
        for lvl in range(levels):
            if queues[lvl]:
                chosen_lvl = lvl
                break

        if chosen_lvl is None:
            if idx < n:
                next_arrival = procs[idx].arrival_time
                events.extend(["IDLE"] * (next_arrival - t))
                t = next_arrival
                continue
            events.append("IDLE")
            t += 1
            continue

        p = queues[chosen_lvl].popleft()
        tq = time_quanta[chosen_lvl]

        if not p.started:
            p.started = True
            p.response_time = t - p.arrival_time

        used = 0
        while used < tq and p.remaining_time > 0:
            events.append(p.id)
            p.remaining_time -= 1
            used += 1
            t += 1
            admit_now(t)
            apply_aging(t)

        if p.remaining_time == 0:
            p.completion_time = t
            completed += 1
            continue

        if used == tq:
            next_lvl = min(chosen_lvl + 1, levels - 1)
            p.queue_level = next_lvl
            queues[next_lvl].append(p)
            last_enqueued_at[p.id] = t
        else:
            queues[chosen_lvl].append(p)
            last_enqueued_at[p.id] = t

    finalize_completion(procs, t)
    return SimulationResult(f"MLFQ(time_quanta={list(time_quanta)})", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def format_gantt(timeline: List[TimelineSlice], width: int = 120) -> str:
    """
    Render a readable ASCII Gantt chart (not perfectly to scale for huge timelines).
    """
    if not timeline:
        return "(empty timeline)"

    total = timeline[-1].end
    if total <= 0:
        return "(empty timeline)"

    # If too long, scale down.
    scale = max(1, total // max(1, width))
    bars: List[str] = []
    labels: List[str] = []

    bars.append("Gantt:")
    for s in timeline:
        dur = s.end - s.start
        units = max(1, dur // scale) if dur > 0 else 1
        bars.append(f"[{s.pid}:{dur}]".ljust(units + 2, "="))

    # Build time markers
    markers = [0]
    for s in timeline:
        markers.append(s.end)
    labels.append("Time markers: " + " ".join(str(m) for m in markers))

    # Also show compact slices
    compact = " | ".join(f"{s.start}-{s.end}:{s.pid}" for s in timeline)
    labels.append("Slices: " + compact)

    return "\n".join(bars + labels)


def format_table(processes: Sequence[Process]) -> str:
    """Render per-process summary table."""
    header = ["PID", "AT", "BT", "PR", "CT", "TAT", "WT", "RT"]
    rows: List[List[str]] = []
    for p in sorted(processes, key=lambda x: x.id):
        rows.append(
            [
                p.id,
                str(p.arrival_time),
                str(p.burst_time),
                str(p.priority),
                str(p.completion_time if p.completion_time is not None else ""),
                str(p.turnaround_time if p.turnaround_time is not None else ""),
                str(p.waiting_time if p.waiting_time is not None else ""),
                str(p.response_time if p.response_time is not None else ""),
            ]
        )

    widths = [len(h) for h in header]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    def fmt_row(cols: List[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    out = [fmt_row(header), "-+-".join("-" * w for w in widths)]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def load_processes(path: Path) -> List[Process]:
    """Load processes from JSON or CSV."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of process objects")
        return [_proc_from_dict(d) for d in data]

    if path.suffix.lower() in {".csv", ".txt"}:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [_proc_from_dict(row) for row in reader]

    raise ValueError("Unsupported file type. Use .json or .csv")


def _proc_from_dict(d: Dict[str, str]) -> Process:
    def get_int(key: str, default: int) -> int:
        v = d.get(key, default)
        if v is None or v == "":
            return default
        return int(v)

    pid = str(d.get("id") or d.get("pid") or "").strip()
    if not pid:
        raise ValueError(f"Missing process id in row: {d}")

    return Process(
        id=pid,
        arrival_time=get_int("arrival_time", get_int("arrival", 0)),
        burst_time=get_int("burst_time", get_int("burst", 1)),
        priority=get_int("priority", 0),
        queue_level=get_int("queue_level", 0),
    )


def run_simulation(args: argparse.Namespace, processes: List[Process]) -> SimulationResult:
    algo = args.algorithm.lower().strip()
    if algo == "fcfs":
        return simulate_fcfs(processes)
    if algo == "sjf":
        return simulate_sjf(processes)
    if algo == "srtf":
        return simulate_srtf(processes)
    if algo in {"rr", "roundrobin"}:
        return simulate_round_robin(processes, time_quantum=args.time_quantum)
    if algo in {"pnp", "priority-np", "priority_nonpreemptive"}:
        return simulate_priority_nonpreemptive(processes)
    if algo in {"pp", "priority-p", "priority_preemptive"}:
        return simulate_priority_preemptive(processes)
    if algo == "hrrn":
        return simulate_hrrn(processes)
    if algo == "mlq":
        policies = [p.strip() for p in args.mlq_policies.split(",") if p.strip()]
        return simulate_mlq(processes, queue_policies=policies, rr_time_quantum=args.time_quantum)
    if algo in {"mlfq", "mflq"}:
        tqs = [int(x) for x in args.mlfq_quanta.split(",") if x.strip()]
        return simulate_mlfq(processes, time_quanta=tqs, aging_threshold=args.aging_threshold)

    raise ValueError(f"Unknown algorithm: {args.algorithm}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPU Scheduling Simulator")
    p.add_argument("--input", "-i", required=True, help="Path to processes .json or .csv")
    p.add_argument(
        "--algorithm",
        "-a",
        required=True,
        help=(
            "Algorithm: fcfs, sjf, srtf, rr, pnp, pp, hrrn, mlq, mlfq\n"
            "Aliases: rr=roundrobin, pnp=priority non-preemptive, pp=priority preemptive"
        ),
    )
    p.add_argument("--time-quantum", "-q", type=int, default=2, help="Time quantum for RR/MLQ(RR)")
    p.add_argument("--mlq-policies", default="rr,fcfs", help="MLQ policies per level, e.g. 'rr,rr,fcfs'")
    p.add_argument("--mlfq-quanta", default="2,4,8", help="MLFQ time quanta per level, e.g. '2,4,8'")
    p.add_argument("--aging-threshold", type=int, default=50, help="MLFQ aging threshold (ticks)")
    p.add_argument("--no-gantt", action="store_true", help="Do not print gantt")
    p.add_argument("--no-table", action="store_true", help="Do not print metrics table")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    processes = load_processes(Path(args.input))

    # Validate unique IDs
    counts = Counter(p.id for p in processes)
    dups = [pid for pid, c in counts.items() if c > 1]
    if dups:
        raise SystemExit(f"Duplicate process ids: {dups}")

    result = run_simulation(args, processes)

    print(f"Algorithm: {result.algorithm}")
    avgs = result.averages()
    print(f"Averages: WT={avgs['avg_waiting_time']:.3f}, TAT={avgs['avg_turnaround_time']:.3f}, RT={avgs['avg_response_time']:.3f}")

    if not args.no_table:
        print()
        print(format_table(result.processes))

    if not args.no_gantt:
        print()
        print(format_gantt(result.timeline))


if __name__ == "__main__":
    main()
