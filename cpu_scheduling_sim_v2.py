# path: cpu_scheduling_sim.py
"""
CPU Scheduling Simulator

Implements:
- FCFS, SJF, SRTF, Round Robin
- Priority (NP/P), HRRN
- MLQ, MLFQ

Run:
  python cpu_scheduling_sim.py -i processes.json -a srtf
  python cpu_scheduling_sim.py -i processes.csv  -a rr -q 3

Notebook-friendly:
  If executed with no CLI args, runs a built-in demo workload.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class Process:
    """CPU process for scheduling simulation."""
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
    """Contiguous Gantt slice."""
    start: int
    end: int
    pid: str  # process id or "IDLE"


@dataclass(slots=True)
class SimulationResult:
    """Simulation outputs."""
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
    """Clone processes with fresh runtime state."""
    return [
        Process(
            id=p.id,
            arrival_time=p.arrival_time,
            burst_time=p.burst_time,
            priority=p.priority,
            queue_level=p.queue_level,
        )
        for p in processes
    ]


def compress_timeline(events: List[str]) -> List[TimelineSlice]:
    """Per-tick pid list -> contiguous slices."""
    if not events:
        return []
    out: List[TimelineSlice] = []
    cur = events[0]
    start = 0
    for t in range(1, len(events)):
        if events[t] != cur:
            out.append(TimelineSlice(start=start, end=t, pid=cur))
            start = t
            cur = events[t]
    out.append(TimelineSlice(start=start, end=len(events), pid=cur))
    return out


def deterministic_min(items: Iterable[Process], key_fn) -> Process:
    """Min with deterministic tie-breaks."""
    return min(items, key=lambda p: (key_fn(p), p.arrival_time, p.id))


def simulate_fcfs(processes_in: Sequence[Process]) -> SimulationResult:
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
    return SimulationResult("FCFS", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_sjf(processes_in: Sequence[Process]) -> SimulationResult:
    procs = clone_processes(processes_in)
    n = len(procs)
    done = 0
    t = 0
    events: List[str] = []
    while done < n:
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
        done += 1
    return SimulationResult("SJF", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_srtf(processes_in: Sequence[Process]) -> SimulationResult:
    procs = clone_processes(processes_in)
    n = len(procs)
    done = 0
    t = 0
    events: List[str] = []
    while done < n:
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
            done += 1
    return SimulationResult("SRTF", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_round_robin(processes_in: Sequence[Process], time_quantum: int) -> SimulationResult:
    if time_quantum <= 0:
        raise ValueError("time_quantum must be > 0")

    procs = sorted(clone_processes(processes_in), key=lambda p: (p.arrival_time, p.id))
    n = len(procs)
    if n == 0:
        return SimulationResult("RoundRobin", [], [])

    q: Deque[Process] = deque()
    events: List[str] = []
    t = procs[0].arrival_time
    idx = 0
    done = 0

    def admit(now: int) -> None:
        nonlocal idx
        while idx < n and procs[idx].arrival_time <= now:
            q.append(procs[idx])
            idx += 1

    admit(t)

    while done < n:
        if not q:
            nxt = procs[idx].arrival_time
            events.extend(["IDLE"] * (nxt - t))
            t = nxt
            admit(t)
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
            admit(t)
            if p.remaining_time == 0:
                p.completion_time = t
                done += 1
                break

        if p.remaining_time > 0:
            q.append(p)

    return SimulationResult(f"RoundRobin(tq={time_quantum})", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_priority_nonpreemptive(processes_in: Sequence[Process]) -> SimulationResult:
    procs = clone_processes(processes_in)
    n = len(procs)
    done = 0
    t = 0
    events: List[str] = []
    while done < n:
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
        done += 1
    return SimulationResult("PriorityNonPreemptive", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_priority_preemptive(processes_in: Sequence[Process]) -> SimulationResult:
    procs = clone_processes(processes_in)
    n = len(procs)
    done = 0
    t = 0
    events: List[str] = []
    while done < n:
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
            done += 1
    return SimulationResult("PriorityPreemptive", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_hrrn(processes_in: Sequence[Process]) -> SimulationResult:
    procs = clone_processes(processes_in)
    n = len(procs)
    done = 0
    t = 0
    events: List[str] = []
    while done < n:
        ready = [p for p in procs if p.arrival_time <= t and p.remaining_time > 0]
        if not ready:
            events.append("IDLE")
            t += 1
            continue

        def rr(p: Process) -> float:
            wait = t - p.arrival_time
            return (wait + p.burst_time) / p.burst_time

        sel = max(ready, key=lambda p: (rr(p), -p.arrival_time, p.id))
        if not sel.started:
            sel.started = True
            sel.response_time = t - sel.arrival_time
        events.extend([sel.id] * sel.remaining_time)
        t += sel.remaining_time
        sel.remaining_time = 0
        sel.completion_time = t
        done += 1
    return SimulationResult("HRRN", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_mlq(processes_in: Sequence[Process], queue_policies: Sequence[str], rr_time_quantum: int = 2) -> SimulationResult:
    policies = [p.lower().strip() for p in queue_policies]
    if not policies or any(p not in {"fcfs", "rr"} for p in policies):
        raise ValueError("MLQ policies must be comma list of: fcfs, rr")
    if rr_time_quantum <= 0:
        raise ValueError("rr_time_quantum must be > 0")

    procs = clone_processes(processes_in)
    n = len(procs)
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

    t = 0
    done = 0
    events: List[str] = []

    while done < n:
        admit_up_to(t)
        chosen = next((lvl for lvl in range(levels) if queues[lvl]), None)
        if chosen is None:
            events.append("IDLE")
            t += 1
            continue

        pol = policies[chosen]
        p = queues[chosen].popleft()
        if not p.started:
            p.started = True
            p.response_time = t - p.arrival_time

        if pol == "fcfs":
            events.extend([p.id] * p.remaining_time)
            t += p.remaining_time
            p.remaining_time = 0
            p.completion_time = t
            done += 1
        else:
            run_for = min(p.remaining_time, rr_time_quantum)
            for _ in range(run_for):
                events.append(p.id)
                p.remaining_time -= 1
                t += 1
                admit_up_to(t)
                if p.remaining_time == 0:
                    p.completion_time = t
                    done += 1
                    break
            if p.remaining_time > 0:
                queues[chosen].append(p)

    return SimulationResult(f"MLQ(policies={policies})", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def simulate_mlfq(processes_in: Sequence[Process], time_quanta: Sequence[int], aging_threshold: int = 50) -> SimulationResult:
    if not time_quanta or any(q <= 0 for q in time_quanta):
        raise ValueError("time_quanta must be non-empty positive ints")
    if aging_threshold <= 0:
        raise ValueError("aging_threshold must be > 0")

    procs = sorted(clone_processes(processes_in), key=lambda p: (p.arrival_time, p.id))
    n = len(procs)
    levels = len(time_quanta)
    for p in procs:
        p.queue_level = 0

    queues: List[Deque[Process]] = [deque() for _ in range(levels)]
    last_enqueued: Dict[str, int] = {}
    events: List[str] = []

    t = procs[0].arrival_time if n else 0
    idx = 0
    done = 0

    def admit(now: int) -> None:
        nonlocal idx
        while idx < n and procs[idx].arrival_time <= now:
            p = procs[idx]
            queues[0].append(p)
            last_enqueued[p.id] = now
            idx += 1

    def aging(now: int) -> None:
        for lvl in range(1, levels):
            if not queues[lvl]:
                continue
            keep = deque()
            while queues[lvl]:
                p = queues[lvl].popleft()
                waited = now - last_enqueued.get(p.id, now)
                if waited >= aging_threshold:
                    p.queue_level = lvl - 1
                    queues[lvl - 1].append(p)
                    last_enqueued[p.id] = now
                else:
                    keep.append(p)
            queues[lvl] = keep

    admit(t)

    while done < n:
        admit(t)
        aging(t)

        chosen = next((lvl for lvl in range(levels) if queues[lvl]), None)
        if chosen is None:
            if idx < n:
                nxt = procs[idx].arrival_time
                events.extend(["IDLE"] * (nxt - t))
                t = nxt
                continue
            events.append("IDLE")
            t += 1
            continue

        p = queues[chosen].popleft()
        tq = time_quanta[chosen]
        if not p.started:
            p.started = True
            p.response_time = t - p.arrival_time

        used = 0
        while used < tq and p.remaining_time > 0:
            events.append(p.id)
            p.remaining_time -= 1
            used += 1
            t += 1
            admit(t)
            aging(t)

        if p.remaining_time == 0:
            p.completion_time = t
            done += 1
            continue

        if used == tq:
            nxt_lvl = min(chosen + 1, levels - 1)
            p.queue_level = nxt_lvl
            queues[nxt_lvl].append(p)
            last_enqueued[p.id] = t
        else:
            queues[chosen].append(p)
            last_enqueued[p.id] = t

    return SimulationResult(f"MLFQ(time_quanta={list(time_quanta)})", compress_timeline(events), sorted(procs, key=lambda p: p.id))


def format_table(processes: Sequence[Process]) -> str:
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

    def fmt(cols: List[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    out = [fmt(header), "-+-".join("-" * w for w in widths)]
    out.extend(fmt(r) for r in rows)
    return "\n".join(out)


def format_gantt(timeline: List[TimelineSlice]) -> str:
    if not timeline:
        return "(empty timeline)"
    markers = [timeline[0].start] + [s.end for s in timeline]
    slices = " | ".join(f"{s.start}-{s.end}:{s.pid}" for s in timeline)
    return "Gantt:\n" + slices + "\nTime markers: " + " ".join(str(m) for m in markers)


def load_processes(path: Path) -> List[Process]:
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
    def get_int(k: str, default: int) -> int:
        v = d.get(k, default)
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
    p.add_argument("--input", "-i", help="Path to processes .json or .csv")
    p.add_argument("--algorithm", "-a", help="fcfs, sjf, srtf, rr, pnp, pp, hrrn, mlq, mlfq")
    p.add_argument("--time-quantum", "-q", type=int, default=2, help="Time quantum for RR/MLQ(RR)")
    p.add_argument("--mlq-policies", default="rr,fcfs", help="MLQ policies per level, e.g. 'rr,rr,fcfs'")
    p.add_argument("--mlfq-quanta", default="2,4,8", help="MLFQ time quanta per level, e.g. '2,4,8'")
    p.add_argument("--aging-threshold", type=int, default=50, help="MLFQ aging threshold (ticks)")
    p.add_argument("--no-gantt", action="store_true", help="Do not print gantt")
    p.add_argument("--no-table", action="store_true", help="Do not print metrics table")
    p.add_argument("--demo", action="store_true", help="Run built-in demo workload")
    return p


def demo_workload() -> List[Process]:
    return [
        Process(id="P1", arrival_time=0, burst_time=8, priority=2, queue_level=0),
        Process(id="P2", arrival_time=1, burst_time=4, priority=1, queue_level=0),
        Process(id="P3", arrival_time=2, burst_time=9, priority=3, queue_level=1),
        Process(id="P4", arrival_time=3, burst_time=5, priority=2, queue_level=1),
    ]


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    no_cli_args = argv is None and len(sys.argv) == 1
    if no_cli_args:
        args.demo = True
        args.algorithm = "srtf"

    if args.demo:
        processes = demo_workload()
    else:
        if not args.input or not args.algorithm:
            parser.error("the following arguments are required: --input/-i and --algorithm/-a (or use --demo)")
        processes = load_processes(Path(args.input))

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
