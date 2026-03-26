**# CPU Scheduling Simulator**



**A small, deterministic \*\*CPU scheduling simulator\*\* for common OS scheduling algorithms.**



**Implements:**

**- \*\*FCFS\*\* (non-preemptive)**

**- \*\*SJF\*\* (non-preemptive)**

**- \*\*SRTF\*\* (preemptive)**

**- \*\*Round Robin\*\* (preemptive)**

**- \*\*Priority\*\* non-preemptive / preemptive (lower number = higher priority)**

**- \*\*HRRN\*\* (non-preemptive)**

**- \*\*MLQ\*\* (multi-level queue; fixed levels)**

**- \*\*MLFQ/ MFLQ\*\* (multi-level feedback queue; dynamic levels + aging)**



**Outputs:**

**- Per-process metrics table: \*\*CT, TAT, WT, RT\*\***

**- Averages: \*\*Avg WT, Avg TAT, Avg RT\*\***

**- Compact \*\*Gantt timeline\*\* (slices + time markers)**



**---**



**## Requirements**



**- Python \*\*3.10+\*\* (tested on 3.12)**



**No external dependencies.**



**---**



**## Files**



**- `cpu\_scheduling\_sim.py` — main simulator script**

**- `README.md` — this file**



**---**



**## Quick start**



**### 1) Run the built-in demo (no input file needed)**



**```bash**

**python cpu\_scheduling\_sim.py**

**```**

