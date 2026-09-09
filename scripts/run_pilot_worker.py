#!/usr/bin/env python3
"""Apply a process-local GPU allocator budget before entering the frozen trainer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-gib', type=float, default=28.)
    parser.add_argument('trainer', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer = args.trainer.resolve()
    if trainer != ROOT / 'scripts/train_parent_group_pilot.py':
        raise ValueError('This worker only launches the frozen parent-group pilot trainer')
    if not 0 < args.memory_gib <= 28.:
        raise ValueError('Allocator budget must be positive and at most 28 GiB per worker')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    total = torch.cuda.get_device_properties(0).total_memory
    allowed = int(args.memory_gib * 2**30)
    if allowed >= total:
        raise ValueError('Allocator limit must leave space for other processes')
    torch.cuda.set_per_process_memory_fraction(allowed / total, device=0)
    receipt = {'pid': os.getpid(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'allocator_limit_bytes': allowed, 'allocator_limit_gib': args.memory_gib,
               'cuda_context_and_non_allocator_memory_not_in_limit': True,
               'trainer_sha256': hashlib.sha256(trainer.read_bytes()).hexdigest(),
               'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'trainer_arguments': args.arguments}
    output = ROOT / 'results/parent_group_pilot_v1/resource_launches'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{os.getpid()}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'gpu_resource_limit': receipt}), flush=True)
    sys.argv = [str(trainer), *args.arguments]
    runpy.run_path(str(trainer), run_name='__main__')


if __name__ == '__main__':
    main()
