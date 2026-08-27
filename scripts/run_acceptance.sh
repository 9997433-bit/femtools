#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0

status=0
run_check() {
    "$@" || {
        code=$?
        if [[ ${status} -eq 0 ]]; then
            status=${code}
        fi
    }
}

printf 'Running functional acceptance tests\n'
run_check "${PYTHON}" -m pytest "${ROOT}/tests" -m "not perf"

printf 'Running deterministic API boundary probes\n'
run_check "${PYTHON}" "${ROOT}/scripts/probe_boundaries.py"

if [[ "${FEMTOOLS_PERF:-0}" == "1" ]]; then
    printf 'Running opt-in scaling tests\n'
    run_check "${PYTHON}" -m pytest "${ROOT}/tests/perf" -m perf
else
    printf 'Scaling tests skipped (set FEMTOOLS_PERF=1 to enable)\n'
fi

exit "${status}"
