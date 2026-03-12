#!/usr/bin/env python3
"""nsys_query.py -- Run SQL queries on nsys-exported SQLite databases.

Supports two communication analysis modes:
- mock: use mock D2H memcpy as the communication critical path
- real: use NCCL kernels as the communication critical path
"""

import csv
import os
import sqlite3
import sys
import time


def log(msg):
    print(msg, flush=True)


def create_indices(db_path):
    label = os.path.basename(db_path)
    log(f"  Creating indices on {label}...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    indices = [
        ("idx_memcpy_copykind", "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind, deviceId, start, end"),
        ("idx_memcpy_copykind_bytes", "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind, bytes"),
        ("idx_kernel_name", "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"),
        ("idx_kernel_device_time", "CUPTI_ACTIVITY_KIND_KERNEL", "deviceId, start, end"),
        ("idx_stringids_value", "StringIds", "value"),
        ("idx_nvtx_text_time", "NVTX_EVENTS", "text, start, end"),
    ]
    for name, table, cols in indices:
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols})")
        except Exception as e:
            log(f"    Warning: index {name}: {e}")
    conn.commit()
    conn.close()
    log(f"  Indices created in {time.time() - t0:.1f}s")


def run_query(db_path, query, output_path, label, params=None):
    log(f"\n--- {label} ---")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query, params or {})
    rows = cursor.fetchall()
    elapsed = time.time() - t0

    if not rows:
        log(f"  (no results) [{elapsed:.1f}s]")
        conn.close()
        return

    keys = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in rows:
            writer.writerow(list(row))

    log(",".join(keys))
    for row in rows:
        log(",".join(str(v) for v in row))
    log(f"  -> Saved to {os.path.basename(output_path)} [{elapsed:.1f}s]")
    conn.close()


def fetch_scalar(conn, query, params=None, default=0):
    row = conn.execute(query, params or {}).fetchone()
    if row is None:
        return default
    return row[0] if row[0] is not None else default


def detect_mock_d2h_bytes(db_path):
    conn = sqlite3.connect(db_path)
    query = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
)
SELECT bytes, COUNT(*) AS cnt
FROM memcpy_base
WHERE copyKind = 2 AND bytes > 1000000
GROUP BY bytes
ORDER BY cnt DESC, bytes DESC
LIMIT 1;
"""
    row = conn.execute(query).fetchone()
    denoise_windows = fetch_scalar(
        conn,
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='SGL_DENOISING_LOOP' AND end IS NOT NULL",
        default=0,
    )
    prefetch_ranges = fetch_scalar(
        conn,
        "SELECT COUNT(*) FROM NVTX_EVENTS "
        "WHERE text IN ('SGL_PREFETCH_H2D','SGL_PREFETCH_H2D_BG','SGL_PREFETCH_H2D_SYNC','SGL_PREFETCH_H2D_CHUNK') "
        "AND end IS NOT NULL",
        default=0,
    )
    mock_nvtx_ranges = fetch_scalar(
        conn,
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='SGL_MOCK_COMM_PCIE' AND end IS NOT NULL",
        default=0,
    )
    conn.close()

    if row is None:
        return {
            "bytes": None,
            "count": 0,
            "denoise_windows": denoise_windows,
            "prefetch_ranges": prefetch_ranges,
            "mock_nvtx_ranges": mock_nvtx_ranges,
        }
    return {
        "bytes": int(row[0]),
        "count": int(row[1]),
        "denoise_windows": denoise_windows,
        "prefetch_ranges": prefetch_ranges,
        "mock_nvtx_ranges": mock_nvtx_ranges,
    }


H2D_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
)
SELECT
    'H2D' as direction,
    COUNT(*) as num_transfers,
    SUM(bytes) as total_bytes,
    SUM(bytes) / 1e9 as total_gb,
    SUM(end - start) / 1e9 as total_time_s,
    (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9) as avg_bw_gbps,
    AVG(bytes / 1e6) as avg_size_mb,
    MIN(bytes / 1e6) as min_size_mb,
    MAX(bytes / 1e6) as max_size_mb,
    AVG((end - start) / 1e6) as avg_dur_ms,
    MIN((end - start) / 1e6) as min_dur_ms,
    MAX((end - start) / 1e6) as max_dur_ms
FROM memcpy_base
WHERE copyKind = 1;
"""

D2H_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
)
SELECT
    'D2H' as direction,
    COUNT(*) as num_transfers,
    SUM(bytes) as total_bytes,
    SUM(bytes) / 1e9 as total_gb,
    SUM(end - start) / 1e9 as total_time_s,
    (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9) as avg_bw_gbps,
    AVG(bytes / 1e6) as avg_size_mb,
    MIN(bytes / 1e6) as min_size_mb,
    MAX(bytes / 1e6) as max_size_mb,
    AVG((end - start) / 1e6) as avg_dur_ms,
    MIN((end - start) / 1e6) as min_dur_ms,
    MAX((end - start) / 1e6) as max_dur_ms
FROM memcpy_base
WHERE copyKind = 2;
"""

MOCK_D2H_TOP_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
)
SELECT
    bytes,
    COUNT(*) as count,
    SUM(bytes) / 1e9 as total_gb,
    AVG((end - start) / 1e6) as avg_dur_ms
FROM memcpy_base
WHERE copyKind = 2 AND bytes > 1000000
GROUP BY bytes
ORDER BY count DESC, bytes DESC
LIMIT 10;
"""

PREFETCH_H2D_STATS_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.*
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
)
SELECT
    COUNT(*) as prefetch_h2d_count,
    SUM(bytes) / 1e9 as prefetch_h2d_total_gb,
    SUM(end - start) / 1e9 as prefetch_h2d_total_time_s,
    (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9) as prefetch_h2d_avg_bw_gbps,
    AVG(bytes / 1e6) as prefetch_h2d_avg_size_mb,
    AVG((end - start) / 1e6) as prefetch_h2d_avg_dur_ms,
    SUM(CASE WHEN bytes < 4194304 THEN 1 ELSE 0 END) as small_lt4mb_count,
    CASE
        WHEN COUNT(*) = 0 THEN NULL
        ELSE 100.0 * SUM(CASE WHEN bytes < 4194304 THEN 1 ELSE 0 END) / COUNT(*)
    END as small_lt4mb_ratio_pct,
    SUM(CASE WHEN bytes > 67108864 THEN 1 ELSE 0 END) as large_gt64mb_count,
    CASE
        WHEN COUNT(*) = 0 THEN NULL
        ELSE 100.0 * SUM(CASE WHEN bytes > 67108864 THEN 1 ELSE 0 END) / COUNT(*)
    END as large_gt64mb_ratio_pct
FROM prefetch_h2d;
"""

H2D_BUCKETS_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.*
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
)
SELECT
    CASE
        WHEN bytes < 4194304 THEN '<4MB'
        WHEN bytes < 16777216 THEN '4-16MB'
        WHEN bytes < 67108864 THEN '16-64MB'
        ELSE '>64MB'
    END as size_bucket,
    COUNT(*) as count,
    SUM(bytes) / 1e9 as total_gb,
    AVG((bytes / 1e6) / ((end - start) / 1e9)) as avg_bw_mbps,
    SUM(end - start) / 1e9 as total_time_s
FROM prefetch_h2d
GROUP BY size_bucket
ORDER BY total_gb DESC;
"""

PREFETCH_MODE_BREAKDOWN_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.*
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
),
tagged AS (
    SELECT
        m.*,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM NVTX_EVENTS n
                WHERE n.text = 'SGL_PREFETCH_H2D_SYNC'
                  AND n.end IS NOT NULL
                  AND n.start < m.end
                  AND n.end > m.start
            ) THEN 'sync'
            WHEN EXISTS (
                SELECT 1
                FROM NVTX_EVENTS n
                WHERE n.text = 'SGL_PREFETCH_H2D_BG'
                  AND n.end IS NOT NULL
                  AND n.start < m.end
                  AND n.end > m.start
            ) THEN 'background'
            WHEN EXISTS (
                SELECT 1
                FROM NVTX_EVENTS n
                WHERE n.text IN (
                      'SGL_PREFETCH_H2D',
                      'SGL_PREFETCH_H2D_BG',
                      'SGL_PREFETCH_H2D_SYNC',
                      'SGL_PREFETCH_H2D_CHUNK'
                )
                  AND n.end IS NOT NULL
                  AND n.start < m.end
                  AND n.end > m.start
            ) THEN 'background'
            ELSE 'unlabeled'
        END as prefetch_mode
    FROM prefetch_h2d m
)
SELECT
    prefetch_mode,
    COUNT(*) as copy_count,
    SUM(bytes) / 1e9 as total_gb,
    SUM(end - start) / 1e9 as total_time_s,
    (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9) as avg_bw_gbps,
    AVG(bytes / 1e6) as avg_size_mb,
    AVG((end - start) / 1e6) as avg_dur_ms
FROM tagged
GROUP BY prefetch_mode
ORDER BY total_gb DESC;
"""

PREFETCH_ENSURE_MEMORY_RATIO_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.bytes, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
),
ensure_ranges AS (
    SELECT n.start, n.end
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND n.text = 'SGL_PREFETCH_ENSURE_READY'
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < n.end AND d.end > n.start
            )
      )
),
tagged AS (
    SELECT
        h.*,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM ensure_ranges e
                WHERE e.start < h.end
                  AND e.end > h.start
            ) THEN 1
            ELSE 0
        END as in_ensure_window
    FROM prefetch_h2d h
)
SELECT
    COUNT(*) as total_copy_count,
    SUM(bytes) as total_prefetch_bytes,
    SUM(bytes) / 1e9 as total_prefetch_gb,
    SUM(CASE WHEN in_ensure_window = 1 THEN 1 ELSE 0 END) as ensure_copy_count,
    SUM(CASE WHEN in_ensure_window = 1 THEN bytes ELSE 0 END) as ensure_prefetch_bytes,
    SUM(CASE WHEN in_ensure_window = 1 THEN bytes ELSE 0 END) / 1e9 as ensure_prefetch_gb,
    CASE
        WHEN SUM(bytes) = 0 THEN NULL
        ELSE 100.0 * SUM(CASE WHEN in_ensure_window = 1 THEN bytes ELSE 0 END) / SUM(bytes)
    END as ensure_prefetch_ratio_pct
FROM tagged;
"""

PREFETCH_WAIT_BREAKDOWN_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
prefetch_wait_events AS (
    SELECT
        n.text,
        (n.end - n.start) / 1e6 as dur_ms
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND n.text IN (
            'SGL_PREFETCH_WAIT_COMM',
            'SGL_PREFETCH_WAIT_COMM_BG',
            'SGL_PREFETCH_WAIT_COMM_SYNC',
            'SGL_PREFETCH_WAIT_LOCK_BG',
            'SGL_PREFETCH_WAIT_LOCK_SYNC',
            'SGL_PREFETCH_ENSURE_READY'
      )
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < n.end AND d.end > n.start
            )
      )
)
SELECT
    text as marker,
    COUNT(*) as num_ranges,
    SUM(dur_ms) as total_ms,
    AVG(dur_ms) as avg_ms,
    MIN(dur_ms) as min_ms,
    MAX(dur_ms) as max_ms
FROM prefetch_wait_events
GROUP BY text
ORDER BY total_ms DESC;
"""

MOCK_COMM_TIMING_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
mock_timing AS (
    SELECT
        n.text,
        (n.end - n.start) / 1e6 as dur_ms
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND n.text IN ('SGL_MOCK_COMM_PCIE', 'SGL_MOCK_COMM_ACTIVE')
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < n.end AND d.end > n.start
            )
      )
)
SELECT
    text as marker,
    COUNT(*) as num_ranges,
    SUM(dur_ms) as total_ms,
    AVG(dur_ms) as avg_ms,
    MIN(dur_ms) as min_ms,
    MAX(dur_ms) as max_ms
FROM mock_timing
GROUP BY text
ORDER BY total_ms DESC;
"""

MOCK_D2H_OVERLAP_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
),
mock_d2h AS (
    SELECT
        m.start,
        m.end,
        m.bytes,
        m.deviceId,
        (m.bytes / 1e6) / ((m.end - m.start) / 1e9) as bw_mbps,
        (m.end - m.start) / 1e6 as dur_ms
    FROM memcpy_base m
    WHERE m.copyKind = 2
      AND m.bytes > 1000000
      AND (:mock_bytes <= 0 OR m.bytes = :mock_bytes)
),
mock_tagged AS (
    SELECT
        d.*,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM prefetch_h2d h
                WHERE h.deviceId = d.deviceId
                  AND h.start < d.end
                  AND h.end > d.start
            ) THEN 1
            ELSE 0
        END as overlap_prefetch_h2d
    FROM mock_d2h d
)
SELECT
    'during_prefetch_h2d' as context,
    COUNT(*) as mock_d2h_count,
    SUM(bytes) / 1e9 as total_mock_d2h_gb,
    AVG(bw_mbps) as avg_mock_d2h_bw_mbps,
    AVG(dur_ms) as avg_mock_d2h_dur_ms,
    MIN(dur_ms) as min_mock_d2h_dur_ms,
    MAX(dur_ms) as max_mock_d2h_dur_ms
FROM mock_tagged
WHERE overlap_prefetch_h2d = 1
UNION ALL
SELECT
    'without_prefetch_h2d' as context,
    COUNT(*) as mock_d2h_count,
    SUM(bytes) / 1e9 as total_mock_d2h_gb,
    AVG(bw_mbps) as avg_mock_d2h_bw_mbps,
    AVG(dur_ms) as avg_mock_d2h_dur_ms,
    MIN(dur_ms) as min_mock_d2h_dur_ms,
    MAX(dur_ms) as max_mock_d2h_dur_ms
FROM mock_tagged
WHERE overlap_prefetch_h2d = 0;
"""

MOCK_D2H_OVERLAP_RATIO_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
),
mock_d2h AS (
    SELECT
        m.start,
        m.end,
        m.bytes,
        m.deviceId,
        (m.bytes / 1e6) / ((m.end - m.start) / 1e9) as bw_mbps,
        (m.end - m.start) / 1e6 as dur_ms
    FROM memcpy_base m
    WHERE m.copyKind = 2
      AND m.bytes > 1000000
      AND (:mock_bytes <= 0 OR m.bytes = :mock_bytes)
),
mock_tagged AS (
    SELECT
        d.*,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM prefetch_h2d h
                WHERE h.deviceId = d.deviceId
                  AND h.start < d.end
                  AND h.end > d.start
            ) THEN 1
            ELSE 0
        END as overlap_prefetch_h2d
    FROM mock_d2h d
)
SELECT
    COUNT(*) as total_mock_d2h_count,
    SUM(CASE WHEN overlap_prefetch_h2d = 1 THEN 1 ELSE 0 END) as overlapped_mock_d2h_count,
    CASE
        WHEN COUNT(*) = 0 THEN NULL
        ELSE 100.0 * SUM(CASE WHEN overlap_prefetch_h2d = 1 THEN 1 ELSE 0 END) / COUNT(*)
    END as overlapped_mock_d2h_ratio_pct,
    AVG(CASE WHEN overlap_prefetch_h2d = 1 THEN dur_ms END) as avg_dur_overlapped_ms,
    AVG(CASE WHEN overlap_prefetch_h2d = 0 THEN dur_ms END) as avg_dur_non_overlapped_ms,
    AVG(CASE WHEN overlap_prefetch_h2d = 1 THEN bw_mbps END) as avg_bw_overlapped_mbps,
    AVG(CASE WHEN overlap_prefetch_h2d = 0 THEN bw_mbps END) as avg_bw_non_overlapped_mbps
FROM mock_tagged;
"""

REAL_COMM_OVERLAP_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < m.end AND d.end > m.start
            )
      )
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
          AND (
                (SELECT COUNT(*) FROM denoise_windows) = 0
                OR EXISTS (
                    SELECT 1 FROM denoise_windows d
                    WHERE d.start < m.end AND d.end > m.start
                )
          )
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.deviceId
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < m.end AND d.end > m.start
            )
      )
),
real_comm AS (
    SELECT
        k.start,
        k.end,
        k.deviceId,
        (k.end - k.start) / 1e6 as dur_ms
    FROM kernel_base k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE s.value LIKE '%nccl%'
),
real_tagged AS (
    SELECT
        d.*,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM prefetch_h2d h
                WHERE h.deviceId = d.deviceId
                  AND h.start < d.end
                  AND h.end > d.start
            ) THEN 1
            ELSE 0
        END as overlap_prefetch_h2d
    FROM real_comm d
)
SELECT
    'during_prefetch_h2d' as context,
    COUNT(*) as comm_count,
    SUM(dur_ms) as total_comm_ms,
    AVG(dur_ms) as avg_comm_dur_ms,
    MIN(dur_ms) as min_comm_dur_ms,
    MAX(dur_ms) as max_comm_dur_ms
FROM real_tagged
WHERE overlap_prefetch_h2d = 1
UNION ALL
SELECT
    'without_prefetch_h2d' as context,
    COUNT(*) as comm_count,
    SUM(dur_ms) as total_comm_ms,
    AVG(dur_ms) as avg_comm_dur_ms,
    MIN(dur_ms) as min_comm_dur_ms,
    MAX(dur_ms) as max_comm_dur_ms
FROM real_tagged
WHERE overlap_prefetch_h2d = 0;
"""

REAL_COMM_OVERLAP_RATIO_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < m.end AND d.end > m.start
            )
      )
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
          AND (
                (SELECT COUNT(*) FROM denoise_windows) = 0
                OR EXISTS (
                    SELECT 1 FROM denoise_windows d
                    WHERE d.start < m.end AND d.end > m.start
                )
          )
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.deviceId
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
      AND (
            (SELECT COUNT(*) FROM denoise_windows) = 0
            OR EXISTS (
                SELECT 1 FROM denoise_windows d
                WHERE d.start < m.end AND d.end > m.start
            )
      )
),
real_comm AS (
    SELECT
        k.start,
        k.end,
        k.deviceId,
        (k.end - k.start) / 1e6 as dur_ms
    FROM kernel_base k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE s.value LIKE '%nccl%'
),
real_tagged AS (
    SELECT
        d.*,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM prefetch_h2d h
                WHERE h.deviceId = d.deviceId
                  AND h.start < d.end
                  AND h.end > d.start
            ) THEN 1
            ELSE 0
        END as overlap_prefetch_h2d
    FROM real_comm d
)
SELECT
    COUNT(*) as total_comm_count,
    SUM(CASE WHEN overlap_prefetch_h2d = 1 THEN 1 ELSE 0 END) as overlapped_comm_count,
    CASE
        WHEN COUNT(*) = 0 THEN NULL
        ELSE 100.0 * SUM(CASE WHEN overlap_prefetch_h2d = 1 THEN 1 ELSE 0 END) / COUNT(*)
    END as overlapped_comm_ratio_pct,
    AVG(CASE WHEN overlap_prefetch_h2d = 1 THEN dur_ms END) as avg_dur_overlapped_ms,
    AVG(CASE WHEN overlap_prefetch_h2d = 0 THEN dur_ms END) as avg_dur_non_overlapped_ms
FROM real_tagged;
"""

MOCK_D2H_PREFETCH_ORDER_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
),
mock_d2h AS (
    SELECT m.start, m.end, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 2
      AND m.bytes > 1000000
      AND (:mock_bytes <= 0 OR m.bytes = :mock_bytes)
),
mock_with_neighbors AS (
    SELECT
        m.start,
        m.end,
        m.deviceId,
        CASE
            WHEN EXISTS (
                SELECT 1 FROM prefetch_h2d p
                WHERE p.deviceId = m.deviceId
                  AND p.start < m.end
                  AND p.end > m.start
            ) THEN 1 ELSE 0
        END as overlap_prefetch_h2d,
        (
            SELECT p.end
            FROM prefetch_h2d p
            WHERE p.deviceId = m.deviceId
              AND p.end <= m.start
            ORDER BY p.end DESC
            LIMIT 1
        ) as prev_prefetch_end_ns,
        (
            SELECT p.start
            FROM prefetch_h2d p
            WHERE p.deviceId = m.deviceId
              AND p.start >= m.end
            ORDER BY p.start
            LIMIT 1
        ) as next_prefetch_start_ns
    FROM mock_d2h m
)
SELECT
    COUNT(*) as total_mock_d2h_count,
    SUM(overlap_prefetch_h2d) as overlapped_mock_d2h_count,
    CASE
        WHEN COUNT(*) = 0 THEN NULL
        ELSE 100.0 * SUM(overlap_prefetch_h2d) / COUNT(*)
    END as overlapped_ratio_pct,
    SUM(CASE WHEN prev_prefetch_end_ns IS NOT NULL THEN 1 ELSE 0 END) as has_prev_prefetch_count,
    SUM(CASE WHEN next_prefetch_start_ns IS NOT NULL THEN 1 ELSE 0 END) as has_next_prefetch_count,
    AVG(CASE WHEN prev_prefetch_end_ns IS NOT NULL THEN (start - prev_prefetch_end_ns) / 1e6 END) as avg_gap_from_prev_prefetch_end_ms,
    AVG(CASE WHEN next_prefetch_start_ns IS NOT NULL THEN (next_prefetch_start_ns - end) / 1e6 END) as avg_gap_to_next_prefetch_start_ms,
    SUM(
        CASE
            WHEN prev_prefetch_end_ns IS NOT NULL
             AND (start - prev_prefetch_end_ns) < 1000000
            THEN 1 ELSE 0
        END
    ) as prev_gap_lt1ms_count,
    SUM(
        CASE
            WHEN next_prefetch_start_ns IS NOT NULL
             AND (next_prefetch_start_ns - end) < 1000000
            THEN 1 ELSE 0
        END
    ) as next_gap_lt1ms_count
FROM mock_with_neighbors;
"""

NCCL_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
)
SELECT
    SUBSTR(s.value, 1, 80) as kernel_name,
    COUNT(*) as count,
    SUM(k.end - k.start) / 1e9 as total_time_s,
    AVG((k.end - k.start) / 1e6) as avg_dur_ms,
    MIN((k.end - k.start) / 1e6) as min_dur_ms,
    MAX((k.end - k.start) / 1e6) as max_dur_ms
FROM kernel_base k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%nccl%'
GROUP BY kernel_name
ORDER BY total_time_s DESC;
"""

H2D_TIMELINE_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
)
SELECT
    deviceId,
    (start / 1000000000) as time_bucket_s,
    COUNT(*) as transfers,
    SUM(bytes) / 1e6 as total_mb,
    AVG((bytes / 1e6) / ((end - start) / 1e9)) as avg_bw_mbps
FROM memcpy_base
WHERE copyKind = 1 AND bytes > 100000
GROUP BY deviceId, time_bucket_s
ORDER BY deviceId, time_bucket_s;
"""

DENOISING_STEP_DURATION_QUERY = """
WITH denoising_steps AS (
    SELECT
        CASE
            WHEN n.text GLOB 'SGL_DENOISING_STEP_*'
            THEN CAST(SUBSTR(n.text, LENGTH('SGL_DENOISING_STEP_') + 1) AS INTEGER)
            ELSE CAST(SUBSTR(n.text, LENGTH('denoising_step_') + 1) AS INTEGER)
        END as step_idx,
        (n.end - n.start) / 1e6 as dur_ms
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND (
          n.text GLOB 'SGL_DENOISING_STEP_*'
          OR n.text GLOB 'denoising_step_*'
      )
)
SELECT
    step_idx,
    COUNT(*) as num_ranges,
    AVG(dur_ms) as avg_ms,
    MIN(dur_ms) as min_ms,
    MAX(dur_ms) as max_ms
FROM denoising_steps
GROUP BY step_idx
ORDER BY step_idx;
"""

MOCK_DENOISING_STEP_COMPONENT_QUERY = """
WITH step_ranges AS (
    SELECT
        CASE
            WHEN n.text GLOB 'SGL_DENOISING_STEP_*'
            THEN CAST(SUBSTR(n.text, LENGTH('SGL_DENOISING_STEP_') + 1) AS INTEGER)
            ELSE CAST(SUBSTR(n.text, LENGTH('denoising_step_') + 1) AS INTEGER)
        END as step_idx,
        n.start as step_start,
        n.end as step_end,
        (n.end - n.start) / 1e6 as step_total_ms
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND (
          n.text GLOB 'SGL_DENOISING_STEP_*'
          OR n.text GLOB 'denoising_step_*'
      )
),
denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
mock_sig AS (
    SELECT m.bytes as bytes
    FROM memcpy_base m
    WHERE m.copyKind = 2
      AND m.bytes > 1000000
    GROUP BY m.bytes
    ORDER BY COUNT(*) DESC, m.bytes DESC
    LIMIT 1
),
comm_ranges AS (
    SELECT m.start, m.end
    FROM memcpy_base m
    JOIN mock_sig s ON m.bytes = s.bytes
    WHERE m.copyKind = 2
),
ensure_ranges AS (
    SELECT n.start, n.end
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND n.text = 'SGL_PREFETCH_ENSURE_READY'
),
step_comm AS (
    SELECT
        s.step_idx,
        SUM(
            (MIN(s.step_end, c.end) - MAX(s.step_start, c.start)) / 1e6
        ) as comm_ms
    FROM step_ranges s
    JOIN comm_ranges c
      ON c.start < s.step_end
     AND c.end > s.step_start
    GROUP BY s.step_idx
),
step_ensure AS (
    SELECT
        s.step_idx,
        SUM(
            (MIN(s.step_end, e.end) - MAX(s.step_start, e.start)) / 1e6
        ) as ensure_ms
    FROM step_ranges s
    JOIN ensure_ranges e
      ON e.start < s.step_end
     AND e.end > s.step_start
    GROUP BY s.step_idx
),
step_kernel AS (
    SELECT
        s.step_idx,
        SUM(
            (MIN(s.step_end, k.end) - MAX(s.step_start, k.start)) / 1e6
        ) as kernel_ms
    FROM step_ranges s
    JOIN kernel_base k
      ON k.start < s.step_end
     AND k.end > s.step_start
    GROUP BY s.step_idx
)
SELECT
    s.step_idx,
    s.step_total_ms,
    COALESCE(sk.kernel_ms, 0.0) as kernel_ms,
    COALESCE(sc.comm_ms, 0.0) as comm_ms,
    COALESCE(se.ensure_ms, 0.0) as ensure_ms,
    (s.step_total_ms - COALESCE(sc.comm_ms, 0.0) - COALESCE(se.ensure_ms, 0.0)) as residual_ms
FROM step_ranges s
LEFT JOIN step_comm sc ON sc.step_idx = s.step_idx
LEFT JOIN step_ensure se ON se.step_idx = s.step_idx
LEFT JOIN step_kernel sk ON sk.step_idx = s.step_idx
ORDER BY s.step_idx;
"""

REAL_DENOISING_STEP_COMPONENT_QUERY = """
WITH step_ranges AS (
    SELECT
        CASE
            WHEN n.text GLOB 'SGL_DENOISING_STEP_*'
            THEN CAST(SUBSTR(n.text, LENGTH('SGL_DENOISING_STEP_') + 1) AS INTEGER)
            ELSE CAST(SUBSTR(n.text, LENGTH('denoising_step_') + 1) AS INTEGER)
        END as step_idx,
        n.start as step_start,
        n.end as step_end,
        (n.end - n.start) / 1e6 as step_total_ms
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND (
          n.text GLOB 'SGL_DENOISING_STEP_*'
          OR n.text GLOB 'denoising_step_*'
      )
),
denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
),
comm_ranges AS (
    SELECT k.start, k.end
    FROM kernel_base k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE s.value LIKE '%nccl%'
),
ensure_ranges AS (
    SELECT n.start, n.end
    FROM NVTX_EVENTS n
    WHERE n.end IS NOT NULL
      AND n.text = 'SGL_PREFETCH_ENSURE_READY'
),
step_comm AS (
    SELECT
        s.step_idx,
        SUM(
            (MIN(s.step_end, c.end) - MAX(s.step_start, c.start)) / 1e6
        ) as comm_ms
    FROM step_ranges s
    JOIN comm_ranges c
      ON c.start < s.step_end
     AND c.end > s.step_start
    GROUP BY s.step_idx
),
step_ensure AS (
    SELECT
        s.step_idx,
        SUM(
            (MIN(s.step_end, e.end) - MAX(s.step_start, e.start)) / 1e6
        ) as ensure_ms
    FROM step_ranges s
    JOIN ensure_ranges e
      ON e.start < s.step_end
     AND e.end > s.step_start
    GROUP BY s.step_idx
),
step_kernel AS (
    SELECT
        s.step_idx,
        SUM(
            (MIN(s.step_end, k.end) - MAX(s.step_start, k.start)) / 1e6
        ) as kernel_ms
    FROM step_ranges s
    JOIN kernel_base k
      ON k.start < s.step_end
     AND k.end > s.step_start
    GROUP BY s.step_idx
)
SELECT
    s.step_idx,
    s.step_total_ms,
    COALESCE(sk.kernel_ms, 0.0) as kernel_ms,
    COALESCE(sc.comm_ms, 0.0) as comm_ms,
    COALESCE(se.ensure_ms, 0.0) as ensure_ms,
    (s.step_total_ms - COALESCE(sc.comm_ms, 0.0) - COALESCE(se.ensure_ms, 0.0)) as residual_ms
FROM step_ranges s
LEFT JOIN step_comm sc ON sc.step_idx = s.step_idx
LEFT JOIN step_ensure se ON se.step_idx = s.step_idx
LEFT JOIN step_kernel sk ON sk.step_idx = s.step_idx
ORDER BY s.step_idx;
"""

IDLE_GAPS_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
),
ordered_kernels AS (
    SELECT
        deviceId,
        start,
        end,
        ROW_NUMBER() OVER (PARTITION BY deviceId ORDER BY start) as rn
    FROM kernel_base
),
gaps AS (
    SELECT
        a.deviceId,
        (b.start - a.end) / 1e6 as gap_ms,
        a.end as gap_start,
        b.start as gap_end
    FROM ordered_kernels a
    JOIN ordered_kernels b ON a.deviceId = b.deviceId AND b.rn = a.rn + 1
    WHERE (b.start - a.end) > 1000000
)
SELECT
    deviceId,
    COUNT(*) as num_gaps,
    SUM(gap_ms) as total_gap_ms,
    AVG(gap_ms) as avg_gap_ms,
    MIN(gap_ms) as min_gap_ms,
    MAX(gap_ms) as max_gap_ms
FROM gaps
GROUP BY deviceId
ORDER BY deviceId;
"""

IDLE_GAP_BREAKDOWN_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT k.*
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
       )
),
ordered_kernels AS (
    SELECT
        deviceId,
        start,
        end,
        ROW_NUMBER() OVER (PARTITION BY deviceId ORDER BY start) as rn
    FROM kernel_base
),
gaps AS (
    SELECT
        a.deviceId,
        a.end as gap_start_ns,
        b.start as gap_end_ns,
        (b.start - a.end) / 1e6 as gap_ms
    FROM ordered_kernels a
    JOIN ordered_kernels b ON a.deviceId = b.deviceId AND b.rn = a.rn + 1
    WHERE (b.start - a.end) > 1000000
),
memcpy_base AS (
    SELECT m.*
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE (SELECT COUNT(*) FROM denoise_windows) = 0
       OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < m.end AND d.end > m.start
       )
),
prefetch_marked_streams AS (
    SELECT DISTINCT m.streamId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND EXISTS (
          SELECT 1
          FROM NVTX_EVENTS n
          WHERE n.text IN (
                'SGL_PREFETCH_H2D',
                'SGL_PREFETCH_H2D_BG',
                'SGL_PREFETCH_H2D_SYNC',
                'SGL_PREFETCH_H2D_CHUNK'
          )
            AND n.end IS NOT NULL
            AND n.start < m.end
            AND n.end > m.start
      )
),
prefetch_streams AS (
    SELECT streamId FROM prefetch_marked_streams
    UNION
    SELECT streamId
    FROM (
        SELECT m.streamId
        FROM memcpy_base m
        WHERE m.copyKind = 1
          AND m.bytes >= :prefetch_min_bytes
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )
    WHERE NOT EXISTS (SELECT 1 FROM prefetch_marked_streams)
),
prefetch_h2d AS (
    SELECT m.start, m.end, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 1
      AND m.bytes >= :prefetch_min_bytes
      AND m.streamId IN (SELECT streamId FROM prefetch_streams)
),
mock_d2h AS (
    SELECT m.start, m.end, m.deviceId
    FROM memcpy_base m
    WHERE m.copyKind = 2
      AND m.bytes > 1000000
      AND (:mock_bytes <= 0 OR m.bytes = :mock_bytes)
)
SELECT
    g.deviceId,
    COUNT(*) as num_gaps,
    SUM(g.gap_ms) as total_gap_ms,
    SUM(
        CASE WHEN EXISTS (
            SELECT 1 FROM prefetch_h2d h
            WHERE h.deviceId = g.deviceId
              AND h.start < g.gap_end_ns
              AND h.end > g.gap_start_ns
        ) THEN g.gap_ms ELSE 0 END
    ) as gap_with_prefetch_h2d_ms,
    SUM(
        CASE WHEN EXISTS (
            SELECT 1 FROM mock_d2h d
            WHERE d.deviceId = g.deviceId
              AND d.start < g.gap_end_ns
              AND d.end > g.gap_start_ns
        ) THEN g.gap_ms ELSE 0 END
    ) as gap_with_mock_d2h_ms,
    SUM(
        CASE WHEN NOT EXISTS (
            SELECT 1 FROM memcpy_base m
            WHERE m.deviceId = g.deviceId
              AND m.start < g.gap_end_ns
              AND m.end > g.gap_start_ns
        ) THEN g.gap_ms ELSE 0 END
    ) as gap_without_memcpy_ms
FROM gaps g
GROUP BY g.deviceId
ORDER BY g.deviceId;
"""

LARGE_GAPS_QUERY = """
WITH denoise_windows AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'SGL_DENOISING_LOOP' AND end IS NOT NULL
),
kernel_base AS (
    SELECT
        k.deviceId,
        k.start,
        k.end,
        k.demangledName
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE k.deviceId = 0
      AND (
        (SELECT COUNT(*) FROM denoise_windows) = 0
        OR EXISTS (
            SELECT 1 FROM denoise_windows d
            WHERE d.start < k.end AND d.end > k.start
        )
      )
),
ordered_kernels AS (
    SELECT
        deviceId,
        start,
        end,
        demangledName,
        LAG(end) OVER (PARTITION BY deviceId ORDER BY start) as prev_end,
        LAG(demangledName) OVER (PARTITION BY deviceId ORDER BY start) as prev_name_id
    FROM kernel_base
),
top_gaps AS (
    SELECT
        deviceId,
        (start - prev_end) / 1e6 as gap_ms,
        prev_name_id as before_name_id,
        demangledName as after_name_id
    FROM ordered_kernels
    WHERE prev_end IS NOT NULL
      AND (start - prev_end) > :gap_threshold_ns
    ORDER BY gap_ms DESC
    LIMIT :topk
)
SELECT
    g.deviceId,
    g.gap_ms,
    SUBSTR(s1.value, 1, 60) as before_kernel,
    SUBSTR(s2.value, 1, 60) as after_kernel
FROM top_gaps g
LEFT JOIN StringIds s1 ON g.before_name_id = s1.id
LEFT JOIN StringIds s2 ON g.after_name_id = s2.id
ORDER BY g.gap_ms DESC;
"""


def main():
    if len(sys.argv) not in (4, 5):
        log(f"Usage: {sys.argv[0]} <new_db> <old_db> <output_dir> [mock|real]")
        sys.exit(1)

    new_db = sys.argv[1]
    old_db = sys.argv[2]
    output_dir = sys.argv[3]
    comm_mode = sys.argv[4] if len(sys.argv) == 5 else "mock"
    if comm_mode not in {"mock", "real"}:
        log(f"ERROR: unsupported comm mode: {comm_mode}")
        sys.exit(1)

    for db in [new_db, old_db]:
        if not os.path.exists(db):
            log(f"ERROR: Database not found: {db}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    log("Verifying database schema and tracing markers...")
    db_info = {}
    for db in [new_db, old_db]:
        info = detect_mock_d2h_bytes(db)
        conn = sqlite3.connect(db)
        nccl_count = fetch_scalar(
            conn,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "WHERE k.demangledName IN (SELECT id FROM StringIds WHERE value LIKE '%ncclDevKernel%')",
            default=0,
        )
        conn.close()
        info["nccl_kernels"] = nccl_count
        db_info[db] = info
        log(
            f"  {os.path.basename(db)}: denoise_windows={info['denoise_windows']}, "
            f"prefetch_nvtx={info['prefetch_ranges']}, "
            f"mock_nvtx={info['mock_nvtx_ranges']}, "
            f"mock_d2h_bytes={info['bytes']}, mock_d2h_count={info['count']}, "
            f"nccl_kernels={info['nccl_kernels']}"
        )

    marker_status_path = os.path.join(output_dir, "marker_status.csv")
    with open(marker_status_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "db",
                "denoise_windows",
                "prefetch_nvtx_ranges",
                "mock_nvtx_ranges",
                "detected_mock_d2h_bytes",
                "detected_mock_d2h_count",
                "nccl_kernels",
            ]
        )
        writer.writerow(
            [
                "new",
                os.path.basename(new_db),
                db_info[new_db]["denoise_windows"],
                db_info[new_db]["prefetch_ranges"],
                db_info[new_db]["mock_nvtx_ranges"],
                db_info[new_db]["bytes"] if db_info[new_db]["bytes"] is not None else "",
                db_info[new_db]["count"],
                db_info[new_db]["nccl_kernels"],
            ]
        )
        writer.writerow(
            [
                "old",
                os.path.basename(old_db),
                db_info[old_db]["denoise_windows"],
                db_info[old_db]["prefetch_ranges"],
                db_info[old_db]["mock_nvtx_ranges"],
                db_info[old_db]["bytes"] if db_info[old_db]["bytes"] is not None else "",
                db_info[old_db]["count"],
                db_info[old_db]["nccl_kernels"],
            ]
        )
    log(f"  Marker status saved: {os.path.basename(marker_status_path)}")

    log("\nCreating indices (one-time, speeds up overlap queries)...")
    for db in [new_db, old_db]:
        create_indices(db)
    log("")

    if comm_mode == "mock":
        base_queries = [
            (
                "QUERY 2: mock D2H DURING vs WITHOUT prefetch H2D overlap",
                MOCK_D2H_OVERLAP_QUERY,
                "comm_overlap",
            ),
            (
                "QUERY 2a: mock D2H overlap ratio summary",
                MOCK_D2H_OVERLAP_RATIO_QUERY,
                "comm_overlap_ratio",
            ),
            (
                "QUERY 4a: DENOISING step duration (per-step NVTX)",
                DENOISING_STEP_DURATION_QUERY,
                "denoising_step_duration",
            ),
            (
                "QUERY 4b: DENOISING step component split (step_total/mock_comm)",
                MOCK_DENOISING_STEP_COMPONENT_QUERY,
                "denoising_step_components",
            ),
        ]
    else:
        base_queries = [
            (
                "QUERY 2: NCCL communication DURING vs WITHOUT prefetch H2D overlap",
                REAL_COMM_OVERLAP_QUERY,
                "comm_overlap",
            ),
            (
                "QUERY 2a: NCCL communication overlap ratio summary",
                REAL_COMM_OVERLAP_RATIO_QUERY,
                "comm_overlap_ratio",
            ),
            (
                "QUERY 4a: DENOISING step duration (per-step NVTX)",
                DENOISING_STEP_DURATION_QUERY,
                "denoising_step_duration",
            ),
            (
                "QUERY 4b: DENOISING step component split (step_total/real_comm)",
                REAL_DENOISING_STEP_COMPONENT_QUERY,
                "denoising_step_components",
            ),
        ]

    total_t0 = time.time()
    prefetch_min_mb = float(os.getenv("SGLANG_NSYS_PREFETCH_MIN_MB", "1.0"))
    prefetch_min_bytes = max(1, int(prefetch_min_mb * 1024 * 1024))
    log(f"  Using prefetch event threshold: >= {prefetch_min_mb:.3f} MB")

    for title, query, prefix in base_queries:
        log(f"\n{'=' * 72}")
        log(f"  {title}")
        log(f"{'=' * 72}")

        new_params = None
        old_params = None
        if comm_mode == "mock" and prefix in {
            "comm_overlap",
            "comm_overlap_ratio",
        }:
            new_params = {
                "mock_bytes": db_info[new_db]["bytes"] or -1,
                "prefetch_min_bytes": prefetch_min_bytes,
            }
            old_params = {
                "mock_bytes": db_info[old_db]["bytes"] or -1,
                "prefetch_min_bytes": prefetch_min_bytes,
            }
        elif prefix in {"comm_overlap", "comm_overlap_ratio"}:
            new_params = {"prefetch_min_bytes": prefetch_min_bytes}
            old_params = {"prefetch_min_bytes": prefetch_min_bytes}
        run_query(
            new_db,
            query,
            os.path.join(output_dir, f"{prefix}_new.csv"),
            "New Offload",
            params=new_params,
        )
        run_query(
            old_db,
            query,
            os.path.join(output_dir, f"{prefix}_old.csv"),
            "Old Offload",
            params=old_params,
        )

    log(f"\n{'=' * 72}")
    log("  ANALYSIS COMPLETE")
    log(f"{'=' * 72}")
    log(f"\n  Total query time: {time.time() - total_t0:.1f}s")
    log(f"  Output directory: {output_dir}/")
    log(f"  Communication mode: {comm_mode}")
    log("  Key files:")
    log("    comm_overlap_*.csv                -- communication during vs without prefetch H2D")
    log("    comm_overlap_ratio_*.csv          -- overlap ratio + duration split")
    log("    denoising_step_duration_*.csv     -- per denoising-step duration")
    log("    denoising_step_components_*.csv   -- per-step step_total/comm split")


if __name__ == "__main__":
    main()
