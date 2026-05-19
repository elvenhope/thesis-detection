"""
query_results.py
Demonstrates reading back from the SQLite database.
Prints a summary table and per-class detail for the best model.

Run after migrate_json_to_db.py:
    python query_results.py
"""

import os
from results_db import ResultsDatabase

DB_PATH = os.path.join(os.path.dirname(__file__), "results", "benchmark.db")


def main():
    db = ResultsDatabase(DB_PATH)

    # 1. Summary table (ordered by mAP descending)
    print("=" * 72)
    print("  BENCHMARK RESULTS SUMMARY (from SQLite)")
    print("=" * 72)
    print(f"  {'Model':<45} {'mAP@0.5':>8} {'FPS':>8}")
    print("-" * 72)

    rows = db.get_all_results_summary()
    for r in rows:
        print(f"  {r['model']:<45} "
              f"{r['map_50']:>7.2%} "
              f"{r['fps']:>7.2f}")

    # 2. Per-class AP for the top model
    print()
    print("=" * 72)
    print("  PER-CLASS AP: best model (experiment_id = 1)")
    print("=" * 72)

    # Find the experiment_id with highest mAP
    db.cursor.execute("""
        SELECT e.experiment_id, m.name
        FROM   overall_result o
        JOIN   experiment e ON e.experiment_id = o.experiment_id
        JOIN   model m      ON m.model_id = e.model_id
        ORDER BY o.map_50 DESC
        LIMIT 1
    """)
    best = db.cursor.fetchone()
    if best:
        exp_id, model_name = best
        print(f"  Model: {model_name}")
        print("-" * 72)
        per_class = db.get_per_class_for_experiment(exp_id)
        for cls, ap in sorted(per_class.items(),
                              key=lambda x: x[1], reverse=True):
            bar = "#" * int(ap * 50)
            print(f"  {cls:<15} {ap:>7.2%}  {bar}")

    db.close()


if __name__ == "__main__":
    main()
