"""
results_db.py

Tables
------
model           : detector metadata (name, type, backbone, size)
dataset         : benchmark dataset info (VOC 2007 splits)
object_class    : the 20 Pascal VOC categories
experiment      : one benchmark run linking model + dataset + config
overall_result  : aggregate metrics (mAP, FPS, RAM) per experiment
per_class_result: AP per class per experiment
"""

import sqlite3
import os
from datetime import datetime


class ResultsDatabase:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()

    def init_schema(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS model (
                model_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL UNIQUE,
                type            TEXT    NOT NULL,
                architecture    TEXT    NOT NULL,
                backbone        TEXT,
                model_size_mb   REAL,
                pretrained_on   TEXT,
                fine_tuned_on   TEXT
            );

            CREATE TABLE IF NOT EXISTS dataset (
                dataset_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                version         TEXT,
                split           TEXT    NOT NULL,
                num_images      INTEGER NOT NULL,
                num_classes     INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS object_class (
                class_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL UNIQUE,
                num_test_instances INTEGER
            );

            CREATE TABLE IF NOT EXISTS experiment (
                experiment_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id            INTEGER NOT NULL,
                dataset_id          INTEGER NOT NULL,
                run_timestamp       TEXT    NOT NULL,
                hardware            TEXT,
                confidence_threshold REAL,
                nms_iou_threshold   REAL,
                FOREIGN KEY (model_id)   REFERENCES model(model_id),
                FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id)
            );

            CREATE TABLE IF NOT EXISTS overall_result (
                result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id   INTEGER NOT NULL UNIQUE,
                map_50          REAL    NOT NULL,
                fps             REAL,
                mean_inference_ms REAL,
                std_ms          REAL,
                ram_delta_mb    REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id)
            );

            CREATE TABLE IF NOT EXISTS per_class_result (
                per_class_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id   INTEGER NOT NULL,
                class_id        INTEGER NOT NULL,
                average_precision REAL  NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id),
                FOREIGN KEY (class_id)      REFERENCES object_class(class_id)
            );
        """)
        self.conn.commit()

    def insert_model(self, name, type_, architecture, backbone=None,
                     model_size_mb=None, pretrained_on=None,
                     fine_tuned_on=None):
        self.cursor.execute("""
            INSERT OR IGNORE INTO model
                (name, type, architecture, backbone,
                 model_size_mb, pretrained_on, fine_tuned_on)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, type_, architecture, backbone,
              model_size_mb, pretrained_on, fine_tuned_on))
        self.conn.commit()
        self.cursor.execute(
            "SELECT model_id FROM model WHERE name = ?", (name,))
        return self.cursor.fetchone()[0]

    def insert_dataset(self, name, version, split, num_images, num_classes):
        self.cursor.execute("""
            INSERT INTO dataset
                (name, version, split, num_images, num_classes)
            VALUES (?, ?, ?, ?, ?)
        """, (name, version, split, num_images, num_classes))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_object_class(self, name, num_test_instances=None):
        self.cursor.execute("""
            INSERT OR IGNORE INTO object_class
                (name, num_test_instances)
            VALUES (?, ?)
        """, (name, num_test_instances))
        self.conn.commit()
        self.cursor.execute(
            "SELECT class_id FROM object_class WHERE name = ?", (name,))
        return self.cursor.fetchone()[0]

    def insert_experiment(self, model_id, dataset_id, hardware=None,
                          confidence_threshold=None, nms_iou_threshold=None):
        self.cursor.execute("""
            INSERT INTO experiment
                (model_id, dataset_id, run_timestamp, hardware,
                 confidence_threshold, nms_iou_threshold)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_id, dataset_id,
              datetime.now().isoformat(),
              hardware, confidence_threshold, nms_iou_threshold))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_overall_result(self, experiment_id, map_50, fps=None,
                              mean_inference_ms=None, std_ms=None,
                              ram_delta_mb=None):
        self.cursor.execute("""
            INSERT INTO overall_result
                (experiment_id, map_50, fps,
                 mean_inference_ms, std_ms, ram_delta_mb)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (experiment_id, map_50, fps,
              mean_inference_ms, std_ms, ram_delta_mb))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_per_class_result(self, experiment_id, class_id,
                                average_precision):
        self.cursor.execute("""
            INSERT INTO per_class_result
                (experiment_id, class_id, average_precision)
            VALUES (?, ?, ?)
        """, (experiment_id, class_id, average_precision))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_all_results_summary(self):
        self.cursor.execute("""
            SELECT m.name, o.map_50, o.fps,
                   o.mean_inference_ms, o.ram_delta_mb
            FROM   overall_result o
            JOIN   experiment e ON e.experiment_id = o.experiment_id
            JOIN   model m      ON m.model_id = e.model_id
            ORDER BY o.map_50 DESC
        """)
        cols = ["model", "map_50", "fps", "mean_inference_ms", "ram_delta_mb"]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def get_per_class_for_experiment(self, experiment_id):
        self.cursor.execute("""
            SELECT oc.name, pcr.average_precision
            FROM   per_class_result pcr
            JOIN   object_class oc ON oc.class_id = pcr.class_id
            WHERE  pcr.experiment_id = ?
            ORDER BY oc.name
        """, (experiment_id,))
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    def close(self):
        self.conn.commit()
        self.conn.close()