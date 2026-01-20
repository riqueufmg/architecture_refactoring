import os
import json
import subprocess
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

from src.utils.designite_runner import DesigniteRunner

class MetricsDepsParser:
    def __init__(self, project_name: str, classes_path: str | None = None):
        self.project_name = project_name

        self.output_path = Path(os.getenv("METRICS_PATH"), project_name)
        self.project_path = Path(os.getenv("REPOSITORIES_PATH"), project_name)

        self.runner = DesigniteRunner(self.project_path, self.output_path, classes_path)

    @staticmethod
    def normalize_columns(df: pd.DataFrame):
        old_cols = df.columns
        new_cols = []
        mapping = {}

        for c in old_cols:
            clean = str(c).strip().replace("\ufeff", "")
            normalized = clean.lower().replace(" ", "_")
            new_cols.append(normalized)
            mapping[normalized] = clean

        df.columns = new_cols
        return df, mapping

    @staticmethod
    def _safe_int(v, default=0) -> int:
        try:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return default
            return int(float(v))
        except Exception:
            return default

    @staticmethod
    def _safe_float(v, default=0.0) -> float:
        try:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return default
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _safe_rel_file(file_value: str | None, project_path: str | Path) -> str:
        if not file_value:
            return ""
        file_str = str(file_value)
        project_str = str(project_path) if project_path else ""

        if project_str and project_str in file_str:
            try:
                return file_str[file_str.index(project_str):]
            except ValueError:
                return file_str
        return file_str

    @staticmethod
    def parse_class_metrics(row: dict, project_path: str) -> dict:
        return {
            "package": row.get("package", ""),
            "class": row.get("class", ""),
            "file": MetricsDepsParser._safe_rel_file(row.get("file"), project_path),
            "metrics": {
                "nof": MetricsDepsParser._safe_int(row.get("nof", 0), 0),
                "nopf": MetricsDepsParser._safe_int(row.get("nopf", 0), 0),
                "nom": MetricsDepsParser._safe_int(row.get("nom", 0), 0),
                "nopm": MetricsDepsParser._safe_int(row.get("nopm", 0), 0),
                "loc": MetricsDepsParser._safe_int(row.get("loc", 0), 0),
                "wmc": MetricsDepsParser._safe_int(row.get("wmc", 0), 0),
                "nc": MetricsDepsParser._safe_int(row.get("nc", 0), 0),
                "dit": MetricsDepsParser._safe_int(row.get("dit", 0), 0),
                "lcom": MetricsDepsParser._safe_float(row.get("lcom", 0), 0.0),
                "fanin": MetricsDepsParser._safe_int(row.get("fan-in", 0), 0),
                "fanout": MetricsDepsParser._safe_int(row.get("fan-out", 0), 0),
            },
            "methods": [],
            "dependencies": [],
        }

    @staticmethod
    def group_classes_by_package(class_rows: list[dict]) -> list[dict]:
        packages_dict = defaultdict(list)

        for cls in class_rows:
            package = cls.get("package") or "default_package"
            packages_dict[package].append(cls)

        packages_list = []
        for pkg_name, classes in packages_dict.items():
            pkg_metrics = {
                "num_classes": len(classes),
                "loc": sum((c.get("metrics") or {}).get("loc", 0) for c in classes),
                "Ce": 0,
                "Ca": 0,
            }

            packages_list.append(
                {
                    "package": pkg_name,
                    "metrics": pkg_metrics,
                    "classes": classes,
                    "dependencies": [],
                }
            )

        return packages_list

    @staticmethod
    def classname_to_package(class_name: str) -> str:
        parts = (class_name or "").split(".")
        # remove trailing segments that look like class names (start with uppercase)
        while parts and parts[-1] and parts[-1][0].isupper():
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def parser_dependencies(graph_path: str | Path):
        tree = ET.parse(str(graph_path))
        root = tree.getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

        package_dependencies = defaultdict(set)
        class_dependencies = defaultdict(set)

        for edge in root.findall(".//g:edge", ns):
            source = edge.attrib.get("source")
            target = edge.attrib.get("target")
            if not source or not target:
                continue

            source_pkg = MetricsDepsParser.classname_to_package(source)
            target_pkg = MetricsDepsParser.classname_to_package(target)

            if source_pkg and target_pkg and source_pkg != target_pkg:
                package_dependencies[source_pkg].add(target_pkg)

            if source != target:
                class_dependencies[source].add(target)

        package_dependencies = {k: sorted(list(v)) for k, v in package_dependencies.items()}
        class_dependencies = {k: sorted(list(v)) for k, v in class_dependencies.items()}

        return package_dependencies, class_dependencies

    @staticmethod
    def calculate_afferent_coupling(package_dependencies: dict[str, list[str]]) -> dict[str, int]:
        afferent = defaultdict(int)
        for _, targets in package_dependencies.items():
            for target_pkg in targets:
                afferent[target_pkg] += 1
        return dict(afferent)

    @staticmethod
    def attach_dependencies(packages: list[dict], package_dependencies: dict, class_dependencies: dict):
        package_index = {pkg["package"]: pkg for pkg in packages}

        valid_classes = {
            f'{pkg["package"]}.{cls["class"]}'
            for pkg in packages
            for cls in pkg.get("classes", [])
            if pkg.get("package") and cls.get("class")
        }

        # Package outgoing deps + Ce
        for source_pkg, targets in package_dependencies.items():
            if source_pkg in package_index:
                valid_targets = [t for t in targets if t in package_index]
                package_index[source_pkg]["dependencies"] = valid_targets
                package_index[source_pkg]["metrics"]["Ce"] = len(valid_targets)

        # Package incoming deps + Ca
        afferent_coupling = defaultdict(int)
        for source_pkg, targets in package_dependencies.items():
            if source_pkg not in package_index:
                continue
            for target_pkg in targets:
                if target_pkg in package_index:
                    afferent_coupling[target_pkg] += 1

        for pkg_name, pkg in package_index.items():
            pkg["metrics"]["Ca"] = afferent_coupling.get(pkg_name, 0)

        # Class deps filtered to valid classes
        for pkg in packages:
            for cls_obj in pkg.get("classes", []):
                class_name = f'{pkg["package"]}.{cls_obj["class"]}'
                raw_deps = class_dependencies.get(class_name, [])
                cls_obj["dependencies"] = [dep for dep in raw_deps if dep in valid_classes]

        return packages

    @staticmethod
    def parse_method_metrics(row: dict) -> dict:
        is_test_raw = row.get("istest", 0)
        try:
            is_test = bool(int(is_test_raw))
        except Exception:
            is_test = bool(is_test_raw)

        return {
            "method_name": row.get("method", row.get("method_name", "")),
            "line_num": MetricsDepsParser._safe_int(row.get("line_no", 0), 0),
            "metrics": {
                "loc": MetricsDepsParser._safe_int(row.get("loc", 0), 0),
                "cc": MetricsDepsParser._safe_int(row.get("cc", 0), 0),
                "pc": MetricsDepsParser._safe_int(row.get("pc", 0), 0),
            },
        }

    @staticmethod
    def attach_methods_to_classes(packages: list[dict], method_rows: list[dict]):
        class_index = {}
        for pkg in packages:
            for cls in pkg.get("classes", []):
                cls.setdefault("methods", [])
                class_index[(pkg.get("package"), cls.get("class"))] = cls

        not_attached = 0

        for row in method_rows:
            pkg_name = row.get("package")
            cls_name = row.get("class")

            if not pkg_name or not cls_name:
                not_attached += 1
                continue

            key = (pkg_name, cls_name)
            if key in class_index:
                class_index[key]["methods"].append(MetricsDepsParser.parse_method_metrics(row))
            else:
                not_attached += 1

        if not_attached:
            print(f"Warning: {not_attached} métodos não foram anexados (package/class não encontrados).")

        return packages

    def collect_metrics(self) -> dict:
        try:
            # Run Designite
            self.runner.run()

            out_dir = Path(self.output_path)
            class_csv = out_dir / "TypeMetrics.csv"
            method_csv = out_dir / "MethodMetrics.csv"
            graph_path = out_dir / "DependencyGraph.graphml"

            missing = [p.name for p in [class_csv, method_csv, graph_path] if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Designite output missing files in {out_dir}: {', '.join(missing)}"
                )

            # ---- Classes
            class_df = pd.read_csv(class_csv)
            class_df, _ = self.normalize_columns(class_df)
            class_rows = class_df.to_dict(orient="records")

            class_dicts = [
                self.parse_class_metrics(row, self.project_path)
                for row in class_rows
                if "/test/" not in ((row.get("file") or "").replace("\\", "/"))
            ]

            packages = self.group_classes_by_package(class_dicts)

            # ---- Methods
            method_df = pd.read_csv(method_csv)
            method_df, _ = self.normalize_columns(method_df)
            method_rows = [
                row
                for row in method_df.to_dict(orient="records")
                if "/test/" not in ((row.get("file") or "").replace("\\", "/"))
            ]

            packages = self.attach_methods_to_classes(packages, method_rows)

            # ---- Dependencies
            package_dependencies, class_dependencies = self.parser_dependencies(graph_path)
            packages = self.attach_dependencies(packages, package_dependencies, class_dependencies)

            final_json = {
                "project": Path(self.project_path).name,
                "summary": {
                    "total_packages": len(packages),
                    "total_classes": sum(len(p.get("classes", [])) for p in packages),
                },
                "packages": packages,
            }

            output_file = out_dir / "project_metrics.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_json, f, indent=4, ensure_ascii=False)

            print(f"Metrics collected and saved to {output_file}")
            return final_json

        except subprocess.CalledProcessError as e:
            print("Designite failed to run.")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            raise