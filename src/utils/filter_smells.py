import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import pandas as pd


class FilterSmells:
    """
    Filters project_metrics_improved.json to keep only packages/classes that actually have the target smell.

    - God Component, Unstable Dependency: filter at PACKAGE level using ArchitectureSmells.csv
    - Hub-like Modularization, Insufficient Modularization: filter at CLASS level using DesignSmells.csv
    """

    PACKAGE_LEVEL_SMELLS: Set[str] = {"God Component", "Unstable Dependency"}
    CLASS_LEVEL_SMELLS: Set[str] = {"Hub-like Modularization", "Insufficient Modularization"}

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.metrics_dir = Path(os.getenv("METRICS_PATH"), project_name)

        self.project_metrics_file = self.metrics_dir / "project_metrics_improved.json"
        self.arch_smells_file = self.metrics_dir / "ArchitectureSmells.csv"
        self.design_smells_file = self.metrics_dir / "DesignSmells.csv"

    # ----------------------------
    # Public API
    # ----------------------------
    def filter_by_smell(
        self,
        smell_type: str,
        output_file: str | Path | None = None,
        prune_dependencies: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns a filtered JSON dict and optionally writes it to disk.

        Args:
            smell_type: One of:
                - "God Component"
                - "Unstable Dependency"
                - "Hub-like Modularization"
                - "Insufficient Modularization"
            output_file: If provided, writes filtered JSON to this path.
            prune_dependencies: If True, removes dependencies that point outside the filtered slice.

        Output JSON additions:
            - smell_type
            - filter_level: "package" | "class"
            - summary updated
        """
        improved = self._load_project_metrics()

        level, targets = self._load_smell_targets(smell_type)
        filtered = self._filter_improved_json(
            improved_json=improved,
            smell_type=smell_type,
            level=level,
            targets=targets,
            prune_dependencies=prune_dependencies,
        )

        if output_file is not None:
            out_path = Path(output_file)
        else:
            # default output location
            safe_name = smell_type.lower().replace(" ", "_").replace("-", "_")
            out_path = self.metrics_dir / f"smell_input_{safe_name}.json"

        self._write_json(out_path, filtered)
        return filtered

    # ----------------------------
    # Loading / IO
    # ----------------------------
    def _load_project_metrics(self) -> Dict[str, Any]:
        if not self.project_metrics_file.exists():
            raise FileNotFoundError(f"Missing: {self.project_metrics_file}")
        with open(self.project_metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ----------------------------
    # Target extraction from CSV
    # ----------------------------
    def _load_smell_targets(self, smell_type: str) -> Tuple[str, object]:
        """
        Returns:
            level: "package" or "class"
            targets:
                - package-level => set[str] of package names
                - class-level => set[tuple[str,str]] of (package, class)
        """
        if smell_type in self.PACKAGE_LEVEL_SMELLS:
            return "package", self._load_architecture_smell_packages(smell_type)

        if smell_type in self.CLASS_LEVEL_SMELLS:
            return "class", self._load_design_smell_classes(smell_type)

        raise ValueError(
            f"Unsupported smell_type: {smell_type}. "
            f"Use one of: {sorted(self.PACKAGE_LEVEL_SMELLS | self.CLASS_LEVEL_SMELLS)}"
        )

    def _load_architecture_smell_packages(self, smell_type: str) -> Set[str]:
        if not self.arch_smells_file.exists():
            raise FileNotFoundError(f"Missing: {self.arch_smells_file}")

        df = pd.read_csv(self.arch_smells_file)
        # expected columns: Project,Package,Smell,Description
        df = df[(df["Project"] == self.project_name) & (df["Smell"] == smell_type)]

        return set(df["Package"].dropna().astype(str).tolist())

    def _load_design_smell_classes(self, smell_type: str) -> Set[Tuple[str, str]]:
        if not self.design_smells_file.exists():
            raise FileNotFoundError(f"Missing: {self.design_smells_file}")

        df = pd.read_csv(self.design_smells_file)
        # expected columns: Project,Package,Class,Smell,Description,File
        df = df[(df["Project"] == self.project_name) & (df["Smell"] == smell_type)]

        pairs: Set[Tuple[str, str]] = set()
        for pkg, cls in zip(df["Package"].fillna(""), df["Class"].fillna("")):
            pkg = str(pkg).strip()
            cls = str(cls).strip()
            if pkg and cls:
                pairs.add((pkg, cls))
        return pairs

    # ----------------------------
    # Filtering logic
    # ----------------------------
    def _filter_improved_json(
        self,
        improved_json: Dict[str, Any],
        smell_type: str,
        level: str,
        targets: object,
        prune_dependencies: bool,
    ) -> Dict[str, Any]:
        packages_in = improved_json.get("packages", []) or []

        kept_packages: List[Dict[str, Any]] = []
        allowed_pkg_names: Set[str] = set()

        # Precompute for class-level: allowed (package, class) pairs
        class_targets: Optional[Set[Tuple[str, str]]] = None
        pkg_targets: Optional[Set[str]] = None

        if level == "package":
            pkg_targets = targets  # type: ignore[assignment]
        elif level == "class":
            class_targets = targets  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown filter level: {level}")

        # 1) Keep packages/classes based on targets
        for pkg in packages_in:
            pkg_name = (pkg.get("package") or "").strip()
            if not pkg_name:
                continue

            if level == "package":
                assert pkg_targets is not None
                if pkg_name not in pkg_targets:
                    continue

                kept_packages.append(dict(pkg))
                allowed_pkg_names.add(pkg_name)

            else:
                assert class_targets is not None
                classes = pkg.get("classes", []) or []
                new_classes: List[Dict[str, Any]] = []

                for cls in classes:
                    cls_name = (cls.get("class") or "").strip()
                    if not cls_name:
                        continue
                    if (pkg_name, cls_name) in class_targets:
                        new_classes.append(dict(cls))

                if new_classes:
                    new_pkg = dict(pkg)
                    new_pkg["classes"] = new_classes
                    kept_packages.append(new_pkg)
                    allowed_pkg_names.add(pkg_name)

        # 2) Optionally prune dependencies to keep the slice self-contained
        if prune_dependencies and kept_packages:
            self._prune_dependencies(kept_packages, level=level, allowed_pkg_names=allowed_pkg_names)

        # 3) Build output
        out = dict(improved_json)
        out["packages"] = kept_packages
        out["summary"] = {
            "total_packages": len(kept_packages),
            "total_classes": sum(len(p.get("classes", [])) for p in kept_packages),
        }
        out["smell_type"] = smell_type
        out["filter_level"] = level
        out["targets_count"] = len(targets) if hasattr(targets, "__len__") else None
        return out

    @staticmethod
    def _prune_dependencies(
        kept_packages: List[Dict[str, Any]],
        level: str,
        allowed_pkg_names: Set[str],
    ) -> None:
        """
        Prunes:
          - package dependencies: keep only deps to packages that are still kept
          - class dependencies (only for class-level filtering): keep only deps to classes that are still kept
        """
        # Package dependencies
        for pkg in kept_packages:
            deps = pkg.get("dependencies", []) or []
            pkg["dependencies"] = [d for d in deps if d in allowed_pkg_names]

        if level != "class":
            return

        # Build set of kept class FQNs
        kept_class_fqns: Set[str] = set()
        for pkg in kept_packages:
            pkg_name = pkg.get("package") or ""
            for cls in pkg.get("classes", []) or []:
                cls_name = cls.get("class") or ""
                if pkg_name and cls_name:
                    kept_class_fqns.add(f"{pkg_name}.{cls_name}")

        # Prune class dependencies
        for pkg in kept_packages:
            for cls in pkg.get("classes", []) or []:
                cdeps = cls.get("dependencies", []) or []
                cls["dependencies"] = [d for d in cdeps if d in kept_class_fqns]
