from pathlib import Path
from util.Fqn import Fqn

def enrich_plan_with_visibility_ops(plan: dict, state: dict) -> dict:
    internal_deps = state.get("internal_deps") or []
    target_files = state.get("target_files") or []
    target_source_root = state.get("target_source_root") or ""
    target_name = state.get("target_name") or ""

    # FQN -> current repository-relative file path
    class_to_file: dict[str, str] = {}
    for f in target_files:
        fqn = Fqn(target_name)._java_file_to_fqn(f, target_source_root)
        class_to_file[fqn] = f

    def fqn_to_java_path(fqn: str) -> str:
        return str(
            Path(target_source_root)
            / Path(*fqn.split("."))
        ) + ".java"

    # Build relation map in both directions.
    related_by_class: dict[str, set[str]] = {}

    for src, dst in internal_deps:
        related_by_class.setdefault(src, set()).add(dst)
        related_by_class.setdefault(dst, set()).add(src)

    original_blocks = plan.get("blocks") or []
    enriched_blocks = []
    next_id = 1

    for block in original_blocks:
        ops = block.get("ops") or []

        move_ops = [
            op for op in ops
            if (op.get("op") or "").strip() == "MOVE_CLASS"
        ]

        if not move_ops:
            block["id"] = next_id
            enriched_blocks.append(block)
            next_id += 1
            continue

        moved_old_fqns: list[str] = []
        moved_new_fqns: list[str] = []

        for op in move_ops:
            inputs = op.get("inputs") or []
            outputs = op.get("outputs") or []

            if not inputs or not outputs:
                continue

            moved_old_fqns.append(inputs[0])
            moved_new_fqns.append(outputs[0])

        moved_old_set = set(moved_old_fqns)
        moved_new_set = set(moved_new_fqns)

        if not moved_old_fqns:
            block["id"] = next_id
            enriched_blocks.append(block)
            next_id += 1
            continue

        related_remaining_classes: set[str] = set()

        for old_fqn in moved_old_fqns:
            for related in related_by_class.get(old_fqn, set()):
                if related in class_to_file and related not in moved_old_set:
                    related_remaining_classes.add(related)

        new_ops = [
            op for op in ops
            if (op.get("op") or "").strip() in {"CREATE_PACKAGE", "MOVE_CLASS"}
        ]

        visibility_inputs = sorted(moved_new_set | related_remaining_classes)

        new_ops.append({
            "op": "UPDATE_VISIBILITY",
            "inputs": visibility_inputs,
            "outputs": [],
            "details": (
                "After moving the whole cluster to the destination package, "
                "update only the minimum required visibility in moved classes "
                "and related remaining classes so the project can compile. "
                "Do not change behavior. Do not move additional classes."
            ),
            "risk": "medium",
            "api_change": True,
        })

        new_files = set(block.get("files") or [])

        for old_fqn in moved_old_fqns:
            if old_fqn in class_to_file:
                new_files.add(class_to_file[old_fqn])

        for new_fqn in moved_new_fqns:
            new_files.add(fqn_to_java_path(new_fqn))

        for related in related_remaining_classes:
            if related in class_to_file:
                new_files.add(class_to_file[related])

        block["id"] = next_id
        block["goal"] = block.get("goal") or (
            "Move cohesive cluster: " + ", ".join(moved_old_fqns)
        )
        block["files"] = sorted(new_files)
        block["ops"] = new_ops

        enriched_blocks.append(block)
        next_id += 1

    plan["blocks"] = enriched_blocks
    return plan