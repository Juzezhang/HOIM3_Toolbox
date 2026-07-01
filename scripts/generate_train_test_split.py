"""Generate deterministic train/test split for HOI-M3 dataset.

Hard constraint: every object in train must also appear in test,
except singleton objects (appearing in only 1 sequence total).

Algorithm: 4 phases
  1. Mandatory group picks (config groups with unique objects)
  2. Greedy coverage (cover remaining objects, rarest first)
  3. Fill to per-scene 20% targets
  4. Verify & repair constraint violations
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/simurgh/group/juze/datasets/HOI-M3")
INPUT_PATH = BASE / "sequence_contents.json"
OUTPUT_PATH = BASE / "train_test_split.json"


def scene_of(seq_name: str) -> str:
    """Extract scene type from sequence name (e.g. 'bedroom_data01' -> 'bedroom')."""
    return seq_name.rsplit("_data", 1)[0]


def config_key(objects: list[str]) -> tuple[str, ...]:
    """Canonical key for a set of objects (sorted tuple)."""
    return tuple(sorted(objects))


def build_indexes(data: dict):
    """Build all lookup structures from sequence_contents data."""
    obj_to_seqs = defaultdict(set)      # object -> set of sequence names
    seq_to_objs = {}                    # sequence -> set of objects
    seq_to_scene = {}                   # sequence -> scene type
    seq_to_num_humans = {}              # sequence -> num_humans
    scene_seqs = defaultdict(list)      # scene -> list of sequences
    config_groups = defaultdict(list)   # config_key -> list of sequences

    for seq_name, info in data.items():
        objs = set(info["objects"])
        seq_to_objs[seq_name] = objs
        seq_to_scene[seq_name] = scene_of(seq_name)
        seq_to_num_humans[seq_name] = info["num_humans"]
        scene_seqs[scene_of(seq_name)].append(seq_name)
        config_groups[config_key(info["objects"])].append(seq_name)
        for obj in objs:
            obj_to_seqs[obj].add(seq_name)

    return obj_to_seqs, seq_to_objs, seq_to_scene, seq_to_num_humans, scene_seqs, config_groups


def phase1_mandatory(obj_to_seqs, config_groups, seq_to_scene, seq_to_num_humans, rng):
    """Pick 1 sequence from each config group that is the sole group for some object.

    A config group is mandatory if it contains a multi-sequence object that
    appears in no other config group.
    """
    # Find multi-sequence objects
    multi_objs = {obj for obj, seqs in obj_to_seqs.items() if len(seqs) > 1}

    # Map each multi-seq object to the set of config groups it appears in
    obj_to_groups = defaultdict(set)
    for key, seqs in config_groups.items():
        for obj in key:
            if obj in multi_objs:
                obj_to_groups[obj].add(key)

    # Mandatory groups: object confined to exactly 1 config group
    mandatory_groups = {}
    for obj, groups in obj_to_groups.items():
        if len(groups) == 1:
            group_key = next(iter(groups))
            mandatory_groups[group_key] = mandatory_groups.get(group_key, set()) | {obj}

    test_seqs = set()
    used_groups = set()
    covered_objs = set()

    for group_key in sorted(mandatory_groups.keys()):
        seqs = config_groups[group_key]
        # Pick one sequence, preferring diversity in num_humans
        pick = rng.choice(sorted(seqs))
        test_seqs.add(pick)
        used_groups.add(group_key)
        covered_objs.update(set(group_key) & multi_objs)

    return test_seqs, used_groups, covered_objs


def phase2_greedy_coverage(obj_to_seqs, config_groups, seq_to_objs, seq_to_scene,
                           seq_to_num_humans, scene_seqs, test_seqs, used_groups,
                           covered_objs, scene_targets, rng):
    """Cover remaining multi-sequence objects greedily, rarest first."""
    multi_objs = {obj for obj, seqs in obj_to_seqs.items() if len(seqs) > 1}
    uncovered = multi_objs - covered_objs

    while uncovered:
        # Pick rarest uncovered object
        target_obj = min(uncovered, key=lambda o: (len(obj_to_seqs[o]), o))

        # Find candidate groups containing this object that haven't been used
        candidate_groups = []
        for key, seqs in config_groups.items():
            if key in used_groups:
                continue
            if target_obj in key:
                candidate_groups.append(key)

        if not candidate_groups:
            # Object is already covered by a used group member's objects
            uncovered.discard(target_obj)
            continue

        # Score each candidate group
        best_group = None
        best_score = (-1, -1, -1)
        for key in candidate_groups:
            new_objs = len((set(key) & multi_objs) - covered_objs)
            # Pick a representative sequence to check scene
            rep_seq = config_groups[key][0]
            scene = seq_to_scene[rep_seq]
            scene_test_count = sum(1 for s in test_seqs if seq_to_scene[s] == scene)
            scene_needs = max(0, scene_targets.get(scene, 0) - scene_test_count)
            score = (1 if scene_needs > 0 else 0, new_objs, scene_needs)
            if score > best_score:
                best_score = score
                best_group = key

        # Pick a sequence from best group
        seqs = config_groups[best_group]
        pick = rng.choice(sorted(seqs))
        test_seqs.add(pick)
        used_groups.add(best_group)
        covered_objs.update(set(best_group) & multi_objs)
        uncovered -= covered_objs

    return test_seqs, used_groups, covered_objs


def phase3_fill_targets(config_groups, seq_to_scene, seq_to_num_humans,
                        scene_seqs, test_seqs, used_groups, scene_targets, rng):
    """Fill remaining test slots to reach per-scene targets, preferring num_humans diversity."""
    for scene in sorted(scene_targets.keys()):
        target = scene_targets[scene]
        current = [s for s in test_seqs if seq_to_scene[s] == scene]
        needed = target - len(current)
        if needed <= 0:
            continue

        # Existing num_humans in test for this scene
        existing_nh = {seq_to_num_humans[s] for s in current}

        # Candidate groups not yet used, belonging to this scene
        candidates = []
        for key, seqs in config_groups.items():
            if key in used_groups:
                continue
            scene_seqs_in_group = [s for s in seqs if seq_to_scene[s] == scene]
            if scene_seqs_in_group:
                candidates.append((key, scene_seqs_in_group))

        # Sort candidates: prefer groups that add num_humans diversity
        def candidate_sort_key(item):
            key, seqs = item
            nh_values = {seq_to_num_humans[s] for s in seqs}
            adds_diversity = 1 if nh_values - existing_nh else 0
            return (-adds_diversity, key)

        candidates.sort(key=candidate_sort_key)

        for key, seqs in candidates:
            if needed <= 0:
                break
            # Pick sequence that adds most num_humans diversity
            seqs_sorted = sorted(seqs, key=lambda s: (
                0 if seq_to_num_humans[s] not in existing_nh else 1, s))
            pick = seqs_sorted[0]
            test_seqs.add(pick)
            used_groups.add(key)
            existing_nh.add(seq_to_num_humans[pick])
            needed -= 1

    return test_seqs, used_groups


def phase4_repair(data, test_seqs, obj_to_seqs, seq_to_objs, seq_to_scene, rng):
    """Verify train_objects ⊆ test_objects (ignoring singletons). Swap to fix violations."""
    all_seqs = set(data.keys())
    train_seqs = all_seqs - test_seqs
    singletons = {obj for obj, seqs in obj_to_seqs.items() if len(seqs) == 1}

    train_objs = set()
    for s in train_seqs:
        train_objs.update(seq_to_objs[s] - singletons)

    test_objs = set()
    for s in test_seqs:
        test_objs.update(seq_to_objs[s] - singletons)

    violations = train_objs - test_objs
    swaps = 0
    while violations:
        # Find a train sequence containing a violating object
        target_obj = min(violations)
        # Find train seq with this object
        train_candidates = [s for s in train_seqs if target_obj in seq_to_objs[s]]
        # Find test seq in same scene to swap out
        for tc in sorted(train_candidates):
            scene = seq_to_scene[tc]
            test_in_scene = sorted(s for s in test_seqs if seq_to_scene[s] == scene)
            for ts in test_in_scene:
                # Try swapping tc into test, ts into train
                new_test = (test_seqs - {ts}) | {tc}
                new_train = (train_seqs - {tc}) | {ts}
                new_test_objs = set()
                for s in new_test:
                    new_test_objs.update(seq_to_objs[s] - singletons)
                new_train_objs = set()
                for s in new_train:
                    new_train_objs.update(seq_to_objs[s] - singletons)
                new_violations = new_train_objs - new_test_objs
                if len(new_violations) < len(violations):
                    test_seqs.discard(ts)
                    test_seqs.add(tc)
                    train_seqs.discard(tc)
                    train_seqs.add(ts)
                    violations = new_violations
                    swaps += 1
                    break
            else:
                continue
            break
        else:
            # No beneficial swap found; force-add a violating train seq to test
            tc = sorted(train_candidates)[0]
            test_seqs.add(tc)
            train_seqs.discard(tc)
            # Recompute violations
            test_objs = set()
            for s in test_seqs:
                test_objs.update(seq_to_objs[s] - singletons)
            train_objs = set()
            for s in train_seqs:
                train_objs.update(seq_to_objs[s] - singletons)
            violations = train_objs - test_objs
            swaps += 1

    return test_seqs, swaps


def validate_and_report(data, test_seqs, obj_to_seqs, seq_to_objs, seq_to_scene):
    """Print summary tables and validate constraints."""
    all_seqs = set(data.keys())
    train_seqs = all_seqs - test_seqs
    singletons = {obj for obj, seqs in obj_to_seqs.items() if len(seqs) == 1}

    print(f"\n{'='*60}")
    print(f"HOI-M3 Train/Test Split Summary")
    print(f"{'='*60}")
    print(f"Total sequences: {len(all_seqs)}")
    print(f"Train: {len(train_seqs)}  |  Test: {len(test_seqs)}")
    print(f"Overall test ratio: {len(test_seqs)/len(all_seqs):.1%}")

    # Per-scene breakdown
    scenes = sorted(set(seq_to_scene.values()))
    print(f"\n{'Scene':<15} {'Total':>6} {'Train':>6} {'Test':>6} {'Test%':>6}")
    print("-" * 45)
    for scene in scenes:
        total = sum(1 for s in all_seqs if seq_to_scene[s] == scene)
        train = sum(1 for s in train_seqs if seq_to_scene[s] == scene)
        test = sum(1 for s in test_seqs if seq_to_scene[s] == scene)
        pct = test / total * 100 if total else 0
        print(f"{scene:<15} {total:>6} {train:>6} {test:>6} {pct:>5.1f}%")

    # Object coverage
    train_objs = set()
    for s in train_seqs:
        train_objs.update(seq_to_objs[s])
    test_objs = set()
    for s in test_seqs:
        test_objs.update(seq_to_objs[s])

    train_only = train_objs - test_objs
    train_only_non_singleton = train_only - singletons
    print(f"\nUnique objects: {len(train_objs | test_objs)}")
    print(f"  In train: {len(train_objs)}")
    print(f"  In test:  {len(test_objs)}")
    print(f"  Singletons (train-only by design): {sorted(singletons)}")
    print(f"  Constraint violations (non-singleton train-only): {len(train_only_non_singleton)}")
    if train_only_non_singleton:
        print(f"    VIOLATED: {sorted(train_only_non_singleton)}")

    # num_humans distribution
    print(f"\nnum_humans distribution:")
    for split_name, split_seqs in [("Train", train_seqs), ("Test", test_seqs)]:
        nh_counts = Counter(data[s]["num_humans"] for s in split_seqs)
        dist = ", ".join(f"{k}h:{v}" for k, v in sorted(nh_counts.items()))
        print(f"  {split_name}: {dist}")

    print(f"{'='*60}\n")
    return len(train_only_non_singleton)


def main():
    parser = argparse.ArgumentParser(description="Generate HOI-M3 train/test split")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Target fraction of sequences for test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH),
                        help=f"Input sequence_contents.json path (default: {INPUT_PATH})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help=f"Output train_test_split.json path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sequences from {args.input}")

    (obj_to_seqs, seq_to_objs, seq_to_scene, seq_to_num_humans,
     scene_seqs, config_groups) = build_indexes(data)

    # Compute per-scene targets
    scene_targets = {}
    for scene, seqs in scene_seqs.items():
        scene_targets[scene] = round(len(seqs) * args.test_ratio)
    total_target = sum(scene_targets.values())
    print(f"Scene targets: {dict(sorted(scene_targets.items()))} (total: {total_target})")
    print(f"Config groups: {len(config_groups)}")

    # Phase 1
    test_seqs, used_groups, covered_objs = phase1_mandatory(
        obj_to_seqs, config_groups, seq_to_scene, seq_to_num_humans, rng)
    multi_objs = {obj for obj, seqs in obj_to_seqs.items() if len(seqs) > 1}
    print(f"Phase 1 (mandatory):  {len(test_seqs)} test seqs, "
          f"{len(covered_objs)}/{len(multi_objs)} multi-seq objects covered")

    # Phase 2
    test_seqs, used_groups, covered_objs = phase2_greedy_coverage(
        obj_to_seqs, config_groups, seq_to_objs, seq_to_scene,
        seq_to_num_humans, scene_seqs, test_seqs, used_groups,
        covered_objs, scene_targets, rng)
    print(f"Phase 2 (greedy):     {len(test_seqs)} test seqs, "
          f"{len(covered_objs)}/{len(multi_objs)} multi-seq objects covered")

    # Phase 3
    test_seqs, used_groups = phase3_fill_targets(
        config_groups, seq_to_scene, seq_to_num_humans,
        scene_seqs, test_seqs, used_groups, scene_targets, rng)
    print(f"Phase 3 (fill):       {len(test_seqs)} test seqs")

    # Phase 4
    test_seqs, swaps = phase4_repair(data, test_seqs, obj_to_seqs, seq_to_objs, seq_to_scene, rng)
    print(f"Phase 4 (repair):     {swaps} swaps performed, {len(test_seqs)} test seqs")

    # Validate
    violations = validate_and_report(data, test_seqs, obj_to_seqs, seq_to_objs, seq_to_scene)

    # Write output
    all_seqs = set(data.keys())
    train_seqs = sorted(all_seqs - test_seqs)
    test_seqs_sorted = sorted(test_seqs)

    output = {"train": train_seqs, "test": test_seqs_sorted}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote split to {output_path}")
    print(f"  train: {len(train_seqs)} sequences")
    print(f"  test:  {len(test_seqs_sorted)} sequences")

    if violations > 0:
        print(f"\nWARNING: {violations} constraint violations remain!")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
