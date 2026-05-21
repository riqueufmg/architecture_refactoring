The target is a Java package affected by God Component.

Your task is to reduce the God Component smell by moving exactly ONE cohesive cluster of classes from the target package to a new subpackage.

# Mandatory cluster movement rule

This run is valid only if you move a cohesive cluster of 2 to 4 classes.

You must not move only one class.
You must not move more than four classes.
You must not move all classes from the target package.
All moved classes must go to the same new subpackage.

If you cannot identify a safe cluster of 4 to 20 classes, choose the safest available cluster of 2 classes. Do not fall back to moving a single class.

# Cluster selection rules

- Select exactly ONE cohesive cluster.
- The cluster must contain 5 to 10 classes.
- Prefer classes with strong internal relationships.
- Prefer classes that interact mostly with each other rather than with many unrelated classes.
- Prefer clusters that can compile after limited import, call-site, and visibility updates.
- Avoid moving highly central parser/state-machine classes alone.
- Avoid clusters that would require exposing many package-private members from classes that remain in the original package.

# Movement rules

- Move classes only from the target package to a subpackage of the target package.
- All moved classes must go to the same destination package.
- The destination package must be a meaningful subpackage name.
- Do not move classes to unrelated packages.
- Do not introduce facades.
- Do not create unnecessary abstractions.
- API changes are allowed if they are necessary for this refactoring.

# Implementation rules

- Apply the refactoring directly to the repository.
- Update package declarations for all moved classes.
- Update imports and fully qualified references as needed.
- Update call sites only when required by the move.
- Make the minimum visibility changes needed to preserve compilation.
- Do not perform broad visibility changes unrelated to the selected cluster.
- Do not edit build files to hide failures.
- Do not delete classes or tests to reduce package size.
- Do not suppress smell detection artificially.
- Do not perform unrelated cleanups.

# Invalid outcomes

The refactoring is invalid if:

- only one class is moved;
- classes are moved to different destination packages;
- classes are moved outside a subpackage of the target package;
- unrelated cleanup dominates the diff;
- source code is deleted only to reduce size;
- build files are changed to hide failures.

# Expected final response

Explicitly report:

- selected cluster;
- destination package;
- classes moved;
- why the cluster is cohesive;
- files changed;
- validation command executed, if any.
