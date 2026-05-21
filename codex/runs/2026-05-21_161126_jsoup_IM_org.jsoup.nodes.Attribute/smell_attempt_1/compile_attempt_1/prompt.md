You are an autonomous Java refactoring agent.

# Task

Refactor the following Java project to remove the target architecture/code smell.

# Target smell

Smell: Insufficient Modularization
Smell code: IM

Target type: class
Target: org.jsoup.nodes.Attribute

# Repository

The repository is available locally in the current working directory.

# Goal

Modify the Java source code to remove the specified smell while preserving the observable behavior of the project.

# Validation command

After your changes, the project must compile and pass the following validation command:

mvn -q -Djapicmp.skip=true -Drat.skip=true -Dcheckstyle.skip=true -Dspotbugs.skip=true -Dpmd.skip=true -DskipITs clean verify

# Refactoring constraints

- Prefer small, behavior-preserving refactorings.
- Do not rewrite unrelated parts of the system.
- Do not remove public APIs unless strictly necessary.
- Update tests only when required to preserve compilation and behavior after the refactoring.
- Do not disable Maven plugins.
- Do not edit build files to hide failures.
- Do not suppress smell detection artificially.
- Do not delete source files only to reduce metrics.
- Preserve package conventions and Java style.
- Make the minimum set of changes needed to remove the smell.

# Test update rules

- Update test files when they are affected by the refactoring.
- If moved classes are referenced in tests, update package declarations, imports, fully qualified names, and call sites in the tests.
- Do not rewrite tests unnecessarily.
- Do not remove or weaken assertions.
- Do not delete tests to make the build pass.
- Do not disable tests.
- Keep test changes limited to what is required by the refactoring.

# Smell-specific guidance

The target is a Java class affected by Insufficient Modularization.

Focus on decomposing responsibilities inside the class.

Possible strategies:

- extract cohesive helper methods;
- extract cohesive helper classes;
- move nested responsibilities to new classes;
- reduce method count or class size when behavior-preserving;
- improve separation of concerns.

Avoid:

- cosmetic-only changes;
- renaming-only changes;
- deleting methods;
- moving code without reducing the modularization problem.


# Expected final response

At the end, summarize:

1. files changed;
2. refactoring strategy used;
3. expected impact on the smell;
4. validation command executed, if any;
5. any remaining risks.
