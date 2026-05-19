You are an autonomous Java refactoring agent.

# Task

Refactor the following Java project to remove the target architecture/code smell.

# Target smell

Smell: God Component
Smell code: GC

Target type: package
Target: org.jsoup.nodes

# Repository

The repository is available locally in the current working directory.

# Goal

Modify the Java source code to remove the specified smell while preserving the observable behavior of the project.

# Validation command

After your changes, the project must compile and pass the following validation command:

mvn -q -Dmaven.test.skip=true -Djapicmp.skip=true -Drat.skip=true -Dcheckstyle.skip=true -Dspotbugs.skip=true -Dpmd.skip=true -DskipITs clean verify

# Refactoring constraints

- Prefer small, behavior-preserving refactorings.
- Do not rewrite unrelated parts of the system.
- Do not remove public APIs unless strictly necessary.
- Do not change tests unless required by a behavior-preserving refactoring.
- Do not disable Maven plugins.
- Do not edit build files to hide failures.
- Do not suppress smell detection artificially.
- Do not delete source files only to reduce metrics.
- Preserve package conventions and Java style.
- Make the minimum set of changes needed to remove the smell.

# Smell-specific guidance

The target is a Java package affected by God Component.

Focus on reducing excessive concentration of classes and responsibilities in the package.

Possible strategies:

- identify cohesive groups of classes;
- create meaningful subpackages;
- move classes to subpackages;
- update imports and package declarations;
- preserve public APIs where possible;
- fix visibility issues caused by package moves.

Avoid:

- moving classes randomly;
- creating artificial packages with no conceptual cohesion;
- editing build configuration to hide errors;
- deleting classes or tests to reduce package size.


# Expected final response

At the end, summarize:

1. files changed;
2. refactoring strategy used;
3. expected impact on the smell;
4. validation command executed, if any;
5. any remaining risks.
