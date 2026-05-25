You are an autonomous Java refactoring agent.

# Task

Refactor the following Java project to remove the target architecture/code smell.

# Target smell

Smell: Insufficient Modularization
Smell code: IM

Target type: class
Target: org.jsoup.nodes.Element

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

Your task is to reduce the Insufficient Modularization smell by decomposing the target class into smaller, cohesive units while preserving behavior and passing the full validation command.

# Detection goal

This run is successful only if the target class is likely to fall below BOTH Insufficient Modularization thresholds:

- public methods < 20
- total methods < 30

Do not make cosmetic-only changes. The refactoring must meaningfully reduce the responsibilities directly implemented by the target class.

# Main refactoring rule

Select exactly ONE cohesive responsibility currently implemented by the target class and extract it into one or more focused collaborators.

Prefer one of these strategies:

- Extract Class
- Extract helper class
- Move Method
- Move Field
- Extract Method, only when it helps prepare or support a larger decomposition
- Introduce a small collaborator used by the original class

Do not try to refactor the entire class at once. Prefer a conservative decomposition that compiles and preserves tests.

# Responsibility selection rules

- Select a responsibility represented by a cohesive group of methods and fields.
- Prefer private or package-private methods when possible.
- Prefer helper behavior that is internally cohesive and has limited external dependencies.
- Prefer method groups that share fields, constants, parameters, or domain concepts.
- Prefer responsibilities that can be extracted without changing the public API.
- Avoid extracting methods that are heavily intertwined with unrelated state.
- Avoid extracting behavior that would require broad changes across unrelated classes.

# Public API preservation rules

- Do not remove public methods only to reduce the method count.
- Do not rename public methods unless absolutely required.
- Do not change public method signatures unless absolutely required.
- If a public method must remain for compatibility, keep it as a delegating wrapper when possible.
- Moving implementation out of the target class is preferred over deleting or hiding behavior.
- API changes are allowed only when necessary and must be reflected in all affected production code and tests.

# Implementation rules

- Apply the refactoring directly to the repository.
- Create new classes only when they represent a cohesive extracted responsibility.
- Keep new classes in an appropriate package close to the target class.
- Update imports and fully qualified references as needed.
- Update call sites only when required by the refactoring.
- Make the minimum visibility changes needed to preserve compilation and tests.
- Do not perform broad visibility changes unrelated to the extracted responsibility.
- Do not edit build files to hide failures.
- Do not delete source code only to reduce method count.
- Do not delete or disable tests.
- Do not suppress smell detection artificially.
- Do not perform unrelated cleanups.
- Avoid large rewrites not required for removing the smell.

# Test update rules

The validation command runs tests. Therefore, tests must be preserved and updated when necessary.

- Update test files when they are affected by the refactoring.
- If moved or extracted behavior changes imports, constructors, package names, or call sites used by tests, update the tests accordingly.
- Do not remove tests to make the build pass.
- Do not disable tests.
- Do not weaken assertions.
- Do not replace meaningful assertions with trivial ones.
- Keep test changes limited to what is required by the refactoring.
- If behavior is preserved through delegation, existing tests should continue to pass with minimal changes.

# Recommended strategy

1. Inspect the target class.
2. Identify one cohesive responsibility implemented by several methods and possibly related fields.
3. Extract that responsibility into a focused collaborator.
4. Keep the target class as the public entry point when needed.
5. Replace direct implementation in the target class with delegation to the new collaborator.
6. Update production call sites and tests only when required.
7. Run the configured validation command.

# Invalid outcomes

The refactoring is invalid if:

- the change is cosmetic only;
- the change is mostly renaming;
- public methods are deleted only to reduce method count;
- tests are deleted, disabled, or weakened;
- build files are changed to hide failures;
- unrelated cleanup dominates the diff;
- new classes are generic dumping grounds instead of cohesive collaborators;
- the target class keeps essentially the same responsibilities after the change.

# Expected final response

Explicitly report:

- selected responsibility;
- extracted class or collaborator, if any;
- methods or fields moved/extracted;
- how the target class was simplified;
- files changed, including test files if any;
- validation command executed, if any;
- remaining risks.

# Expected final response

At the end, summarize:

1. files changed;
2. refactoring strategy used;
3. expected impact on the smell;
4. validation command executed, if any;
5. any remaining risks.
