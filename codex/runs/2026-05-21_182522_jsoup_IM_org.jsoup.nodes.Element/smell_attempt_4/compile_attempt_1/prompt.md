You are an autonomous Java refactoring agent.

The previous refactoring attempt compiled successfully, but the target smell is still reported by Designite.

# Original refactoring goal

Smell: Insufficient Modularization
Smell code: IM

Target type: class
Target: org.jsoup.nodes.Element

# Task

Continue refactoring the current repository state to remove the target smell.

# Rules

- Do not revert the previous successful refactoring.
- Preserve compilation.
- Keep changes focused on removing the target smell.
- Do not edit build files to hide failures.
- Do not disable Maven plugins.
- Do not delete source files or tests only to reduce metrics.
- Do not suppress smell detection artificially.
- Follow the smell-specific guidance strictly.

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

# Validation command

mvn -q -Djapicmp.skip=true -Drat.skip=true -Dcheckstyle.skip=true -Dspotbugs.skip=true -Dpmd.skip=true -DskipITs clean verify

# Designite result

Designite still reports the target smell after the previous attempt.

# Expected final response

Summarize:

1. additional refactoring strategy used;
2. files changed;
3. expected impact on the smell;
4. validation command executed, if any;
5. remaining risks.
