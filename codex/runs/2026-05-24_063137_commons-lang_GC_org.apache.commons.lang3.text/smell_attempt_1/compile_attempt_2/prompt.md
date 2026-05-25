You are an autonomous Java refactoring agent.

Your previous refactoring attempt failed to compile.

# Original refactoring goal

Smell: God Component
Smell code: GC

Target type: package
Target: org.apache.commons.lang3.text

# Task

Fix the compilation errors while preserving the original refactoring goal.

# Rules

- Do not revert the entire refactoring unless absolutely necessary.
- Do not edit build files to hide failures.
- Do not disable Maven plugins.
- Do not delete source files to make the build pass.
- Do not delete or weaken tests.
- Keep changes focused on fixing compilation errors.
- Preserve the moved/refactored structure whenever possible.
- Update imports, package declarations, call sites, and visibility only as needed.

# Validation command

mvn -q -Djapicmp.skip=true -Drat.skip=true -Dcheckstyle.skip=true -Dspotbugs.skip=true -Dpmd.skip=true -DskipITs clean verify

# Maven error log

[ERROR] COMPILATION ERROR : 
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/commons-lang/src/main/java/org/apache/commons/lang3/text/format/ExtendedMessageFormat.java:[459,19] cannot find symbol
  symbol:   variable StrMatcher
  location: class org.apache.commons.lang3.text.format.ExtendedMessageFormat
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.15.0:compile (default-compile) on project commons-lang3: Compilation failure
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/commons-lang/src/main/java/org/apache/commons/lang3/text/format/ExtendedMessageFormat.java:[459,19] cannot find symbol
[ERROR]   symbol:   variable StrMatcher
[ERROR]   location: class org.apache.commons.lang3.text.format.ExtendedMessageFormat
[ERROR] 
[ERROR] -> [Help 1]
[ERROR] 
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR] 
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException

# Expected final response

Summarize:

1. compilation errors fixed;
2. files changed;
3. whether the original refactoring goal was preserved;
4. remaining risks.
