You are an autonomous Java refactoring agent.

Your previous refactoring attempt failed to compile.

# Original refactoring goal

Smell: God Component
Smell code: GC

Target type: package
Target: org.jsoup.nodes

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

WARNING: :matchText selector is deprecated and will be removed in jsoup 1.24.1. Use Element#selectNodes(String, Class) with selector ::textnode and class TextNode instead.
[ERROR] Tests run: 73, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 96.73 s <<< FAILURE! -- in org.jsoup.integration.HttpClientConnectTest
[ERROR] org.jsoup.integration.HttpClientConnectTest.multipleParsesOkAfterReadFully -- Time elapsed: 62.70 s <<< ERROR!
java.net.http.HttpTimeoutException: request timed out
	at java.net.http/jdk.internal.net.http.HttpClientImpl.send(HttpClientImpl.java:954)
	at java.net.http/jdk.internal.net.http.HttpClientFacade.send(HttpClientFacade.java:133)
	at org.jsoup.helper.HttpClientExecutor.execute(HttpClientExecutor.java:90)
	at org.jsoup.helper.HttpConnection$Response.execute(HttpConnection.java:894)
	at org.jsoup.helper.HttpConnection$Response.execute(HttpConnection.java:868)
	at org.jsoup.helper.HttpConnection.execute(HttpConnection.java:366)
	at org.jsoup.integration.ConnectTest.multipleParsesOkAfterReadFully(ConnectTest.java:409)
	at java.base/java.lang.reflect.Method.invoke(Method.java:580)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)

[ERROR] Errors: 
[ERROR]   HttpClientConnectTest>ConnectTest.multipleParsesOkAfterReadFully:409 » HttpTimeout request timed out
[ERROR] Tests run: 1971, Failures: 0, Errors: 1, Skipped: 0
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:3.5.5:test (default-test) on project jsoup: 
[ERROR] 
[ERROR] See /data/henrique/langchain_prototype/codex/data/repositories/jsoup/target/surefire-reports for the individual test results.
[ERROR] See dump files (if any exist) [date].dump, [date]-jvmRun[N].dump and [date].dumpstream.
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
