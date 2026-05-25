You are an autonomous Java refactoring agent.

Your previous refactoring attempt failed to compile.

# Original refactoring goal

Smell: God Component
Smell code: GC

Target type: package
Target: org.apache.commons.io.output

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

	at org.mockito.internal.creation.MockSettingsImpl.build(MockSettingsImpl.java:234)
	at org.mockito.internal.MockitoCore.mock(MockitoCore.java:86)
	at org.mockito.Mockito.mock(Mockito.java:2037)
	at org.mockito.Mockito.mock(Mockito.java:1952)
	at org.apache.commons.io.output.TeeOutputStreamTest.testIOExceptionOnClose(TeeOutputStreamTest.java:50)
	at java.base/java.lang.reflect.Method.invoke(Method.java:580)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.lang.IllegalStateException: Failed to load interface org.mockito.plugins.MockMaker implementation declared in java.lang.CompoundEnumeration@352c308
	at org.mockito.internal.configuration.plugins.PluginInitializer.loadImpl(PluginInitializer.java:56)
	at org.mockito.internal.configuration.plugins.PluginLoader.loadPlugin(PluginLoader.java:65)
	at org.mockito.internal.configuration.plugins.PluginLoader.loadPlugin(PluginLoader.java:50)
	at org.mockito.internal.configuration.plugins.PluginRegistry.<init>(PluginRegistry.java:27)
	at org.mockito.internal.configuration.plugins.Plugins.<clinit>(Plugins.java:22)
	at org.mockito.internal.MockitoCore.<clinit>(MockitoCore.java:73)
	at org.mockito.Mockito.<clinit>(Mockito.java:1669)
	... 4 more
Caused by: java.lang.reflect.InvocationTargetException
	at java.base/jdk.internal.reflect.DirectConstructorHandleAccessor.newInstance(DirectConstructorHandleAccessor.java:74)
	at java.base/java.lang.reflect.Constructor.newInstanceWithCaller(Constructor.java:501)
	at java.base/java.lang.reflect.Constructor.newInstance(Constructor.java:485)
	at org.mockito.internal.configuration.plugins.PluginInitializer.loadImpl(PluginInitializer.java:51)
	... 10 more
Caused by: org.mockito.exceptions.base.MockitoInitializationException: 
Could not initialize inline Byte Buddy mock maker.

It appears as if your JDK does not supply a working agent attachment mechanism.
Java               : 23
JVM vendor name    : Amazon.com Inc.
JVM vendor version : 23.0.2+7-FR
JVM name           : OpenJDK 64-Bit Server VM
JVM version        : 23.0.2+7-FR
JVM info           : mixed mode, sharing
OS name            : Linux
OS version         : 5.15.0-177-generic

	at org.mockito.internal.creation.bytebuddy.InlineDelegateByteBuddyMockMaker.<init>(InlineDelegateByteBuddyMockMaker.java:244)
	at org.mockito.internal.creation.bytebuddy.InlineByteBuddyMockMaker.<init>(InlineByteBuddyMockMaker.java:23)
	at java.base/jdk.internal.reflect.DirectConstructorHandleAccessor.newInstance(DirectConstructorHandleAccessor.java:62)
	... 13 more
Caused by: java.lang.IllegalStateException: 
Mockito could not self-attach a Java agent to the current VM. This feature is required for inline mocking.
This error occured due to an I/O error during the creation of this agent: java.io.IOException: No space left on device

Potentially, the current VM does not support the instrumentation API correctly
	at org.mockito.internal.creation.bytebuddy.InlineDelegateByteBuddyMockMaker.<clinit>(InlineDelegateByteBuddyMockMaker.java:176)
	... 15 more
Caused by: java.io.IOException: No space left on device
	at java.base/java.io.FileOutputStream.writeBytes(Native Method)
	at java.base/java.io.FileOutputStream.write(FileOutputStream.java:400)
	at java.base/java.util.zip.DeflaterOutputStream.deflate(DeflaterOutputStream.java:280)
	at java.base/java.util.zip.ZipOutputStream.closeEntry(ZipOutputStream.java:285)
	at java.base/java.util.zip.ZipOutputStream.finish(ZipOutputStream.java:390)
	at java.base/java.util.zip.DeflaterOutputStream.close(DeflaterOutputStream.java:249)
	at java.base/java.util.zip.ZipOutputStream.close(ZipOutputStream.java:407)
	at org.mockito.internal.creation.bytebuddy.InlineDelegateByteBuddyMockMaker.<clinit>(InlineDelegateByteBuddyMockMaker.java:156)
	... 15 more

[ERROR] org.apache.commons.io.output.TeeOutputStreamTest.testIOExceptionOnCloseBranch -- Time elapsed: 0.003 s <<< ERROR!
java.lang.IllegalStateException: Could not initialize plugin: interface org.mockito.plugins.MockMaker (alternate: null)
	at org.mockito.internal.configuration.plugins.PluginLoader$1.invoke(PluginLoader.java:84)
	at jdk.proxy2/jdk.proxy2.$Proxy11.isTypeMockable(Unknown Source)
	at org.mockito.internal.util.MockUtil.typeMockabilityOf(MockUtil.java:78)
	at org.mockito.internal.util.MockCreationValidator.validateType(MockCreationValidator.java:22)
	at org.mockito.internal.creation.MockSettingsImpl.validatedSettings(MockSettingsImpl.java:267)
	at org.mockito.internal.creation.MockSettingsImpl.build(MockSettingsImpl.java:234)
	at org.mockito.internal.MockitoCore.mock(MockitoCore.java:86)
	at org.mockito.Mockito.mock(Mockito.java:2037)
	at org.mockito.Mockito.mock(Mockito.java:1952)
	at org.apache.commons.io.output.TeeOutputStreamTest.testIOExceptionOnCloseBranch(TeeOutputStreamTest.java:64)
	at java.base/java.lang.reflect.Method.invoke(Method.java:580)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.lang.IllegalStateException: Failed to load interface org.mockito.plugins.MockMaker implementation declared in java.lang.CompoundEnumeration@352c308
	at org.mockito.internal.configuration.plugins.PluginInitializer.loadImpl(PluginInitializer.java:56)
	at org.mockito.internal.configuration.plugins.PluginLoader.loadPlugin(PluginLoader.java:65)
	at org.mockito.internal.configuration.plugins.PluginLoader.loadPlugin(PluginLoader.java:50)
	at org.mockito.internal.configuration.plugins.PluginRegistry.<init>(PluginRegistry.java:27)
	at org.mockito.internal.configuration.plugins.Plugins.<clinit>(Plugins.java:22)
	at org.mockito.internal.MockitoCore.<clinit>(MockitoCore.java:73)
	at org.mockito.Mockito.<clinit>(Mockito.java:1669)
	at org.apache.commons.io.output.TeeOutputStreamTest.testIOExceptionOnClose(TeeOutputStreamTest.java:50)
	... 3 more
Caused by: java.lang.reflect.InvocationTargetException
	at java.base/jdk.internal.reflect.DirectConstructorHandleAccessor.newInstance(DirectConstructorHandleAccessor.java:74)
	at java.base/java.lang.reflect.Constructor.newInstanceWithCaller(Constructor.java:501)
	at java.base/java.lang.reflect.Constructor.newInstance(Constructor.java:485)
	at org.mockito.internal.configuration.plugins.PluginInitializer.loadImpl(PluginInitializer.java:51)
	... 10 more
Caused by: org.mockito.exceptions.base.MockitoInitializationException: 
Could not initialize inline Byte Buddy mock maker.

It appears as if your JDK does not supply a working agent attachment mechanism.
Java               : 23
JVM vendor name    : Amazon.com Inc.
JVM vendor version : 23.0.2+7-FR
JVM name           : OpenJDK 64-Bit Server VM
JVM version        : 23.0.2+7-FR
JVM info           : mixed mode, sharing
OS name            : Linux
OS version         : 5.15.0-177-generic

	at org.mockito.internal.creation.bytebuddy.InlineDelegateByteBuddyMockMaker.<init>(InlineDelegateByteBuddyMockMaker.java:244)
	at org.mockito.internal.creation.bytebuddy.InlineByteBuddyMockMaker.<init>(InlineByteBuddyMockMaker.java:23)
	at java.base/jdk.internal.reflect.DirectConstructorHandleAccessor.newInstance(DirectConstructorHandleAccessor.java:62)
	... 13 more
Caused by: java.lang.IllegalStateException: 
Mockito could not self-attach a Java agent to the current VM. This feature is required for inline mocking.
This error occured due to an I/O error during the creation of this agent: java.io.IOException: No space left on device

Potentially, the current VM does not support the instrumentation API correctly
	at org.mockito.internal.creation.bytebuddy.InlineDelegateByteBuddyMockMaker.<clinit>(InlineDelegateByteBuddyMockMaker.java:176)
	... 15 more
Caused by: java.io.IOException: No space left on device
	at java.base/java.io.FileOutputStream.writeBytes(Native Method)
	at java.base/java.io.FileOutputStream.write(FileOutputStream.java:400)
	at java.base/java.util.zip.DeflaterOutputStream.deflate(DeflaterOutputStream.java:280)
	at java.base/java.util.zip.ZipOutputStream.closeEntry(ZipOutputStream.java:285)
	at java.base/java.util.zip.ZipOutputStream.finish(ZipOutputStream.java:390)
	at java.base/java.util.zip.DeflaterOutputStream.close(DeflaterOutputStream.java:249)
	at java.base/java.util.zip.ZipOutputStream.close(ZipOutputStream.java:407)
	at org.mockito.internal.creation.bytebuddy.InlineDelegateByteBuddyMockMaker.<clinit>(InlineDelegateByteBuddyMockMaker.java:156)
	... 15 more

OpenJDK 64-Bit Server VM warning: Insufficient space for shared memory file:
   1229293
Try using the -Djava.io.tmpdir= option to select an alternate temp location.

OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
[ERROR] Errors: 
[ERROR]   MemoryMappedFileInputStreamTest.testSkipOutOfCurrentBuffer » ExtensionConfiguration Failed to create default temp directory
[ERROR]   MemoryMappedFileInputStreamTest.testSmallFileBuilder » ExtensionConfiguration Failed to create default temp directory
[ERROR]   MemoryMappedFileInputStreamTest.testSmallPath » ExtensionConfiguration Failed to create default temp directory
[ERROR]   MemoryMappedFileInputStreamTest.testSmallPathBuilder » ExtensionConfiguration Failed to create default temp directory
[ERROR]   TailerCloseTest.testCloseTailer:91 » IO No space left on device
[ERROR]   TailerTest » ExtensionConfiguration Failed to create default temp directory
[ERROR]   ChunkedOutputStreamTest.testBuildSetPath:75 » IO No space left on device
[ERROR]   TeeOutputStreamTest.testIOExceptionOnClose:50 » IllegalState Could not initialize plugin: interface org.mockito.plugins.MockMaker (alternate: null)
[ERROR]   TeeOutputStreamTest.testIOExceptionOnCloseBranch:64 » IllegalState Could not initialize plugin: interface org.mockito.plugins.MockMaker (alternate: null)
[ERROR] Tests run: 6278, Failures: 0, Errors: 9, Skipped: 26
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:3.5.5:test (default-test) on project commons-io: 
[ERROR] 
[ERROR] See /data/henrique/langchain_prototype/codex/data/repositories/commons-io/target/surefire-reports for the individual test results.
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
