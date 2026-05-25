You are an autonomous Java refactoring agent.

Your previous refactoring attempt failed to compile.

# Original refactoring goal

Smell: God Component
Smell code: GC

Target type: package
Target: org.apache.commons.io.channels

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

	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 38 more

[ERROR] org.apache.commons.io.build.PathOriginTest.testGetRandomAccessFile(OpenOption)[9] -- Time elapsed: 0.004 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.Optional.ifPresent(Optional.java:178)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.Iterator.forEachRemaining(Iterator.java:133)
	at java.base/java.util.Spliterators$IteratorSpliterator.forEachRemaining(Spliterators.java:1939)
	at java.base/java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:570)
	at java.base/java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:560)
	at java.base/java.util.stream.ForEachOps$ForEachOp.evaluateSequential(ForEachOps.java:151)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.evaluateSequential(ForEachOps.java:174)
	at java.base/java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:265)
	at java.base/java.util.stream.ReferencePipeline.forEach(ReferencePipeline.java:636)
	at java.base/java.util.stream.ReferencePipeline$7$1FlatMap.accept(ReferencePipeline.java:294)
	at java.base/java.util.ArrayList$ArrayListSpliterator.forEachRemaining(ArrayList.java:1709)
	at java.base/java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:570)
	at java.base/java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:560)
	at java.base/java.util.stream.ForEachOps$ForEachOp.evaluateSequential(ForEachOps.java:151)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.evaluateSequential(ForEachOps.java:174)
	at java.base/java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:265)
	at java.base/java.util.stream.ReferencePipeline.forEach(ReferencePipeline.java:636)
	at java.base/java.util.stream.ReferencePipeline$7$1FlatMap.accept(ReferencePipeline.java:294)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.ArrayList$ArrayListSpliterator.forEachRemaining(ArrayList.java:1709)
	at java.base/java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:570)
	at java.base/java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:560)
	at java.base/java.util.stream.ForEachOps$ForEachOp.evaluateSequential(ForEachOps.java:151)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.evaluateSequential(ForEachOps.java:174)
	at java.base/java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:265)
	at java.base/java.util.stream.ReferencePipeline.forEach(ReferencePipeline.java:636)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-1563268250838258077: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 38 more

[ERROR] org.apache.commons.io.build.PathOriginTest.testGetRandomAccessFile(OpenOption)[10] -- Time elapsed: 0.004 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.Optional.ifPresent(Optional.java:178)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.Iterator.forEachRemaining(Iterator.java:133)
	at java.base/java.util.Spliterators$IteratorSpliterator.forEachRemaining(Spliterators.java:1939)
	at java.base/java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:570)
	at java.base/java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:560)
	at java.base/java.util.stream.ForEachOps$ForEachOp.evaluateSequential(ForEachOps.java:151)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.evaluateSequential(ForEachOps.java:174)
	at java.base/java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:265)
	at java.base/java.util.stream.ReferencePipeline.forEach(ReferencePipeline.java:636)
	at java.base/java.util.stream.ReferencePipeline$7$1FlatMap.accept(ReferencePipeline.java:294)
	at java.base/java.util.ArrayList$ArrayListSpliterator.forEachRemaining(ArrayList.java:1709)
	at java.base/java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:570)
	at java.base/java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:560)
	at java.base/java.util.stream.ForEachOps$ForEachOp.evaluateSequential(ForEachOps.java:151)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.evaluateSequential(ForEachOps.java:174)
	at java.base/java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:265)
	at java.base/java.util.stream.ReferencePipeline.forEach(ReferencePipeline.java:636)
	at java.base/java.util.stream.ReferencePipeline$7$1FlatMap.accept(ReferencePipeline.java:294)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:215)
	at java.base/java.util.ArrayList$ArrayListSpliterator.forEachRemaining(ArrayList.java:1709)
	at java.base/java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:570)
	at java.base/java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:560)
	at java.base/java.util.stream.ForEachOps$ForEachOp.evaluateSequential(ForEachOps.java:151)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.evaluateSequential(ForEachOps.java:174)
	at java.base/java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:265)
	at java.base/java.util.stream.ReferencePipeline.forEach(ReferencePipeline.java:636)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-13402287234027235001: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 38 more

[ERROR] org.apache.commons.io.build.PathOriginTest.testGetReadableByteChannel -- Time elapsed: 0.003 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-2084480844072819238: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 5 more

OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
[ERROR] Errors: 
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[10] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[4] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[5] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[6] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[7] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[8] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetRandomAccessFile(OpenOption)[9] » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathOriginTest.testGetReadableByteChannel » ExtensionConfiguration Failed to create default temp directory
[ERROR]   CloseShieldChannelTest.testWrapFileChannel(Path) » ParameterResolution Failed to resolve parameter [java.nio.file.Path arg0] in method [void org.apache.commons.io.channels.CloseShieldChannelTest.testWrapFileChannel(java.nio.file.Path) throws java.io.IOException]: Failed to create default temp directory
[ERROR]   FileChannelsTest.testContentEqualsSeekableByteChannel:279 » IO No space left on device
[ERROR]   FileChannelsTest.testContentEqualsSeekableByteChannel:268 » IO No space left on device
[ERROR]   FileChannelsTest.testContentEqualsSeekableByteChannel:268 » IO No space left on device
[ERROR]   FileChannelsTest.testContentEqualsSeekableByteChannel:268 » IO No space left on device
[ERROR]   FileChannelsTest.testContentEqualsSeekableByteChannel:268 » IO No space left on device
[ERROR]   PathFileComparatorTest.testCaseSensitivity » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testComparator » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testReverseComparator » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testSortArray » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testSortArrayNull » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testSortList » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testSortListNull » ExtensionConfiguration Failed to create default temp directory
[ERROR]   PathFileComparatorTest.testToString » ExtensionConfiguration Failed to create default temp directory
[ERROR] Tests run: 6304, Failures: 0, Errors: 22, Skipped: 26
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
