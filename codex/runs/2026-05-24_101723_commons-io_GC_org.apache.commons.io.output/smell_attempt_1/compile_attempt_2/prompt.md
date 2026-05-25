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

	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-13858833909537087378: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 5 more

[ERROR] org.apache.commons.io.output.FileWriterWithEncodingTest.testDifferentEncoding -- Time elapsed: 0.008 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-998337424094656090: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 5 more

[ERROR] org.apache.commons.io.output.FileWriterWithEncodingTest.testSameEncoding_string_string_constructor -- Time elapsed: 0.005 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-12367513551402417936: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 5 more

[ERROR] org.apache.commons.io.output.FileWriterWithEncodingTest.testSameEncoding_Charset_constructor -- Time elapsed: 0.004 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-5906340987060921994: No space left on device
	at java.base/sun.nio.fs.UnixException.translateToIOException(UnixException.java:100)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:106)
	at java.base/sun.nio.fs.UnixException.rethrowAsIOException(UnixException.java:111)
	at java.base/sun.nio.fs.UnixFileSystemProvider.createDirectory(UnixFileSystemProvider.java:463)
	at java.base/java.nio.file.Files.createDirectory(Files.java:700)
	at java.base/java.nio.file.TempFileHelper.create(TempFileHelper.java:134)
	at java.base/java.nio.file.TempFileHelper.createTempDirectory(TempFileHelper.java:171)
	at java.base/java.nio.file.Files.createTempDirectory(Files.java:1018)
	... 5 more

[ERROR] org.apache.commons.io.output.FileWriterWithEncodingTest.testConstructor_File_existingFile_withContent -- Time elapsed: 0.002 s <<< ERROR!
org.junit.jupiter.api.extension.ExtensionConfigurationException: Failed to create default temp directory
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.Collections$UnmodifiableCollection.forEach(Collections.java:1117)
	at java.base/java.util.Collections$SingletonList.forEach(Collections.java:5188)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
	at java.base/java.util.ArrayList.forEach(ArrayList.java:1597)
Caused by: java.nio.file.FileSystemException: /tmp/junit-2982690931044369169: No space left on device
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
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
[ERROR] Tests run: 153, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 10.89 s <<< FAILURE! -- in org.apache.commons.io.channels.FileChannelsTest
[ERROR] org.apache.commons.io.channels.FileChannelsTest.testContentEqualsEmpty(int, FileChannelType)[1] -- Time elapsed: 0.168 s <<< ERROR!
java.io.IOException: No space left on device
	at java.base/sun.nio.ch.UnixFileDispatcherImpl.write0(Native Method)
	at java.base/sun.nio.ch.UnixFileDispatcherImpl.write(UnixFileDispatcherImpl.java:65)
	at java.base/sun.nio.ch.IOUtil.writeFromNativeBuffer(IOUtil.java:137)
	at java.base/sun.nio.ch.IOUtil.write(IOUtil.java:102)
	at java.base/sun.nio.ch.IOUtil.write(IOUtil.java:72)
	at java.base/sun.nio.ch.FileChannelImpl.implWrite(FileChannelImpl.java:371)
	at java.base/sun.nio.ch.FileChannelImpl.write(FileChannelImpl.java:351)
	at java.base/sun.nio.ch.ChannelOutputStream.writeFully(ChannelOutputStream.java:68)
	at java.base/sun.nio.ch.ChannelOutputStream.write(ChannelOutputStream.java:105)
	at java.base/java.nio.channels.Channels$WritableByteChannelImpl.write(Channels.java:394)
	at org.apache.commons.io.IOUtils.write(IOUtils.java:3734)
	at org.apache.commons.io.FileUtils.writeStringToFile(FileUtils.java:3559)
	at org.apache.commons.io.FileUtils.writeStringToFile(FileUtils.java:3542)
	at org.apache.commons.io.channels.FileChannelsTest.testContentEqualsEmpty(FileChannelsTest.java:157)
	at java.base/java.lang.reflect.Method.invoke(Method.java:580)
	at java.base/java.util.Optional.ifPresent(Optional.java:178)
	at java.base/java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
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

OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
OpenJDK 64-Bit Server VM warning: Sharing is only supported for boot loader classes because bootstrap classpath has been appended
[ERROR] Errors: 
[ERROR]   FileChannelsTest.testContentEqualsEmpty:157 » IO No space left on device
[ERROR]   FilesUncheckTest.testCreateDirectory:137 » UncheckedIO java.nio.file.FileSystemException: /tmp/junit-10080221325274347637/newdir: No space left on device
[ERROR]   FilesUncheckTest.testCreateTempDirectoryPathStringFileAttributeOfQArray:158 » UncheckedIO java.nio.file.FileSystemException: /tmp/junit-10080221325274347637/prefix10693851378383612226: No space left on device
[ERROR]   FilesUncheckTest.testCreateTempDirectoryStringFileAttributeOfQArray:163 » UncheckedIO java.nio.file.FileSystemException: /tmp/prefix3651051187612085418: No space left on device
[ERROR]   FilesUncheckTest.testNewBufferedWriterPathCharsetOpenOptionArray:279 » UncheckedIO java.io.IOException: No space left on device
[ERROR]   FilesUncheckTest.testNewBufferedWriterPathOpenOptionArray:290 » UncheckedIO java.io.IOException: No space left on device
[ERROR]   FilesUncheckTest.testNewOutputStream:357 » UncheckedIO java.io.IOException: No space left on device
[ERROR]   FilesUncheckTest.testWritePathByteArrayOpenOptionArray:474 » UncheckedIO java.io.IOException: No space left on device
[ERROR]   FilesUncheckTest.testWritePathIterableOfQextendsCharSequenceCharsetOpenOptionArray:481 » UncheckedIO java.io.IOException: No space left on device
[ERROR]   FilesUncheckTest.testWritePathIterableOfQextendsCharSequenceOpenOptionArray:488 » UncheckedIO java.io.IOException: No space left on device
[ERROR]   FileWriterWithEncodingTest.testConstructor_File_directory » ExtensionConfiguration Failed to create default temp directory
[ERROR]   FileWriterWithEncodingTest.testConstructor_File_existingFile_withContent » ExtensionConfiguration Failed to create default temp directory
[ERROR]   FileWriterWithEncodingTest.testDifferentEncoding » ExtensionConfiguration Failed to create default temp directory
[ERROR]   FileWriterWithEncodingTest.testSameEncoding_Charset_constructor » ExtensionConfiguration Failed to create default temp directory
[ERROR]   FileWriterWithEncodingTest.testSameEncoding_null_CharsetEncoder_constructor » ExtensionConfiguration Failed to create default temp directory
[ERROR]   FileWriterWithEncodingTest.testSameEncoding_string_string_constructor » ExtensionConfiguration Failed to create default temp directory
[ERROR] Tests run: 6304, Failures: 0, Errors: 16, Skipped: 26
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
