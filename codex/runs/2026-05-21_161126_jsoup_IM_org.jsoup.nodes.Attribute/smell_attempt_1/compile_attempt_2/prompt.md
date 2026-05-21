You are an autonomous Java refactoring agent.

Your previous refactoring attempt failed to compile.

# Original refactoring goal

Smell: Insufficient Modularization
Smell code: IM

Target type: class
Target: org.jsoup.nodes.Attribute

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

[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[245,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[282,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[282,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[295,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[295,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[305,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[305,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[329,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[329,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[346,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[346,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[350,17] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[357,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[357,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[369,19] cannot find symbol
[ERROR]   symbol:   variable CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[372,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[372,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[385,22] cannot find symbol
[ERROR]   symbol:   variable CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[400,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[400,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[412,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[412,38] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[428,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[428,39] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[430,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[430,37] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[496,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[496,38] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[511,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[511,38] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[528,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[528,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[535,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[535,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[542,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[542,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[549,9] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/CharacterReaderTest.java:[549,33] cannot find symbol
[ERROR]   symbol:   class CharacterReader
[ERROR]   location: class org.jsoup.parser.CharacterReaderTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[58,9] cannot find symbol
[ERROR]   symbol:   class Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[58,25] cannot find symbol
[ERROR]   symbol:   variable Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[79,9] cannot find symbol
[ERROR]   symbol:   class Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[79,54] cannot find symbol
[ERROR]   symbol:   variable preserveCase
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[79,25] cannot find symbol
[ERROR]   symbol:   variable Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[125,9] cannot find symbol
[ERROR]   symbol:   class Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[125,25] cannot find symbol
[ERROR]   symbol:   variable Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[169,9] cannot find symbol
[ERROR]   symbol:   class Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[169,25] cannot find symbol
[ERROR]   symbol:   variable Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[375,31] cannot find symbol
[ERROR]   symbol:   variable CharacterReader
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[495,9] cannot find symbol
[ERROR]   symbol:   class Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[495,25] cannot find symbol
[ERROR]   symbol:   variable Parser
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/parser/HtmlParserTest.java:[496,70] cannot find symbol
[ERROR]   symbol:   variable Tag
[ERROR]   location: class org.jsoup.parser.HtmlParserTest
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
