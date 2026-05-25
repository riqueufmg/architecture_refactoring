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

[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/ElementTest.java:[3442,55] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.ElementTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/ElementTest.java:[3445,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.ElementTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/ElementTest.java:[3445,56] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.ElementTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/ElementTest.java:[3525,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.ElementTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/ElementTest.java:[3525,26] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.ElementTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeStreamTest.java:[44,45] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeStreamTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[287,36] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[288,32] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[321,33] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[322,17] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[322,38] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[365,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[365,25] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[366,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[366,25] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[421,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[421,26] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[424,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[447,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[447,27] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[450,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[450,26] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[473,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[473,26] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[485,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[485,29] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[495,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[495,26] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[516,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/NodeTest.java:[516,26] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.NodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[57,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[57,27] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[60,35] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[73,101] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[73,76] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[73,52] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/PrinterTest.java:[73,28] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.PrinterTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[22,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[22,28] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[23,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[23,28] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[24,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[24,30] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[25,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[25,29] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[26,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[26,29] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[41,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[41,30] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[44,9] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/test/java/org/jsoup/nodes/TextNodeTest.java:[44,24] cannot find symbol
[ERROR]   symbol:   class TextNode
[ERROR]   location: class org.jsoup.nodes.TextNodeTest
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
