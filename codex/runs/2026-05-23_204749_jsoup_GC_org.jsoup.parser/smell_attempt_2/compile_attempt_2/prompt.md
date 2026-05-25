You are an autonomous Java refactoring agent.

Your previous refactoring attempt failed to compile.

# Original refactoring goal

Smell: God Component
Smell code: GC

Target type: package
Target: org.jsoup.parser

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
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/Parser.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/TreeBuilder.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/ParseSettings.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[4,24] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[346,15] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[105,5] cannot find symbol
  symbol:   class TagSet
  location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/TreeBuilder.java:[270,22] method valueOf in class org.jsoup.parser.tag.TagSet cannot be applied to given types;
  required: java.lang.String,java.lang.String,java.lang.String,boolean
  found:    java.lang.String,java.lang.String,java.lang.String,boolean
  reason: valueOf(java.lang.String,@org.jspecify.annotations.Nullable java.lang.String,java.lang.String,boolean) is not public in org.jsoup.parser.tag.TagSet; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/TreeBuilder.java:[274,22] method valueOf in class org.jsoup.parser.tag.TagSet cannot be applied to given types;
  required: java.lang.String,java.lang.String,java.lang.String,boolean
  found:    java.lang.String,java.lang.String,java.lang.String,boolean
  reason: valueOf(java.lang.String,@org.jspecify.annotations.Nullable java.lang.String,java.lang.String,boolean) is not public in org.jsoup.parser.tag.TagSet; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/TagSet.java:[147,59] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[57,36] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[68,36] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[104,40] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[347,32] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[348,32] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[138,68] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[140,68] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[482,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[490,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[494,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[498,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[626,51] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[1812,122] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[118,51] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[318,16] setSeenSelfClose() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[346,21] setSeenSelfClose() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[381,44] namespace is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[701,20] hasParserOption(int) is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[716,37] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[723,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[727,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[727,67] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[731,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[731,67] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[735,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[745,43] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[811,55] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[827,39] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[827,75] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[830,21] hasParserOption(int) is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[843,41] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[67,49] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[106,20] cannot find symbol
  symbol:   class TagSet
  location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[164,9] cannot find symbol
  symbol:   class Tag
  location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[234,44] cannot find symbol
  symbol:   variable Tag
  location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.15.0:compile (compile-java8-base) on project jsoup: Compilation failure: Compilation failure: 
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/Parser.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/TreeBuilder.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/ParseSettings.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[4,24] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[346,15] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[3,28] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[105,5] cannot find symbol
[ERROR]   symbol:   class TagSet
[ERROR]   location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/TreeBuilder.java:[270,22] method valueOf in class org.jsoup.parser.tag.TagSet cannot be applied to given types;
[ERROR]   required: java.lang.String,java.lang.String,java.lang.String,boolean
[ERROR]   found:    java.lang.String,java.lang.String,java.lang.String,boolean
[ERROR]   reason: valueOf(java.lang.String,@org.jspecify.annotations.Nullable java.lang.String,java.lang.String,boolean) is not public in org.jsoup.parser.tag.TagSet; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/TreeBuilder.java:[274,22] method valueOf in class org.jsoup.parser.tag.TagSet cannot be applied to given types;
[ERROR]   required: java.lang.String,java.lang.String,java.lang.String,boolean
[ERROR]   found:    java.lang.String,java.lang.String,java.lang.String,boolean
[ERROR]   reason: valueOf(java.lang.String,@org.jspecify.annotations.Nullable java.lang.String,java.lang.String,boolean) is not public in org.jsoup.parser.tag.TagSet; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/TagSet.java:[147,59] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[57,36] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[68,36] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[104,40] normalName(java.lang.String) is not public in org.jsoup.parser.ParseSettings; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[347,32] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/tag/Tag.java:[348,32] org.jsoup.parser.TokeniserState is not public in org.jsoup.parser; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[138,68] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[140,68] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[482,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[490,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[494,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[498,70] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[626,51] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java:[1812,122] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[118,51] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[318,16] setSeenSelfClose() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[346,21] setSeenSelfClose() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[381,44] namespace is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[701,20] hasParserOption(int) is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[716,37] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[723,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[727,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[727,67] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[731,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[731,67] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[735,44] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[745,43] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[811,55] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[827,39] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[827,75] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[830,21] hasParserOption(int) is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java:[843,41] org.jsoup.parser.tag.HtmlTagOptions is not public in org.jsoup.parser.tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[67,49] textState() is not public in org.jsoup.parser.tag.Tag; cannot be accessed from outside package
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[106,20] cannot find symbol
[ERROR]   symbol:   class TagSet
[ERROR]   location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[164,9] cannot find symbol
[ERROR]   symbol:   class Tag
[ERROR]   location: class org.jsoup.parser.XmlTreeBuilder
[ERROR] /data/henrique/langchain_prototype/codex/data/repositories/jsoup/src/main/java/org/jsoup/parser/XmlTreeBuilder.java:[234,44] cannot find symbol
[ERROR]   symbol:   variable Tag
[ERROR]   location: class org.jsoup.parser.XmlTreeBuilder
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
