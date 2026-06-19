# GOAL

Remove the design smell: Insufficient Modularization from the TARGET CLASS.

Insufficient Modularization is characterized by one or more of the following conditions:

1. Too many public methods.
2. Too many total methods.
3. Excessive lines of code.

For this MVP, the remediation priority is hierarchical:

1. First, reduce the number of public methods in the target class.
2. If reducing public methods is not safe or not applicable, reduce the total number of methods in the target class.
3. If reducing methods is not safe or not sufficient, reduce LOC by extracting cohesive implementation details.

The main objective is to extract cohesive responsibilities from the target class into new classes. Each extracted class must represent a coherent responsibility, not merely a generic utility holder.

Behavior must be preserved. However, preserving the original target-class API is not mandatory when changing that API is necessary to remove Insufficient Modularization and the required updates can be handled within the files in the block and mechanically added related tests.

# ALLOWED STRATEGY

- Each block must contain exactly one EXTRACT_CLASS operation.
- Consequently, each block may create at most one new production class.
- The EXTRACT_CLASS operation must represent a cohesive cluster of methods and fields that belong to the same responsibility.
- Prefer extracting members in this priority order:
  1. public methods that form a cohesive responsibility;
  2. package-private or protected methods and fields that support that responsibility;
  3. private methods and fields that support that responsibility;
  4. fields, nested classes, or helper-only logic, only when they meaningfully reduce LOC or support the extracted responsibility.
- The broader goal is to reduce the number of methods remaining in the target class, especially public methods.
- Do not prefer private-helper extraction when a cohesive cluster of public methods can be moved safely.
- Move public methods when they clearly belong to the extracted responsibility and moving them reduces the target class public method count.
- Move methods and fields to the extracted class when doing so reduces the target class size, method count, complexity, or responsibility concentration.
- Do not keep delegating wrappers solely to preserve the original target-class API if the corresponding call sites can be updated within the block or by mechanically added related tests.
- Create compatibility wrappers only when they are strictly necessary to preserve behavior for production code that cannot be updated within the block.
- Avoid generic utility classes. Prefer responsibility-oriented names such as SafelistProtocols, SafelistEnforcedAttributes, AttributeValidation, TokeniserRawDataReader, etc.
- New classes must usually be placed in the same package as the target class unless the input clearly justifies otherwise.
- Plan only production-code refactorings.
- Do not include src/test files, UPDATE_TESTS operations, or test-specific changes. Related tests will be added by a deterministic post-processing step.
- The deterministic post-processing step will append UPDATE_TESTS as the final operation of each block when related tests are found.

# REFACTORING PRIORITY

When choosing what to extract, follow this decision process:

1. If the target class has too many public methods, identify a cohesive responsibility implemented by public methods and extract that responsibility first.
2. If no safe public-method cluster exists, identify a cohesive cluster of methods that reduces the total method count.
3. If method extraction is not safe or insufficient, extract cohesive fields, nested classes, or helper logic to reduce LOC.
4. Avoid extracting only fields or nested value objects when the target class still has a cohesive group of movable public methods.
5. A successful plan should aim to reduce at least one of the following, in this order:
   - public method count;
   - total method count;
   - LOC.

# ALLOWED OPS

EXTRACT_CLASS, MOVE_METHOD, MOVE_FIELD, ADD_OR_UPDATE_IMPORTS, UPDATE_CALL_SITES

Use NO_OP only when no safe refactoring plan can be produced at all.
Do not include NO_OP inside a block that already contains real refactoring operations.
Do not use NO_OP as a placeholder, filler, or block-size adjustment.

# CONSTRAINTS

1. Reference only packages, classes, methods, and fields present in the input.
2. Each block must be small and independently compilable.
3. Each block must contain exactly one EXTRACT_CLASS operation.
4. Each EXTRACT_CLASS operation must output exactly one new production class.
5. Do not create multiple new classes in the same block.
6. Do not split a single cohesive responsibility across multiple blocks.
7. Do not create generic utility classes unless the extracted responsibility is genuinely utility-like.
8. Do not duplicate moved logic between the target class and the extracted class.
9. Remove moved methods and fields from the target class unless a compatibility wrapper is strictly required.
10. Keep blocks small: one cohesive extraction per block.
11. The files list must include only production files. Do not include src/test files.
12. The ops list must not include UPDATE_TESTS; it will be added mechanically after planning.
13. Each block must explain how the extraction is expected to reduce the target class public method count, total method count, or LOC.

# INPUT

{
  "smell": "Insufficient Modularization",
  "target_type": "class",
  "target_name": "org.jsoup.parser.Parser",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/new/data/runs/20260618_152016_72670bf0/planner/designite",
    "smells_csv": "DesignSmells.csv",
    "target_has_smell": true
  },
  "target_file": "src/main/java/org/jsoup/parser/Parser.java",
  "target_source_root": "src/main/java",
  "target_code": "package org.jsoup.parser;\n\nimport org.jsoup.helper.Validate;\nimport org.jsoup.nodes.Document;\nimport org.jsoup.nodes.Element;\nimport org.jsoup.nodes.Node;\nimport org.jspecify.annotations.Nullable;\n\nimport java.io.Reader;\nimport java.io.StringReader;\nimport java.util.List;\nimport java.util.concurrent.locks.ReentrantLock;\n\n/**\n Parses HTML or XML into a {@link org.jsoup.nodes.Document}. Generally, it is simpler to use one of the parse methods in\n {@link org.jsoup.Jsoup}.\n <p>Note that a given Parser instance object is threadsafe, but not concurrent. (Concurrent parse calls will\n synchronize.) To reuse a Parser configuration in a multithreaded environment, use {@link #newInstance()} to make\n copies.</p>\n */\npublic class Parser implements Cloneable {\n    public static final String NamespaceHtml = \"http://www.w3.org/1999/xhtml\";\n    public static final String NamespaceXml = \"http://www.w3.org/XML/1998/namespace\";\n    public static final String NamespaceMathml = \"http://www.w3.org/1998/Math/MathML\";\n    public static final String NamespaceSvg = \"http://www.w3.org/2000/svg\";\n\n    private final TreeBuilder treeBuilder;\n    private final ParserConfig config;\n    private final ReentrantLock lock = new ReentrantLock();\n\n    /**\n     * Create a new Parser, using the specified TreeBuilder\n     * @param treeBuilder TreeBuilder to use to parse input into Documents.\n     */\n    public Parser(TreeBuilder treeBuilder) {\n        this.treeBuilder = treeBuilder;\n        this.config = new ParserConfig();\n        // initialize defaults from the treeBuilder\n        this.config.settings(treeBuilder.defaultSettings())\n                   .setTrackErrors(0)\n                   .setMaxDepth(treeBuilder.defaultMaxDepth());\n        // supply default tag set supplier so config.tagSet() can fall back to the builder's default\n        this.config.setDefaultTagSetSupplier(() -> treeBuilder.defaultTagSet());\n    }\n\n    /**\n     Creates a new Parser as a deep copy of this; including initializing a new TreeBuilder. Allows independent (multi-threaded) use.\n     @return a copied parser\n     */\n    public Parser newInstance() {\n        return new Parser(this);\n    }\n\n    @SuppressWarnings(\"MethodDoesntCallSuperMethod\") // because we use the copy constructor instead\n    @Override\n    public Parser clone() {\n        return new Parser(this);\n    }\n\n    private Parser(Parser copy) {\n        treeBuilder = copy.treeBuilder.newInstance(); // because extended\n        config = new ParserConfig(copy.config);\n        // ensure default supplier uses our (new) treeBuilder\n        this.config.setDefaultTagSetSupplier(() -> treeBuilder.defaultTagSet());\n    }\n\n    /**\n     Parse the contents of a String.\n\n     @param html HTML to parse\n     @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     @return parsed Document\n     */\n    public Document parseInput(String html, String baseUri) {\n        return parseInput(new StringReader(html), baseUri);\n    }\n\n    /**\n     Parse the contents of Reader.\n\n     @param inputHtml HTML to parse\n     @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     @return parsed Document\n     @throws java.io.UncheckedIOException if an I/O error occurs in the Reader\n     */\n    public Document parseInput(Reader inputHtml, String baseUri) {\n        try {\n            lock.lock(); // using a lock vs synchronized to support loom threads\n            return treeBuilder.parse(inputHtml, baseUri, this);\n        } finally {\n            lock.unlock();\n        }\n    }\n\n    /**\n     Parse a fragment of HTML into a list of nodes. The context element, if supplied, supplies parsing context.\n\n     @param fragment the fragment of HTML to parse\n     @param context (optional) the element that this HTML fragment is being parsed for (i.e. for inner HTML).\n     @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     @return list of nodes parsed from the input HTML.\n     */\n    public List<Node> parseFragmentInput(String fragment, @Nullable Element context, String baseUri) {\n        return parseFragmentInput(new StringReader(fragment), context, baseUri);\n    }\n\n    /**\n     Parse a fragment of HTML into a list of nodes. The context element, if supplied, supplies parsing context.\n\n     @param fragment the fragment of HTML to parse\n     @param context (optional) the element that this HTML fragment is being parsed for (i.e. for inner HTML).\n     @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     @return list of nodes parsed from the input HTML.\n     @throws java.io.UncheckedIOException if an I/O error occurs in the Reader\n     */\n    public List<Node> parseFragmentInput(Reader fragment, @Nullable Element context, String baseUri) {\n        try {\n            lock.lock();\n            return treeBuilder.parseFragment(fragment, context, baseUri, this);\n        } finally {\n            lock.unlock();\n        }\n    }\n\n    // gets & sets\n    /**\n     * Get the TreeBuilder currently in use.\n     * @return current TreeBuilder.\n     */\n    public TreeBuilder getTreeBuilder() {\n        return treeBuilder;\n    }\n\n    /**\n     * Access the Parser configuration holder. This consolidated object contains parse settings, error tracking,\n     * position tracking, tag set and max depth. Use this to configure parser behaviour.\n     *\n     * Example: Parser html = Parser.htmlParser(); html.config().setTrackErrors(100);\n     */\n    public ParserConfig config() { return config; }\n\n    /**\n     * Convenience factory to obtain a configuration facade for this Parser. The returned ParserConfigSupport\n     * allows chained configuration calls and exposes the legacy convenience API that previously lived on Parser.\n     */\n    public ParserConfigSupport configSupport() { return new ParserConfigSupport(this.config); }\n\n    /**\n     * Check if parse error tracking is enabled.\n     * @return current track error state.\n     */\n    @Deprecated\n    public boolean isTrackErrors() {\n        return config.isTrackErrors();\n    }\n\n    /**\n     * Enable or disable parse error tracking for the next parse.\n     * @param maxErrors the maximum number of errors to track. Set to 0 to disable.\n     * @return this, for chaining\n     */\n    @Deprecated\n    public Parser setTrackErrors(int maxErrors) {\n        config.setTrackErrors(maxErrors);\n        return this;\n    }\n\n    /**\n     * Retrieve the parse errors, if any, from the last parse.\n     * @return list of parse errors, up to the size of the maximum errors tracked.\n     * @see #setTrackErrors(int)\n     */\n    @Deprecated\n    public ParseErrorList getErrors() {\n        return config.getErrors();\n    }\n\n    /**\n     Test if position tracking is enabled. If it is, Nodes will have a Position to track where in the original input\n     source they were created from. By default, tracking is not enabled.\n     * @return current track position setting\n     */\n    @Deprecated\n    public boolean isTrackPosition() {\n        return config.isTrackPosition();\n    }\n\n    /**\n     Enable or disable source position tracking. If enabled, Nodes will have a Position to track where in the original\n     input source they were created from.\n     @param trackPosition position tracking setting; {@code true} to enable\n     @return this Parser, for chaining\n     */\n    @Deprecated\n    public Parser setTrackPosition(boolean trackPosition) {\n        config.setTrackPosition(trackPosition);\n        return this;\n    }\n\n    /**\n     Update the ParseSettings of this Parser, to control the case sensitivity of tags and attributes.\n     * @param settings the new settings\n     * @return this Parser\n     */\n    @Deprecated\n    public Parser settings(ParseSettings settings) {\n        config.settings(settings);\n        return this;\n    }\n\n    /**\n     Gets the current ParseSettings for this Parser\n     * @return current ParseSettings\n     */\n    @Deprecated\n    public ParseSettings settings() {\n        return config.settings();\n    }\n\n    /**\n     Set the parser's maximum stack depth (maximum number of open elements). When reached, new open elements will be\n     removed to prevent excessive nesting. Defaults to 512 for the HTML parser, and unlimited for the XML\n     parser.\n\n     @param maxDepth maximum parser depth; must be >= 1\n     @return this Parser, for chaining\n     */\n    @Deprecated\n    public Parser setMaxDepth(int maxDepth) {\n        config.setMaxDepth(maxDepth);\n        return this;\n    }\n\n    /**\n     * Get the maximum parser depth (maximum number of open elements).\n     * @return the current max parser depth\n     */\n    @Deprecated\n    public int getMaxDepth() {\n        return config.getMaxDepth();\n    }\n\n    /**\n     Set a custom TagSet to use for this Parser. This allows you to define your own tags, and control how they are\n     parsed. For example, you can set a tag to preserve whitespace, or to be treated as a block tag.\n     <p>You can start with the {@link TagSet#Html()} defaults and customize, or a new empty TagSet.</p>\n\n     @param tagSet the TagSet to use. This gets copied, so that changes that the parse makes (tags found in the document will be added) do not clobber the original TagSet.\n     @return this Parser\n     @since 1.20.1\n     */\n    @Deprecated\n    public Parser tagSet(TagSet tagSet) {\n        config.tagSet(tagSet);\n        return this;\n    }\n\n    /**\n     Get the current TagSet for this Parser, which will be either this parser's default, or one that you have set.\n     @return the current TagSet. After the parse, this will contain any new tags that were found in the document.\n     @since 1.20.1\n     */\n    @Deprecated\n    public TagSet tagSet() {\n        return config.tagSet();\n    }\n\n    public String defaultNamespace() {\n        return getTreeBuilder().defaultNamespace();\n    }\n\n    // static parse functions below\n    /**\n     * Parse HTML into a Document.\n     *\n     * @param html HTML to parse\n     * @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     *\n     * @return parsed Document\n     */\n    public static Document parse(String html, String baseUri) {\n        return Parsers.parse(html, baseUri);\n    }\n\n    /**\n     * Parse a fragment of HTML into a list of nodes. The context element, if supplied, supplies parsing context.\n     *\n     * @param fragmentHtml the fragment of HTML to parse\n     * @param context (optional) the element that this HTML fragment is being parsed for (i.e. for inner HTML). This\n     * provides stack context (for implicit element creation).\n     * @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     *\n     * @return list of nodes parsed from the input HTML. Note that the context element, if supplied, is not modified.\n     */\n    public static List<Node> parseFragment(String fragmentHtml, Element context, String baseUri) {\n        return Parsers.parseFragment(fragmentHtml, context, baseUri);\n    }\n\n    /**\n     * Parse a fragment of HTML into a list of nodes. The context element, if supplied, supplies parsing context.\n     *\n     * @param fragmentHtml the fragment of HTML to parse\n     * @param context (optional) the element that this HTML fragment is being parsed for (i.e. for inner HTML). This\n     * provides stack context (for implicit element creation).\n     * @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     * @param errorList list to add errors to\n     *\n     * @return list of nodes parsed from the input HTML. Note that the context element, if supplied, is not modified.\n     */\n    public static List<Node> parseFragment(String fragmentHtml, Element context, String baseUri, ParseErrorList errorList) {\n        return Parsers.parseFragment(fragmentHtml, context, baseUri, errorList);\n    }\n\n    /**\n     * Parse a fragment of XML into a list of nodes.\n     *\n     * @param fragmentXml the fragment of XML to parse\n     * @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     * @return list of nodes parsed from the input XML.\n     */\n    public static List<Node> parseXmlFragment(String fragmentXml, String baseUri) {\n        return Parsers.parseXmlFragment(fragmentXml, baseUri);\n    }\n\n    /**\n     * Parse a fragment of HTML into the {@code body} of a Document.\n     *\n     * @param bodyHtml fragment of HTML\n     * @param baseUri base URI of document (i.e. original fetch location), for resolving relative URLs.\n     *\n     * @return Document, with empty head, and HTML parsed into body\n     */\n    public static Document parseBodyFragment(String bodyHtml, String baseUri) {\n        return Parsers.parseBodyFragment(bodyHtml, baseUri);\n    }\n\n    /**\n     Utility method to unescape HTML entities from a string.\n     <p>To track errors while unescaping, use\n     {@link #unescape(String, boolean)} with a Parser instance that has error tracking enabled.</p>\n\n     @param string HTML escaped string\n     @param inAttribute if the string is to be escaped in strict mode (as attributes are)\n     @return an unescaped string\n     @see #unescape(String, boolean)\n     */\n    public static String unescapeEntities(String string, boolean inAttribute) {\n        return Parsers.unescapeEntities(string, inAttribute);\n    }\n\n    /**\n     Utility method to unescape HTML entities from a string, using this {@code Parser}'s configuration (for example, to\n     collect errors while unescaping).\n\n     @param string HTML escaped string\n     @param inAttribute if the string is to be escaped in strict mode (as attributes are)\n     @return an unescaped string\n     @see #setTrackErrors(int)\n     @see #unescapeEntities(String, boolean)\n     */\n    public String unescape(String string, boolean inAttribute) {\n        Validate.notNull(string);\n        if (string.indexOf('&') < 0) return string; // nothing to unescape\n        this.treeBuilder.initialiseParse(new StringReader(string), \"\", this);\n        Tokeniser tokeniser = new Tokeniser(this.treeBuilder);\n        return tokeniser.unescapeEntities(inAttribute);\n    }\n\n    // builders\n\n    /**\n     * Create a new HTML parser. This parser treats input as HTML5, and enforces the creation of a normalised document,\n     * based on a knowledge of the semantics of the incoming tags.\n     * @return a new HTML parser.\n     */\n    public static Parser htmlParser() {\n        return Parsers.htmlParser();\n    }\n\n    /**\n     * Create a new XML parser. This parser assumes no knowledge of the incoming tags and does not treat it as HTML,\n     * rather creates a simple tree directly from the input.\n     * @return a new simple XML parser.\n     */\n    public static Parser xmlParser() {\n        return Parsers.xmlParser();\n    }\n}\n"
}

# OUTPUT SCHEMA

{
  "smell_type": "Insufficient Modularization",
  "target_level": "class",
  "target": "<class FQN from input>",
  "blocks": [
    {
      "id": 1,
      "goal": "...",
      "expected_impact": {
        "primary_metric": "public_methods|total_methods|loc",
        "public_methods_moved": 0,
        "total_methods_moved": 0,
        "loc_reduction_expected": "low|medium|high",
        "rationale": "..."
      },
      "files": ["..."],
      "ops": [
        {
          "op": "EXTRACT_CLASS",
          "inputs": ["..."],
          "outputs": ["..."],
          "details": "...",
          "risk": "low|medium|high"
        },
        {
          "op": "<allowed op>",
          "inputs": ["..."],
          "outputs": ["..."],
          "details": "...",
          "risk": "low|medium|high"
        }
      ]
    }
  ]
}