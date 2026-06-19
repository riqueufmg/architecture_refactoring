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
  "target_name": "org.jsoup.internal.StringUtil",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/new/data/runs/20260618_141125_9db7c92e/planner/designite",
    "smells_csv": "DesignSmells.csv",
    "target_has_smell": true
  },
  "target_file": "src/main/java/org/jsoup/internal/StringUtil.java",
  "target_source_root": "src/main/java",
  "target_code": "package org.jsoup.internal;\n\nimport org.jsoup.helper.Validate;\nimport org.jspecify.annotations.Nullable;\n\nimport java.net.MalformedURLException;\nimport java.net.URL;\nimport java.util.Arrays;\nimport java.util.Collection;\nimport java.util.Iterator;\nimport java.util.regex.Pattern;\nimport java.util.stream.Collector;\nimport java.util.stream.Collectors;\n\n/**\n A minimal String utility class. Designed for <b>internal</b> jsoup use only - the API and outcome may change without\n notice.\n */\npublic final class StringUtil {\n    // memoised padding up to 21 (blocks 0 to 20 spaces)\n    static final String[] padding = {\"\", \" \", \"  \", \"   \", \"    \", \"     \", \"      \", \"       \", \"        \",\n        \"         \", \"          \", \"           \", \"            \", \"             \", \"              \", \"               \",\n        \"                \", \"                 \", \"                  \", \"                   \", \"                    \"};\n\n    /**\n     * Join a collection of strings by a separator\n     * @param strings collection of string objects\n     * @param sep string to place between strings\n     * @return joined string\n     */\n    public static String join(Collection<?> strings, String sep) {\n        return join(strings.iterator(), sep);\n    }\n\n    /**\n     * Join a collection of strings by a separator\n     * @param strings iterator of string objects\n     * @param sep string to place between strings\n     * @return joined string\n     */\n    public static String join(Iterator<?> strings, String sep) {\n        if (!strings.hasNext())\n            return \"\";\n\n        String start = strings.next().toString();\n        if (!strings.hasNext()) // only one, avoid builder\n            return start;\n\n        StringJoiner j = new StringJoiner(sep);\n        j.add(start);\n        while (strings.hasNext()) {\n            j.add(strings.next());\n        }\n        return j.complete();\n    }\n\n    /**\n     * Join an array of strings by a separator\n     * @param strings collection of string objects\n     * @param sep string to place between strings\n     * @return joined string\n     */\n    public static String join(String[] strings, String sep) {\n        return join(Arrays.asList(strings), sep);\n    }\n\n    /**\n     A StringJoiner allows incremental / filtered joining of a set of stringable objects.\n     @since 1.14.1\n     */\n    public static class StringJoiner {\n        @Nullable StringBuilder sb = borrowBuilder(); // sets null on builder release so can't accidentally be reused\n        final String separator;\n        boolean first = true;\n\n        /**\n         Create a new joiner, that uses the specified separator. MUST call {@link #complete()} or will leak a thread\n         local string builder.\n\n         @param separator the token to insert between strings\n         */\n        public StringJoiner(String separator) {\n            this.separator = separator;\n        }\n\n        /**\n         Add another item to the joiner, will be separated\n         */\n        public StringJoiner add(Object stringy) {\n            Validate.notNull(sb); // don't reuse\n            if (!first)\n                sb.append(separator);\n            sb.append(stringy);\n            first = false;\n            return this;\n        }\n\n        /**\n         Append content to the current item; not separated\n         */\n        public StringJoiner append(Object stringy) {\n            Validate.notNull(sb); // don't reuse\n            sb.append(stringy);\n            return this;\n        }\n\n        /**\n         Return the joined string, and release the builder back to the pool. This joiner cannot be reused.\n         */\n        public String complete() {\n            String string = releaseBuilder(sb);\n            sb = null;\n            return string;\n        }\n    }\n\n    /**\n     * Returns space padding (up to the default max of 30). Use {@link #padding(int, int)} to specify a different limit.\n     * @param width amount of padding desired\n     * @return string of spaces * width\n     * @see #padding(int, int) \n      */\n    public static String padding(int width) {\n        return padding(width, 30);\n    }\n\n    /**\n     * Returns space padding, up to a max of maxPaddingWidth.\n     * @param width amount of padding desired\n     * @param maxPaddingWidth maximum padding to apply. Set to {@code -1} for unlimited.\n     * @return string of spaces * width\n     */\n    public static String padding(int width, int maxPaddingWidth) {\n        Validate.isTrue(width >= 0, \"width must be >= 0\");\n        Validate.isTrue(maxPaddingWidth >= -1);\n        if (maxPaddingWidth != -1)\n            width = Math.min(width, maxPaddingWidth);\n        if (width < padding.length)\n            return padding[width];        \n        char[] out = new char[width];\n        for (int i = 0; i < width; i++)\n            out[i] = ' ';\n        return String.valueOf(out);\n    }\n\n    /**\n     * Tests if a string is blank: null, empty, or only whitespace (\" \", \\r\\n, \\t, etc)\n     * @param string string to test\n     * @return if string is blank\n     */\n    public static boolean isBlank(@Nullable String string) {\n        if (string == null || string.isEmpty())\n            return true;\n\n        int l = string.length();\n        for (int i = 0; i < l; i++) {\n            if (!StringUtil.isWhitespace(string.codePointAt(i)))\n                return false;\n        }\n        return true;\n    }\n\n    /**\n     Tests if a string starts with a newline character\n     @param string string to test\n     @return if its first character is a newline\n     */\n    public static boolean startsWithNewline(final String string) {\n        if (string == null || string.length() == 0)\n            return false;\n        return string.charAt(0) == '\\n';\n    }\n\n    /**\n     * Tests if a string is numeric, i.e. contains only ASCII digit characters\n     * @param string string to test\n     * @return true if only digit chars, false if empty or null or contains non-digit chars\n     */\n    public static boolean isNumeric(String string) {\n        if (string == null || string.length() == 0)\n            return false;\n\n        int l = string.length();\n        for (int i = 0; i < l; i++) {\n            if (!isDigit(string.charAt(i)))\n                return false;\n        }\n        return true;\n    }\n\n    /**\n     * Tests if a code point is \"whitespace\" as defined in the HTML spec. Used for output HTML.\n     * @param c code point to test\n     * @return true if code point is whitespace, false otherwise\n     * @see #isActuallyWhitespace(int)\n     */\n    public static boolean isWhitespace(int c){\n        return c == ' ' || c == '\\t' || c == '\\n' || c == '\\f' || c == '\\r';\n    }\n\n    /**\n     * Tests if a code point is \"whitespace\" as defined by what it looks like. Used for Element.text etc.\n     * @param c code point to test\n     * @return true if code point is whitespace, false otherwise\n     */\n    public static boolean isActuallyWhitespace(int c){\n        return c == ' ' || c == '\\t' || c == '\\n' || c == '\\f' || c == '\\r' || c == 160;\n        // 160 is &nbsp; (non-breaking space). Not in the spec but expected.\n    }\n\n    public static boolean isInvisibleChar(int c) {\n        return c == 8203 || c == 173; // zero width sp, soft hyphen\n        // previously also included zw non join, zw join - but removing those breaks semantic meaning of text\n    }\n\n    /**\n     * Normalise the whitespace within this string; multiple spaces collapse to a single, and all whitespace characters\n     * (e.g. newline, tab) convert to a simple space.\n     * @param string content to normalise\n     * @return normalised string\n     */\n    public static String normaliseWhitespace(String string) {\n        StringBuilder sb = StringUtil.borrowBuilder();\n        appendNormalisedWhitespace(sb, string, false);\n        return StringUtil.releaseBuilder(sb);\n    }\n\n    /**\n     * After normalizing the whitespace within a string, appends it to a string builder.\n     * @param accum builder to append to\n     * @param string string to normalize whitespace within\n     * @param stripLeading set to true if you wish to remove any leading whitespace\n     */\n    public static void appendNormalisedWhitespace(StringBuilder accum, String string, boolean stripLeading) {\n        boolean lastWasWhite = false;\n        boolean reachedNonWhite = false;\n\n        int len = string.length();\n        int c;\n        for (int i = 0; i < len; i+= Character.charCount(c)) {\n            c = string.codePointAt(i);\n            if (isActuallyWhitespace(c)) {\n                if ((stripLeading && !reachedNonWhite) || lastWasWhite)\n                    continue;\n                accum.append(' ');\n                lastWasWhite = true;\n            }\n            else if (!isInvisibleChar(c)) {\n                accum.appendCodePoint(c);\n                lastWasWhite = false;\n                reachedNonWhite = true;\n            }\n        }\n    }\n\n    public static boolean in(final String needle, final String... haystack) {\n        final int len = haystack.length;\n        for (int i = 0; i < len; i++) {\n            if (haystack[i].equals(needle))\n               return true;\n        }\n        return false;\n    }\n\n    public static boolean inSorted(String needle, String[] haystack) {\n        if (haystack.length <= 8) { // Parser benchmarking showed that it's faster to scan than binary search for these lengths\n            for (int i = 0; i < haystack.length; i++) {\n                if (needle.equals(haystack[i]))\n                    return true;\n            }\n            return false;\n        }\n        return Arrays.binarySearch(haystack, needle) >= 0;\n    }\n\n    /**\n     Tests that a String contains only ASCII characters.\n     @param string scanned string\n     @return true if all characters are in range 0 - 127\n     */\n    public static boolean isAscii(String string) {\n        Validate.notNull(string);\n        for (int i = 0; i < string.length(); i++) {\n            int c = string.charAt(i);\n            if (c > 127) { // ascii range\n                return false;\n            }\n        }\n        return true;\n    }\n\n    private static final Pattern extraDotSegmentsPattern = Pattern.compile(\"^/(?>(?>\\\\.\\\\.?/)+)\");\n    /**\n     * Create a new absolute URL, from a provided existing absolute URL and a relative URL component.\n     * @param base the existing absolute base URL\n     * @param relUrl the relative URL to resolve. (If it's already absolute, it will be returned)\n     * @return the resolved absolute URL\n     * @throws MalformedURLException if an error occurred generating the URL\n     */\n    public static URL resolve(URL base, String relUrl) throws MalformedURLException {\n        relUrl = stripControlChars(relUrl);\n        // workaround: java resolves '//path/file + ?foo' to '//path/?foo', not '//path/file?foo' as desired\n        if (relUrl.startsWith(\"?\"))\n            relUrl = base.getPath() + relUrl;\n        // workaround: //example.com + ./foo = //example.com/./foo, not //example.com/foo\n        URL url = new URL(base, relUrl);\n        String fixedFile = extraDotSegmentsPattern.matcher(url.getFile()).replaceFirst(\"/\");\n        if (url.getRef() != null) {\n            fixedFile = fixedFile + \"#\" + url.getRef();\n        }\n        return new URL(url.getProtocol(), url.getHost(), url.getPort(), fixedFile);\n    }\n\n    /**\n     * Create a new absolute URL, from a provided existing absolute URL and a relative URL component.\n     * @param baseUrl the existing absolute base URL\n     * @param relUrl the relative URL to resolve. (If it's already absolute, it will be returned)\n     * @return an absolute URL if one was able to be generated, or the empty string if not\n     */\n    public static String resolve(String baseUrl, String relUrl) {\n        // workaround: java will allow control chars in a path URL and may treat as relative, but Chrome / Firefox will strip and may see as a scheme. Normalize to browser's view.\n        baseUrl = stripControlChars(baseUrl); relUrl = stripControlChars(relUrl);\n        try {\n            URL base;\n            try {\n                base = new URL(baseUrl);\n            } catch (MalformedURLException e) {\n                // the base is unsuitable, but the attribute/rel may be abs on its own, so try that\n                URL abs = new URL(relUrl);\n                return abs.toExternalForm();\n            }\n            return resolve(base, relUrl).toExternalForm();\n        } catch (MalformedURLException e) {\n            // it may still be valid, just that Java doesn't have a registered stream handler for it, e.g. tel\n            // we test here vs at start to normalize supported URLs (e.g. HTTP -> http)\n            return validUriScheme.matcher(relUrl).find() ? relUrl : \"\";\n        }\n    }\n    private static final Pattern validUriScheme = Pattern.compile(\"^[a-zA-Z][a-zA-Z0-9+-.]*:\");\n\n    private static final Pattern controlChars = Pattern.compile(\"[\\\\x00-\\\\x1f]*\"); // matches ascii 0 - 31, to strip from url\n    private static String stripControlChars(final String input) {\n        return controlChars.matcher(input).replaceAll(\"\");\n    }\n\n    private static final int InitBuilderSize = 1024;\n    private static final int MaxBuilderSize = 8 * 1024;\n    private static final SoftPool<StringBuilder> BuilderPool = new SoftPool<>(\n        () -> new StringBuilder(InitBuilderSize));\n\n    /**\n     * Maintains cached StringBuilders in a flyweight pattern, to minimize new StringBuilder GCs. The StringBuilder is\n     * prevented from growing too large.\n     * <p>\n     * Care must be taken to release the builder once its work has been completed, with {@link #releaseBuilder}\n     * @return an empty StringBuilder\n     */\n    public static StringBuilder borrowBuilder() {\n        return BuilderPool.borrow();\n    }\n\n    /**\n     * Release a borrowed builder. Care must be taken not to use the builder after it has been returned, as its\n     * contents may be changed by this method, or by a concurrent thread.\n     * @param sb the StringBuilder to release.\n     * @return the string value of the released String Builder (as an incentive to release it!).\n     */\n    public static String releaseBuilder(StringBuilder sb) {\n        Validate.notNull(sb);\n        String string = sb.toString();\n        releaseBuilderVoid(sb);\n        return string;\n    }\n\n    /**\n     Releases a borrowed builder, but does not call .toString() on it. Useful in case you already have that string.\n     @param sb the StringBuilder to release.\n     @see #releaseBuilder(StringBuilder)\n     */\n    public static void releaseBuilderVoid(StringBuilder sb) {\n        // if it hasn't grown too big, reset it and return it to the pool:\n        if (sb.length() <= MaxBuilderSize) {\n            sb.delete(0, sb.length()); // make sure it's emptied on release\n            BuilderPool.release(sb);\n        }\n    }\n\n    /**\n     * Return a {@link Collector} similar to the one returned by {@link Collectors#joining(CharSequence)},\n     * but backed by jsoup's {@link StringJoiner}, which allows for more efficient garbage collection.\n     *\n     * @param delimiter The delimiter for separating the strings.\n     * @return A {@code Collector} which concatenates CharSequence elements, separated by the specified delimiter\n     */\n    public static Collector<CharSequence, ?, String> joining(String delimiter) {\n        return Collector.of(() -> new StringJoiner(delimiter),\n            StringJoiner::add,\n            (j1, j2) -> {\n                j1.append(j2.complete());\n                return j1;\n            },\n            StringJoiner::complete);\n    }\n\n    public static boolean isAsciiLetter(char c) {\n        return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z';\n    }\n\n    public static boolean isDigit(char c) {\n        return c >= '0' && c <= '9';\n    }\n\n    public static boolean isHexDigit(char c) {\n        return isDigit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F';\n    }\n}\n"
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