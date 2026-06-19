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
  "target_name": "org.apache.commons.lang3.RandomUtils",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/new/data/runs/20260618_124233_d7c81793/planner/designite",
    "smells_csv": "DesignSmells.csv",
    "target_has_smell": true
  },
  "target_file": "src/main/java/org/apache/commons/lang3/RandomUtils.java",
  "target_source_root": "src/main/java",
  "target_code": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *     https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\npackage org.apache.commons.lang3;\n\nimport java.security.NoSuchAlgorithmException;\nimport java.security.SecureRandom;\nimport java.security.Security;\nimport java.util.Random;\nimport java.util.concurrent.ThreadLocalRandom;\nimport java.util.function.Supplier;\n\nimport org.apache.commons.lang3.exception.UncheckedException;\n\n/**\n * Supplements the standard {@link Random} class.\n * <p>\n * Use {@link #secure()} to get the singleton instance based on {@link SecureRandom#SecureRandom()} which uses a secure random number generator implementing the\n * default random number algorithm.\n * </p>\n * <p>\n * Use {@link #secureStrong()} to get the singleton instance based on {@link SecureRandom#getInstanceStrong()} which uses an instance that was selected by using\n * the algorithms/providers specified in the {@code securerandom.strongAlgorithms} {@link Security} property.\n * </p>\n * <p>\n * Use {@link #insecure()} to get the singleton instance based on {@link ThreadLocalRandom#current()} <strong>which is not cryptographically secure</strong>. In addition,\n * instances do not use a cryptographically random seed unless the {@linkplain System#getProperty system property} {@code java.util.secureRandomSeed} is set to\n * {@code true}.\n * </p>\n * <p>\n * Starting in version 3.17.0, the method {@link #secure()} uses {@link SecureRandom#SecureRandom()} instead of {@link SecureRandom#getInstanceStrong()}, and\n * adds {@link #secureStrong()}.\n * </p>\n * <p>\n * Starting in version 3.16.0, this class uses {@link #secure()} for static methods and adds {@link #insecure()}.\n * </p>\n * <p>\n * Starting in version 3.15.0, this class uses {@link SecureRandom#getInstanceStrong()} for static methods.\n * </p>\n * <p>\n * Before version 3.15.0, this class used {@link ThreadLocalRandom#current()} for static methods, which is not cryptographically secure.\n * </p>\n * <p>\n * Please note that the Apache Commons project provides a component dedicated to pseudo-random number generation, namely\n * <a href=\"https://commons.apache.org/proper/commons-rng/\">Commons RNG</a>, that may be a better choice for applications with more stringent requirements\n * (performance and/or correctness).\n * </p>\n *\n * @see #secure()\n * @see #secureStrong()\n * @see #insecure()\n * @see SecureRandom#SecureRandom()\n * @see SecureRandom#getInstanceStrong()\n * @see ThreadLocalRandom#current()\n * @see RandomStringUtils\n * @since 3.3\n */\npublic class RandomUtils {\n\n    private static final RandomUtils INSECURE = new RandomUtils(ThreadLocalRandom::current);\n\n    private static final RandomUtils SECURE = new RandomUtils(SecureRandom::new);\n\n    private static final Supplier<Random> SECURE_STRONG_SUPPLIER = () -> RandomUtils.SECURE_RANDOM_STRONG.get();\n\n    private static final RandomUtils SECURE_STRONG = new RandomUtils(SECURE_STRONG_SUPPLIER);\n\n    private static final ThreadLocal<SecureRandom> SECURE_RANDOM_STRONG = ThreadLocal.withInitial(() -> {\n        try {\n            return SecureRandom.getInstanceStrong();\n        } catch (final NoSuchAlgorithmException e) {\n            throw new UncheckedException(e);\n        }\n    });\n\n    /**\n     * Gets the singleton instance based on {@link ThreadLocalRandom#current()}; <b>which is not cryptographically\n     * secure</b>; for more secure processing use {@link #secure()} or {@link #secureStrong()}.\n     * <p>\n     * The method {@link ThreadLocalRandom#current()} is called on-demand.\n     * </p>\n     *\n     * @return the singleton instance based on {@link ThreadLocalRandom#current()}.\n     * @see ThreadLocalRandom#current()\n     * @see #secure()\n     * @see #secureStrong()\n     * @since 3.17.0\n     */\n    public static RandomUtils insecure() {\n        return INSECURE;\n    }\n\n    /**\n     * Generates a random boolean value.\n     *\n     * @return the random boolean.\n     * @since 3.5\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static boolean nextBoolean() {\n        return secure().randomBoolean();\n    }\n\n    /**\n     * Generates an array of random bytes.\n     *\n     * @param count the size of the returned array.\n     * @return the random byte array.\n     * @throws IllegalArgumentException if {@code count} is negative.\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static byte[] nextBytes(final int count) {\n        return secure().randomBytes(count);\n    }\n\n    /**\n     * Generates a random double between 0 (inclusive) and {@link Double#MAX_VALUE} ({@value Double#MAX_VALUE} exclusive).\n     *\n     * @return the random double.\n     * @see #nextDouble(double, double)\n     * @since 3.5\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static double nextDouble() {\n        return secure().randomDouble();\n    }\n\n    /**\n     * Generates a random double within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random double.\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static double nextDouble(final double startInclusive, final double endExclusive) {\n        return secure().randomDouble(startInclusive, endExclusive);\n    }\n\n    /**\n     * Generates a random float between 0 (inclusive) and {@link Float#MAX_VALUE} ({@value Float#MAX_VALUE} exclusive).\n     *\n     * @return the random float.\n     * @see #nextFloat(float, float)\n     * @since 3.5\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static float nextFloat() {\n        return secure().randomFloat();\n    }\n\n    /**\n     * Generates a random float within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random float.\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static float nextFloat(final float startInclusive, final float endExclusive) {\n        return secure().randomFloat(startInclusive, endExclusive);\n    }\n\n    /**\n     * Generates a random int between 0 (inclusive) and {@link Integer#MAX_VALUE} ({@value Integer#MAX_VALUE} exclusive).\n     *\n     * @return the random integer.\n     * @see #nextInt(int, int)\n     * @since 3.5\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static int nextInt() {\n        return secure().randomInt();\n    }\n\n    /**\n     * Generates a random integer within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random integer.\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static int nextInt(final int startInclusive, final int endExclusive) {\n        return secure().randomInt(startInclusive, endExclusive);\n    }\n\n    /**\n     * Generates a random long between 0 (inclusive) and {@link Long#MAX_VALUE} ({@value Long#MAX_VALUE} exclusive).\n     *\n     * @return the random long.\n     * @see #nextLong(long, long)\n     * @since 3.5\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static long nextLong() {\n        return secure().randomLong();\n    }\n\n    /**\n     * Generates a random long within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random long.\n     * @deprecated Use {@link #secure()}, {@link #secureStrong()}, or {@link #insecure()}.\n     */\n    @Deprecated\n    public static long nextLong(final long startInclusive, final long endExclusive) {\n        return secure().randomLong(startInclusive, endExclusive);\n    }\n\n    /**\n     * Gets the singleton instance based on {@link SecureRandom#SecureRandom()} which uses the default algorithm\n     * and provider of {@link SecureRandom}.\n     * <p>\n     * The method {@link SecureRandom#SecureRandom()} is called on-demand.\n     * </p>\n     *\n     * @return the singleton instance based on {@link SecureRandom#SecureRandom()}.\n     * @see SecureRandom#SecureRandom()\n     * @since 3.16.0\n     */\n    public static RandomUtils secure() {\n        return SECURE;\n    }\n\n    static SecureRandom secureRandom() {\n        return SECURE_RANDOM_STRONG.get();\n    }\n\n    /**\n     * Gets the singleton instance based on {@link SecureRandom#getInstanceStrong()} which uses an algorithms/providers\n     * specified in the {@code securerandom.strongAlgorithms} {@link Security} property.\n     * <p>\n     * The method {@link SecureRandom#getInstanceStrong()} is called on-demand.\n     * </p>\n     *\n     * @return the singleton instance based on {@link SecureRandom#getInstanceStrong()}.\n     * @see SecureRandom#getInstanceStrong()\n     * @since 3.17.0\n     */\n    public static RandomUtils secureStrong() {\n        return SECURE_STRONG;\n    }\n\n    private final Supplier<Random> random;\n\n    /**\n     * {@link RandomUtils} instances should NOT be constructed in standard programming. Instead, the class should be\n     * used as {@code RandomUtils.nextBytes(5);}.\n     * <p>\n     * This constructor is public to permit tools that require a JavaBean instance to operate.\n     * </p>\n     *\n     * @deprecated TODO Make private in 4.0.\n     */\n    @Deprecated\n    public RandomUtils() {\n        this(SECURE_STRONG_SUPPLIER);\n    }\n\n    private RandomUtils(final Supplier<Random> random) {\n        this.random = random;\n    }\n\n    Random random() {\n        return random.get();\n    }\n\n    /**\n     * Generates a random boolean value.\n     *\n     * @return the random boolean.\n     * @since 3.16.0\n     */\n    public boolean randomBoolean() {\n        return random().nextBoolean();\n    }\n\n    /**\n     * Generates an array of random bytes.\n     *\n     * @param count the size of the returned array.\n     * @return the random byte array.\n     * @throws IllegalArgumentException if {@code count} is negative.\n     * @since 3.16.0\n     */\n    public byte[] randomBytes(final int count) {\n        Validate.isTrue(count >= 0, \"Count cannot be negative.\");\n        final byte[] result = new byte[count];\n        random().nextBytes(result);\n        return result;\n    }\n\n    /**\n     * Generates a random double between 0 (inclusive) and {@link Double#MAX_VALUE} ({@value Double#MAX_VALUE} exclusive).\n     *\n     * @return the random double.\n     * @see #randomDouble(double, double)\n     * @since 3.16.0\n     */\n    public double randomDouble() {\n        return randomDouble(0, Double.MAX_VALUE);\n    }\n\n    /**\n     * Generates a random double within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random double.\n     * @since 3.16.0\n     */\n    public double randomDouble(final double startInclusive, final double endExclusive) {\n        Validate.isTrue(endExclusive >= startInclusive, \"Start value must be smaller or equal to end value.\");\n        Validate.isTrue(startInclusive >= 0, \"Both range values must be non-negative.\");\n        if (startInclusive == endExclusive) {\n            return startInclusive;\n        }\n        return startInclusive + (endExclusive - startInclusive) * random().nextDouble();\n    }\n\n    /**\n     * Generates a random float between 0 (inclusive) and Float.MAX_VALUE (exclusive).\n     *\n     * @return the random float.\n     * @see #randomFloat(float, float)\n     * @since 3.16.0\n     */\n    public float randomFloat() {\n        return randomFloat(0, Float.MAX_VALUE);\n    }\n\n    /**\n     * Generates a random float within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random float.\n     * @since 3.16.0\n     */\n    public float randomFloat(final float startInclusive, final float endExclusive) {\n        Validate.isTrue(endExclusive >= startInclusive, \"Start value must be smaller or equal to end value.\");\n        Validate.isTrue(startInclusive >= 0, \"Both range values must be non-negative.\");\n        if (startInclusive == endExclusive) {\n            return startInclusive;\n        }\n        return startInclusive + (endExclusive - startInclusive) * random().nextFloat();\n    }\n\n    /**\n     * Generates a random int between 0 (inclusive) and {@link Integer#MAX_VALUE} ({@value Integer#MAX_VALUE} exclusive).\n     *\n     * @return the random integer.\n     * @see #randomInt(int, int)\n     * @since 3.16.0\n     */\n    public int randomInt() {\n        return randomInt(0, Integer.MAX_VALUE);\n    }\n\n    /**\n     * Generates a random integer within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random integer.\n     * @since 3.16.0\n     */\n    public int randomInt(final int startInclusive, final int endExclusive) {\n        Validate.isTrue(endExclusive >= startInclusive, \"Start value must be smaller or equal to end value.\");\n        Validate.isTrue(startInclusive >= 0, \"Both range values must be non-negative.\");\n        if (startInclusive == endExclusive) {\n            return startInclusive;\n        }\n        return startInclusive + random().nextInt(endExclusive - startInclusive);\n    }\n\n    /**\n     * Generates a random long between 0 (inclusive) and {@link Long#MAX_VALUE} ({@value Long#MAX_VALUE}, exclusive).\n     *\n     * @return the random long.\n     * @see #randomLong(long, long)\n     * @since 3.16.0\n     */\n    public long randomLong() {\n        return randomLong(Long.MAX_VALUE);\n    }\n\n    /**\n     * Generates a {@code long} value between 0 (inclusive) and the specified value (exclusive).\n     *\n     * @param n Bound on the random number to be returned. Must be positive.\n     * @return a random {@code long} value between 0 (inclusive) and {@code n} (exclusive).\n     */\n    private long randomLong(final long n) {\n        // Extracted from o.a.c.rng.core.BaseProvider.nextLong(long)\n        long bits;\n        long val;\n        do {\n            bits = random().nextLong() >>> 1;\n            val = bits % n;\n        } while (bits - val + n - 1 < 0);\n        return val;\n    }\n\n    /**\n     * Generates a random long within the specified range.\n     *\n     * @param startInclusive the smallest value that can be returned, must be non-negative.\n     * @param endExclusive   the upper bound (not included).\n     * @throws IllegalArgumentException if {@code startInclusive > endExclusive} or if {@code startInclusive} is negative.\n     * @return the random long.\n     * @since 3.16.0\n     */\n    public long randomLong(final long startInclusive, final long endExclusive) {\n        Validate.isTrue(endExclusive >= startInclusive, \"Start value must be smaller or equal to end value.\");\n        Validate.isTrue(startInclusive >= 0, \"Both range values must be non-negative.\");\n        if (startInclusive == endExclusive) {\n            return startInclusive;\n        }\n        return startInclusive + randomLong(endExclusive - startInclusive);\n    }\n\n    @Override\n    public String toString() {\n        return \"RandomUtils [random=\" + random() + \"]\";\n    }\n\n}\n"
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