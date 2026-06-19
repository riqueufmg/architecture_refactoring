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
  "target_name": "org.apache.commons.lang3.mutable.MutableByte",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/new/data/runs/20260619_115219_6b37c634/planner/designite",
    "smells_csv": "DesignSmells.csv",
    "target_has_smell": true
  },
  "target_file": "src/main/java/org/apache/commons/lang3/mutable/MutableByte.java",
  "target_source_root": "src/main/java",
  "target_code": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\npackage org.apache.commons.lang3.mutable;\n\nimport java.util.concurrent.atomic.AtomicInteger;\n\n/**\n * A mutable {@code byte} wrapper.\n * <p>\n * This class was created before the introduction of the {@link java.util.concurrent.atomic} package and the {@link AtomicInteger} class.\n * </p>\n * <p>\n * Note that as MutableByte does not extend Byte, it is not treated by String.format as a Byte parameter.\n * </p>\n *\n * @see Byte\n * @see AtomicInteger\n * @since 2.1\n */\npublic class MutableByte extends Number implements Comparable<MutableByte>, Mutable<Number> {\n\n    /**\n     * Required for serialization support.\n     *\n     * @see java.io.Serializable\n     */\n    private static final long serialVersionUID = -1585823265L;\n\n    /** The mutable value. */\n    private byte value;\n\n    /**\n     * Constructs a new MutableByte with the default value of zero.\n     */\n    public MutableByte() {\n    }\n\n    /**\n     * Constructs a new MutableByte with the specified value.\n     *\n     * @param value  the initial value to store\n     */\n    public MutableByte(final byte value) {\n        this.value = value;\n    }\n\n    /**\n     * Constructs a new MutableByte with the specified value.\n     *\n     * @param value  the initial value to store, not null.\n     * @throws NullPointerException if the object is null.\n     */\n    public MutableByte(final Number value) {\n        this.value = value.byteValue();\n    }\n\n    /**\n     * Constructs a new MutableByte parsing the given string.\n     *\n     * @param value  the string to parse, not null.\n     * @throws NumberFormatException if the string cannot be parsed into a byte, see {@link Byte#parseByte(String)}.\n     * @since 2.5\n     */\n    public MutableByte(final String value) {\n        this.value = Byte.parseByte(value);\n    }\n\n    /**\n     * Adds a value to the value of this instance.\n     *\n     * @param operand  the value to add, not null.\n     * @since 2.2\n     */\n    public void add(final byte operand) {\n        this.value += operand;\n    }\n\n    /**\n     * Adds a value to the value of this instance.\n     *\n     * @param operand  the value to add, not null.\n     * @throws NullPointerException if the object is null.\n     * @since 2.2\n     */\n    public void add(final Number operand) {\n        this.value += operand.byteValue();\n    }\n\n    /**\n     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance\n     * immediately after the addition operation. This method is not thread safe.\n     *\n     * @param operand the quantity to add, not null.\n     * @return the value associated with this instance after adding the operand.\n     * @since 3.5\n     */\n    public byte addAndGet(final byte operand) {\n        this.value += operand;\n        return value;\n    }\n\n    /**\n     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance\n     * immediately after the addition operation. This method is not thread safe.\n     *\n     * @param operand the quantity to add, not null.\n     * @throws NullPointerException if {@code operand} is null.\n     * @return the value associated with this instance after adding the operand.\n     * @since 3.5\n     */\n    public byte addAndGet(final Number operand) {\n        this.value += operand.byteValue();\n        return value;\n    }\n\n    // shortValue relies on Number implementation\n    /**\n     * Returns the value of this MutableByte as a byte.\n     *\n     * @return the numeric value represented by this object after conversion to type byte.\n     */\n    @Override\n    public byte byteValue() {\n        return value;\n    }\n\n    /**\n     * Compares this mutable to another in ascending order.\n     *\n     * @param other  the other mutable to compare to, not null.\n     * @return negative if this is less, zero if equal, positive if greater.\n     */\n    @Override\n    public int compareTo(final MutableByte other) {\n        return Byte.compare(this.value, other.value);\n    }\n\n    /**\n     * Decrements the value.\n     *\n     * @since 2.2\n     */\n    public void decrement() {\n        value--;\n    }\n\n    /**\n     * Decrements this instance's value by 1; this method returns the value associated with the instance\n     * immediately after the decrement operation. This method is not thread safe.\n     *\n     * @return the value associated with the instance after it is decremented.\n     * @since 3.5\n     */\n    public byte decrementAndGet() {\n        value--;\n        return value;\n    }\n\n    /**\n     * Returns the value of this MutableByte as a double.\n     *\n     * @return the numeric value represented by this object after conversion to type double.\n     */\n    @Override\n    public double doubleValue() {\n        return value;\n    }\n\n    /**\n     * Compares this object to the specified object. The result is {@code true} if and only if the argument is\n     * not {@code null} and is a {@link MutableByte} object that contains the same {@code byte} value\n     * as this object.\n     *\n     * @param obj  the object to compare with, null returns false.\n     * @return {@code true} if the objects are the same; {@code false} otherwise.\n     */\n    @Override\n    public boolean equals(final Object obj) {\n        if (obj instanceof MutableByte) {\n            return value == ((MutableByte) obj).byteValue();\n        }\n        return false;\n    }\n\n    /**\n     * Returns the value of this MutableByte as a float.\n     *\n     * @return the numeric value represented by this object after conversion to type float.\n     */\n    @Override\n    public float floatValue() {\n        return value;\n    }\n\n    /**\n     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance\n     * immediately prior to the addition operation. This method is not thread safe.\n     *\n     * @param operand the quantity to add, not null.\n     * @return the value associated with this instance immediately before the operand was added.\n     * @since 3.5\n     */\n    public byte getAndAdd(final byte operand) {\n        final byte last = value;\n        this.value += operand;\n        return last;\n    }\n\n    /**\n     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance\n     * immediately prior to the addition operation. This method is not thread safe.\n     *\n     * @param operand the quantity to add, not null.\n     * @throws NullPointerException if {@code operand} is null.\n     * @return the value associated with this instance immediately before the operand was added.\n     * @since 3.5\n     */\n    public byte getAndAdd(final Number operand) {\n        final byte last = value;\n        this.value += operand.byteValue();\n        return last;\n    }\n\n    /**\n     * Decrements this instance's value by 1; this method returns the value associated with the instance\n     * immediately prior to the decrement operation. This method is not thread safe.\n     *\n     * @return the value associated with the instance before it was decremented.\n     * @since 3.5\n     */\n    public byte getAndDecrement() {\n        final byte last = value;\n        value--;\n        return last;\n    }\n\n    /**\n     * Increments this instance's value by 1; this method returns the value associated with the instance\n     * immediately prior to the increment operation. This method is not thread safe.\n     *\n     * @return the value associated with the instance before it was incremented.\n     * @since 3.5\n     */\n    public byte getAndIncrement() {\n        final byte last = value;\n        value++;\n        return last;\n    }\n\n    /**\n     * Gets the value as a Byte instance.\n     *\n     * @return the value as a Byte, never null.\n     * @deprecated Use {@link #get()}.\n     */\n    @Deprecated\n    @Override\n    public Byte getValue() {\n        return Byte.valueOf(this.value);\n    }\n\n    /**\n     * Returns a suitable hash code for this mutable.\n     *\n     * @return a suitable hash code.\n     */\n    @Override\n    public int hashCode() {\n        return value;\n    }\n\n    /**\n     * Increments the value.\n     *\n     * @since 2.2\n     */\n    public void increment() {\n        value++;\n    }\n\n    /**\n     * Increments this instance's value by 1; this method returns the value associated with the instance\n     * immediately after the increment operation. This method is not thread safe.\n     *\n     * @return the value associated with the instance after it is incremented.\n     * @since 3.5\n     */\n    public byte incrementAndGet() {\n        value++;\n        return value;\n    }\n\n    /**\n     * Returns the value of this MutableByte as an int.\n     *\n     * @return the numeric value represented by this object after conversion to type int.\n     */\n    @Override\n    public int intValue() {\n        return value;\n    }\n\n    /**\n     * Returns the value of this MutableByte as a long.\n     *\n     * @return the numeric value represented by this object after conversion to type long.\n     */\n    @Override\n    public long longValue() {\n        return value;\n    }\n\n    /**\n     * Sets the value.\n     *\n     * @param value  the value to set.\n     */\n    public void setValue(final byte value) {\n        this.value = value;\n    }\n\n    /**\n     * Sets the value from any Number instance.\n     *\n     * @param value  the value to set, not null.\n     * @throws NullPointerException if the object is null.\n     */\n    @Override\n    public void setValue(final Number value) {\n        this.value = value.byteValue();\n    }\n\n    /**\n     * Subtracts a value from the value of this instance.\n     *\n     * @param operand  the value to subtract, not null.\n     * @since 2.2\n     */\n    public void subtract(final byte operand) {\n        this.value -= operand;\n    }\n\n    /**\n     * Subtracts a value from the value of this instance.\n     *\n     * @param operand  the value to subtract, not null.\n     * @throws NullPointerException if the object is null.\n     * @since 2.2\n     */\n    public void subtract(final Number operand) {\n        this.value -= operand.byteValue();\n    }\n\n    /**\n     * Gets this mutable as an instance of Byte.\n     *\n     * @return a Byte instance containing the value from this mutable.\n     */\n    public Byte toByte() {\n        return Byte.valueOf(byteValue());\n    }\n\n    /**\n     * Returns the String value of this mutable.\n     *\n     * @return the mutable value as a string.\n     */\n    @Override\n    public String toString() {\n        return String.valueOf(value);\n    }\n\n}\n"
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