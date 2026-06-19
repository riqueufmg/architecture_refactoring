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
  "target_name": "org.apache.commons.lang3.BitField",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/new/data/runs/20260618_163902_730c2872/planner/designite",
    "smells_csv": "DesignSmells.csv",
    "target_has_smell": true
  },
  "target_file": "src/main/java/org/apache/commons/lang3/BitField.java",
  "target_source_root": "src/main/java",
  "target_code": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\npackage org.apache.commons.lang3;\n\n/**\n * Supports operations on bit-mapped fields. Instances of this class can be used to store a flag or data within an {@code int}, {@code short} or {@code byte}.\n * <p>\n * Each {@link BitField} is constructed with a mask value, which indicates the bits that will be used to store and retrieve the data for that field. For\n * instance, the mask {@code 0xFF} indicates the least-significant byte should be used to store the data.\n * </p>\n * <p>\n * As an example, consider a car painting machine that accepts paint instructions as integers. Bit fields can be used to encode this:\n * </p>\n *\n * <pre>\n *\n * // blue, green and red are 1 byte values (0-255) stored in the three least\n * // significant bytes\n * BitField blue = new BitField(0xFF);\n *\n * BitField green = new BitField(0xFF00);\n *\n * BitField red = new BitField(0xFF0000);\n *\n * // anyColor is a flag triggered if any color is used\n * BitField anyColor = new BitField(0xFFFFFF);\n *\n * // isMetallic is a single bit flag\n * BitField isMetallic = new BitField(0x1000000);\n * </pre>\n * <p>\n * Using these {@link BitField} instances, a paint instruction can be encoded into an integer:\n * </p>\n *\n * <pre>\n * int paintInstruction = 0;\n * paintInstruction = red.setValue(paintInstruction, 35);\n * paintInstruction = green.setValue(paintInstruction, 100);\n * paintInstruction = blue.setValue(paintInstruction, 255);\n * </pre>\n * <p>\n * Flags and data can be retrieved from the integer:\n * </p>\n *\n * <pre>\n * // Prints true if red, green or blue is non-zero\n * System.out.println(anyColor.isSet(paintInstruction)); // prints true\n * // Prints value of red, green and blue\n * System.out.println(red.getValue(paintInstruction)); // prints 35\n * System.out.println(green.getValue(paintInstruction)); // prints 100\n * System.out.println(blue.getValue(paintInstruction)); // prints 255\n * // Prints true if isMetallic was set\n * System.out.println(isMetallic.isSet(paintInstruction)); // prints false\n * </pre>\n *\n * @since 2.0\n */\npublic class BitField {\n\n    private final long mask;\n\n    private final int shiftCount;\n\n    /**\n     * Creates a BitField instance.\n     *\n     * @param mask the mask specifying which bits apply to this BitField. Bits that are set in this mask are the bits that this BitField operates on.\n     */\n    public BitField(final int mask) {\n        this.mask = Integer.toUnsignedLong(mask);\n        this.shiftCount = this.mask == 0 ? 0 : Long.numberOfTrailingZeros(this.mask);\n    }\n\n    /**\n     * Creates a BitField instance.\n     *\n     * @param mask the mask specifying which bits apply to this BitField. Bits that are set in this mask are the bits that this BitField operates on.\n     * @since 3.21.0\n     */\n    public BitField(final long mask) {\n        this.mask = mask;\n        this.shiftCount = mask == 0 ? 0 : Long.numberOfTrailingZeros(mask);\n    }\n\n    /**\n     * Clears the bits.\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @return the value of holder with the specified bits cleared (set to {@code 0}).\n     */\n    public int clear(final int holder) {\n        return (int) (holder & ~mask);\n    }\n\n    /**\n     * Clears the bits.\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @return the value of holder with the specified bits cleared (set to {@code 0}).\n     * @since 3.21.0\n     */\n    public long clear(final long holder) {\n        return holder & ~mask;\n    }\n\n    /**\n     * Clears the bits.\n     *\n     * @param holder the byte data containing the bits we're interested in.\n     * @return the value of holder with the specified bits cleared (set to {@code 0}).\n     */\n    public byte clearByte(final byte holder) {\n        return (byte) clear(holder);\n    }\n\n    /**\n     * Clears the bits.\n     *\n     * @param holder the short data containing the bits we're interested in.\n     * @return the value of holder with the specified bits cleared (set to {@code 0}).\n     */\n    public short clearShort(final short holder) {\n        return (short) clear(holder);\n    }\n\n    /**\n     * Gets the value for the specified BitField, unshifted.\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @return the selected bits.\n     */\n    public int getRawValue(final int holder) {\n        return (int) (holder & mask);\n    }\n\n    /**\n     * Gets the value for the specified BitField, unshifted.\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @return the selected bits.\n     * @since 3.21.0\n     */\n    public long getRawValue(final long holder) {\n        return holder & mask;\n    }\n\n    /**\n     * Obtains the value for the specified BitField, unshifted.\n     *\n     * @param holder the short data containing the bits we're interested in.\n     * @return the selected bits.\n     */\n    public short getShortRawValue(final short holder) {\n        return (short) getRawValue(holder);\n    }\n\n    /**\n     * Gets the value for the specified BitField, appropriately shifted right, as a short.\n     * <p>\n     * Many users of a BitField will want to treat the specified bits as an int value, and will not want to be aware that the value is stored as a BitField (and\n     * so shifted left so many bits).\n     * </p>\n     *\n     * @param holder the short data containing the bits we're interested in.\n     * @return the selected bits, shifted right appropriately.\n     * @see #setShortValue(short,short)\n     */\n    public short getShortValue(final short holder) {\n        return (short) getValue(holder);\n    }\n\n    /**\n     * Gets the value for the specified BitField, appropriately shifted right.\n     * <p>\n     * Many users of a BitField will want to treat the specified bits as an int value, and will not want to be aware that the value is stored as a BitField (and\n     * so shifted left so many bits).\n     * </p>\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @return the selected bits, shifted right appropriately.\n     * @see #setValue(int,int)\n     */\n    public int getValue(final int holder) {\n        return getRawValue(holder) >> shiftCount;\n    }\n\n    /**\n     * Gets the value for the specified BitField, appropriately shifted right.\n     * <p>\n     * Many users of a BitField will want to treat the specified bits as an long value, and will not want to be aware that the value is stored as a BitField (and\n     * so shifted left so many bits).\n     * </p>\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @return the selected bits, shifted right appropriately.\n     * @see #setValue(long,long)\n     * @since 3.21.0\n     */\n    public long getValue(final long holder) {\n        return getRawValue(holder) >> shiftCount;\n    }\n\n    /**\n     * Tests whether all of the bits are set or not.\n     * <p>\n     * This is a stricter test than {@link #isSet(int)}, in that all of the bits in a multi-bit set must be set for this method to return {@code true}.\n     * </p>\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @return {@code true} if all of the bits are set, else {@code false}.\n     */\n    public boolean isAllSet(final int holder) {\n        return (holder & mask) == mask;\n    }\n\n    /**\n     * Tests whether all of the bits are set or not.\n     * <p>\n     * This is a stricter test than {@link #isSet(long)}, in that all of the bits in a multi-bit set must be set for this method to return {@code true}.\n     * </p>\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @return {@code true} if all of the bits are set, else {@code false}.\n     * @since 3.21.0\n     */\n    public boolean isAllSet(final long holder) {\n        return (holder & mask) == mask;\n    }\n\n    /**\n     * Tests whether the field is set or not.\n     * <p>\n     * This is most commonly used for a single-bit field, which is often used to represent a boolean value; the results of using it for a multi-bit field is to\n     * determine whether <em>any</em> of its bits are set.\n     * </p>\n     *\n     * @param holder the int data containing the bits we're interested in\n     * @return {@code true} if any of the bits are set, else {@code false}\n     */\n    public boolean isSet(final int holder) {\n        return (holder & mask) != 0;\n    }\n\n    /**\n     * Tests whether the field is set or not.\n     * <p>\n     * This is most commonly used for a single-bit field, which is often used to represent a boolean value; the results of using it for a multi-bit field is to\n     * determine whether <em>any</em> of its bits are set.\n     * </p>\n     *\n     * @param holder the long data containing the bits we're interested in\n     * @return {@code true} if any of the bits are set, else {@code false}\n     * @since 3.21.0\n     */\n    public boolean isSet(final long holder) {\n        return (holder & mask) != 0;\n    }\n\n    /**\n     * Sets the bits.\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @return the value of holder with the specified bits set to {@code 1}.\n     */\n    public int set(final int holder) {\n        return (int) (holder | mask);\n    }\n\n    /**\n     * Sets the bits.\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @return the value of holder with the specified bits set to {@code 1}.\n     * @since 3.21.0\n     */\n    public long set(final long holder) {\n        return holder | mask;\n    }\n\n    /**\n     * Sets a boolean BitField.\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @param flag   indicating whether to set or clear the bits.\n     * @return the value of holder with the specified bits set or cleared.\n     */\n    public int setBoolean(final int holder, final boolean flag) {\n        return flag ? set(holder) : clear(holder);\n    }\n\n    /**\n     * Sets a boolean BitField.\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @param flag   indicating whether to set or clear the bits.\n     * @return the value of holder with the specified bits set or cleared.\n     * @since 3.21.0\n     */\n    public long setBoolean(final long holder, final boolean flag) {\n        return flag ? set(holder) : clear(holder);\n    }\n\n    /**\n     * Sets the bits.\n     *\n     * @param holder the byte data containing the bits we're interested in\n     * @return the value of holder with the specified bits set to {@code 1}\n     */\n    public byte setByte(final byte holder) {\n        return (byte) set(holder);\n    }\n\n    /**\n     * Sets a boolean BitField.\n     *\n     * @param holder the byte data containing the bits we're interested in.\n     * @param flag   indicating whether to set or clear the bits.\n     * @return the value of holder with the specified bits set or cleared.\n     */\n    public byte setByteBoolean(final byte holder, final boolean flag) {\n        return flag ? setByte(holder) : clearByte(holder);\n    }\n\n    /**\n     * Sets the bits.\n     *\n     * @param holder the short data containing the bits we're interested in.\n     * @return the value of holder with the specified bits set to {@code 1}.\n     */\n    public short setShort(final short holder) {\n        return (short) set(holder);\n    }\n\n    /**\n     * Sets a boolean BitField.\n     *\n     * @param holder the short data containing the bits we're interested in.\n     * @param flag   indicating whether to set or clear the bits.\n     * @return the value of holder with the specified bits set or cleared.\n     */\n    public short setShortBoolean(final short holder, final boolean flag) {\n        return flag ? setShort(holder) : clearShort(holder);\n    }\n\n    /**\n     * Sets the bits with new values.\n     *\n     * @param holder the short data containing the bits we're interested in\n     * @param value  the new value for the specified bits\n     * @return the value of holder with the bits from the value parameter replacing the old bits\n     * @see #getShortValue(short)\n     */\n    public short setShortValue(final short holder, final short value) {\n        return (short) setValue(holder, value);\n    }\n\n    /**\n     * Sets the bits with new values.\n     *\n     * @param holder the int data containing the bits we're interested in.\n     * @param value  the new value for the specified bits.\n     * @return the value of holder with the bits from the value parameter replacing the old bits.\n     * @see #getValue(int)\n     */\n    public int setValue(final int holder, final int value) {\n        return (int) (holder & ~mask | value << shiftCount & mask);\n    }\n\n    /**\n     * Sets the bits with new values.\n     *\n     * @param holder the long data containing the bits we're interested in.\n     * @param value  the new value for the specified bits.\n     * @return the value of holder with the bits from the value parameter replacing the old bits.\n     * @see #getValue(long)\n     * @since 3.21.0\n     */\n    public long setValue(final long holder, final long value) {\n        return holder & ~mask | value << shiftCount & mask;\n    }\n}\n"
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