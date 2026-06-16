# TASK

Apply the staged refactoring block to the provided Java source files.

You must implement the requested block while preserving behavior.

# IMPORTANT RULES

- Modify only files listed in allowed_files.
- Return the full new content for every modified file.
- Do not return partial patches.
- Do not edit tests.
- Do not change unrelated code.
- Do not undo previous refactorings.
- If the block cannot be safely applied, return an empty files_to_write list and explain the reason in "notes".

# OUTPUT FORMAT

Return ONLY valid JSON in this format:

{
  "files_to_write": [
    {
      "path": "src/main/java/...",
      "content": "full file content here"
    }
  ],
  "files_to_delete": [],
  "notes": "short explanation"
}

# INPUT

{
  "repo_path": "/data/henrique/langchain_prototype/new/data/repositories/commons-lang",
  "target": {
    "smell": "GC",
    "smell_name": "God Component",
    "target_type": "package",
    "target_name": "org.apache.commons.lang3.function"
  },
  "block": {
    "id": 1,
    "goal": "Move a cohesive cluster of primitive functional interfaces into a new subpackage to reduce God Component size.",
    "files": [
      "src/main/java/org/apache/commons/lang3/function/BooleanConsumer.java",
      "src/main/java/org/apache/commons/lang3/function/ByteConsumer.java",
      "src/main/java/org/apache/commons/lang3/function/ByteSupplier.java",
      "src/main/java/org/apache/commons/lang3/function/primitives/BooleanConsumer.java",
      "src/main/java/org/apache/commons/lang3/function/primitives/ByteConsumer.java",
      "src/main/java/org/apache/commons/lang3/function/primitives/ByteSupplier.java"
    ],
    "ops": [
      {
        "op": "CREATE_PACKAGE",
        "inputs": [],
        "outputs": [
          "org.apache.commons.lang3.function.primitives"
        ],
        "details": "Create the destination subpackage if it does not exist.",
        "risk": "low",
        "api_change": false
      },
      {
        "op": "MOVE_CLASS",
        "inputs": [
          "org.apache.commons.lang3.function.BooleanConsumer"
        ],
        "outputs": [
          "org.apache.commons.lang3.function.primitives.BooleanConsumer"
        ],
        "details": "Move this primitive functional interface to the primitives subpackage.",
        "risk": "medium",
        "api_change": true
      },
      {
        "op": "MOVE_CLASS",
        "inputs": [
          "org.apache.commons.lang3.function.ByteConsumer"
        ],
        "outputs": [
          "org.apache.commons.lang3.function.primitives.ByteConsumer"
        ],
        "details": "Move this primitive functional interface to the primitives subpackage.",
        "risk": "medium",
        "api_change": true
      },
      {
        "op": "MOVE_CLASS",
        "inputs": [
          "org.apache.commons.lang3.function.ByteSupplier"
        ],
        "outputs": [
          "org.apache.commons.lang3.function.primitives.ByteSupplier"
        ],
        "details": "Move this primitive functional interface to the primitives subpackage.",
        "risk": "medium",
        "api_change": true
      },
      {
        "op": "UPDATE_VISIBILITY",
        "inputs": [
          "org.apache.commons.lang3.function.primitives.BooleanConsumer",
          "org.apache.commons.lang3.function.primitives.ByteConsumer",
          "org.apache.commons.lang3.function.primitives.ByteSupplier"
        ],
        "outputs": [],
        "details": "After moving the whole cluster to the destination package, update only the minimum required visibility in moved classes and related remaining classes so the project can compile. Do not change behavior. Do not move additional classes.",
        "risk": "medium",
        "api_change": true
      }
    ]
  },
  "allowed_files": [
    "src/main/java/org/apache/commons/lang3/function/BooleanConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/ByteConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/ByteSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/primitives/BooleanConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/primitives/ByteConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/primitives/ByteSupplier.java"
  ],
  "executor_existing_files": [
    "src/main/java/org/apache/commons/lang3/function/BooleanConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/ByteConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/ByteSupplier.java"
  ],
  "executor_new_files": [
    "src/main/java/org/apache/commons/lang3/function/primitives/BooleanConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/primitives/ByteConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/primitives/ByteSupplier.java"
  ],
  "executor_rejected_files": [],
  "files_context": [
    {
      "path": "src/main/java/org/apache/commons/lang3/function/BooleanConsumer.java",
      "exists": "false",
      "content": ""
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/ByteConsumer.java",
      "exists": "false",
      "content": ""
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/ByteSupplier.java",
      "exists": "false",
      "content": ""
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/primitives/BooleanConsumer.java",
      "exists": "true",
      "content": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\npackage org.apache.commons.lang3.function.primitives;\n\nimport java.util.Objects;\nimport java.util.function.IntConsumer;\n\n/**\n * A functional interface like {@link IntConsumer} but for {@code boolean}.\n *\n * @see IntConsumer\n * @since 3.13.0\n */\n@FunctionalInterface\npublic interface BooleanConsumer {\n\n    /** NOP singleton */\n    BooleanConsumer NOP = t -> { /* NOP */ };\n\n    /**\n     * Gets the NOP singleton.\n     *\n     * @return The NOP singleton.\n     */\n    static BooleanConsumer nop() {\n        return NOP;\n    }\n\n    /**\n     * Accepts the given arguments.\n     *\n     * @param value the input argument\n     */\n    void accept(boolean value);\n\n    /**\n     * Returns a composed {@link BooleanConsumer} that performs, in sequence, this operation followed by the {@code after}\n     * operation. If performing either operation throws an exception, it is relayed to the caller of the composed operation.\n     * If performing this operation throws an exception, the {@code after} operation will not be performed.\n     *\n     * @param after the operation to perform after this operation\n     * @return a composed {@link BooleanConsumer} that performs in sequence this operation followed by the {@code after}\n     *         operation\n     * @throws NullPointerException if {@code after} is null\n     */\n    default BooleanConsumer andThen(final BooleanConsumer after) {\n        Objects.requireNonNull(after);\n        return (final boolean t) -> {\n            accept(t);\n            after.accept(t);\n        };\n    }\n}\n"
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/primitives/ByteConsumer.java",
      "exists": "true",
      "content": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\npackage org.apache.commons.lang3.function.primitives;\n\nimport java.util.Objects;\nimport java.util.function.IntConsumer;\n\n/**\n * A functional interface like {@link IntConsumer} but for {@code byte}.\n *\n * @see IntConsumer\n * @since 3.19.0\n */\n@FunctionalInterface\npublic interface ByteConsumer {\n\n    /** NOP singleton */\n    ByteConsumer NOP = t -> {\n        /* NOP */ };\n\n    /**\n     * Gets the NOP singleton.\n     *\n     * @return The NOP singleton.\n     */\n    static ByteConsumer nop() {\n        return NOP;\n    }\n\n    /**\n     * Accepts the given arguments.\n     *\n     * @param value the input argument\n     */\n    void accept(byte value);\n\n    /**\n     * Returns a composed {@link ByteConsumer} that performs, in sequence, this operation followed by the {@code after} operation. If performing either\n     * operation throws an exception, it is relayed to the caller of the composed operation. If performing this operation throws an exception, the {@code after}\n     * operation will not be performed.\n     *\n     * @param after the operation to perform after this operation\n     * @return a composed {@link ByteConsumer} that performs in sequence this operation followed by the {@code after} operation\n     * @throws NullPointerException if {@code after} is null\n     */\n    default ByteConsumer andThen(final ByteConsumer after) {\n        Objects.requireNonNull(after);\n        return (final byte t) -> {\n            accept(t);\n            after.accept(t);\n        };\n    }\n}\n"
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/primitives/ByteSupplier.java",
      "exists": "true",
      "content": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\npackage org.apache.commons.lang3.function.primitives;\n\nimport java.util.function.IntSupplier;\n\n/**\n * A functional interface like {@link IntSupplier}, but for a byte.\n *\n * @since 3.19\n */\n@FunctionalInterface\npublic interface ByteSupplier {\n\n    /**\n     * Supplies a byte.\n     *\n     * @return a result.\n     */\n    byte getAsByte();\n}\n"
    }
  ],
  "feedback": "",
  "attempt": 0,
  "move_class_constraints": {
    "moved_old_files": [],
    "moved_new_files": [],
    "rule": "When MOVE_CLASS is present, OpenRewrite has already moved the classes before this executor runs. Do not recreate or write files listed in moved_old_files. Edit destination files and related remaining files only."
  }
}