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
    "goal": "Move the small cohesive cluster of arity-3 functional interfaces to a new subpackage to reduce package size.",
    "files": [
      "src/main/java/org/apache/commons/lang3/function/TriConsumer.java",
      "src/main/java/org/apache/commons/lang3/function/TriFunction.java",
      "src/main/java/org/apache/commons/lang3/function/triple/TriConsumer.java",
      "src/main/java/org/apache/commons/lang3/function/triple/TriFunction.java"
    ],
    "ops": [
      {
        "op": "CREATE_PACKAGE",
        "inputs": [],
        "outputs": [
          "org.apache.commons.lang3.function.triple"
        ],
        "details": "Create the destination subpackage if it does not exist.",
        "risk": "low",
        "api_change": false
      },
      {
        "op": "MOVE_CLASS",
        "inputs": [
          "org.apache.commons.lang3.function.TriConsumer"
        ],
        "outputs": [
          "org.apache.commons.lang3.function.triple.TriConsumer"
        ],
        "details": "Move this arity-3 consumer functional interface to the destination package.",
        "risk": "low",
        "api_change": true
      },
      {
        "op": "MOVE_CLASS",
        "inputs": [
          "org.apache.commons.lang3.function.TriFunction"
        ],
        "outputs": [
          "org.apache.commons.lang3.function.triple.TriFunction"
        ],
        "details": "Move this arity-3 function functional interface to the destination package.",
        "risk": "low",
        "api_change": true
      },
      {
        "op": "UPDATE_VISIBILITY",
        "inputs": [
          "org.apache.commons.lang3.function.triple.TriConsumer",
          "org.apache.commons.lang3.function.triple.TriFunction"
        ],
        "outputs": [],
        "details": "After moving the whole cluster to the destination package, update only the minimum required visibility in moved classes and related remaining classes so the project can compile. Do not change behavior. Do not move additional classes.",
        "risk": "medium",
        "api_change": true
      }
    ]
  },
  "allowed_files": [
    "src/main/java/org/apache/commons/lang3/function/TriConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/TriFunction.java",
    "src/main/java/org/apache/commons/lang3/function/triple/TriConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/triple/TriFunction.java"
  ],
  "executor_existing_files": [
    "src/main/java/org/apache/commons/lang3/function/TriConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/TriFunction.java"
  ],
  "executor_new_files": [
    "src/main/java/org/apache/commons/lang3/function/triple/TriConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/triple/TriFunction.java"
  ],
  "executor_rejected_files": [],
  "files_context": [
    {
      "path": "src/main/java/org/apache/commons/lang3/function/TriConsumer.java",
      "exists": "false",
      "content": ""
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/TriFunction.java",
      "exists": "false",
      "content": ""
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/triple/TriConsumer.java",
      "exists": "true",
      "content": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\npackage org.apache.commons.lang3.function.triple;\n\nimport java.util.Objects;\nimport java.util.function.Consumer;\n\n/**\n * Represents an operation that accepts three input arguments and returns no result. This is the three-arity\n * specialization of {@link Consumer}. Unlike most other functional interfaces, {@link TriConsumer} is expected to\n * operate via side effects.\n *\n * <p>\n * This is a {@link FunctionalInterface} whose functional method is {@link #accept(Object, Object, Object)}.\n * </p>\n * <p>\n * Provenance: Apache Log4j 2.7\n * </p>\n *\n * @param <T> type of the first argument\n * @param <U> type of the second argument\n * @param <V> type of the third argument\n * @since 3.13.0\n */\n@FunctionalInterface\npublic interface TriConsumer<T, U, V> {\n\n    /**\n     * Performs the operation given the specified arguments.\n     *\n     * @param k the first input argument\n     * @param v the second input argument\n     * @param s the third input argument\n     */\n    void accept(T k, U v, V s);\n\n    /**\n     * Returns a composed {@link TriConsumer} that performs, in sequence, this operation followed by the {@code after}\n     * operation. If performing either operation throws an exception, it is relayed to the caller of the composed\n     * operation. If performing this operation throws an exception, the {@code after} operation will not be performed.\n     *\n     * @param after the operation to perform after this operation.\n     * @return a composed {@link TriConsumer} that performs in sequence this operation followed by the {@code after}\n     *         operation.\n     * @throws NullPointerException if {@code after} is null.\n     */\n    default TriConsumer<T, U, V> andThen(final TriConsumer<? super T, ? super U, ? super V> after) {\n        Objects.requireNonNull(after);\n        return (t, u, v) -> {\n            accept(t, u, v);\n            after.accept(t, u, v);\n        };\n    }\n\n}\n"
    },
    {
      "path": "src/main/java/org/apache/commons/lang3/function/triple/TriFunction.java",
      "exists": "true",
      "content": "/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n * contributor license agreements.  See the NOTICE file distributed with\n * this work for additional information regarding copyright ownership.\n * The ASF licenses this file to You under the Apache License, Version 2.0\n * (the \"License\"); you may not use this file except in compliance with\n * the License.  You may obtain a copy of the License at\n *\n *      https://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\npackage org.apache.commons.lang3.function.triple;\n\nimport java.util.Objects;\nimport java.util.function.Function;\n\n/**\n * Represents a function that accepts three arguments and produces a result. This is the three-arity specialization of\n * {@link Function}.\n *\n * <p>\n * This is a <a href=\"package-summary.html\">functional interface</a> whose functional method is\n * {@link #apply(Object, Object, Object)}.\n * </p>\n *\n * @param <T> the type of the first argument to the function\n * @param <U> the type of the second argument to the function\n * @param <V> the type of the third argument to the function\n * @param <R> the type of the result of the function\n * @see Function\n * @since 3.12.0\n */\n@FunctionalInterface\npublic interface TriFunction<T, U, V, R> {\n\n    /**\n     * Returns a composed function that first applies this function to its input, and then applies the {@code after}\n     * function to the result. If evaluation of either function throws an exception, it is relayed to the caller of the\n     * composed function.\n     *\n     * @param <W> the type of output of the {@code after} function, and of the composed function\n     * @param after the function to apply after this function is applied\n     * @return a composed function that first applies this function and then applies the {@code after} function\n     * @throws NullPointerException if after is null\n     */\n    default <W> TriFunction<T, U, V, W> andThen(final Function<? super R, ? extends W> after) {\n        Objects.requireNonNull(after);\n        return (final T t, final U u, final V v) -> after.apply(apply(t, u, v));\n    }\n\n    /**\n     * Applies this function to the given arguments.\n     *\n     * @param t the first function argument\n     * @param u the second function argument\n     * @param v the third function argument\n     * @return the function result\n     */\n    R apply(T t, U u, V v);\n}\n"
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