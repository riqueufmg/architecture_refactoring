# TASK

Apply the staged refactoring block to the provided Java source files.

You must implement the requested block while preserving behavior.

# IMPORTANT RULES

- Modify only files listed in allowed_files.
- Return the full new content for every modified file.
- Do not return partial patches.
- You may edit test files only when they are listed in allowed_files and the block contains UPDATE_TESTS.
- Do not edit unrelated tests.
- If a listed test file does not require changes, do not include it in files_to_write.
- Do not change unrelated code.
- Do not undo previous refactorings.
- If the block cannot be safely applied, return an empty files_to_write list and explain the reason in "notes".
- Return only files that actually changed.
- Do not include unchanged files in files_to_write.

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
  "repo_path": "/data/henrique/langchain_prototype/new/data/repositories/jsoup",
  "target": {
    "smell": "IM",
    "smell_name": "Insufficient Modularization",
    "target_type": "class",
    "target_name": "org.jsoup.parser.TokeniserState"
  },
  "block": {
    "id": 1,
    "goal": "Introduce a small helper class to house TokeniserState's private static helper routines (reduce enum size while preserving API).",
    "files": [
      "src/main/java/org/jsoup/parser/TokeniserStateHelper.java"
    ],
    "ops": [
      {
        "op": "EXTRACT_CLASS",
        "inputs": [
          "src/main/java/org/jsoup/parser/TokeniserState.java"
        ],
        "outputs": [
          "src/main/java/org/jsoup/parser/TokeniserStateHelper.java"
        ],
        "details": "Create new class org.jsoup.parser.TokeniserStateHelper in same package. Move implementations of the following private static helper methods into this helper as public static methods with identical signatures (but in TokeniserStateHelper): handleDataEndTag(Tokeniser, CharacterReader, TokeniserState), readRawData(Tokeniser, CharacterReader, TokeniserState, TokeniserState), readCharRef(Tokeniser, TokeniserState), readEndTag(Tokeniser, CharacterReader, TokeniserState, TokeniserState), handleDataDoubleEscapeTag(Tokeniser, CharacterReader, TokeniserState, TokeniserState), readDoctypeInternalSubset(Tokeniser, CharacterReader, TokeniserState). Add an import for org.jsoup.internal.StringUtil for readDoctypeInternalSubset. Methods should reference Tokeniser.replacementChar directly where needed (instead of TokeniserState's replacementChar), and otherwise use only types present in the original source. Do not change method semantics or visibility for Tokeniser (keep helper methods static so they can be called from TokeniserState).",
        "risk": "medium"
      },
      {
        "op": "ADD_OR_UPDATE_IMPORTS",
        "inputs": [
          "src/main/java/org/jsoup/parser/TokeniserStateHelper.java"
        ],
        "outputs": [
          "src/main/java/org/jsoup/parser/TokeniserStateHelper.java"
        ],
        "details": "Ensure TokeniserStateHelper has 'package org.jsoup.parser;' and imports required: org.jsoup.internal.StringUtil (used by readDoctypeInternalSubset). No other external imports required because types are in same package.",
        "risk": "low"
      }
    ],
    "test_files": []
  },
  "allowed_files": [
    "src/main/java/org/jsoup/parser/TokeniserStateHelper.java"
  ],
  "executor_existing_files": [],
  "executor_new_files": [
    "src/main/java/org/jsoup/parser/TokeniserStateHelper.java"
  ],
  "executor_rejected_files": [],
  "files_context": [
    {
      "path": "src/main/java/org/jsoup/parser/TokeniserStateHelper.java",
      "exists": "false",
      "content": ""
    }
  ],
  "feedback": "EXECUTOR_NO_CHANGES: The executor returned valid JSON, but produced no files_to_write or files_to_delete. Revise the same block and generate concrete file changes.",
  "attempt": 4,
  "move_class_constraints": {
    "moved_old_files": [],
    "moved_new_files": [],
    "rule": "When MOVE_CLASS is present, OpenRewrite has already moved the classes before this executor runs. Do not recreate or write files listed in moved_old_files. Edit destination files and related remaining files only."
  }
}