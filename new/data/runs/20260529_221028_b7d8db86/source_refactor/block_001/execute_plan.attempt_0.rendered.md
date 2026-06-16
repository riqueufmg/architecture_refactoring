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
  "repo_path": "/data/henrique/langchain_prototype/new/data/repositories/jsoup",
  "target": {
    "smell": "IM",
    "smell_name": "Insufficient Modularization",
    "target_type": "class",
    "target_name": "org.jsoup.nodes.Attribute"
  },
  "block": {
    "id": 1,
    "goal": "Introduce a small helper class to hold attribute key validation/coercion patterns and logic, reducing Attribute's size and grouping cohesive responsibilities.",
    "files": [
      "src/main/java/org/jsoup/nodes/AttributeKeyUtil.java"
    ],
    "ops": [
      {
        "op": "EXTRACT_CLASS",
        "inputs": [],
        "outputs": [
          "src/main/java/org/jsoup/nodes/AttributeKeyUtil.java"
        ],
        "details": "Create a new package-private helper class AttributeKeyUtil in org.jsoup.nodes that centralizes pattern constants and validation/coercion logic for attribute keys (xml/html). This class provides: xmlKeyReplace, htmlKeyReplace, isValidXmlKey, isValidHtmlKey, coerceXmlKey, coerceHtmlKey. The class is package-private so no public API changes are introduced.\n\nExample content (new file):\n\npackage org.jsoup.nodes;\n\nimport java.util.regex.Pattern;\n\nclass AttributeKeyUtil {\n    static final Pattern xmlKeyReplace = Pattern.compile(\"[^-a-zA-Z0-9_:.]+\");\n    static final Pattern htmlKeyReplace = Pattern.compile(\"[\\\\x00-\\\\x1f\\\\x7f-\\\\x9f \\\\\\\"'/=]+\");\n\n    static boolean isValidXmlKey(String key) {\n        final int length = key.length();\n        if (length == 0) return false;\n        char c = key.charAt(0);\n        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == ':'))\n            return false;\n        for (int i = 1; i < length; i++) {\n            c = key.charAt(i);\n            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == ':' || c == '.'))\n                return false;\n        }\n        return true;\n    }\n\n    static boolean isValidHtmlKey(String key) {\n        final int length = key.length();\n        if (length == 0) return false;\n        for (int i = 0; i < length; i++) {\n            char c = key.charAt(i);\n            if ((c <= 0x1f) || (c >= 0x7f && c <= 0x9f) || c == ' ' || c == '\"' || c == '\\'' || c == '/' || c == '=')\n                return false;\n        }\n        return true;\n    }\n\n    static String coerceXmlKey(String key) {\n        key = xmlKeyReplace.matcher(key).replaceAll(\"_\");\n        return isValidXmlKey(key) ? key : null;\n    }\n\n    static String coerceHtmlKey(String key) {\n        key = htmlKeyReplace.matcher(key).replaceAll(\"_\");\n        return isValidHtmlKey(key) ? key : null;\n    }\n}\n",
        "risk": "low"
      }
    ]
  },
  "allowed_files": [
    "src/main/java/org/jsoup/nodes/AttributeKeyUtil.java"
  ],
  "executor_existing_files": [
    "src/main/java/org/jsoup/nodes/AttributeKeyUtil.java"
  ],
  "executor_new_files": [],
  "executor_rejected_files": [],
  "files_context": [
    {
      "path": "src/main/java/org/jsoup/nodes/AttributeKeyUtil.java",
      "exists": "true",
      "content": "package org.jsoup.nodes;\n\nimport java.util.regex.Pattern;\n\nclass AttributeKeyUtil {\n    static final Pattern xmlKeyReplace = Pattern.compile(\"[^-a-zA-Z0-9_:.]+\");\n    static final Pattern htmlKeyReplace = Pattern.compile(\"[\\\\x00-\\\\x1f\\\\x7f-\\\\x9f \\\"'/=]+\");\n\n    static boolean isValidXmlKey(String key) {\n        final int length = key.length();\n        if (length == 0) return false;\n        char c = key.charAt(0);\n        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == ':'))\n            return false;\n        for (int i = 1; i < length; i++) {\n            c = key.charAt(i);\n            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == ':' || c == '.'))\n                return false;\n        }\n        return true;\n    }\n\n    static boolean isValidHtmlKey(String key) {\n        final int length = key.length();\n        if (length == 0) return false;\n        for (int i = 0; i < length; i++) {\n            char c = key.charAt(i);\n            if ((c <= 0x1f) || (c >= 0x7f && c <= 0x9f) || c == ' ' || c == '\"' || c == '\\'' || c == '/' || c == '=')\n                return false;\n        }\n        return true;\n    }\n\n    static String coerceXmlKey(String key) {\n        key = xmlKeyReplace.matcher(key).replaceAll(\"_\");\n        return isValidXmlKey(key) ? key : null;\n    }\n\n    static String coerceHtmlKey(String key) {\n        key = htmlKeyReplace.matcher(key).replaceAll(\"_\");\n        return isValidHtmlKey(key) ? key : null;\n    }\n}\n"
    }
  ],
  "feedback": "",
  "attempt": 0
}