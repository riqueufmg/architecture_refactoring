# GOAL
Refactor ONE cohesive cluster from the TARGET PACKAGE to reduce the God Component smell.

A single run must move only ONE cluster. All classes in the selected cluster must be moved to the SAME new subpackage.

# INPUT DATA
- `target_files`: files currently in the target package.
- `internal_deps`: dependencies between classes inside the target package.
- `incoming_deps`: external classes that depend on classes in the target package.
- `outgoing_deps`: target package classes depending on external classes.

# TASK
Choose exactly ONE cohesive cluster of classes from the target package.

Then generate a refactoring plan where:
- all classes in the cluster move to the same new subpackage;
- the plan contains exactly ONE block;
- that block moves all classes in the selected cluster together;
- the block contains 2 to 10 MOVE_CLASS operations.

# CLUSTER MOVEMENT STRATEGY

For God Component refactoring, each block must move one cohesive cluster, not a single isolated class.

A cluster must contain 2 to 4 classes.

All classes in the selected cluster must be moved to the same destination package.

Prefer clusters whose classes have strong internal relationships and many package-private interactions with each other.

Avoid moving highly central parser/state-machine classes alone.

Avoid selecting clusters that would require exposing many package-private members from classes that remain in the original package.

Each block must contain:
- optionally one CREATE_PACKAGE operation;
- two to four MOVE_CLASS operations;
- no UPDATE_VISIBILITY operation. UPDATE_VISIBILITY will be added automatically later.

Do not create one block per class.

Do not move all package classes in one block.

Prefer small cohesive clusters that can compile after limited visibility updates.

# PACKAGE RULES
- Move classes only from the TARGET PACKAGE to a subpackage of the TARGET PACKAGE.
- All moved classes must use the same destination package.
- Do not move classes to unrelated packages.
- Do not introduce facades.
- API changes are allowed.

# BLOCK RULES
- The plan must contain exactly ONE block.
- This single block represents the whole selected cluster.
- The block must contain 2 to 4 MOVE_CLASS operations.
- All MOVE_CLASS operations must move classes to the same destination package.
- The block may include one CREATE_PACKAGE operation before the MOVE_CLASS operations.
- The block files list must include the original source file path of every moved class.
- Do not create one block per class.
- Do not split the selected cluster across multiple blocks.
- Do not update external files in the plan; OpenRewrite will handle references/imports later.
- The executor will handle minimal package-private visibility changes in related internal files.

# ALLOWED OPS
CREATE_PACKAGE, MOVE_CLASS, ADD_OR_UPDATE_IMPORTS, UPDATE_CALL_SITES, UPDATE_VISIBILITY, NO_OP.

# OUTPUT
Return ONLY valid JSON.

{
  "smell_type": "<copied from input>",
  "target_level": "package",
  "target": "<target_name from input>",
  "selected_cluster": [
    "<class FQN>"
  ],
  "destination_package": "<new subpackage FQN>",
  "cluster_reason": "<why this cluster is cohesive and relatively safe>",
  "risk": "low|medium|high",
  "blocks": [
    {
      "id": 1,
      "goal": "Move one class from the selected cluster.",
      "files": [
        "<source file path from target_files>"
      ],
      "ops": [
        {
          "op": "CREATE_PACKAGE",
          "inputs": [],
          "outputs": ["<destination package FQN>"],
          "details": "Create the destination subpackage if it does not exist.",
          "risk": "low",
          "api_change": false
        },
        {
          "op": "MOVE_CLASS",
          "inputs": ["<old class FQN>"],
          "outputs": ["<new class FQN>"],
          "details": "Move this class to the destination package.",
          "risk": "low|medium|high",
          "api_change": true
        }
      ]
    }
  ]
}

# INPUT
```json
{
  "smell": "God Component",
  "target_type": "package",
  "target_name": "org.jsoup.parser",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/new/data/runs/20260616_105452_ab1479a5/planner/designite",
    "smells_csv": "ArchitectureSmells.csv",
    "target_has_smell": true
  },
  "target_source_root": "src/main/java",
  "target_files": [
    "src/main/java/org/jsoup/parser/CharacterReader.java",
    "src/main/java/org/jsoup/parser/HtmlTagOptions.java",
    "src/main/java/org/jsoup/parser/HtmlTreeBuilder.java",
    "src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java",
    "src/main/java/org/jsoup/parser/ParseError.java",
    "src/main/java/org/jsoup/parser/ParseErrorList.java",
    "src/main/java/org/jsoup/parser/ParseSettings.java",
    "src/main/java/org/jsoup/parser/Parser.java",
    "src/main/java/org/jsoup/parser/StreamParser.java",
    "src/main/java/org/jsoup/parser/Tag.java",
    "src/main/java/org/jsoup/parser/TagSet.java",
    "src/main/java/org/jsoup/parser/Token.java",
    "src/main/java/org/jsoup/parser/TokenData.java",
    "src/main/java/org/jsoup/parser/TokenQueue.java",
    "src/main/java/org/jsoup/parser/Tokeniser.java",
    "src/main/java/org/jsoup/parser/TokeniserState.java",
    "src/main/java/org/jsoup/parser/TreeBuilder.java",
    "src/main/java/org/jsoup/parser/XmlTreeBuilder.java",
    "src/main/java/org/jsoup/parser/package-info.java"
  ],
  "internal_deps": [
    [
      "org.jsoup.parser.TokenQueue",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.Tokeniser"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.TagSet"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.ParseErrorList"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.parser.ParseError",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.TreeBuilder"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Tokeniser"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.parser.TagSet"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.HtmlTreeBuilderState"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.HtmlTagOptions"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.TreeBuilder"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.ParseErrorList"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.TagSet"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.HtmlTreeBuilder"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.XmlTreeBuilder"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.Tokeniser"
    ],
    [
      "org.jsoup.parser.ParserIT",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.ParseErrorList"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.TreeBuilder"
    ],
    [
      "org.jsoup.parser.TokeniserState",
      "org.jsoup.parser.Tokeniser"
    ],
    [
      "org.jsoup.parser.TokeniserState",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.parser.TokeniserState",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.parser.TokeniserState",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.HtmlTreeBuilder"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.TreeBuilder"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.TagSet",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.parser.TagSet",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.parser.TreeBuilder"
    ],
    [
      "org.jsoup.parser.Tag",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.Tag",
      "org.jsoup.parser.HtmlTagOptions"
    ],
    [
      "org.jsoup.parser.Tag",
      "org.jsoup.parser.TagSet"
    ],
    [
      "org.jsoup.parser.Tag",
      "org.jsoup.parser.TokeniserState"
    ]
  ],
  "incoming_deps": [
    [
      "org.jsoup.nodes.FormElement",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.nodes.Document",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.nodes.Document",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.parser.Token.Doctype",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.nodes.PseudoTextElement",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.select.Evaluator.MatchText",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.helper.CookieUtil",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.parser.Token.Tag",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.parser.Token.Tag",
      "org.jsoup.parser.TreeBuilder"
    ],
    [
      "org.jsoup.parser.Token.Tag",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.Token.Tag",
      "org.jsoup.parser.Tokeniser"
    ],
    [
      "org.jsoup.select.QueryParserAttributeHelper",
      "org.jsoup.parser.TokenQueue"
    ],
    [
      "org.jsoup.helper.W3CDom",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.Token.Character",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.parser.Token.Character",
      "org.jsoup.parser.TokeniserState"
    ],
    [
      "org.jsoup.parser.Token.Character",
      "org.jsoup.parser.Tokeniser"
    ],
    [
      "org.jsoup.nodes.Printer.Pretty",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.nodes.Node",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.helper.HttpConnection",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.helper.W3CDom.W3CBuilder",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.nodes.NodeUtils",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.helper.HttpConnection.Request",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.select.Evaluator",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.Connection",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.select.QueryParser",
      "org.jsoup.parser.TokenQueue"
    ],
    [
      "org.jsoup.safety.Cleaner",
      "org.jsoup.parser.ParseErrorList"
    ],
    [
      "org.jsoup.safety.Cleaner",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.Token.EOF",
      "org.jsoup.parser.Token"
    ],
    [
      "org.jsoup.nodes.Element.TextAccumulator",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.nodes.Comment",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.JsoupCleaner",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.parser.Token.Comment",
      "org.jsoup.parser.TokenData"
    ],
    [
      "org.jsoup.nodes.Element",
      "org.jsoup.parser.Tag"
    ],
    [
      "org.jsoup.nodes.Element",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.nodes.Element",
      "org.jsoup.parser.TokenQueue"
    ],
    [
      "org.jsoup.nodes.Element",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.select.Selector",
      "org.jsoup.parser.TokenQueue"
    ],
    [
      "org.jsoup.helper.DataUtil",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.helper.DataUtil",
      "org.jsoup.parser.StreamParser"
    ],
    [
      "org.jsoup.Jsoup",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.integration.ParserSoakIT",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.nodes.Attributes",
      "org.jsoup.parser.ParseSettings"
    ],
    [
      "org.jsoup.nodes.Entities",
      "org.jsoup.parser.Parser"
    ],
    [
      "org.jsoup.nodes.Entities",
      "org.jsoup.parser.CharacterReader"
    ],
    [
      "org.jsoup.nodes.Printer",
      "org.jsoup.parser.Tag"
    ]
  ],
  "outgoing_deps": [
    [
      "org.jsoup.parser.TokenQueue",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.TokenQueue",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.ParseSettings",
      "org.jsoup.internal.Normalizer"
    ],
    [
      "org.jsoup.parser.ParseSettings",
      "org.jsoup.nodes.Attributes"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.select.NodeVisitor"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.internal.LineMap"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.nodes.Attributes"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.nodes.Node"
    ],
    [
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.nodes.NodeInternals"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.TokenType"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.nodes.Attributes"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.Doctype"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.StartTag"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.EndTag"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.Comment"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.Character"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token.XmlDecl"
    ],
    [
      "org.jsoup.parser.Token",
      "org.jsoup.nodes.Range"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.select.Elements"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.Attributes"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.LeafNode"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.Comment"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.CDataNode"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.DataNode"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.TextNode"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.DocumentType"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.XmlDeclaration"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.nodes.Entities"
    ],
    [
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.internal.SharedConstants"
    ],
    [
      "org.jsoup.parser.TokenData",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.FormElement"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.internal.Normalizer"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.Attributes"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.Comment"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.Node"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.CDataNode"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.DataNode"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.TextNode"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.Parser",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.ParserIT",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.ParserIT",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.HtmlTagOptions",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.nodes.Entities"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.TokeniserState",
      "org.jsoup.nodes.DocumentType"
    ],
    [
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.internal.SoftPool"
    ],
    [
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.internal.LineMap"
    ],
    [
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.parser.CharacterReader.CharPredicate"
    ],
    [
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.nodes.DocumentType"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.internal.StringUtil"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.nodes.Attributes"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.HtmlTreeBuilderState.Constants"
    ],
    [
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.nodes.Range"
    ],
    [
      "org.jsoup.parser.TagSet",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.TagSet",
      "org.jsoup.internal.SharedConstants"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.parser.StreamParser.ElementIterator"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.nodes.Document"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.nodes.Element"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.helper.Validate"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.select.Selector"
    ],
    [
      "org.jsoup.parser.StreamParser",
      "org.jsoup.select.Evaluator"
    ]
  ]
}
```