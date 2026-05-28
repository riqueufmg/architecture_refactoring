# GOAL

Remove the architecture smell: Insufficient Modularization from the TARGET CLASS.

For this MVP, prefer API-preserving refactorings.

Allowed strategy:
- Extract cohesive private methods and helper logic into new classes.
- Move private helper methods when this reduces the complexity or size of the target class.
- Do not move public methods.
- Do not change public API.
- Do not require production call-site updates.
- Do not edit tests.

# ALLOWED OPS

# ALLOWED OPS

EXTRACT_CLASS, EXTRACT_INTERFACE, MOVE_METHOD, MOVE_FIELD, INTRODUCE_FACADE, DEPENDENCY_INVERSION, REPLACE_DEPENDENCY, CREATE_PACKAGE, MOVE_CLASS, ADD_OR_UPDATE_IMPORTS, UPDATE_CALL_SITES

Use NO_OP only when no safe refactoring plan can be produced at all.
Do not include NO_OP inside a block that already contains real refactoring operations.
Do not use NO_OP as a placeholder, filler, or block-size adjustment.

# CONSTRAINTS

1. Reference only packages, classes, methods, and fields present in the input.
2. Each block must be small and independently compilable.
3. New classes must be placed in the same package as the target class unless the input clearly justifies otherwise.
4. Do not create compatibility wrappers.
5. Keep blocks small: 3–8 ops per block.

# INPUT

{
  "smell": "God Component",
  "target_type": "package",
  "target_name": "org.apache.commons.lang3.function",
  "designite": {
    "dir": "/data/henrique/langchain_prototype/ArchAgent/new/data/runs/20260528_143406_0d9629df/planner/designite",
    "smells_csv": "ArchitectureSmells.csv",
    "target_has_smell": true
  },
  "target_source_root": "src/main/java",
  "target_files": [
    "src/main/java/org/apache/commons/lang3/function/BooleanConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/ByteConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/ByteSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/Consumers.java",
    "src/main/java/org/apache/commons/lang3/function/Failable.java",
    "src/main/java/org/apache/commons/lang3/function/FailableBiConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableBiFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableBiPredicate.java",
    "src/main/java/org/apache/commons/lang3/function/FailableBooleanSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableByteConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableByteSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableCallable.java",
    "src/main/java/org/apache/commons/lang3/function/FailableConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleBinaryOperator.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoublePredicate.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleToIntFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleToLongFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableDoubleUnaryOperator.java",
    "src/main/java/org/apache/commons/lang3/function/FailableFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntBinaryOperator.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntPredicate.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntToDoubleFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntToFloatFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntToLongFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableIntUnaryOperator.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongBinaryOperator.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongPredicate.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongToDoubleFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongToIntFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableLongUnaryOperator.java",
    "src/main/java/org/apache/commons/lang3/function/FailableObjDoubleConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableObjIntConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailableObjLongConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/FailablePredicate.java",
    "src/main/java/org/apache/commons/lang3/function/FailableRunnable.java",
    "src/main/java/org/apache/commons/lang3/function/FailableShortSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableSupplier.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToBooleanFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToDoubleBiFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToDoubleFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToIntBiFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToIntFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToLongBiFunction.java",
    "src/main/java/org/apache/commons/lang3/function/FailableToLongFunction.java",
    "src/main/java/org/apache/commons/lang3/function/Functions.java",
    "src/main/java/org/apache/commons/lang3/function/IntToCharFunction.java",
    "src/main/java/org/apache/commons/lang3/function/MethodInvokers.java",
    "src/main/java/org/apache/commons/lang3/function/Predicates.java",
    "src/main/java/org/apache/commons/lang3/function/Suppliers.java",
    "src/main/java/org/apache/commons/lang3/function/ToBooleanBiFunction.java",
    "src/main/java/org/apache/commons/lang3/function/TriConsumer.java",
    "src/main/java/org/apache/commons/lang3/function/TriFunction.java",
    "src/main/java/org/apache/commons/lang3/function/package-info.java"
  ],
  "internal_deps": [
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableBiConsumer"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableDoubleConsumer"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableIntConsumer"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableLongConsumer"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableBiFunction"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableFunction"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableDoubleBinaryOperator"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableBiPredicate"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableCallable"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailablePredicate"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableRunnable"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableBooleanSupplier"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableDoubleSupplier"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableIntSupplier"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableLongSupplier"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.function.FailableShortSupplier"
    ],
    [
      "org.apache.commons.lang3.function.FailableBiFunction",
      "org.apache.commons.lang3.function.FailableFunction"
    ],
    [
      "org.apache.commons.lang3.function.MethodInvokers",
      "org.apache.commons.lang3.function.FailableBiConsumer"
    ],
    [
      "org.apache.commons.lang3.function.MethodInvokers",
      "org.apache.commons.lang3.function.FailableBiFunction"
    ],
    [
      "org.apache.commons.lang3.function.MethodInvokers",
      "org.apache.commons.lang3.function.FailableFunction"
    ],
    [
      "org.apache.commons.lang3.function.MethodInvokers",
      "org.apache.commons.lang3.function.FailableSupplier"
    ]
  ],
  "incoming_deps": [
    [
      "org.apache.commons.lang3.tuple.Pair",
      "org.apache.commons.lang3.function.FailableBiConsumer"
    ],
    [
      "org.apache.commons.lang3.tuple.Pair",
      "org.apache.commons.lang3.function.FailableBiFunction"
    ],
    [
      "org.apache.commons.lang3.concurrent.AtomicInitializer",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.concurrent.AtomicInitializer",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.time.DurationUtils",
      "org.apache.commons.lang3.function.FailableBiConsumer"
    ],
    [
      "org.apache.commons.lang3.time.DurationUtils",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.time.DurationUtils",
      "org.apache.commons.lang3.function.FailableRunnable"
    ],
    [
      "org.apache.commons.lang3.concurrent.LazyInitializer",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.concurrent.LazyInitializer",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.Functions",
      "org.apache.commons.lang3.function.FailableBooleanSupplier"
    ],
    [
      "org.apache.commons.lang3.Functions",
      "org.apache.commons.lang3.function.FailableRunnable"
    ],
    [
      "org.apache.commons.lang3.Functions",
      "org.apache.commons.lang3.function.Failable"
    ],
    [
      "org.apache.commons.lang3.ThreadUtils",
      "org.apache.commons.lang3.function.Predicates"
    ],
    [
      "org.apache.commons.lang3.ArrayFill",
      "org.apache.commons.lang3.function.FailableIntFunction"
    ],
    [
      "org.apache.commons.lang3.AutoCloseables",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.AutoCloseables",
      "org.apache.commons.lang3.function.Consumers"
    ],
    [
      "org.apache.commons.lang3.event.EventListenerSupport",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.ArrayUtils",
      "org.apache.commons.lang3.function.FailableFunction"
    ],
    [
      "org.apache.commons.lang3.concurrent.AtomicSafeInitializer",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.concurrent.AtomicSafeInitializer",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.Strings",
      "org.apache.commons.lang3.function.ToBooleanBiFunction"
    ],
    [
      "org.apache.commons.lang3.SystemProperties",
      "org.apache.commons.lang3.function.Suppliers"
    ],
    [
      "org.apache.commons.lang3.ObjectUtils",
      "org.apache.commons.lang3.function.Suppliers"
    ],
    [
      "org.apache.commons.lang3.StringUtils",
      "org.apache.commons.lang3.function.Suppliers"
    ],
    [
      "org.apache.commons.lang3.concurrent.AbstractConcurrentInitializer",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.concurrent.AbstractConcurrentInitializer",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.concurrent.AbstractConcurrentInitializer.AbstractBuilder",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.concurrent.AbstractConcurrentInitializer.AbstractBuilder",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.event.EventListenerSupport.ProxyInvocationHandler",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.AppendableJoiner.Builder",
      "org.apache.commons.lang3.function.FailableBiConsumer"
    ],
    [
      "org.apache.commons.lang3.AppendableJoiner",
      "org.apache.commons.lang3.function.FailableBiConsumer"
    ],
    [
      "org.apache.commons.lang3.concurrent.BackgroundInitializer",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.concurrent.BackgroundInitializer",
      "org.apache.commons.lang3.function.FailableConsumer"
    ],
    [
      "org.apache.commons.lang3.time.StopWatch",
      "org.apache.commons.lang3.function.FailableSupplier"
    ],
    [
      "org.apache.commons.lang3.time.StopWatch",
      "org.apache.commons.lang3.function.FailableRunnable"
    ]
  ],
  "outgoing_deps": [
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.exception.ExceptionUtils"
    ],
    [
      "org.apache.commons.lang3.function.Failable",
      "org.apache.commons.lang3.stream.Streams"
    ]
  ]
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
      "files": ["..."],
      "ops": [
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