The target is a Java package affected by God Component.

Focus on reducing excessive concentration of classes and responsibilities in the package.

Possible strategies:

- identify cohesive groups of classes;
- create meaningful subpackages;
- move classes to subpackages;
- update imports and package declarations;
- preserve public APIs where possible;
- fix visibility issues caused by package moves.

Avoid:

- moving classes randomly;
- creating artificial packages with no conceptual cohesion;
- editing build configuration to hide errors;
- deleting classes or tests to reduce package size.
