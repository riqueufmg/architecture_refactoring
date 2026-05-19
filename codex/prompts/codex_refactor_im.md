The target is a Java class affected by Insufficient Modularization.

Focus on decomposing responsibilities inside the class.

Possible strategies:

- extract cohesive helper methods;
- extract cohesive helper classes;
- move nested responsibilities to new classes;
- reduce method count or class size when behavior-preserving;
- improve separation of concerns.

Avoid:

- cosmetic-only changes;
- renaming-only changes;
- deleting methods;
- moving code without reducing the modularization problem.
