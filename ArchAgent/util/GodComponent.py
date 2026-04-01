import xml.etree.ElementTree as ET
from typing import List, Tuple

class GodComponent:
    def __init__(self, target_fqn: str):
        self.target_fqn = target_fqn

    def _get_package_dependencies(self, graphml_path: str) -> Tuple[List[str], List[str]]:
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

        tree = ET.parse(graphml_path)
        root = tree.getroot()

        dependencies = []

        target_package = self.target_fqn

        for edge in root.findall(".//g:edge", ns):
            source = edge.get("source")
            target = edge.get("target")

            if not source or not target:
                continue

            if self._get_package_name(target) == target_package or self._get_package_name(source) == target_package:
                dependencies.append((source, target))

        return dependencies

    def _get_package_name(self, fqn: str) -> str:
        return ".".join(
            f for f in fqn.split(".") if f and not f[0].isupper()
        )

target_fqn = "org.jsoup.integration"
file = "data/DependencyGraph.graphml"

gc = GodComponent(target_fqn)
dependencies = gc._get_package_dependencies(file)

print(f"Dependencies for {target_fqn}:")
for dep in dependencies:
    print(f"  {dep}")