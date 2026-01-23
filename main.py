import os
from dotenv import load_dotenv
from src.utils.prompt import Prompt
from pathlib import Path
from src.utils.metrics_deps_parser import MetricsDepsParser
from src.utils.filter_smells import FilterSmells

def main():
    load_dotenv()

    projects_list = [
        {
            "project_name": "byte-buddy",
            "classes_path": "data/repositories/byte-buddy/byte-buddy-agent/target;data/repositories/byte-buddy/byte-buddy-android/target;data/repositories/byte-buddy/byte-buddy-android-test/target;data/repositories/byte-buddy/byte-buddy-benchmark/target;data/repositories/byte-buddy/byte-buddy-dep/target;data/repositories/byte-buddy/byte-buddy-gradle-plugin/target;data/repositories/byte-buddy/byte-buddy-maven-plugin/target"
        },
        {
            "project_name": "commons-io",
            "classes_path": "data/repositories/commons-io/target"
        },
        {
            "project_name": "commons-lang",
            "classes_path": "data/repositories/commons-lang/target"
        },
        {
            "project_name": "google-java-format",
            "classes_path": "data/repositories/google-java-format/core/target"
        },
        {
            "project_name": "gson",
            "classes_path": "data/repositories/gson/metrics/target;data/repositories/gson/test-jpms/target;data/repositories/gson/test-graal-native-image/target;data/repositories/gson/gson/target;data/repositories/gson/extras/target;data/repositories/gson/test-shrinker/target;data/repositories/gson/target;data/repositories/gson/proto/target"
        },
        {
            "project_name": "javaparser",
            "classes_path": "data/repositories/javaparser/javaparser-core/target;data/repositories/javaparser/javaparser-core-generators/target;data/repositories/javaparser/javaparser-core-metamodel-generator/target;data/repositories/javaparser/javaparser-core-serialization/target;data/repositories/javaparser/javaparser-core-testing/target;data/repositories/javaparser/javaparser-core-testing-bdd/target;data/repositories/javaparser/javaparser-symbol-solver-core/target;data/repositories/javaparser/javaparser-symbol-solver-testing/target;"
        },
        {
            "project_name": "jimfs",
            "classes_path": "data/repositories/jimfs/jimfs/target"
        },
        {
            "project_name": "jitwatch",
            "classes_path": "data/repositories/jitwatch/core/target;data/repositories/jitwatch/ui/target"
        },
        {
            "project_name": "jsoup",
            "classes_path": "data/repositories/jsoup/target/classes"
        },
        {
            "project_name": "zxing",
            "classes_path": "data/repositories/zxing/target;data/repositories/zxing/core/target;data/repositories/zxing/javase/target"
        },
    ]

    for project_data in projects_list:
        #mdp = MetricsDepsParser(project_data['project_name'])
        #mdp.collect_metrics()

        fs = FilterSmells(project_data['project_name'])
        # pacote-level
        fs.filter_by_smell("God Component")
        fs.filter_by_smell("Unstable Dependency")

        # classe-level
        fs.filter_by_smell("Insufficient Modularization")
        fs.filter_by_smell("Hub-like Modularization")

    '''prompt_manager = Prompt()

    file = open(metrics_dir / "ResultHandler.json", "r")
    input = file.read()
    file.close()

    prompt = prompt_manager.load_custom_prompt("agent_planner", input)

    file = open(output_dir / "Jsoup_output.prompt", "w")
    file.write(prompt)
    file.close()'''

if __name__ == "__main__":
    main()