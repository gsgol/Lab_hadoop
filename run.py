import os
import time
import json
import subprocess

def upload_data():
    subprocess.run(["docker", "exec", "namenode", "hdfs", "dfs", "-mkdir", "-p", "/input"])
    subprocess.run(["docker", "exec", "namenode", "hdfs", "dfs", "-put", "-f", "/Online_Retail_Sample_100k.csv", "/input/"])

def execute_job(optimization=False):
    command = [
        "docker", "exec", "spark",
        "spark-submit", "--master", "spark://spark:7077",
        "/app/Spark_application.py"
    ]
    if optimization:
        command.extend(["--optimized"])
    start = time.time()
    subprocess.run(command)
    return round(time.time() - start, 2)

def prepare_workspace():
    if not os.path.exists("results"):
        os.makedirs("results")

def measure_usage(container="spark"):
    output = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}:{{.MemUsage}}"],
        capture_output=True, text=True
    )
    for line in output.stdout.strip().split("\n"):
        if container in line:
            return line.split(":")[1].split("/")[0].strip()
    return "0MiB"

def manage_cluster(config, tag):
    subprocess.run(["docker-compose", "-f", config, "up", "-d"])
    time.sleep(60)
    upload_data()
    prepare_workspace()

    normal_time = execute_job()
    normal_mem = measure_usage()
    with open(f"results/stats_{tag}.json", "w") as file:
        json.dump({
            "setup": tag,
            "tuning": False,
            "duration": normal_time,
            "memory": normal_mem,
            "timestamp": time.ctime()
        }, file, indent=2)

    optimized_time = execute_job(True)
    optimized_mem = measure_usage()
    with open(f"results/stats_{tag}_tuned.json", "w") as file:
        json.dump({
            "setup": tag,
            "tuning": True,
            "duration": optimized_time,
            "memory": optimized_mem,
            "timestamp": time.ctime()
        }, file, indent=2)

    subprocess.run(["docker-compose", "-f", config, "down"])

if __name__ == "__main__":
    manage_cluster("docker-compose.yaml", "SingleNode")
    manage_cluster("docker-compose_3dn.yaml", "MultiNode")