""" 
Docker operations 
"""
import subprocess
from typing import List, Dict, Optional, Tuple

def run_docker_image(
    image: str,
    command: Optional[List[str]] = None,
    volumes: Optional[Dict[str, str]] = None,
    workdir: Optional[str] = None,
    envs: Optional[Dict[str, str]] = None,
    platform: Optional[str] = None,
    detach: bool = False,
) -> Tuple[int, str, str]:
    """
    Run a docker image with the given command and options.
        e.g., docker run -v $PWD/NumberUtilsA.java:/diff/NumberUtilsA.java -v $PWD/NumberUtils.java:/diff/NumberUtils.java -p 4567:4567 gumtreediff/gumtree webdiff /diff/NumberUtilsA.java /diff/NumberUtils.java
    
    Returns (exit_code, stdout, stderr).
    """
    docker_cmd = ["docker", "run", "--rm"]

    if platform:
        # for our platform usage: 
        #   gumtree: --platform linux/amd64  
        docker_cmd += ["--platform", platform]

    if workdir:
        docker_cmd += ["-w", workdir]

    if envs:
        for key, val in envs.items():
            docker_cmd += ["-e", f"{key}={val}"]

    if volumes:
        # volume matching: can be either a directory path or file path 
        for host_path, container_path in volumes.items():
            docker_cmd += ["-v", f"{host_path}:{container_path}"]

    if detach:
        # run the container background or not (return container id)
        docker_cmd.append("-d")

    docker_cmd.append(image)

    if command:
        docker_cmd += command

    # Execution
    print (" ".join(docker_cmd))
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            check=False  # don't raise exception on non-zero exit
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return -1, "", "Docker command not found."
    except Exception as e:
        return -1, "", str(e)
