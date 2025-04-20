with open("requirements.txt", "w", encoding="utf-8") as f:
    import subprocess
    freeze = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    f.write(freeze)
