import os
import subprocess

def run_command(command):
    print(f"\n Running: {command}\n")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise Exception(f"Failed: {command}")

def main():
    print("========== SPEECH SSL PIPELINE ==========")

  
    run_command("python scripts/train_duration_probe.py")

    
    run_command("python scripts/eval_duration_probe.py")

   
    run_command("python scripts/run_cka_analysis.py")

    
    run_command("python scripts/run_layerwise_cka.py")

   
    run_command("python scripts/representation_analysis.py")

   
    run_command("python scripts/visualize_embeddings.py")

    print("\n Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()