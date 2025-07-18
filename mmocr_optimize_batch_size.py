#!/usr/bin/env python3
# NOTE: doesn't work - constanly skips batch sizes because of fake OOM when in fact just cannot find file
# TODO: fix dict construction error:
# 📋 Traceback (most recent call last):
# 📋 raise_for_execution_errors(nb, output_path)
# 📋 File "/home/bonting/micromamba/envs/bonting-id/lib/python3.11/site-packages/papermill/execute.py", line 251, in raise_for_execution_errors
# 📋 raise error
# 📋 papermill.exceptions.PapermillExecutionError:
# 📋 KeyError                                  Traceback (most recent call last)
# 📋 24 # Optionally, smoke test on 1 epoch
# 📋 KeyError: 'configs/textrecog/svtr_custom/svtr_cegdr_dict-extend.py'

"""
Comprehensive batch size optimization for all MMOCR detection models using real-time monitoring.

Usage:
    python optimize_all_models.py --configs_folder configs/textdet/              # Optimize all models in folder
    python optimize_all_models.py --configs_folder configs/textdet/ dbnetpp      # Optimize single model
    python optimize_all_models.py --configs_folder configs/textdet/ textsnake    # Optimize single model

Features:
- Automatic config discovery from specified folder
- Parse initial batch size directly from config files
- Dynamic batch size range adjustment (2x smaller to 2x larger than default)
- Adaptive binary search with range expansion/contraction
- Real-time training output monitoring with regex pattern matching
- Immediate optimization when GPU utilization hits 90-99% target
- Automatic config file updates when optimal batch size is found
- Progress tracking with timestamps and emojis
- Stops immediately when optimal utilization is achieved
"""

assert False, "TODO: fix dict construction error (see file Notes)"

import subprocess
import time
import re
import shutil
import threading
import queue
import os
import glob
import argparse
from datetime import datetime
import yaml

# Constants
MAX_BATCH_SIZE = 768
TARGET_UTILIZATION = 0.9
MIN_UTILIZATION = 0.8

# Simplified training progress regex: capture only memory usage (MB)
TRAINING_PATTERN = re.compile(
    r'Epoch\(train\).*?memory:\s*(\d+)',
    re.IGNORECASE
)

def discover_configs(configs_folder):
    """Automatically discover config files in the specified folder"""
    if not os.path.exists(configs_folder):
        raise ValueError(f"Config folder does not exist: {configs_folder}")
    
    # Look for _base_*.py files which contain batch size configurations
    base_config_pattern = os.path.join(configs_folder, "**", "_base_*.py")
    base_configs = glob.glob(base_config_pattern, recursive=True)
    
    if not base_configs:
        raise ValueError(f"No _base_*.py config files found in {configs_folder}")
    
    models = {}
    
    for base_config in base_configs:
        # Extract model name from path
        # e.g., configs/textdet/dbnetpp_custom/_base_dbnetpp_cegdr.py -> dbnetpp
        path_parts = base_config.replace(configs_folder, "").strip("/").split("/")
        
        # Try to extract model name from folder name or filename
        model_name = None
        for part in path_parts:
            if "_custom" in part:
                model_name = part.replace("_custom", "")
                break
        
        if not model_name:
            # Extract from filename
            filename = os.path.basename(base_config)
            match = re.search(r'_base_(\w+)_', filename)
            if match:
                model_name = match.group(1)
        
        if not model_name:
            print(f"Warning: Could not determine model name for {base_config}, skipping")
            continue
        
        # Find corresponding main config file
        config_dir = os.path.dirname(base_config)
        main_config_pattern = os.path.join(config_dir, f"{model_name}_*.py")
        main_configs = glob.glob(main_config_pattern)
        
        # Filter out _base_ files
        main_configs = [f for f in main_configs if "_base_" not in f]
        
        if not main_configs:
            print(f"Warning: No main config file found for {model_name}, skipping")
            continue
        
        main_config = main_configs[0]  # Take the first one
        
        # Convert to relative path for config_line
        config_line = os.path.relpath(main_config, ".").replace("\\", "/")
        
        models[model_name] = {
            "config_path": base_config,
            "config_line": config_line,
            "initial_batch_size": None,  # Will be parsed from config
            "min_batch_size": None,      # Will be set based on initial
            "max_batch_size": None       # Will be set based on initial
        }
    
    print(f"Discovered {len(models)} models:")
    for name, config in models.items():
        print(f"  - {name}: {config['config_path']}")
    
    return models

def parse_batch_size_from_config(config_path):
    """Parse initial batch size from config file"""
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Look for train_dataloader batch_size
        match = re.search(r'batch_size\s*=\s*(\d+)', content)
        if match:
            return int(match.group(1))
        
        # Alternative patterns
        match = re.search(r'samples_per_gpu\s*=\s*(\d+)', content)
        if match:
            return int(match.group(1))
        
        print(f"Warning: Could not find batch_size in {config_path}, using default 8")
        return 8
        
    except Exception as e:
        print(f"Error parsing batch size from {config_path}: {e}")
        return 8

def setup_model_configs(models):
    """Setup batch size ranges for discovered models"""
    for name, config in models.items():
        # Parse initial batch size from config
        initial_batch = parse_batch_size_from_config(config["config_path"])
        config["initial_batch_size"] = initial_batch
        
        # Set dynamic ranges: 2x smaller to 2x larger than default
        config["min_batch_size"] = max(1, initial_batch // 2)
        config["max_batch_size"] = min(MAX_BATCH_SIZE, initial_batch * 2)
        
        print(f"Model {name}: initial={initial_batch}, range=[{config['min_batch_size']}-{config['max_batch_size']}]")

def create_config_list_file(models, config_list_file):
    """Create config list file from discovered models"""
    os.makedirs(os.path.dirname(config_list_file), exist_ok=True)
    
    with open(config_list_file, 'w') as f:
        f.write("# MMOCR Detection Model Configuration List\n")
        f.write("# Auto-discovered from configs folder\n\n")
        
        for name, config in models.items():
            f.write(f"# {name.upper()} models\n")
            f.write(f"- {config['config_line']}\n\n")

def run_command(cmd, print_output=True):
    """Run a shell command and return output"""
    if print_output:
        print(f"Executing: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if print_output and result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if print_output and result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode, result.stdout, result.stderr

def get_gpu_utilization():
    """Get current GPU memory utilization ratio"""
    cmd = "nvidia-smi | grep -oP '\\d+MiB\\s*/\\s*\\d+MiB' | grep -oP '\\d+' | tr '\\n' ' ' | awk '{if ($2>0) printf \"%.6f\\n\", $1/$2; else print \"0\"}'"
    returncode, stdout, stderr = run_command(cmd, print_output=False)
    if returncode == 0 and stdout.strip():
        return float(stdout.strip())
    return 0.0

def update_model_batch_size(model_name, batch_size, models):
    """Update batch size in a specific model config"""
    config_path = models[model_name]["config_path"]
    
    # Backup original
    backup_path = f"{config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(config_path, backup_path)
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update train_dataloader batch_size
        content = re.sub(r'(\s*batch_size=)\d+', f'\\g<1>{batch_size}', content)
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"Updated {model_name} batch size to {batch_size}")
        return backup_path
    except Exception as e:
        print(f"Error updating batch size for {model_name}: {e}")
        return None

def update_config_list_for_model(model_name, models, config_list_file):
    """Update config list to only enable specific model"""
    backup_file = f"{config_list_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(config_list_file, backup_file)
    
    with open(config_list_file, 'w') as f:
        f.write("# MMOCR Detection Model Configuration List\n")
        f.write("# Uncomment the models you want to train\n\n")
        
        for name, config in models.items():
            f.write(f"# {name.upper()} models\n")
            if name == model_name:
                f.write(f"- {config['config_line']}\n\n")
            else:
                f.write(f"# - {config['config_line']}\n\n")
    
    return backup_file

def monitor_training_output(process, output_queue, stop_event):
    """Monitor training process output in a separate thread"""
    try:
        for line in iter(process.stdout.readline, b''):
            line_str = line.decode('utf-8').strip()
            output_queue.put(line_str)
            if stop_event.is_set():
                break
    except Exception as e:
        output_queue.put(f"ERROR: {e}")
    finally:
        process.stdout.close()

def test_batch_size_for_model(model_name, batch_size, models, config_list_file):
    """Test a batch size for a specific model with real-time monitoring"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} with batch size: {batch_size}")
    print(f"{'='*60}")

    # ---------------- GPU CLEANUP ---------------------
    print(f"🧹 Cleaning GPU memory before test...")
    run_command("fuser -k /dev/nvidia* 2>/dev/null || true", print_output=False)
    time.sleep(3)  # short pause for cleanup
    print(f"GPU utilization after cleanup: {get_gpu_utilization():.4f}")
    # --------------------------------------------------
    
    # Update config
    config_backup = update_model_batch_size(model_name, batch_size, models)
    if not config_backup:
        return False, 0.0
    
    # Run training command
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cmd = [
        'papermill', 'mmocr_det_cegdr_finetune_pretrained.ipynb',
        '/dev/null',
        '--kernel', 'python3', '--log-output',
        '-p', 'SMOKE_TEST', 'True',
        '-p', 'NUM_MODELS', '1',
        '-p', 'CONFIG_LIST', config_list_file
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training with real-time monitoring...")
    print(f"Command: {' '.join(cmd)}")
    
    # Start process
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=False,
            bufsize=1
        )
    except Exception as e:
        print(f"✗ Failed to start process: {e}")
        return False, 0.0
    
    # Setup monitoring
    output_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_training_output, 
        args=(process, output_queue, stop_event)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Monitor for training progress
    start_time = time.time()
    max_wait_time = 300  # 5 minutes max wait
    training_started = False
    oom_detected = False
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring output for training progress...")
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check timeout
        if elapsed > max_wait_time:
            print(f"⏱ Timeout after {elapsed:.1f} seconds")
            break
        
        # Check if process ended
        if process.poll() is not None:
            print(f"🔚 Process ended with return code: {process.returncode}")
            break
        
        # Check output queue
        try:
            line = output_queue.get(timeout=1)
            
            # Print line for debugging
            if any(keyword in line.lower() for keyword in ['error', 'traceback', 'failed', 'epoch']):
                print(f"📋 {line}")
            
            # Check for OOM
            if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                print(f"💥 OOM detected: {line}")
                oom_detected = True
                break
            
            # Check for training progress
            match = TRAINING_PATTERN.search(line)
            if match:
                memory_usage = int(match.group(1))
                print(f"🎯 TRAINING STARTED! Memory usage: {memory_usage}MB")
                training_started = True
                break
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Error monitoring output: {e}")
            break
    
    # Cleanup
    stop_event.set()
    if process.poll() is None:
        print(f"🛑 Terminating process...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=2)
    
    # Evaluate results
    if oom_detected:
        print(f"✗ Out of memory error detected")
        return False, 0.0
    elif training_started:
        # Wait a moment for GPU usage to stabilize
        time.sleep(5)
        gpu_util = get_gpu_utilization()
        print(f"✓ Training started successfully! GPU utilization: {gpu_util:.4f} ({gpu_util*100:.1f}%)")
        return True, gpu_util
    else:
        print(f"? Training status unclear after {elapsed:.1f} seconds")
        return False, 0.0

def adaptive_binary_search_optimal_batch_size(model_name, models, config_list_file):
    """Use adaptive binary search to find optimal batch size with range expansion/contraction"""
    model_config = models[model_name]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting adaptive binary search for {model_name.upper()}")
    print(f"🎯 Target: GPU utilization between 90-99%")
    
    # Store all tested results
    all_results = {}
    
    # Initial range
    min_batch = model_config["min_batch_size"]
    max_batch = model_config["max_batch_size"]
    
    optimal_batch = None
    optimal_util = 0.0
    range_iteration = 0
    
    while True:
        range_iteration += 1
        print(f"\n🔍 Range iteration {range_iteration}: testing range [{min_batch}, {max_batch}]")
        
        # Binary search within current range
        left, right = min_batch, max_batch
        binary_iteration = 0
        range_results = {}
        all_oom = True
        all_low_util = True
        
        while left <= right:
            binary_iteration += 1
            mid = (left + right) // 2
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Binary search iteration {binary_iteration}: testing batch size {mid} (range: {left}-{right})")
            
            try:
                success, gpu_util = test_batch_size_for_model(model_name, mid, models, config_list_file)
                range_results[mid] = {"success": success, "gpu_util": gpu_util}
                all_results[mid] = {"success": success, "gpu_util": gpu_util}
                
                if success:
                    all_oom = False
                    print(f"✓ Batch size {mid} works! GPU utilization: {gpu_util:.1%}")
                    
                    # Check if GPU utilization is in optimal range [0.9, 1.0)
                    if 0.9 <= gpu_util < 1.0:
                        print(f"🎯 OPTIMAL BATCH SIZE FOUND: {mid} (GPU utilization: {gpu_util:.1%})")
                        optimal_batch = mid
                        optimal_util = gpu_util
                        
                        # Immediately update config with optimal batch size
                        print(f"💾 Updating config with optimal batch size {mid}...")
                        update_model_batch_size(model_name, mid, models)
                        return optimal_batch, optimal_util, all_results
                        
                    elif gpu_util < 0.9:
                        print(f"📈 GPU utilization too low ({gpu_util:.1%}), trying higher batch size")
                        # Try higher batch size
                        left = mid + 1
                        print(f"  → Trying higher batch sizes (new range: {left}-{right})")
                        
                    else:  # gpu_util >= 1.0 (over 100% - shouldn't happen but safety check)
                        print(f"📉 GPU utilization too high ({gpu_util:.1%}), trying lower batch size")
                        optimal_batch = mid  # This batch size works but is at the edge
                        optimal_util = gpu_util
                        # Try lower batch size to find something more stable
                        right = mid - 1
                        print(f"  → Trying lower batch sizes (new range: {left}-{right})")
                        
                else:
                    print(f"✗ Batch size {mid} failed (OOM), trying lower")
                    # Try lower batch size
                    right = mid - 1
                    print(f"  → Trying lower batch sizes (new range: {left}-{right})")
                    
            except KeyboardInterrupt:
                print("Binary search interrupted by user")
                return optimal_batch, optimal_util, all_results
            except Exception as e:
                print(f"Error testing batch size {mid}: {e}")
                all_results[mid] = {"success": False, "gpu_util": 0.0, "error": str(e)}
                # On error, try lower batch size
                right = mid - 1
        
        # Check if we found any working batch sizes
        working_batches = {k: v for k, v in range_results.items() if v["success"]}
        if working_batches:
            all_low_util = not any(v["gpu_util"] >= 0.9 for v in working_batches.values())
        
        print(f"\n📊 Range [{min_batch}, {max_batch}] completed after {binary_iteration} iterations")
        print(f"Working batch sizes: {list(working_batches.keys()) if working_batches else 'None'}")
        
        # Decide next range based on results
        if all_oom and min_batch > 1:
            # All batch sizes caused OOM - try smaller range
            new_upper = min_batch
            new_lower = max(1, min_batch // 4)
            print(f"🔻 All batch sizes caused OOM, trying smaller range: [{new_lower}, {new_upper}]")
            
            if new_lower >= new_upper:
                print("📉 Reached minimum possible range starting from 1")
                break
                
            min_batch, max_batch = new_lower, new_upper
            
        elif all_low_util and max_batch < MAX_BATCH_SIZE:
            # All batch sizes had low utilization - try larger range
            new_lower = max_batch
            new_upper = min(MAX_BATCH_SIZE, max_batch * 4)
            print(f"🔺 All batch sizes had low utilization, trying larger range: [{new_lower}, {new_upper}]")
            
            if new_lower >= new_upper or new_lower >= MAX_BATCH_SIZE:
                print(f"📈 Reached maximum possible range up to {MAX_BATCH_SIZE}")
                break
                
            min_batch, max_batch = new_lower, new_upper
            
        else:
            # Found some working batch sizes or reached limits
            break
    
    # If we haven't found optimal utilization, find the best working batch size
    if optimal_batch is None:
        successful_results = {k: v for k, v in all_results.items() if v["success"]}
        if successful_results:
            # Find batch size with highest utilization
            best_batch = max(successful_results.keys(), key=lambda x: successful_results[x]["gpu_util"])
            optimal_batch = best_batch
            optimal_util = successful_results[best_batch]["gpu_util"]
            print(f"🔧 Best working batch size found: {optimal_batch} (GPU utilization: {optimal_util:.1%})")
            
            # Update config with best batch size
            print(f"💾 Updating config with best batch size {optimal_batch}...")
            update_model_batch_size(model_name, optimal_batch, models)
        else:
            print("❌ No working batch size found")
    
    return optimal_batch, optimal_util, all_results

def optimize_model(model_name, models, config_list_file):
    """Optimize batch size for a specific model using adaptive binary search"""
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZING {model_name.upper()} - ADAPTIVE BINARY SEARCH")
    print(f"{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting optimization for {model_name.upper()}")
    
    model_config = models[model_name]
    print(f"Initial batch size: {model_config['initial_batch_size']}")
    print(f"Initial search range: {model_config['min_batch_size']} to {model_config['max_batch_size']}")
    
    # Update config list to enable only this model
    config_backup = update_config_list_for_model(model_name, models, config_list_file)
    print(f"Config list backed up to: {config_backup}")
    
    try:
        # Adaptive binary search to find optimal batch size
        optimal_batch, optimal_util, results = adaptive_binary_search_optimal_batch_size(model_name, models, config_list_file)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if optimal_batch:
            if 0.9 <= optimal_util < 1.0:
                print(f"\n🎯 OPTIMAL RESULT for {model_name.upper()}: {optimal_batch} (GPU utilization: {optimal_util:.1%})")
                print(f"✅ Config automatically updated with optimal batch size!")
            else:
                print(f"\n🔧 BEST RESULT for {model_name.upper()}: {optimal_batch} (GPU utilization: {optimal_util:.1%})")
                print(f"⚠️  Not in optimal range (90-99%) but best available")
        else:
            print(f"\n❌ No working batch size found for {model_name.upper()}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Optimization completed in {total_time:.1f} seconds")
        print(f"Total tests performed: {len(results)}")
        
        return optimal_batch, optimal_util, results
        
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted for {model_name}")
        return None, 0.0, {}
    except Exception as e:
        print(f"\nError optimizing {model_name}: {e}")
        return None, 0.0, {}

def main(configs_folder, single_model=None):
    """Main function to optimize all models"""
    print("MMOCR Detection Models Batch Size Optimization - ADAPTIVE BINARY SEARCH")
    print(f"🎯 Target GPU utilization: {TARGET_UTILIZATION:.1%} (90-99%)")
    print(f"📊 Current GPU utilization: {get_gpu_utilization():.4f}")
    
    # Discover models from configs folder
    try:
        models = discover_configs(configs_folder)
        setup_model_configs(models)
    except Exception as e:
        print(f"Error discovering configs: {e}")
        return
    import re
    
    # HACK: hardcoded config list files for now
    if re.search(r'det', configs_folder):
        config_list_file = "notebooks/configs/mmocr_det_model_list.yml"
    elif re.search(r'recog', configs_folder):
        config_list_file = "notebooks/configs/mmocr_recog_model_list.yml"
    else:
        raise ValueError(f"Unknown task: {configs_folder}")
    
    create_config_list_file(models, config_list_file)
    
    # Filter to single model if specified
    if single_model:
        if single_model not in models:
            print(f"Error: Model '{single_model}' not found in discovered models")
            print(f"Available models: {list(models.keys())}")
            return
        models = {single_model: models[single_model]}
    
    print("\nModels to optimize:")
    for name, config in models.items():
        initial_range = config['max_batch_size'] - config['min_batch_size'] + 1
        binary_tests = int(initial_range.bit_length()) + 3  # log2(range) + margin
        print(f"  - {name.upper()}: initial={config['initial_batch_size']}, range=[{config['min_batch_size']}-{config['max_batch_size']}], est. tests=~{binary_tests}")
    
    total_binary_tests = sum(int((config['max_batch_size'] - config['min_batch_size'] + 1).bit_length()) + 3 for config in models.values())
    print(f"\n🚀 Adaptive binary search: Expands/contracts ranges based on OOM and GPU utilization")
    print(f"⚡ Estimated initial tests: ~{total_binary_tests} (may vary with range adjustments)")
    
    # Store results
    optimization_results = {}
    
    # Optimize each model
    for model_name in models.keys():
        try:
            # Clean up GPU memory between models
            print(f"\nCleaning up GPU memory before next model...")
            run_command("fuser -k /dev/nvidia*", print_output=True)
            time.sleep(5)  # Wait for cleanup
            print(f"GPU utilization after cleanup: {get_gpu_utilization():.4f}")

            optimal_batch, optimal_util, test_results = optimize_model(model_name, models, config_list_file)
            optimization_results[model_name] = {
                "optimal_batch_size": optimal_batch,
                "optimal_utilization": optimal_util,
                "test_results": test_results
            }
            
            
        except Exception as e:
            print(f"Error optimizing {model_name}: {e}")
            optimization_results[model_name] = {
                "optimal_batch_size": None,
                "optimal_utilization": 0.0,
                "error": str(e)
            }
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    
    for model_name, result in optimization_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Initial batch size: {models[model_name]['initial_batch_size']}")
        if result["optimal_batch_size"]:
            print(f"  Optimal batch size: {result['optimal_batch_size']}")
            print(f"  GPU utilization: {result['optimal_utilization']:.1%}")
        else:
            print(f"  Optimization failed")
            if "error" in result:
                print(f"  Error: {result['error']}")
    
    # Create final config list with all optimized models
    print(f"\nCreating final optimized config list...")
    with open(config_list_file, 'w') as f:
        f.write("# MMOCR Detection Model Configuration List\n")
        f.write("# Optimized batch sizes - Adaptive binary search\n")
        f.write("# Config files already updated with optimal batch sizes\n\n")
        
        for name, config in models.items():
            f.write(f"# {name.upper()} models\n")
            f.write(f"- {config['config_line']}\n\n")
    
    print("✅ All models optimized! Individual config files updated with optimal batch sizes.")
    print("📋 Final config list created for training all models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize MMOCR detection model batch sizes')
    parser.add_argument('--configs_folder', required=True, help='Path to configs folder (configs/textdet/ or configs/textrecog/)')
    parser.add_argument('model_name', nargs='?', help='Optional: specific model to optimize')
    
    args = parser.parse_args()
    
    # Allow testing just one model via command line argument
    if args.model_name:
        print(f"Testing single model: {args.model_name.upper()}")
        main(args.configs_folder, args.model_name.lower())
    else:
        main(args.configs_folder) 