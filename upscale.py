import os
import time
import subprocess
import argparse
import sys

# Initialize argument parser
parser = argparse.ArgumentParser(description="CodeFormer upscaling with multiprocessing")
parser.add_argument("--frames_folder", help="Frames folder", required=True)
parser.add_argument("--threads", help="Parallel threads", default=4, type=int)
parser.add_argument("--upscale_factor", help="Upscale factor (1-4)", default=1, type=int)
parser.add_argument("--weight", help="CodeFormer weight (0-1)", default=0.85, type=float)
args = parser.parse_args()

frames_folder = os.path.normpath(args.frames_folder)  # Windows compatible
threads = args.threads
upscale_factor = args.upscale_factor
weight = args.weight


def launch_codeformer(input_folder, processes):
    """
    Launch CodeFormer with Windows path compatibility
    """
    # Windows-safe paths
    input_folder = os.path.normpath(input_folder)
    output_folder = os.path.normpath(os.path.join(input_folder, "upscaled"))
    
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # CodeFormer command with proper quoting for Windows
    codeformer_command = (
        f'python codeformer_inference_roop.py '
        f'-i "{input_folder}" '
        f'-o "{output_folder}" '
        f'-w {weight} '
        f'-s {upscale_factor} '
        f'--face_upsample '
        f'--rewrite'
    )
    
    print(f"[INFO] Launching CodeFormer: {os.path.basename(input_folder)}")
    
    try:
        # Suppress console window on Windows
        startupinfo = None
        if sys.platform.startswith('win'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        process = subprocess.Popen(
            codeformer_command, 
            shell=True,
            startupinfo=startupinfo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append((process, os.path.basename(input_folder)))
    except Exception as e:
        print(f"[ERROR] Failed to launch CodeFormer: {e}")


def main():
    """Main upscaling process with Windows compatibility"""
    
    # Validate input
    if not os.path.exists(frames_folder):
        print(f"[ERROR] Frames folder not found: {frames_folder}")
        sys.exit(1)
    
    processing_path = os.path.normpath(os.path.join(frames_folder, "processing"))
    
    if not os.path.exists(processing_path):
        print(f"[ERROR] Processing folder not found: {processing_path}")
        sys.exit(1)
    
    print(f"[INFO] Starting CodeFormer upscaling")
    print(f"[INFO] Processing folder: {processing_path}")
    print(f"[INFO] Upscale factor: {upscale_factor}x")
    print(f"[INFO] CodeFormer weight: {weight}")
    print(f"[INFO] Parallel threads: {threads}")
    
    processes = []
    
    try:
        while True:
            # Check status of running processes
            for process, folder_name in processes[:]:
                if process.poll() is not None:  # Process finished
                    print(f"[INFO] Completed: {folder_name}")
                    processes.remove((process, folder_name))
            
            # Launch new processes if room available
            if len(processes) < threads:
                launch_codeformer(processing_path, processes)
                time.sleep(2)  # Small delay between launches
            
            # Exit if all done
            if len(processes) == 0 and len(processes) < threads:
                break
            
            # Check interval
            time.sleep(5)
        
        print("[INFO] All CodeFormer processes completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[INFO] Upscaling interrupted by user")
        # Terminate remaining processes
        for process, folder_name in processes:
            if process.poll() is None:
                process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
