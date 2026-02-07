
import os
import json
import argparse
from pathlib import Path
from typing import Set, List, Dict

def get_file_size(path: Path) -> int:
    try:
        if path.exists():
            return path.stat().st_size
        return -1 # File not found
    except Exception:
        return -1

def validate_entry(entry: Dict, dataset_dir: Path) -> bool:
    """
    Validates a metadata entry.
    All referenced files must exist AND be > 0 bytes.
    """
    for key in ["source_path", "target_path", "mask_path"]:
        rel_path = entry.get(key)
        if not rel_path:
            continue # Should be fine if optional, but here all are required
            
        abs_path = dataset_dir / rel_path
        size = get_file_size(abs_path)
        
        if size <= 0:
            # File missing (size -1) or empty (size 0)
            return False
            
    return True

def get_all_files(root_dir: Path, subdirs: List[str]) -> Set[str]:
    """
    Recursively finds all files in the specified subdirectories relative to root_dir.
    Returns a set of relative paths (strings).
    """
    found_files = set()
    for subdir in subdirs:
        abs_subdir = root_dir / subdir
        if not abs_subdir.exists():
            continue
            
        for root, _, files in os.walk(abs_subdir):
            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(root_dir)
                found_files.add(str(rel_path))
    return found_files

def process_metadata(dataset_dir: Path, meta_rel_path: str, dry_run: bool) -> Set[str]:
    """
    Reads metadata, filters invalid entries, writes back cleaned metadata (if not dry_run).
    Returns a set of VALID referenced file paths.
    """
    meta_path = dataset_dir / meta_rel_path
    if not meta_path.exists():
        print(f"[WARN] Metadata file not found: {meta_path}")
        return set()
    
    valid_entries = []
    invalid_entries = []
    valid_paths = set()
    
    print(f"Processing metadata: {meta_path}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                if validate_entry(entry, dataset_dir):
                    valid_entries.append(entry)
                    valid_paths.add(entry.get("source_path"))
                    valid_paths.add(entry.get("target_path"))
                    valid_paths.add(entry.get("mask_path"))
                else:
                    invalid_entries.append((i, entry))
            except json.JSONDecodeError:
                print(f"[ERR] Failed to decode line {i} in {meta_path}")
                
    print(f"  Total lines: {len(valid_entries) + len(invalid_entries)}")
    print(f"  Valid entries: {len(valid_entries)}")
    print(f"  Invalid entries (missing/0kb files): {len(invalid_entries)}")
    
    if len(invalid_entries) > 0:
        print("  Example invalid entries:")
        for idx, entry in invalid_entries[:5]:
             print(f"    Line {idx}: {entry.get('target_path')} (likely 0kb or missing)")

    # Rewrite metadata if not dry run
    if not dry_run and len(invalid_entries) > 0:
        # Create backup
        backup_path = meta_path.with_suffix(".jsonl.bak")
        print(f"  Backing up original metadata to {backup_path}")
        import shutil
        shutil.copy2(meta_path, backup_path)
        
        print(f"  Overwriting {meta_path} with {len(valid_entries)} valid entries...")
        with open(meta_path, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write(json.dumps(entry) + "\n")
                
    elif dry_run and len(invalid_entries) > 0:
        print("  [DRY RUN] Would rewrite metadata to exclude invalid entries.")

    return {p for p in valid_paths if p}

def main():
    parser = argparse.ArgumentParser(description="Clean dataset by removing 0kb files and syncing metadata.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="If set, only simulate actions.")
    
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return

    print(f"--- Dataset Cleaning Tool ---")
    print(f"Target: {dataset_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DESTRUCTIVE EXECUTON'}")
    
    # 1. Process Metadata & Collect Valid Paths
    all_valid_paths = set()
    
    # Train
    all_valid_paths.update(process_metadata(dataset_dir, "train/metadata.jsonl", args.dry_run))
    # Validation
    all_valid_paths.update(process_metadata(dataset_dir, "validation/metadata.jsonl", args.dry_run))
    
    print(f"\nTotal valid file references allowed: {len(all_valid_paths)}")
    
    # 2. Scan Disk for Orphans (Unreferenced files)
    # Note: Valid files (referenced but 0kb) were excluded from all_valid_paths by process_metadata
    # So they will appear as orphans here and get deleted!
    check_subdirs = [
        "train/images", "train/masks", 
        "validation/images", "validation/masks"
    ]
    
    print("\nScanning disk for unreferenced files...")
    files_on_disk = get_all_files(dataset_dir, check_subdirs)
    print(f"Total files on disk: {len(files_on_disk)}")
    
    orphans = files_on_disk - all_valid_paths
    
    if not orphans:
        print("\nSuccess! Dataset is perfectly synchronized.")
    else:
        print(f"\nFound {len(orphans)} files to delete (orphans or corrupted/0kb).")
        sorted_orphans = sorted(list(orphans))
        
        if args.dry_run:
            print("[DRY RUN] Files that would be deleted:")
            for o in sorted_orphans[:20]:
                abs_p = dataset_dir / o
                sz = get_file_size(abs_p)
                status = "0KB" if sz == 0 else f"{sz} bytes"
                print(f"  [DELETE] {o} ({status})")
            if len(sorted_orphans) > 20:
                print(f"  ... and {len(sorted_orphans) - 20} more.")
        else:
            print(f"Deleting {len(orphans)} files...")
            count = 0 
            for o in sorted_orphans:
                abs_p = dataset_dir / o
                try:
                    if abs_p.exists():
                        os.remove(abs_p)
                        count += 1
                except Exception as e:
                    print(f"Failed to delete {o}: {e}")
            print(f"Deleted {count} files.")

if __name__ == "__main__":
    main()
