#!/usr/bin/env python3
"""
CogBench — Label Studio Setup Helper

Prints instructions for setting up Label Studio and provides utility
functions for importing tasks and creating annotator accounts.

Usage:
  python setup_label_studio.py              # Print setup instructions
  python setup_label_studio.py --export     # Export tasks JSON for import
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "data", "annotations", "label_studio_config.xml")
SAMPLE_FILE = os.path.join(BASE_DIR, "data", "annotations", "annotation_sample_600.json")
TRAINING_FILE = os.path.join(BASE_DIR, "data", "annotations", "training_set.json")
EXPORT_DIR = os.path.join(BASE_DIR, "data", "annotations", "label_studio_import")


def print_setup_instructions():
    """Print step-by-step Label Studio setup guide."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         CogBench — Label Studio Setup Guide                  ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: Start Label Studio
─────────────────────────────
  source {venv}/bin/activate
  label-studio start --port 8080

  Then open: http://<server-ip>:8080

STEP 2: Create Admin Account
─────────────────────────────
  On first launch, you'll be prompted to create an admin account.
  Use your own email/password.

STEP 3: Create Projects
─────────────────────────────
  Create TWO projects:
  a) "CogBench Training" — for the 36-question calibration round
  b) "CogBench Annotation" — for the 400-question main annotation

  For each project:
  1. Click "Create Project"
  2. Go to "Labeling Setup" → "Code" tab
  3. Paste the contents of:
     {config}
  4. Save

STEP 4: Import Tasks
─────────────────────────────
  Run this script with --export first:
    python setup_label_studio.py --export

  Then in Label Studio:
  a) Training project → Import → Upload "training_tasks.json"
  b) Annotation project → Import → Upload "annotation_tasks.json"

STEP 5: Create Annotator Accounts
─────────────────────────────
  In Label Studio Settings → Members:
  1. Invite each annotator by email
  2. Set role to "Annotator" (not Manager or Admin)
  3. They'll receive a sign-up link

  Annotators can only see tasks assigned to them and submit
  their own annotations. They cannot see other annotators' work.

STEP 6: Configure Overlap
─────────────────────────────
  In Project Settings → Annotation:
  - Set "Annotations per task" = number of annotators (4-5)
  - This ensures every question gets labeled by ALL annotators

STEP 7: Annotator Instructions
─────────────────────────────
  Share the annotation guidelines document with all annotators:
    {guidelines}

  Hotkeys:
    1-6 = Bloom's level (L1 Remember through L6 Create)
    h/m/l = Confidence (High/Medium/Low)
    Ctrl+Enter = Submit annotation

""".format(
        venv=os.path.join(BASE_DIR, "venv"),
        config=CONFIG_FILE,
        guidelines=os.path.join(BASE_DIR, "data", "annotations", "annotation_guidelines.md"),
    ))


def export_tasks_for_import():
    """Export tasks in Label Studio import format."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Training tasks
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE) as f:
            training = json.load(f)
        # Label Studio expects a list of {"data": {...}} objects
        tasks = [item for item in training]
        out_path = os.path.join(EXPORT_DIR, "training_tasks.json")
        with open(out_path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Exported {len(tasks)} training tasks → {out_path}")
    else:
        print(f"WARNING: Training file not found: {TRAINING_FILE}")
        print("  Run sample_for_annotation.py first.")

    # Annotation tasks
    if os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE) as f:
            data = json.load(f)
        tasks = data.get("tasks", [])
        out_path = os.path.join(EXPORT_DIR, "annotation_tasks.json")
        with open(out_path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Exported {len(tasks)} annotation tasks → {out_path}")

        # Save gold-check mapping separately (admin only, not for annotators)
        gold_mapping = data.get("gold_check_mapping", {})
        gold_path = os.path.join(EXPORT_DIR, "gold_check_mapping.json")
        with open(gold_path, "w") as f:
            json.dump(gold_mapping, f, indent=2)
        print(f"Exported {len(gold_mapping)} gold-check answers → {gold_path}")
        print("  (Keep this file private — do NOT share with annotators)")
    else:
        print(f"WARNING: Annotation sample not found: {SAMPLE_FILE}")
        print("  Run sample_for_annotation.py first.")


def main():
    parser = argparse.ArgumentParser(description="CogBench Label Studio Setup")
    parser.add_argument("--export", action="store_true",
                        help="Export tasks as JSON files for Label Studio import")
    args = parser.parse_args()

    if args.export:
        export_tasks_for_import()
    else:
        print_setup_instructions()


if __name__ == "__main__":
    main()
