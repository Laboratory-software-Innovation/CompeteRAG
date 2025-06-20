#By Illya Gordyy and ChatGPT


import os
import sys

from pathlib import Path
import pandas as pd
from transformers import logging
import transformers.modeling_utils as mutils
from transformers.utils import import_utils


from collection    import collect_and_structured
from encoding      import build_index
from similarity    import find_similar_ids as find_similar_from_description
from prompts       import solve_competition_with_code, followup_prompt


# Monkey-patches
mutils.check_torch_load_is_safe = lambda *args, **kwargs: None
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
logging.set_verbosity_error()



# ─────────────────────────────────────────────────────────────────────────────
# Command‐Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def print_usage():
    print("""
        Usage:
        python pipeline.py collect_and_structured
        python pipeline.py build_index
        python pipeline.py find_similar
        python pipeline.py find_similar_desc <description.txt> [notebooks|comps] [top_k]

        Examples:
        python pipeline.py find_similar_desc my_problem.txt notebooks 5
        python pipeline.py find_similar_desc my_problem.txt comps 10
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "collect_and_structured":
        df_struct = collect_and_structured(max_per_keyword=5)
        print(f"[OK] Collected and structured {len(df_struct)} notebooks.")
        sys.exit(0)

    elif cmd == "build_index":
        if not Path("notebooks_structured.csv").exists():
            print("[ERROR] Please run `collect_and_structured` first.")
            sys.exit(1)
        df_struct = pd.read_csv("notebooks_structured.csv")
        build_index(df_struct)
        sys.exit(0)
    elif cmd == "find_similar_desc":
        # Usage: python3 pipeline.py find_similar_desc <desc_and_meta.json> <top_k> [<exclude_competition>]
        if len(sys.argv) < 4:
            print("Usage: python3 pipeline.py find_similar_desc <desc_meta.json> <top_k> [<exclude_competition>]")
            sys.exit(1)

        desc_json    = sys.argv[2]
        top_k        = int(sys.argv[3])
        exclude_comp = sys.argv[4] if len(sys.argv) >= 5 else None

        find_similar_from_description(
            desc_json,
            top_k=top_k,
            exclude_competition=exclude_comp
        )
        sys.exit(0)

    elif cmd == "auto_solve_code":
        if len(sys.argv) < 4:
            print("Usage: python pipeline.py auto_solve_code <slug> <class_col> [top_k] [Keras-Tuner True:1|False:0]")
            sys.exit(1)
        slug      = sys.argv[2]
        class_col = sys.argv[3]
        top_k     = int(sys.argv[4]) if len(sys.argv)>4 else 5
        kt = int(sys.argv[5]) if len(sys.argv) > 5 else 0

        # call the new solver
        notebook_code = solve_competition_with_code(
            class_col     = class_col,
            slug          = slug,
            structured_csv= "notebooks_structured.csv",
            top_k         = top_k,
            kt            = kt, 

        )
        # write out the notebook
        out_path = Path(f"{slug}_solution.py")
        out_path.write_text(notebook_code, encoding="utf-8")
        print(f"[OK] Solution code written to {out_path}")

    elif cmd == "followup":
        # Usage: python pipeline.py followup <solution_file.py>
        if len(sys.argv) != 3:
            print("Usage: python pipeline.py followup slug")
            sys.exit(1)
            
        slug = sys.argv[2]
        print(slug)
        try:
            corrected_code = followup_prompt(str(slug))
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        # Write out the corrected version alongside the original
        orig = Path(str(slug))
        fixed = orig.with_name(orig.stem + "_fixed.py")
        fixed.write_text(corrected_code, encoding="utf-8")
        print(f"[OK] Corrected code written to {fixed}")

    else:
        print_usage()
        sys.exit(1)

